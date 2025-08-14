import numpy as np
import cv2
from PIL import Image
from scipy import ndimage
from skimage.morphology import binary_dilation, binary_closing, disk
from scipy.ndimage import label
from skimage.segmentation import watershed
import os
import json
import subprocess

import InkLayer
proj_dir = os.path.dirname(InkLayer.__file__)
from InkLayer.utils.visualization import generate_pastel_colors, color_sketch_by_masks
from InkLayer.refinement.utils import compute_bbox_iou, compute_mask_bbox, unnormalize_bboxes
from InkLayer.refinement.depth_sort import get_depth_map, sort_sketch_masks
SKETCH_THRESHOLD = 250 # Change this if you want stricter sketch detection


def clean_delicate_mask(mask, isolation_threshold=1, window_size=3):
    # Create a copy to avoid modifying the original
    cleaned = mask.copy()

    # Count neighbors for each pixel
    kernel = np.ones((window_size, window_size), dtype=bool)
    kernel[window_size // 2, window_size // 2] = False  # Don't count the center pixel
    neighbor_count = ndimage.convolve(mask.astype(int), kernel, mode="constant", cval=0)

    # Remove pixels with very few neighbors
    cleaned[neighbor_count <= isolation_threshold] = False

    return cleaned

def composite_and_parse_masks(masks, bboxes, empty_threshold=0.05):
    if not masks:
        return [], []

    height, width = masks[0].shape
    composite = np.zeros((height, width), dtype=np.uint8)
    original_areas = [np.sum(mask > 0) for mask in masks]

    # Create composite by layering masks
    for i in range(len(masks) - 1, -1, -1):
        mask = masks[i]
        composite[mask > 0] = i + 1  # Add 1 to avoid using 0 as a label

    unique_labels = np.unique(composite)[1:]  # Skip background (0)
    parsed_masks = [(composite == label) for label in unique_labels]

    # Track mask info using the actual label values minus 1 (to get back to 0-based index)
    mask_info = [
        {"bbox": bboxes[label - 1], "original_indices": [label - 1]}
        for label in unique_labels
    ]

    final_masks = []
    final_mask_info = []

    for i, (parsed_mask, info) in enumerate(zip(parsed_masks, mask_info)):
        parsed_area = np.sum(parsed_mask)
        original_idx = info["original_indices"][0]

        if parsed_area < empty_threshold * original_areas[original_idx]:
            max_overlap = 0
            best_merge_idx = None
            original_mask = masks[original_idx]

            # Find the mask with maximum overlap that comes before the current mask
            for j in range(original_idx):
                overlap = np.sum(np.logical_and(original_mask, masks[j]))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_merge_idx = j

            if best_merge_idx is not None:
                merge_label = best_merge_idx + 1  # Use the label of the earlier mask
                merge_mask = composite == merge_label

                # Merge the masks
                merged = np.logical_or(merge_mask, original_mask)
                composite[merged] = merge_label
                continue

        final_masks.append(parsed_mask)
        final_mask_info.append(info)

    return final_masks, final_mask_info


def parse_masks_to_disjoint_masks(masks_np, bboxes, sketch_path, depth_map=None):
    sorted_bboxes_indices, depth_scores, containment = sort_sketch_masks(
        masks_np, bboxes, sketch_path, depth_sketch=depth_map
    )

    sorted_masks = [masks_np[i] for i in sorted_bboxes_indices]
    sorted_bboxes = [bboxes[i] for i in sorted_bboxes_indices]
    assert len(sorted_masks) == len(masks_np)

    # Check if any of the masks cover the majority of the sketch, if so, we remove it
    # This is to prevent the case where the mask covers the entire sketch
    num_masks = len(sorted_masks)
    sketch = Image.open(sketch_path).convert("L")
    sketch_array = np.array(sketch)
    sketch_area = np.sum(sketch_array < SKETCH_THRESHOLD)
    for i, mask in enumerate(sorted_masks):
        mask_sketch = np.logical_and(mask > 0, sketch_array < SKETCH_THRESHOLD)
        mask_area = np.sum(mask_sketch > 0)
        if num_masks > 1 and mask_area > 0.9 * sketch_area:
            sorted_masks[i] = np.zeros_like(mask)
            num_masks -= 1
    
    disjoint_masks, mask_info = composite_and_parse_masks(sorted_masks, sorted_bboxes)
    cleaned_masks = [clean_delicate_mask(mask) for mask in disjoint_masks]

    # Convert mask info to use original bbox indices
    final_mask_info = []
    for info in mask_info:
        original_indices = [sorted_bboxes_indices[i] for i in info["original_indices"]]
        final_mask_info.append(
            {
                "bbox": info["bbox"],  # This is already the original bbox
                "original_indices": original_indices,
            }
        )

    return cleaned_masks, sorted_bboxes, final_mask_info


def refine_masks_with_watershed(sketch_image, original_masks, debug=False):
    """
    Refine existing masks by expanding them to cover unlabeled sketch pixels.
    """
    # Convert sketch to binary (ensure black pixels are True)
    sketch_binary = ~(sketch_image > SKETCH_THRESHOLD)

    # Initialize markers
    markers = np.full(sketch_binary.shape, -1, dtype=int)

    # Create combined mask of all original regions
    combined_mask = np.zeros_like(sketch_binary, dtype=bool)
    for mask in original_masks:
        combined_mask |= mask

    # Find unlabeled black pixels
    unlabeled_black = sketch_binary & ~combined_mask

    # Close small gaps in unlabeled regions to help identify continuous areas
    unlabeled_closed = binary_closing(unlabeled_black, disk(3))

    # Label connected components in unlabeled regions
    labeled_regions, num_regions = label(unlabeled_closed)

    # Find sizes of each region
    region_sizes = np.bincount(labeled_regions.ravel())[1:]  # Skip background
    large_regions = np.zeros_like(unlabeled_black, dtype=bool)

    # Mark regions larger than threshold as large regions
    for i, size in enumerate(region_sizes, start=1):
        if size > 50:  # Adjust threshold as needed
            large_regions |= labeled_regions == i

    # Add original masks as markers with more aggressive dilation
    for i, mask in enumerate(original_masks, start=1):
        # More aggressive dilation near large regions
        dilate_size = 3 if np.any(binary_dilation(mask, disk(3)) & large_regions) else 2
        dilated = binary_dilation(mask, disk(dilate_size))
        # Only add dilated pixels that are black and unlabeled
        new_pixels = dilated & unlabeled_black
        markers[new_pixels] = i
        markers[mask] = i

    # Create enhanced distance transform
    distance = ndimage.distance_transform_edt(unlabeled_black)

    # Enhance distance transform in large regions
    distance = np.where(
        large_regions, distance * 3, distance
    )  # Stronger weight for large regions

    # Make it negative and scale
    distance = -distance

    # Add gradient component with less influence in large regions
    gradient = ndimage.gaussian_gradient_magnitude(sketch_binary.astype(float), sigma=1)
    gradient = np.where(large_regions, gradient * 0.01, gradient * 0.1)
    distance += gradient

    # Apply watershed with very low compactness for large regions
    labels = watershed(distance, markers, mask=sketch_binary, compactness=0.01)
    # Create refined masks
    refined_masks = []
    for i, original_mask in enumerate(original_masks, start=1):
        refined_mask = labels == i
        refined_masks.append(refined_mask)

    return refined_masks


def match_masks_to_boxes(masks, boxes):
    """
    Match masks to boxes using IoU between mask bounding boxes and input boxes.
    Returns a mapping from box index to mask index.
    """
    mask_boxes = [compute_mask_bbox(mask) for mask in masks]
    mask_boxes = [box for box in mask_boxes if box is not None]

    # Compute IoU matrix
    iou_matrix = np.zeros((len(boxes), len(mask_boxes)))
    for i, box in enumerate(boxes):
        for j, mask_box in enumerate(mask_boxes):
            iou_matrix[i, j] = compute_bbox_iou(box, mask_box)

    # Match boxes to masks using highest IoU
    box_to_mask = {}
    if iou_matrix.size == 0:
        return None
    while True:
        if np.max(iou_matrix) == 0:
            break
        box_idx, mask_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        box_to_mask[box_idx] = mask_idx
        iou_matrix[box_idx, :] = 0
        iou_matrix[:, mask_idx] = 0

    return box_to_mask


def refine_masks_with_boxes(sketch_image_path, original_masks, boxes):
    """
    Refine existing masks by assigning unlabeled black pixels to masks based on bounding box containment.
    Automatically matches masks to boxes based on overlap.

    Args:
        sketch_image: Binary image where black pixels are True
        original_masks: List of boolean masks
        boxes: List of [x1, y1, x2, y2] bounding boxes
    """
    # Convert sketch to binary (ensure black pixels are True)
    sketch_image = np.array(Image.open(sketch_image_path).convert("L"))
    sketch_binary = ~(sketch_image > SKETCH_THRESHOLD)
    sorted_bboxes_indices = range(len(boxes))  # already sorted from disjoint step

    # Match masks to boxes using IoU
    box_to_mask = match_masks_to_boxes(original_masks, boxes)
    if box_to_mask is None:
        return original_masks

    # Create combined mask of all original regions
    combined_mask = np.zeros_like(sketch_binary, dtype=bool)
    for mask in original_masks:
        combined_mask |= mask

    # Find unlabeled black pixels
    unlabeled_black = sketch_binary & ~combined_mask

    # Create refined masks starting with original masks
    refined_masks = [mask.copy() for mask in original_masks]

    # For each unlabeled pixel, assign it to the most appropriate mask based on boxes
    y_coords, x_coords = np.where(unlabeled_black)
    for y, x in zip(y_coords, x_coords):
        # Check which boxes contain this point
        containing_boxes = []
        for box_idx in sorted_bboxes_indices:
            x1, y1, x2, y2 = boxes[box_idx]
            if x1 <= x <= x2 and y1 <= y <= y2:
                containing_boxes.append(box_idx)

        if containing_boxes:
            # If point is in multiple boxes, assign to the mask with nearest filled pixels
            if len(containing_boxes) > 1:
                min_dist = float("inf")
                best_box_idx = None

                for box_idx in containing_boxes:
                    if box_idx not in box_to_mask:
                        continue
                    mask_idx = box_to_mask[box_idx]
                    # Calculate distance to nearest True pixel in this mask
                    mask_coords = np.where(refined_masks[mask_idx])
                    if len(mask_coords[0]) > 0:  # only if mask has any pixels
                        mask_y, mask_x = mask_coords
                        distances = np.sqrt((mask_y - y) ** 2 + (mask_x - x) ** 2)
                        min_mask_dist = np.min(distances)
                        if min_mask_dist < min_dist:
                            min_dist = min_mask_dist
                            best_box_idx = box_idx

                if best_box_idx is not None and best_box_idx in box_to_mask:
                    refined_masks[box_to_mask[best_box_idx]][y, x] = True
            else:
                # If point is in only one box, assign to that mask if mapping exists
                box_idx = containing_boxes[0]
                if box_idx in box_to_mask:
                    refined_masks[box_to_mask[box_idx]][y, x] = True

    return refined_masks



def create_unlabeled_mask(sketch_path, masks):
    sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
    sketch_mask = (sketch < SKETCH_THRESHOLD).astype(np.uint8)

    # Initialize combined mask for all labeled regions
    if len(masks) > 0:
        labeled_mask = np.zeros_like(masks[0], dtype=np.uint8)

        # Combine all existing masks
        for mask in masks:
            labeled_mask = np.logical_or(labeled_mask, mask)
    else:
        labeled_mask = np.zeros_like(sketch_mask, dtype=np.uint8)

    # Find unlabeled sketch pixels
    unlabeled_mask = np.logical_and(sketch_mask, np.logical_not(labeled_mask)).astype(
        np.uint8
    )

    # Clean up the unlabeled mask using morphological operations
    kernel_open = np.ones((3, 3), np.uint8)
    unlabeled_mask = cv2.morphologyEx(unlabeled_mask, cv2.MORPH_OPEN, kernel_open)

    # Dilate slightly to restore some size
    kernel_dilate = np.ones((2, 2), np.uint8)
    unlabeled_mask = cv2.dilate(unlabeled_mask, kernel_dilate, iterations=1)

    if np.sum(unlabeled_mask) == 0:
        print("No unlabeled pixels found in the sketch.")
        return masks

    # Add the new mask to the list
    result_masks = list(masks)  
    result_masks.append(unlabeled_mask)

    return result_masks


def improve_sam_masks(sketch_image_path, masks_np, bboxes):
    sketch_image_pil = Image.open(sketch_image_path).convert("RGB")
    # print(f"Image size: {sketch_image_pil.size}")
    pastel_colors = generate_pastel_colors(len(masks_np))
    initial_seg_sketch = color_sketch_by_masks(
        sketch_image_pil, masks_np, pastel_colors
    )

    masks_numpy = [mask.astype(bool) for mask in masks_np]
    sketch_numpy = np.array(sketch_image_pil.convert("L"))
    watershed_masks = refine_masks_with_watershed(
        sketch_numpy,
        masks_numpy,
    )
    print(f"Finished watershed refinement")
    watershed_sketch = color_sketch_by_masks(
        sketch_image_pil, watershed_masks, pastel_colors
    )
    bbox_masks = refine_masks_with_boxes(sketch_image_path, watershed_masks, bboxes)
    print(f"Finished bbox refinement")
    final_masks = create_unlabeled_mask(sketch_image_path, bbox_masks) 
    final_sketch = color_sketch_by_masks(
        sketch_image_pil, final_masks, generate_pastel_colors(len(final_masks))
    )

    res_obj = {
        "initial_seg_sketch": initial_seg_sketch,
        "watershed": watershed_sketch,
        "final_seg_sketch": final_sketch,
        "final_masks": final_masks,
    }
    return res_obj


def run_refinement_on_sketch_dir(sketch_dir, bboxes_path, out_base_dir=None):
    if not os.path.exists(sketch_dir):
        print(f"Sketch directory {sketch_dir} does not exist.")
        return

    masks_dir = f"{sketch_dir}/masks_cleaned"
    sketch_path = f"{sketch_dir}/input.png"
    sketch_image = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
    with open(bboxes_path, "r") as f:
        bboxes_data = json.load(f)
    assert len(bboxes_data["bboxes"]) == len(bboxes_data["kept_indices"])
    bboxes = bboxes_data["bboxes"]
    h, w = sketch_image.shape
    bboxes = unnormalize_bboxes(bboxes, h, w)
    kept_indices = bboxes_data["kept_indices"]
    masks_paths = [f"{masks_dir}/mask_{index}.png" for index in kept_indices]
    cleaned_SAM_masks = [
        cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for mask_path in masks_paths
    ]
    depth_map = get_depth_map(sketch_path)
    disjoint_masks_np, sorted_bboxes, final_mask_info = parse_masks_to_disjoint_masks(
        cleaned_SAM_masks, bboxes, sketch_path, depth_map
    )
    
    if out_base_dir is None:
        out_base_dir = sketch_dir 
    
    # Save all disjoint masks
    disjoint_masks_dir = f"{out_base_dir}/masks_disjoint"
    subprocess.run(["rm", "-rf", disjoint_masks_dir])
    os.makedirs(disjoint_masks_dir, exist_ok=True)  
    for i, mask in enumerate(disjoint_masks_np):
        mask_path = f"{disjoint_masks_dir}/mask_{i}.png"
        cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)
    
    res = improve_sam_masks(sketch_path, disjoint_masks_np, sorted_bboxes)

    # Save results
    out_dir = f"{out_base_dir}/masks_final"
    if os.path.exists(out_dir):
        for f in os.listdir(out_dir):
            os.remove(f"{out_dir}/{f}")
    os.makedirs(out_dir, exist_ok=True)
    for i, mask in enumerate(res["final_masks"]):
        mask_path = f"{out_dir}/mask_{i}.png"
        cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)
    # need to normalize depth map to 0-255
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    Image.fromarray(depth_map_normalized).convert("RGB").save(
        f"{out_base_dir}/depth_map.png"
    )
    res["final_seg_sketch"].save(f"{out_base_dir}/segmented_sketch_final.png")

    print(f"Results saved to {out_dir}")
    print(f"Segmented sketch visualized at {out_base_dir}/segmented_sketch_final.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Refine SAM masks on a sketch directory.")
    parser.add_argument("--sketch_dir", type=str, help="Directory containing the sketch and masks.")
    parser.add_argument("--bboxes_path", type=str, help="Path to the bounding boxes JSON file.")
    parser.add_argument("--out_base_dir", type=str, default=None, help="Base directory for output results.")
    
    args = parser.parse_args()
    
    run_refinement_on_sketch_dir(args.sketch_dir, args.bboxes_path, args.out_base_dir)