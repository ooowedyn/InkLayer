import numpy as np
import cv2
import torch
from scipy.spatial import KDTree
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
from InkLayer.third_party.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
from InkLayer.utils.visualization import generate_pastel_colors, color_sketch_by_masks
from InkLayer.refinement.utils import compute_bbox_iou, compute_mask_bbox, sketch_to_01binary, get_binned_frequent, unnormalize_bboxes

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
depth_model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}

encoder = "vitb"  # or 'vits', 'vitb', 'vitg'
depth_model = DepthAnythingV2(**depth_model_configs[encoder])
depth_model.load_state_dict(
    torch.load(f"{proj_dir}/../models/depth_anything_v2_{encoder}.pth", map_location="cpu")
)
depth_model = depth_model.to(DEVICE).eval()

SKETCH_THRESHOLD=250 # This is the threshold to consider a pixel as part of the sketch

def get_depth_map(sketch_path):
    sketch = cv2.imread(sketch_path)
    depth = depth_model.infer_image(sketch)  # HxW raw depth map in numpy
    return depth


def sparse_sketch_sample(binary_edge_map):
    radius = binary_edge_map.shape[0] * 0.01
    # Get all edge coordinates
    edge_points = np.column_stack(np.where(binary_edge_map > 0))
    # number of points we will sample on the edge map
    num_points = int(0.05 * len(edge_points))
    # Convert edge points to a KDTree for efficient distance queries
    tree = KDTree(edge_points)
    sampled_points = []
    remaining_indices = set(range(len(edge_points)))
    while remaining_indices:
        # Take the first available point
        current_index = next(iter(remaining_indices))
        current_point = edge_points[current_index]
        sampled_points.append(tuple(current_point))
        # Find all points within the radius of the current point
        indices_to_exclude = tree.query_ball_point(current_point, radius)
        # Remove these indices from the remaining set
        remaining_indices.difference_update(indices_to_exclude)
    return sampled_points



def get_mask_depth_score(mask, edge_points, depth_map):
    depth_values = []
    depth_value_points = []

    # 1. First get depth from edge points that lie within mask
    for point in edge_points:
        y, x = point
        if mask[y, x]:
            depth_values.append(depth_map[y, x])
            depth_value_points.append([y, x])

    if not depth_values:
        return float("inf")  # Return infinity if no points in mask

    res = get_binned_frequent(depth_values)
    return res


def build_containment_graph(masks, bboxes, depth_scores, image_size):
    """
    Build a directed graph where edges indicate containment, using dynamic epsilon.

    Args:
        masks: List of binary masks
        bboxes: List of bounding boxes (x1,y1,x2,y2)
        depth_scores: List of depth scores for each mask
        image_size: Tuple of (height, width) of the image

    Returns:
        containment_graph: Boolean matrix where containment_graph[i,j] indicates if box i contains box j
    """
    n = len(masks)
    containment_graph = np.zeros((n, n), dtype=bool)

    # Calculate epsilon as a fraction of the image diagonal
    image_diagonal = np.sqrt(image_size[0] ** 2 + image_size[1] ** 2)
    epsilon = image_diagonal * 0.05  # 1% of image diagonal

    for i in range(n):
        for j in range(n):
            if i != j:
                # Check if box i contains box j
                box_i = bboxes[i]
                box_j = bboxes[j]

                # Containment check with epsilon tolerance
                is_contained = (
                    box_i[0] - epsilon <= box_j[0]  # left edge
                    and box_i[1] - epsilon <= box_j[1]  # top edge
                    and box_i[2] + epsilon >= box_j[2]  # right edge
                    and box_i[3] + epsilon >= box_j[3]  # bottom edge
                )

                # Only set containment if the boxes are actually different in size
                # This prevents marking boxes of very similar dimensions as containing each other
                box_i_size = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
                box_j_size = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
                size_diff = abs(box_i_size - box_j_size)

                if is_contained and size_diff > (
                    epsilon * epsilon
                ):  # Use squared epsilon for area comparison
                    containment_graph[i, j] = True

                iou = compute_bbox_iou(box_i, box_j)
                if iou > 0.7:
                    containment_graph[i, j] = True

    return containment_graph

def build_containment_graph_fast(
    masks,
    bboxes,
    depth_scores,
    image_size,
    eps_frac: float = 0.05,   # fraction of image diagonal used as ε
    iou_thr:  float = 0.7,    # IoU threshold that also triggers containment
):
    if len(bboxes) == 0:
        return np.zeros((0, 0), dtype=bool)

    b = np.asarray(bboxes, dtype=float)           
    N = b.shape[0]
    H, W = image_size
    eps  = np.hypot(H, W) * eps_frac             

    # broadcast (N,1,4) vs (1,N,4)
    b1, b2 = b[:, None, :], b[None, :, :]

    # Check if boxes are contained within each otherå
    contained = (
        (b1[..., 0] - eps <= b2[..., 0]) &    # left
        (b1[..., 1] - eps <= b2[..., 1]) &    # top
        (b1[..., 2] + eps >= b2[..., 2]) &    # right
        (b1[..., 3] + eps >= b2[..., 3])      # bottom
    )

    # Ensure that boxes are not too similar in size
    areas      = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])     
    size_diff  = np.abs(areas[:, None] - areas[None, :])       
    contained &= size_diff > (eps ** 2)                       

    # IoU matrix (vectorised)
    ix1 = np.maximum(b1[..., 0], b2[..., 0])
    iy1 = np.maximum(b1[..., 1], b2[..., 1])
    ix2 = np.minimum(b1[..., 2], b2[..., 2])
    iy2 = np.minimum(b1[..., 3], b2[..., 3])

    iw  = np.clip(ix2 - ix1, 0, None)
    ih  = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih

    area1 = areas[:, None]
    area2 = areas[None, :]
    iou   = inter / (area1 + area2 - inter + 1e-6)

    # final containment graph
    graph = contained | (iou > iou_thr)

    # no self-edges
    np.fill_diagonal(graph, False)
    return graph

def sort_sketch_masks(masks, bboxes, sketch_path, depth_sketch=None):
    # Validate inputs
    assert os.path.exists(sketch_path), f"Sketch path {sketch_path} does not exist."
    if depth_sketch is None:
        depth_sketch = get_depth_map(sketch_path)

    # Get image dimensions
    image = cv2.imread(sketch_path)
    h, w = image.shape[:2]

    # Convert sketch to binary and sample points
    binary_sketch = sketch_to_01binary(image)
    sampled_points = sparse_sketch_sample(binary_sketch)

    # Scale bboxes if normalized
    if np.all(np.array(bboxes) <= 1.0):
        bboxes = [box * np.array([w, h, w, h]) for box in bboxes]

    # Get initial depth scores
    depth_scores = []
    for mask in masks:
        score = get_mask_depth_score(mask, sampled_points, depth_sketch)
        depth_scores.append(score)

    # Build containment relationships
    containment = build_containment_graph_fast(masks, bboxes, depth_scores, image.shape[:2])

    # Get initial sorting based on depth scores
    final_order = list(np.argsort(depth_scores)[::-1])
    for _ in range(2):
        for i in range(len(final_order)):
            for j in range(i + 1, len(final_order)):
                idx1 = final_order[i]
                idx2 = final_order[j]

                # ONLY check if the earlier box contains the later box
                if containment[idx1, idx2]:  # if earlier contains later
                    final_order[i], final_order[j] = final_order[j], final_order[i]

    return final_order, depth_scores, containment


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

