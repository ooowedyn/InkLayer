import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from InkLayer.refinement.utils import compute_bbox_iou

def get_dynamic_threshold(sketch_path, base_threshold=8.0):
    # Read image dimensions
    sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
    h, w = sketch.shape

    # Calculate diagonal length
    diagonal = np.sqrt(w**2 + h**2)

    # Scale threshold to be proportional to diagonal
    # Using 1000 as a reference diagonal length for the base_threshold
    reference_diagonal = 1000
    dynamic_threshold = base_threshold * (diagonal / reference_diagonal)

    return dynamic_threshold


def share_corner(box1, box2, epsilon=5.0):
    """
    Check if two bounding boxes share a corner within an epsilon threshold.

    Args:
        box1: First box in format [x1, y1, x2, y2]
        box2: Second box in format [x1, y1, x2, y2]
        epsilon: Distance threshold for considering corners as shared (in pixels)

    Returns:
        bool: True if boxes share at least one corner, False otherwise
    """
    # Extract corners for each box
    # Format: [(x1,y1), (x1,y2), (x2,y1), (x2,y2)]
    corners1 = [
        (box1[0], box1[1]),  # top-left
        (box1[0], box1[3]),  # bottom-left
        (box1[2], box1[1]),  # top-right
        (box1[2], box1[3]),  # bottom-right
    ]

    corners2 = [
        (box2[0], box2[1]),  # top-left
        (box2[0], box2[3]),  # bottom-left
        (box2[2], box2[1]),  # top-right
        (box2[2], box2[3]),  # bottom-right
    ]

    # Check each pair of corners
    for c1 in corners1:
        for c2 in corners2:
            # Calculate Euclidean distance between corners
            dist = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
            if dist <= epsilon:
                return True

    return False


def refine_mask_to_sketch_regions(input_image_path, pred_mask):
    """
    Refine predicted masks to only include regions overlapping with black pixels in a sketch.
    """
    pred_mask_height, pred_mask_width = pred_mask.shape
    sketch_image = (
        Image.open(input_image_path)
        .convert("L")
        .resize((pred_mask_width, pred_mask_height), Image.Resampling.BILINEAR)
    )

    sketch_array = np.array(sketch_image)
    sketch_mask = sketch_array < 250
    pred_mask = pred_mask.astype(bool)
    refined_mask = np.logical_and(pred_mask, sketch_mask)
    return refined_mask


def filter_empty_bbox(sketch_path, bboxes):
    # Load the sketch to get image dimensions
    sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
    img_height, img_width = sketch.shape
    img_area = img_height * img_width

    # Compute kept indices
    kept_indices = []
    for i, box in enumerate(bboxes):
        x_min, y_min, x_max, y_max = box

        # Ensure box coordinates are within image bounds
        x_min = max(0, min(x_min, img_width - 1))
        y_min = max(0, min(y_min, img_height - 1))
        x_max = max(0, min(x_max, img_width - 1))
        y_max = max(0, min(y_max, img_height - 1))

        # Extract the region defined by the box
        region = sketch[y_min : y_max + 1, x_min : x_max + 1]

        # Calculate the number of non-zero (non-empty) pixels in the region
        non_zero_pixels = np.count_nonzero(region)

        # Include the box if it has any non-empty pixel
        if non_zero_pixels > 0:
            kept_indices.append(i)

    return np.array(kept_indices)

def box_contains(box1, box2):
    """Check if box1 contains box2"""
    return (
        box1[0] <= box2[0]
        and box1[1] <= box2[1]
        and box1[2] >= box2[2]
        and box1[3] >= box2[3]
    )
    
def count_contained_boxes(box, all_boxes):
        """Count how many other boxes are contained within the given box"""
        count = 0
        for other_box in all_boxes:
            # Skip if it's the same box
            if np.array_equal(box, other_box):
                continue
            if box_contains(box, other_box):
                count += 1
        return count
    
def filter_full_or_empty_bbox(
    sketch_path, bboxes, size_threshold=0.9, max_contained_boxes=5
):
    sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
    img_height, img_width = sketch.shape
    img_area = img_height * img_width

    if np.max(bboxes) <= 1.0:
        bboxes = bboxes * np.array([img_width, img_height, img_width, img_height])
        bboxes = bboxes.astype(int)

    # Compute kept indices
    kept_indices = []
    for i, box in enumerate(bboxes):
        # Check full 
        # Compute bbox area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        valid_area = box_area / img_area < size_threshold

        # Check empty
        x_min, y_min, x_max, y_max = box

        # Ensure box coordinates are within image bounds
        x_min = int(max(0, min(x_min, img_width - 1)))
        y_min = int(max(0, min(y_min, img_height - 1)))
        x_max = int(max(0, min(x_max, img_width - 1)))
        y_max = int(max(0, min(y_max, img_height - 1)))

        # Extract the region defined by the box
        region = sketch[y_min : y_max + 1, x_min : x_max + 1]

        # Calculate the number of non-zero (non-empty) pixels in the region
        non_zero_pixels = np.count_nonzero(region)

        # Include the box if it has any non-empty pixel
        valid_sketch_content = non_zero_pixels > 0

        # Check number of contained boxes
        contained_boxes = count_contained_boxes(box, bboxes)
        valid_contained_boxes = contained_boxes <= max_contained_boxes

        # Keep boxes that meet all criteria
        if valid_area and valid_sketch_content and valid_contained_boxes:
            kept_indices.append(i)

    return np.array(kept_indices)


def is_contained_bbox(small_box, big_box, epsilon=5.0):
    return (
        small_box[0] >= big_box[0] - epsilon  # x1 larger
        and small_box[1] >= big_box[1] - epsilon  # y1 larger
        and small_box[2] <= big_box[2] + epsilon  # x2 smaller
        and small_box[3] <= big_box[3] + epsilon
    )  # y2 smaller


def content_iou(sketch_path, bboxes, scores, box1_index, box2_index, masks_dir):
    sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
    h, w = sketch.shape
    # Convert box coordinates to float and scale up if necessary
    box1 = bboxes[box1_index].astype(float)
    box2 = bboxes[box2_index].astype(float)
    score1 = scores[box1_index]
    score2 = scores[box2_index]
    # If coordinates are normalized (all values <= 1), scale them up
    if np.all(box1 <= 1.0) and np.all(box2 <= 1.0):
        box1 = box1 * np.array([w, h, w, h])
        box2 = box2 * np.array([w, h, w, h])

    # Calculate bbox areas instead of mask areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Determine which box is larger based on bbox area
    if area1 > area2:
        larger_box = box1
        smaller_box = box2
        larger_index = box1_index
        smaller_index = box2_index
        mask1_path = f"{masks_dir}/mask_{box1_index}.png"
        mask2_path = f"{masks_dir}/mask_{box2_index}.png"
        larger_score = score1
        smaller_score = score2
    else:
        larger_box = box2
        smaller_box = box1
        larger_index = box2_index
        smaller_index = box1_index
        mask1_path = f"{masks_dir}/mask_{box2_index}.png"
        mask2_path = f"{masks_dir}/mask_{box1_index}.png"
        larger_score = score2
        smaller_score = score1

    # Load masks after determining larger/smaller
    mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)
    larger_mask = refine_mask_to_sketch_regions(sketch_path, mask1)
    smaller_mask = refine_mask_to_sketch_regions(sketch_path, mask2)

    # Calculate intersection and union
    intersection = (larger_mask > 0) & (smaller_mask > 0)
    union = (larger_mask > 0) | (smaller_mask > 0)
    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)
    sketch_iou = intersection_sum / union_sum if union_sum > 0 else 0.0  # sketch IoU

    containment_threshold = get_dynamic_threshold(sketch_path)
    bbox_contained = is_contained_bbox(
        smaller_box, larger_box, epsilon=containment_threshold
    )
    bbox_share_corner = share_corner(
        smaller_box, larger_box, epsilon=containment_threshold
    )
    bbox_iou = compute_bbox_iou(smaller_box, larger_box)

    if not bbox_contained or not bbox_share_corner:
        # Invalid merge, zero out
        return 0.0, 0.0, larger_index

    better_idx = larger_index if larger_score > smaller_score else smaller_index

    return sketch_iou, bbox_iou, better_idx


def compute_ious(sketch_path, bboxes, scores, box1_index, remaining_indices, masks_dir):
    sketch_ious = []
    bbox_ious = []
    larger_indices = []  # Track which box to keep for each comparison
    for box2_index in remaining_indices:
        sketch_iou, bbox_iou, larger_index = content_iou(
            sketch_path, bboxes, scores, box1_index, box2_index, masks_dir
        )

        sketch_ious.append(sketch_iou)
        bbox_ious.append(bbox_iou)
        larger_indices.append(larger_index)

    # Convert to numpy arrays for consistency
    return np.array(sketch_ious), np.array(bbox_ious), np.array(larger_indices)

def sketch_nms(
    sketch_path: str,
    bboxes: np.ndarray,
    scores: np.ndarray,
    masks_dir: str,
    sketch_iou_threshold: float,
    bbox_iou_threshold=0.7,
) -> np.ndarray:
    if len(bboxes) == 0:
        return np.array([])

    # Filter out boxes that cover the entire sketch
    sketch_kept_box_indices = filter_full_or_empty_bbox(sketch_path, bboxes)
    if len(sketch_kept_box_indices) == 0:
        return np.array([])
    filtered_bboxes = bboxes[sketch_kept_box_indices]
    filtered_scores = scores[sketch_kept_box_indices]

    # Sort boxes by score, highest first
    order = np.argsort(-filtered_scores)

    # Create a mapping from filtered indices to original indices
    original_indices = sketch_kept_box_indices[order]

    indices = np.arange(len(filtered_bboxes))
    keep = np.ones_like(indices, dtype=bool)
    print(f"Starting NMS")

    # Perform NMS
    for index_idx in tqdm(range(len(indices))):
        i = indices[index_idx]
        if keep[i]:
            remaining_indices = order[i + 1:]  # Get indices of remaining boxes (these are filtered indices)
            if len(remaining_indices) > 0:
                ious_sketch, ious_bbox, larger_indices = compute_ious(
                    sketch_path,
                    filtered_bboxes,
                    filtered_scores,
                    order[i],  # This is a filtered index
                    remaining_indices,  # These are filtered indices
                    masks_dir,
                )
                overlapped = np.where(
                    np.logical_or(
                        ious_sketch > sketch_iou_threshold,
                        ious_bbox > bbox_iou_threshold,
                    )
                )[0]

                # For each overlapped box, check if we should keep the larger one
                for idx, overlap_idx in enumerate(overlapped):
                    current_filtered_idx = order[i]  # Current box (filtered index)
                    compared_filtered_idx = remaining_indices[overlap_idx]  # Compared box (filtered index)
                    larger_filtered_idx = larger_indices[overlap_idx]  # Larger box (filtered index)
                    
                    # Get original indices for logging
                    current_original_idx = original_indices[i]
                    compared_original_idx = sketch_kept_box_indices[compared_filtered_idx]

                    # If the larger box is the one we're comparing against
                    if larger_filtered_idx == compared_filtered_idx:
                        # Keep the larger box (compared) and remove the smaller one (current)
                        keep[i] = False  # Remove current box using filtered index
                        print(f"Removed original index {current_original_idx} due to overlap with {compared_original_idx}")
                        break  # Exit loop since current box is removed
                    else:
                        # Keep the current box and remove the compared one
                        # Find the position of compared_filtered_idx in the order array
                        compared_position = np.where(order == compared_filtered_idx)[0][0]
                        keep[compared_position] = False  # Remove compared box using its position
                        print(f"Removed original index {compared_original_idx} due to overlap with {current_original_idx}")

    # Map the kept indices back to original indices
    final_kept_indices = original_indices[keep]
    # Print out the removed original indices for debugging
    removed_indices = original_indices[~keep]
    print(f"Removed original indices: {removed_indices}")

    print(
        f"Filtered {len(filtered_bboxes)} to {len(final_kept_indices)} boxes with Sketch NMS"
    )
    return final_kept_indices
