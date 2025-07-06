import numpy as np

def sketch_to_01binary(sketch):
    if len(sketch.shape) == 2:
        sketch = sketch[:, :, np.newaxis]
    # returns a binary image where pixels on the edge map are 1 (white)
    thresh = sketch.max() / 2
    binary_sketch = 1.0 * ~(sketch > thresh)
    return binary_sketch[:, :, 0]

def compute_bbox_iou(box1, box2):
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Check if boxes intersect
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    area_i = (x2_i - x1_i) * (y2_i - y1_i)
    area_u = area_1 + area_2 - area_i
    iou = area_i / area_u

    return iou

def compute_mask_bbox(mask):
    """Compute bounding box for a binary mask."""
    y_coords, x_coords = np.where(mask)
    if len(y_coords) == 0:
        return None
    return [np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords)]

def unnormalize_bboxes(bboxes, h, w):
    unnorm_bboxes = []
    for box in bboxes:
        unnorm_box = [
            int(box[0] * w),
            int(box[1] * h),
            int(box[2] * w),
            int(box[3] * h),
        ]
        unnorm_bboxes.append(unnorm_box)
    return unnorm_bboxes

def get_binned_frequent(depth_values, bin_width=0.1):
    depth_array = np.array(depth_values)
    # Round values to handle floating point differences
    binned = np.round(depth_array / bin_width) * bin_width
    # Get most common value
    values, counts = np.unique(binned, return_counts=True)
    return values[np.argmax(counts)]

