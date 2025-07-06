import cv2
import numpy as np
import os
import glob

def calculate_kernel_size(image_shape, factor=0.025):
    kernel_size = int(min(image_shape) * factor)
    kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
    return (kernel_size, kernel_size)
    
def clean_up_mask(binary_mask):
    _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
    kernel_size = calculate_kernel_size(binary_mask.shape, factor=0.025)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed_mask, connectivity=8
    )

    size_threshold = 500 
    aspect_ratio_threshold = 1.1  

    final_mask = np.zeros_like(closed_mask)
    for i in range(1, num_labels):
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(width, height) / (min(width, height) + 1e-5)

        if (
            stats[i, cv2.CC_STAT_AREA] > size_threshold
            or aspect_ratio > aspect_ratio_threshold
        ):
            final_mask[labels == i] = 255

    return final_mask


def run_clean_masks_on_sketch_dir(sketch_dir):
    sam_masks_dir = f"{sketch_dir}/masks"
    if not os.path.exists(sam_masks_dir):
        print(f"Skipping {sam_masks_dir}")
        return
    num_masks = len(glob.glob(f"{sam_masks_dir}/mask_*.png"))
    out_dir = f"{sketch_dir}/masks_cleaned"
    os.makedirs(out_dir, exist_ok=True)
    for mask_i in range(num_masks):
        mask_path = f"{sam_masks_dir}/mask_{mask_i}.png"
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        cleaned_mask = clean_up_mask(mask)
        out_path = f"{out_dir}/mask_{mask_i}.png"
        cv2.imwrite(out_path, cleaned_mask)
    print(f"Processed {num_masks} masks in {sam_masks_dir}")
    return out_dir
