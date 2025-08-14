import cv2
import numpy as np
import glob
from PIL import Image
import os
import subprocess

from InkLayer.inpainting.fill_object_bg_mask import get_mask

def create_sketch_layer_file_for_background_mask(mask_path, output_path):
    """
    Create a temporary sketch file that can be used with the background mask function.
    This converts the mask back to a sketch-like format.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # Invert the mask to create a sketch-like image (black strokes on white background)
    sketch = cv2.bitwise_not(mask)
    cv2.imwrite(output_path, sketch)
    return output_path


def assemble_inpaint_input_at_index(masks_dir, mask_index):
    mask_path = f"{masks_dir}/mask_{mask_index}.png"
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_bbox = mask_to_bbox(mask)
    num_masks = len(glob.glob(f"{masks_dir}/mask_*"))
    # rgb_image = cv2.cvtColor(
    #     ~mask, cv2.COLOR_GRAY2BGR
    # )  # Invert mask to get sketch instead of mask
    original_sketch_path = f"{masks_dir}/../input.png"
    rgb_image = cv2.imread(original_sketch_path, cv2.IMREAD_COLOR)
    # only keep the masked region
    mask = mask.astype(bool)
    rgb_image[~mask] = 255  # Set non-masked regions to white

    if mask_index == 0:  # infront
        return None, rgb_image, None, False, None

    overlap_indices = []
    for i in range(num_masks):
        if i == mask_index:
            continue
        
        if i > mask_index:
            # Skip masks that are after the current mask
            break

        other_mask_path = f"{masks_dir}/mask_{i}.png"
        other_mask = cv2.imread(other_mask_path, cv2.IMREAD_GRAYSCALE)
        other_bbox = mask_to_bbox(other_mask)
        sketch_in_other_bbox = mask_within_bbox(mask, other_bbox)
        bbox_overlap = np.sum(sketch_in_other_bbox) > 0

        if bbox_overlap:
            overlap_indices.append(i)

    if len(overlap_indices) == 0:
        return mask, rgb_image, mask, False, None  # Dummy

    print(f"Processing overlaps for mask {mask_index}: {overlap_indices}")
    
    # Create background masks using the sophisticated algorithm instead of convex hull
    background_masks = []
    for idx in overlap_indices:
        overlap_mask_path = f"{masks_dir}/mask_{idx}.png"
        
        # Create a temporary sketch file for the background mask function
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_sketch_path = tmp.name
        
        try:
            # Convert mask to sketch format for background detection
            create_sketch_layer_file_for_background_mask(overlap_mask_path, temp_sketch_path)
            
            # Get background mask using the sophisticated algorithm
            background_mask, mask_type = create_background_mask_from_sketch(
                temp_sketch_path,
                dilate_iter=10,
                kernel_size=5,
                safety_margin=1,
                stroke_thick=2,
                border_band=3
            )
            background_masks.append(background_mask)
            print(f"Created background mask for overlap {idx}: {mask_type}")
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_sketch_path):
                os.remove(temp_sketch_path)
    
    # Combine all background masks
    combined_background_mask = combine_masks(background_masks)
    
    # Restrict to the current mask's bounding box
    overlap_in_mask_bbox = mask_within_bbox(combined_background_mask, mask_bbox)
    overlap_in_mask_bbox[mask.astype(bool)] = False  # Make sure our mask region is not edited
    
    # Create original sketch mask (non-white pixels from the original sketch)
    original_sketch_mask = (rgb_image[:, :, 0] < 255) | (rgb_image[:, :, 1] < 255) | (rgb_image[:, :, 2] < 255)
    
    rgba_image = mask_transparent_region(rgb_image, overlap_in_mask_bbox)
    edit_highlight_image = create_red_masked_region(mask.astype(bool), overlap_in_mask_bbox)

    return overlap_in_mask_bbox, rgb_image, edit_highlight_image, True, original_sketch_mask


def composite_original_sketch_onto_inpainted(inpainted_image, original_sketch_image, original_sketch_mask):
    """
    Composite the original sketch content onto the inpainted image.
    
    Args:
        inpainted_image: PIL Image from SDXL inpainting
        original_sketch_image: Original sketch as numpy array (BGR)
        original_sketch_mask: Boolean mask indicating where original sketch content exists
        
    Returns:
        PIL Image with original sketch content preserved
    """
    # Convert PIL image to numpy array (RGB)
    inpainted_array = np.array(inpainted_image)
    
    # Convert original sketch from BGR to RGB
    original_sketch_rgb = cv2.cvtColor(original_sketch_image, cv2.COLOR_BGR2RGB)
    
    # Create the final composited image
    final_image = inpainted_array.copy()
    
    # Overlay original sketch content where the mask indicates
    final_image[original_sketch_mask] = original_sketch_rgb[original_sketch_mask]
    
    return Image.fromarray(final_image)



def mask_within_bbox(mask, bbox):
    """
    Modify a mask to only contain True values within the specified bbox.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W)
        bbox (list): Bounding box coordinates [x1, y1, x2, y2]

    Returns:
        np.ndarray: Modified mask with True values only inside bbox
    """
    x1, y1, x2, y2 = bbox

    # Create a copy to avoid modifying the original mask
    modified_mask = mask.copy()

    # Set everything outside the bbox to False
    modified_mask[:y1, :] = False  # Above bbox
    modified_mask[y2:, :] = False  # Below bbox
    modified_mask[:, :x1] = False  # Left of bbox
    modified_mask[:, x2:] = False  # Right of bbox

    return modified_mask


def mask_transparent_region(img_rgb, mask):
    """
    Create an RGBA image where the masked region is transparent.
    """
    # Create alpha channel from inverted mask
    # Convert inverted mask to uint8 (0 or 255)
    alpha = (~mask).astype(np.uint8) * 255

    # Stack RGB channels and alpha channel
    rgba = np.dstack((img_rgb, alpha))

    return rgba


def combine_masks(masks):
    if not masks:
        raise ValueError("Empty list of masks provided")

    # Get shape from first mask
    height, width = masks[0].shape

    # Initialize combined mask with zeros
    combined_mask = np.zeros((height, width), dtype=bool)

    # Combine masks using OR operation
    for mask in masks:
        # Verify mask dimensions match
        if mask.shape != (height, width):
            raise ValueError(
                f"Mask shape mismatch. Expected {(height, width)}, got {mask.shape}"
            )
        combined_mask |= mask

    return combined_mask


def mask_to_bbox(mask):
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    indices = np.where(mask > 0)
    x1, x2 = np.min(indices[1]), np.max(indices[1])
    y1, y2 = np.min(indices[0]), np.max(indices[0])
    bbox = [x1, y1, x2, y2]
    return bbox


def create_background_mask_from_sketch(sketch_image_path, **mask_params):
    """
    Create a background mask using the sophisticated background detection from the first script.
    
    Args:
        sketch_image_path: Path to the sketch image
        **mask_params: Additional parameters to pass to get_mask()
    
    Returns:
        np.ndarray: Binary mask where True indicates background regions
    """
    import tempfile
    import os
    
    # Create a temporary file for get_mask output
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        temp_mask_path = tmp.name
    
    try:
        # Get the background mask using the sophisticated function
        _, mask_type = get_mask(sketch_image_path, temp_mask_path, **mask_params)
        
        # Read the mask (it's colored, so we need to convert to grayscale)
        mask_colored = cv2.imread(temp_mask_path, cv2.IMREAD_COLOR)
        mask_gray = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2GRAY)
        _, background_mask = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
        
        return background_mask.astype(bool), mask_type
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_mask_path):
            os.remove(temp_mask_path)


def create_red_masked_region(base_mask, overlay_mask):
    """
    Create an RGB image where the region covered by overlay_mask is colored red.

    Args:
        base_mask (np.ndarray): Base binary mask
        overlay_mask (np.ndarray): Mask defining region to be colored red

    Returns:
        np.ndarray: RGB image with red overlay region
    """
    # Create RGB image (initialize as all black)
    height, width = base_mask.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_image[base_mask > 0] = [255, 255, 255]
    # Set red channel to 255 where overlay_mask is True/white
    rgb_image[overlay_mask > 0] = [0, 0, 255]  # BGR format for OpenCV

    return rgb_image


def run_inpainting_on_sketch_dir_template(inpaint_func):

    def wrapper(sketch_dir):
        masks_dir = f"{sketch_dir}/masks_final"
        if not os.path.exists(masks_dir):
            print(f"Directory {masks_dir} does not exist. Please run the segmentation step first.")
            exit(1)
        num_masks = len(glob.glob(f"{masks_dir}/mask_*"))
        layers_out_dir = f"{sketch_dir}/complete_layers"
        debug_out_dir = f"{sketch_dir}/complete_layers_process"

        if os.path.exists(layers_out_dir) and len(os.listdir(layers_out_dir)) > 0:
            subprocess.run(["rm", "-rf", layers_out_dir])
        if os.path.exists(debug_out_dir) and len(os.listdir(debug_out_dir)) > 0:
            subprocess.run(["rm", "-rf", debug_out_dir])
    
        os.makedirs(layers_out_dir, exist_ok=True)
        os.makedirs(debug_out_dir, exist_ok=True)
        
        for mask_index in range(num_masks):
            print(f"Processing mask {mask_index}")
            edit_mask, sketch_layer, debug_vis, need_inpaint, original_sketch_mask = (
                assemble_inpaint_input_at_index(masks_dir, mask_index)
            )
            cur_debug_out_dir = f"{debug_out_dir}/mask_{mask_index}"
            os.makedirs(cur_debug_out_dir, exist_ok=True)

            Image.fromarray(sketch_layer).save(f"{cur_debug_out_dir}/sketch_layer.png")
            Image.fromarray(sketch_layer).save(f"{layers_out_dir}/layer_{mask_index}.png")
            debug_vis_path = f"{cur_debug_out_dir}/debug_vis.png"
            if debug_vis is not None:
                Image.fromarray(debug_vis).save(debug_vis_path)
                print(f"Debug visualization saved to {debug_vis_path}")
            if need_inpaint:
                Image.fromarray(edit_mask.astype(np.uint8) * 255).save(
                    f"{cur_debug_out_dir}/edit_mask.png"
                )

                inpainted_image = inpaint_func(
                    input_image=Image.fromarray(sketch_layer),
                    mask_image=Image.fromarray(edit_mask.astype(np.uint8) * 255),
                )
                inpainted_image.save(f"{cur_debug_out_dir}/inpainted_image.png")
                
                # Composite original sketch content back onto the inpainted image
                final_image = composite_original_sketch_onto_inpainted(
                    inpainted_image, sketch_layer, original_sketch_mask
                )
                final_image.save(f"{cur_debug_out_dir}/final_composited.png")
                
                # overwrite the layer with final composited image
                final_image.save(f"{layers_out_dir}/layer_{mask_index}.png")
        return layers_out_dir

    return wrapper