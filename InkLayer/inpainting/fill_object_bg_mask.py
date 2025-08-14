import cv2
import numpy as np
    
def fill_enclosed_regions(mask_binary):
    """
    Fills every interior hole (child contour) in-place.
    mask_binary must be uint8 with values {0,255}.
    """
    cnts, hier = cv2.findContours(mask_binary,
                                  cv2.RETR_CCOMP,  
                                  cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return mask_binary  

    for i, c in enumerate(cnts):
        parent = hier[0][i][3]
        if parent != -1:                        
            # if cv2.contourArea(c) >= min_area:  # optional: skip tiny specks
            cv2.drawContours(mask_binary, [c], -1, 255, thickness=-1)
    return mask_binary                      # add them to the mask

def fill_holes_not_touching_border(mask_binary, min_area=50):
    """
    Fills every interior hole that is completely enclosed
    (none of its pixels lie on the image border).
    mask_binary must be uint8 with values {0,255}.
    """
    h, w = mask_binary.shape
    cnts, hier = cv2.findContours(mask_binary,
                                  cv2.RETR_CCOMP,
                                  cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return mask_binary

    for i, c in enumerate(cnts):
        parent = hier[0][i][3]
        if parent == -1:          # outer contour -> skip
            continue

        # Child contour: check area & whether it touches border
        x, y, cw, ch = cv2.boundingRect(c)
        touches_edge = (x == 0 or y == 0 or x+cw == w or y+ch == h)

        if (not touches_edge) and cv2.contourArea(c) >= min_area:
            cv2.drawContours(mask_binary, [c], -1, 255, thickness=-1)

    return mask_binary


def get_mask(input_path,
               output_path,
               mask_color=(255, 0, 0),  # blue  (BGR)
               dilate_iter=5,          # grow to close gaps
               kernel_size=3,
               safety_margin=0,         # keep ≥1 px over every stroke
               stroke_thick=1,          # thickness for open curves
               border_band=2):          # how many border rows/cols to inspect
    """
    Build a mask that is:
      • a filled silhouette when the drawing is closed, OR
      • just the strokes when the drawing touches the image border.
    """
    gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    # need to flip it
    gray = cv2.bitwise_not(gray)
    if gray is None:
        raise ValueError(f"Cannot read {input_path}")
    _, bin_strokes = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (kernel_size, kernel_size))
    thick = cv2.dilate(bin_strokes, k, iterations=dilate_iter)

    h, w = thick.shape
    touches_border = (
        thick[:border_band, :].any() or            # top
        thick[-border_band:, :].any() or           # bottom
        thick[:, :border_band].any() or            # left
        thick[:, -border_band:].any()              # right
    )

    if touches_border: # this is probably a background line, we don't add background mask
        mask = cv2.dilate(bin_strokes, k, iterations=stroke_thick)
        mask = fill_holes_not_touching_border(mask, min_area=50)   
        coloured = np.zeros((h, w, 3), np.uint8)
        coloured[mask == 255] = mask_color
        cv2.imwrite(output_path, coloured)
        return coloured, 'open-curve'

    flooded = thick.copy()
    cv2.floodFill(flooded, np.zeros((h+2, w+2), np.uint8), (0, 0), 255)
    silhouette = cv2.bitwise_not(flooded) | thick

    #  take largest CC
    cnts, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)

    # shrink just enough so every stroke pixel is still covered
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    min_pad = int(np.floor(dist[bin_strokes == 255].min()))
    shrink_by = max(0, min_pad - safety_margin)
    if shrink_by > 0:
        mask = (dist >= shrink_by).astype(np.uint8) * 255

    mask = fill_enclosed_regions(mask)    
    coloured = np.zeros((h, w, 3), np.uint8)
    coloured[mask == 255] = mask_color
    cv2.imwrite(output_path, coloured)
    # print(f"Saved {output_path}. ")
    return coloured, f'closed-silhouette (shrunk by {shrink_by}px)'


def create_rgba_with_background_mask(input_path, output_path, **mask_params):
    """
    Creates an RGBA image where:
    - Original sketch pixels (black in input) remain black
    - Background mask area becomes white
    - Everything else is transparent
    
    This allows for proper layering of multiple sketch objects.
    
    Args:
        input_path: Path to input black/white sketch image
        output_path: Path for output RGBA PNG file
        **mask_params: Additional parameters to pass to get_mask()
    
    Returns:
        rgba_image: The resulting RGBA image as numpy array
        mask_type: String describing the type of mask created
    """
    # Read original sketch
    original_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if original_gray is None:
        raise ValueError(f"Cannot read {input_path}")
    
    h, w = original_gray.shape
    sketch_pixels = (original_gray < 240).astype(np.uint8) * 255  # Detect non-white pixels
    # And preserve original colors:

    
    # Create a temporary file for get_mask (we don't actually need the colored output)
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        temp_mask_path = tmp.name
    
    try:
        _, mask_type = get_mask(input_path, temp_mask_path, **mask_params)
        
        # Read the mask (it's colored, so we need to convert to grayscale)
        mask_colored = cv2.imread(temp_mask_path, cv2.IMREAD_COLOR)
        mask_gray = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2GRAY)
        _, background_mask = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_mask_path):
            os.remove(temp_mask_path)
    
    # Create RGBA image
    rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Set alpha channel: opaque where we have sketch or background mask
    alpha_mask = np.logical_or(sketch_pixels == 255, background_mask == 255)
    rgba_image[:, :, 3] = alpha_mask.astype(np.uint8) * 255
    
    # Set colors:
    # - Sketch pixels remain sketch pixel colors (can have gray)
    # - Background mask pixels become white (RGB = 255,255,255)
    sketch_mask = sketch_pixels == 255
    original_sketch = cv2.imread(input_path, cv2.IMREAD_COLOR)
    rgba_image[background_mask == 255, :3] = [255, 255, 255]  # background mask becomes white
    rgba_image[sketch_mask, :3] = cv2.cvtColor(original_gray[sketch_mask].reshape(-1, 1), cv2.COLOR_GRAY2BGR).reshape(-1, 3)
    
    # Save as PNG to preserve alpha channel
    if not output_path.lower().endswith('.png'):
        output_path = os.path.splitext(output_path)[0] + '.png'
    
    cv2.imwrite(output_path, rgba_image)
    
    return rgba_image, mask_type

def create_rgba_with_background_mask_on_dir(input_dir, output_dir):
    """
    Processes all images in input_dir and saves RGBA versions to output_dir.
    """
    import os
    import glob
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    input_images = glob.glob(os.path.join(input_dir, '*.png'))
    original_sketch_path = os.path.join(input_dir, '../input.png')  # Assuming this is the original sketch path
    if not os.path.exists(original_sketch_path):
        raise ValueError(f"Original sketch image not found at {original_sketch_path}")
    
    for input_path in input_images:
        base_name = os.path.basename(input_path)
        output_path = os.path.join(output_dir, base_name)
        
        rgba_image, mask_type = create_rgba_with_background_mask(
            input_path=input_path,
            output_path=output_path,
            # original_sketch_path=original_sketch_path
        )
        
        # print(f"Processed {input_path} → {output_path} ({mask_type})")
    print(f"Saved RGBA images to {output_dir}")
    return output_dir

if __name__ == "__main__":
    # Original processing
    import argparse
    parser = argparse.ArgumentParser(description="Create RGBA images with background mask.")
    parser.add_argument("--dir", help="Directory containing input images")
    args = parser.parse_args()

    test_dir = f"{args.dir}/complete_layers"
    out_dir = f"{test_dir}_rgba"
    create_rgba_with_background_mask_on_dir(test_dir, out_dir)