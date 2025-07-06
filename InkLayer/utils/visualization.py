import cv2
import numpy as np
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont
from typing import List, Union


def draw_norm_bbox_on_image(
    image_pil, bboxes, pred_phrases=None, color=(255, 0, 0), thickness=5
):
    res_image = image_pil.copy()
    draw = ImageDraw.Draw(res_image)
    colors = generate_pastel_colors(len(bboxes))
    for i, bbox in enumerate(bboxes):
        color = colors[i]
        x1, y1, x2, y2 = bbox
        if max(x1, y1, x2, y2) <= 1:
            x1, y1, x2, y2 = (
                x1 * image_pil.size[0],
                y1 * image_pil.size[1],
                x2 * image_pil.size[0],
                y2 * image_pil.size[1],
            )
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        if pred_phrases is not None and len(pred_phrases) > i:
            draw.text((x1, y1), pred_phrases[i], fill=color)
    return res_image


def generate_pastel_colors(n_colors):
    """Generate pastel colors and randomize their order"""
    # Generate colors evenly spaced in hue
    hues = [x / n_colors for x in range(n_colors)]
    
    # Convert to RGB with fixed saturation and value for pastel effect
    colors = [mcolors.hsv_to_rgb([hue, 0.7, 0.88]) for hue in hues]
    
    # Convert to RGB int format
    colors = [
        (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        for color in colors
    ]
    
    # Randomize the order of colors to ensure adjacent instances get different colors
    np.random.shuffle(colors)
    
    return colors


def color_sketch_by_masks(sketch_image_pil, seg_masks, colors=None, enhance_factor=1.5, min_opacity=0.2):
    """
    Color a sketch by masks with enhanced visibility for faint strokes.
    
    Parameters:
    - sketch_image_pil: PIL image of the sketch
    - seg_masks: List of segmentation masks
    - colors: Optional list of colors for each mask
    - enhance_factor: Factor to enhance the opacity of strokes (default: 1.5)
    - min_opacity: Minimum opacity for any visible stroke (default: 0.2)
    """
    if colors is None:
        pastel_colors = generate_pastel_colors(len(seg_masks))
    else:
        pastel_colors = colors

    sketch_np = np.array(sketch_image_pil)

    # Convert to grayscale if needed
    if len(sketch_np.shape) == 3:
        sketch_gray = cv2.cvtColor(sketch_np, cv2.COLOR_RGB2GRAY)
    else:
        sketch_gray = sketch_np

    # Initialize final image with white background
    h, w = sketch_np.shape[:2]
    final_colored_sketch = np.ones((h, w, 3), dtype=np.float32) * 255

    # Create a mask for all non-white pixels 
    stroke_mask = sketch_gray < 250
    
    # Get background mask
    background_mask = get_background_idxs(sketch_gray, seg_masks)
    
    # Enhanced opacity calculation
    # 1. Normalize grayscale values from 0-255 to 0-1, then invert so darker pixels have higher opacity
    raw_opacity = (255 - sketch_gray) / 255.0
    
    # 2. Enhance opacity: apply non-linear mapping to make faint strokes more visible
    # First, determine the minimum opacity in the actual strokes (to avoid enhancing pure white)
    stroke_opacity_values = raw_opacity[stroke_mask]
    if len(stroke_opacity_values) > 0:
        min_stroke_opacity = np.min(stroke_opacity_values[stroke_opacity_values > 0])
        max_stroke_opacity = np.max(stroke_opacity_values)
        
        # Only apply enhancement if there are faint strokes
        if max_stroke_opacity > 0.1:  # Some reasonable strokes exist
            # Apply enhancement: power function with enhance_factor
            enhanced_opacity = np.power(raw_opacity, 1.0/enhance_factor)
            
            # Ensure minimum opacity for any visible stroke
            enhanced_opacity = np.where(
                stroke_mask & (raw_opacity > 0.02),  # Only enhance pixels that are actual strokes
                np.maximum(enhanced_opacity, min_opacity),
                enhanced_opacity
            )
        else:
            # If strokes are too faint, use a more aggressive enhancement
            enhanced_opacity = np.where(
                stroke_mask, 
                np.maximum(raw_opacity * 3, min_opacity),
                raw_opacity
            )
    else:
        # Fallback if no stroke pixels detected
        enhanced_opacity = raw_opacity
    
    # Process each segmentation mask
    for i, mask in enumerate(seg_masks):
        # Get color for this segment - convert to numpy array
        color = np.array(pastel_colors[i], dtype=np.float32)
        
        # Create a mask for strokes in this segment
        segment_strokes = np.logical_and(stroke_mask, mask)
        
        # For each stroke pixel in this segment, apply color based on original intensity
        for y in range(h):
            for x in range(w):
                if segment_strokes[y, x]:
                    # Get enhanced pixel opacity
                    pixel_opacity = float(enhanced_opacity[y, x])
                    
                    # Calculate the weighted color
                    weighted_color = color * pixel_opacity
                    weighted_white = np.array([255, 255, 255], dtype=np.float32) * (1 - pixel_opacity)
                    
                    # Set the final color as the sum of the weighted components
                    final_colored_sketch[y, x] = weighted_color + weighted_white
    
    # Handle background strokes (strokes not in any mask)
    background_strokes = np.logical_and(stroke_mask, background_mask)
    for y in range(h):
        for x in range(w):
            if background_strokes[y, x]:
                # Get enhanced pixel opacity
                pixel_opacity = float(enhanced_opacity[y, x])
                
                # Calculate the weighted color for background (black to white)
                weighted_black = np.array([0, 0, 0], dtype=np.float32) * pixel_opacity
                weighted_white = np.array([255, 255, 255], dtype=np.float32) * (1 - pixel_opacity)
                
                # Set the final color
                final_colored_sketch[y, x] = weighted_black + weighted_white

    return Image.fromarray(final_colored_sketch.astype(np.uint8))

def get_background_idxs(sketch, seg_masks):
    """Modified to ensure single-channel output"""
    foreground_mask = np.zeros_like(sketch, dtype=bool)
    if len(foreground_mask.shape) > 2:
        foreground_mask = foreground_mask[
            :, :, 0
        ]  # Take only first channel if multi-channel

    for mask in seg_masks:
        foreground_mask = np.logical_or(foreground_mask, mask)

    return ~foreground_mask  # Return inverted mask


def draw_boxes(
    image: Union[str, Image.Image],
    boxes: List[List[float]],
    scores: List[float] = None,
    labels: List[str] = None,
    line_width: int = 3,
    font_size: int = 16,
    show_scores: bool = True,
    output_path: str = None,
) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image)

    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    img_width, img_height = image.size

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Create a distinct color for each box
    colors = generate_pastel_colors(len(boxes))

    # Draw each box
    for i, box in enumerate(boxes):
        x1 = box[0] * img_width
        y1 = box[1] * img_height
        x2 = box[2] * img_width
        y2 = box[3] * img_height

        draw.rectangle([(x1, y1), (x2, y2)], outline=colors[i], width=line_width)

        label_parts = []
        if labels and i < len(labels):
            label_parts.append(labels[i])
        if show_scores and scores and i < len(scores):
            label_parts.append(f"{scores[i]:.2f}")

        if label_parts:
            label_text = " : ".join(label_parts)

            text_width = (
                font.getsize(label_text)[0]
                if hasattr(font, "getsize")
                else len(label_text) * font_size
            )
            text_height = font_size + 4
            draw.rectangle(
                [(x1, y1 - text_height), (x1 + text_width + 4, y1)], fill=colors[i]
            )
            draw.text(
                (x1 + 2, y1 - text_height + 2), label_text, fill="white", font=font
            )

    if output_path:
        draw_image.save(output_path)

    return draw_image

