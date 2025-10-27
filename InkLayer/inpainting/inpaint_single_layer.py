import os
import cv2
import numpy as np
import torch
from PIL import Image
from InkLayer.inpainting.fill_object_bg_mask import create_rgba_with_background_mask
from InkLayer.inpainting.inpaint_ControlNet import get_controlnet_pipeline, preprocess_image, preprocess_mask, make_inpaint_condition


def inpaint_single_layer(
    image_path: str,
    mask_path: str,
    output_dir: str,
    prompt: str,
    layer_id: str,
    position_data=None,
):
    """
    Run text-guided inpainting for a single sketch layer
    using the **already loaded ControlNet pipeline** (for speed).
    """

    print(f"[Inpaint-ControlNet] Starting text-guided inpainting for layer {layer_id}")

    # 1️⃣ Load image & mask
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    # 2️⃣ Apply mask movement if needed
    if position_data:
        print(f"[Inpaint-ControlNet] Moving mask for layer {layer_id}")
        mask = _move_mask(mask, position_data, image.size)

    # 3️⃣ Preprocess inputs (light denoise + contrast)
    image = preprocess_image(image, enhance_contrast=True, denoise=True)
    mask = preprocess_mask(mask, dilate_iterations=1, blur_radius=1)

    # 4️⃣ Load (cached) ControlNet pipeline once
    pipe = get_controlnet_pipeline()

    # 5️⃣ Prepare conditioning
    target_size = 768
    input_resized = image.resize((target_size, target_size), Image.LANCZOS)
    mask_resized = mask.resize((target_size, target_size), Image.LANCZOS)
    control_image = make_inpaint_condition(input_resized, mask_resized)
    prompt = prompt#+", line sketch style"
    # 6️⃣ Run inpainting with custom prompt
    print(f"[Inpaint-ControlNet] Running with prompt: '{prompt}'")
    generator = torch.Generator(device="cuda").manual_seed(3)

    result = pipe(
        prompt= prompt,
        negative_prompt="blurry, smudged, messy lines, low quality, artifacts, noise, distorted, pixelated",
        image=input_resized,
        mask_image=mask_resized,
        control_image=control_image,
        guidance_scale=7.0,
        num_inference_steps=30,
        controlnet_conditioning_scale=0.6,
        generator=generator,
    ).images[0]

    result = result.resize(image.size, Image.LANCZOS)
    result_path = os.path.join(output_dir, f"inpainted_layer_{layer_id}.png")
    result.save(result_path)

    # numpy 변환
    result_np = np.array(result.convert("RGB"))
    mask_np = np.array(mask.convert("L"))

    # ensure shape 맞춤
    mask_np = cv2.resize(mask_np, (result_np.shape[1], result_np.shape[0]))

    # (1) RGBA 생성: 마스크 영역만 남기고 나머지는 투명
    rgba = np.zeros((result_np.shape[0], result_np.shape[1], 4), dtype=np.uint8)
    inside = mask_np > 128
    rgba[..., :3][inside] = result_np[inside]
    rgba[..., 3][inside] = 255  # 알파 채널

    # RGBA 저장
    layer_rgba_path = os.path.join(output_dir, f"layer_{layer_id}_rgba.png")
    Image.fromarray(rgba, "RGBA").save(layer_rgba_path)
    print(f"[✅] Saved transparent RGBA layer → {layer_rgba_path}")

    return layer_rgba_path


def _move_mask(mask_img: Image.Image, position_data, canvas_size):
    """Move and resize mask according to layer's position info."""
    if isinstance(position_data, list):
        position_data = position_data[0]

    x = int(position_data.get("x", 0))
    y = int(position_data.get("y", 0))
    w = int(position_data.get("width", mask_img.width))
    h = int(position_data.get("height", mask_img.height))

    print(f"[Mask Move] (x={x}, y={y}, w={w}, h={h})")

    mask_resized = mask_img.resize((w, h))
    canvas_w, canvas_h = canvas_size
    mask_canvas = Image.new("L", (canvas_w, canvas_h), 0)
    mask_canvas.paste(mask_resized, (x, y))

    return mask_canvas