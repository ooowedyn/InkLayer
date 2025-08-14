import argparse
from PIL import Image, ImageFilter, ImageEnhance
import torch
import numpy as np
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, DPMSolverMultistepScheduler
import cv2

from InkLayer.inpainting.util import (
    run_inpainting_on_sketch_dir_template
)

_controlnet_pipeline = None

def get_controlnet_pipeline():
    global _controlnet_pipeline
    
    if _controlnet_pipeline is None:
        print("Loading enhanced SD with controlnet pipeline...")
        
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", 
            torch_dtype=torch.float16, 
            variant="fp16"
        )
        
        _controlnet_pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", 
            controlnet=controlnet, 
            torch_dtype=torch.float16, 
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False
        ).to("cuda")
        
        _controlnet_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            _controlnet_pipeline.scheduler.config
        )
        
        # Enable memory optimizations
        try:
            _controlnet_pipeline.enable_xformers_memory_efficient_attention()
        except:
            pass  # xFormers not available
            
        print("pipeline loaded!")
    
    return _controlnet_pipeline

def preprocess_image(image, enhance_contrast=True, denoise=True):
    """Preprocess input image for better inpainting results."""
    if enhance_contrast:
        # Enhance contrast slightly
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
    
    if denoise:
        # Light denoising
        image_array = np.array(image)
        if len(image_array.shape) == 3:
            denoised = cv2.bilateralFilter(image_array, 5, 50, 50)
        else:
            denoised = cv2.bilateralFilter(image_array, 5, 50, 50)
        image = Image.fromarray(denoised)
    
    return image

def preprocess_mask(mask, dilate_iterations=1, blur_radius=1):
    mask_array = np.array(mask.convert('L'))
    if dilate_iterations > 0:
        kernel = np.ones((3,3), np.uint8)
        mask_array = cv2.dilate(mask_array, kernel, iterations=dilate_iterations)
    if blur_radius > 0:
        mask_array = cv2.GaussianBlur(mask_array, (blur_radius*2+1, blur_radius*2+1), 0)
    
    return Image.fromarray(mask_array)

def make_inpaint_condition(init_image, mask_image):
    init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
    mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0
    
    assert init_image.shape[:2] == mask_image.shape[:2], "image and mask must have the same dimensions"
    
    # Set masked pixels to -1.0 (ControlNet convention)
    init_image[mask_image > 0.5] = -1.0
    
    # Convert to tensor format [1, C, H, W]
    init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
    init_image = torch.from_numpy(init_image)
    
    return init_image

def postprocess_result(result_image, original_image, mask_image):
    result_array = np.array(result_image)
    original_array = np.array(original_image)
    mask_array = np.array(mask_image.convert('L')) / 255.0
    
    return _adaptive_threshold_blend(result_array, original_array, mask_array)

def _adaptive_threshold_blend(result_array, original_array, mask_array):
    if len(result_array.shape) == 3:
        result_gray = cv2.cvtColor(result_array, cv2.COLOR_RGB2GRAY)
    else:
        result_gray = result_array.copy()
    
    thresh = cv2.adaptiveThreshold(
        result_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Create a cleaner result by applying the threshold
    if len(result_array.shape) == 3:
        clean_result = np.zeros_like(result_array)
        for i in range(3):
            clean_result[:, :, i] = np.where(thresh > 127, 255, result_array[:, :, i])
    else:
        clean_result = np.where(thresh > 127, 255, result_array)
    
    soft_mask = cv2.GaussianBlur(mask_array, (3, 3), 1)
    soft_mask = np.clip(soft_mask, 0, 1)
    if len(result_array.shape) == 3:
        soft_mask = soft_mask[:, :, np.newaxis]
    
    blended = clean_result * soft_mask + original_array * (1 - soft_mask)
    return Image.fromarray(blended.astype(np.uint8))

def ControlNet_inpaint(input_image, mask_image, 
                      preprocess_input=True,
                      postprocess_output=True):
    pipe = get_controlnet_pipeline()

    original_input = input_image.copy()
    original_mask = mask_image.copy()
    
    # Preprocessing
    if preprocess_input:
        input_image = preprocess_image(input_image)
        mask_image = preprocess_mask(mask_image)
    

    # Try changing this to improve results for specific cases!
    prompt = "high quality black and white line drawing, clean precise lines, detailed sketch, professional illustration, sharp edges"
    negative_prompt = "blurry, smudged, messy lines, low quality, artifacts, noise, distorted, pixelated"
    num_steps = 30
    guidance_scale = 9.0
    controlnet_conditioning_scale = 1.2
    
    generator = torch.Generator(device="cuda").manual_seed(3)
    
    target_size = 768
    input_resized = input_image.resize((target_size, target_size), Image.LANCZOS)
    mask_resized = mask_image.resize((target_size, target_size), Image.LANCZOS)
    
    control_image = make_inpaint_condition(input_resized, mask_resized)
    
    # Run multiple passes for very high quality
    num_passes = 2

    for pass_num in range(num_passes):
        if pass_num > 0:
            # For second pass, use result as input
            input_resized = image.resize((target_size, target_size), Image.LANCZOS)
            control_image = make_inpaint_condition(input_resized, mask_resized)
        
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_resized,
            mask_image=mask_resized,
            control_image=control_image,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
        ).images[0]
    
    image = image.resize(original_input.size, Image.LANCZOS)
    
    if postprocess_output:
        image = postprocess_result(image, original_input, original_mask)
    
    image = image.convert("L").convert("RGB")
    image = image.filter(ImageFilter.UnsharpMask(radius=0.5, percent=150, threshold=3))
    
    return image


run_inpainting_on_sketch_dir = run_inpainting_on_sketch_dir_template(ControlNet_inpaint)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sketch to complete layers with enhanced ControlNet")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing segmented sketch")
    args = parser.parse_args()

    run_inpainting_on_sketch_dir = run_inpainting_on_sketch_dir_template(ControlNet_inpaint)
    sketch_dir = args.dir
    run_inpainting_on_sketch_dir(sketch_dir)