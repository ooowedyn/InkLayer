import argparse
from PIL import Image

# SDXL inpaint
from diffusers import AutoPipelineForInpainting
import torch

from InkLayer.inpainting.util import (
    run_inpainting_on_sketch_dir_template
)


def SDXL_inpaint(input_image, mask_image):
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")
    prompt = "black and white sketch, complete lines"
    generator = torch.Generator(device="cuda").manual_seed(3)
    image = pipe(
        prompt=prompt,
        image=input_image.resize((1024, 1024)),
        mask_image=mask_image.resize((1024, 1024)),
        guidance_scale=8.0,
        num_inference_steps=20,  # steps between 15 and 30 work well for us
        strength=0.99,  
        generator=generator,
    ).images[0]
    
    image = image.resize(input_image.size, Image.LANCZOS)
    image = image.convert("L").convert("RGB")  # Convert to grayscale and back
    return image

run_inpainting_on_sketch_dir = run_inpainting_on_sketch_dir_template(SDXL_inpaint)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sketch to complete layers")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing segmented sketch")
    args = parser.parse_args()
    sketch_dir = args.dir
    run_inpainting_on_sketch_dir(sketch_dir)