import sys 
import os
import glob
import argparse
from InkLayer.runner import run_inklayer_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default=None, help="Path to the input image")
    parser.add_argument("--dir", type=str, default=None, help="Path to the directory containing images")
    parser.add_argument("--out_dir", type=str, default="./output", help="Path to the output directory")
    parser.add_argument("--no_intermediate", default=False, action="store_true", help="If set, skips saving intermediate results")
    parser.add_argument("--inpaint", default=False, action="store_true", help="If set, runs inpainting on the sketches")
    args = parser.parse_args()

    if args.img is None and args.dir is None:
        print("Please provide either an image path or a directory containing images.")
        sys.exit(1)
        
    print(f"Running InkLayer pipeline with parameters: {args}")
        
    # Single image processing
    if args.img:
        run_inklayer_pipeline(args.img, args.out_dir, no_intermediate=args.no_intermediate, inpaint=args.inpaint)
        
    # Directory processing
    elif args.dir:
        sketch_images = sorted(glob.glob(os.path.join(args.dir, "*.png"))) + sorted(glob.glob(os.path.join(args.dir, "*.jpg")))
        print(f"Found {len(sketch_images)} images in directory {args.dir}")
        for sketch_image in sketch_images:
            print(f"Processing {sketch_image}")
            run_inklayer_pipeline(sketch_image, args.out_dir, no_intermediate=args.no_intermediate, inpaint=args.inpaint)
        
if __name__ == "__main__":
    main()