# Third-party imports
import os
from PIL import Image
import sys
import subprocess

# Inklayer imports
from InkLayer.detector.gdino import run_ft_dino_on_sketch
from InkLayer.segmentor.sam import run_SAM
from InkLayer.refinement.mask_cleaner import run_clean_masks_on_sketch_dir
from InkLayer.utils.processing import process_dino_output, save_norm_bboxes
from InkLayer.utils.visualization import draw_norm_bbox_on_image, color_sketch_by_masks
from InkLayer.refinement.bbox_filter import run_postprocess_boxes_on_sketch_dir
from InkLayer.refinement.refiner import run_refinement_on_sketch_dir

def run_inklayer_pipeline(input_path, out_base_dir, no_intermediate=False):
    input_name = os.path.basename(input_path).split(".")[0]
    input_pil = Image.open(input_path).convert("RGB")

    out_dir = os.path.join(out_base_dir, input_name)
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        subprocess.run(["rm", "-r", out_dir])
    os.makedirs(out_dir, exist_ok=True)
    input_pil.save(os.path.join(out_dir, "input.png"))
    
    """
    Run DINO inference on sketch to get bounding boxes 
    """
    dino_out_dict = run_ft_dino_on_sketch(sketch_path=input_path)
    processed_boxes_tensor, pred_phrases = process_dino_output(dino_out_dict, input_pil)
    bboxes_list = processed_boxes_tensor.tolist()
    bboxes_list = [[int(x) for x in bbox] for bbox in bboxes_list]
    bboxes_json = os.path.join(out_dir, "bboxes.json")
    save_norm_bboxes(
        bboxes_list=bboxes_list,
        scores_list=dino_out_dict["scores"],
        input_pil=input_pil,
        out_path=bboxes_json,
    )

    """
    Run SAM inference on image to get pixel masks
    """
    input_pil = Image.open(input_path).convert("RGB")
    masks_np = run_SAM(image_pil=input_pil, boxes_filt=processed_boxes_tensor)
    masks_pils = [Image.fromarray(mask) for mask in masks_np]
    colored_sketch_pil = color_sketch_by_masks(input_pil, masks_pils)

    """ 
    Save initial bboxes and masks 
    """
    masks_dir = os.path.join(out_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    for i, mask in enumerate(masks_pils):
        mask.save(os.path.join(masks_dir, f"mask_{i}.png"))
    colored_sketch_pil.save(os.path.join(out_dir, "segmented_sketch.png"))
    image_w_bboxes = draw_norm_bbox_on_image(input_pil, bboxes_list, pred_phrases)
    image_w_bboxes.save(os.path.join(out_dir, "bboxes.png"))
    input_pil.save(os.path.join(out_dir, "input.png"))

    """
    Refinement modules
    """
    run_clean_masks_on_sketch_dir(out_dir)
    bbox_out_path = run_postprocess_boxes_on_sketch_dir(
        out_dir, sketch_iou_thresh=0.2
    )
    run_refinement_on_sketch_dir(out_dir, bbox_out_path)
    
    """
    Clean up intermediate results if no_intermediate is True
    """
    if no_intermediate:
        for item in os.listdir(out_dir):
            item_path = os.path.join(out_dir, item)
            # Edit this list to keep or remove specific files
            if item not in ["masks_final", "bboxes_final.json", "bboxes_final.png", 
                            "segmented_sketch_final.png", "depth_map.png", "input.png"]:
                if os.path.isdir(item_path):
                    subprocess.run(["rm", "-r", item_path])
                else:
                    os.remove(item_path)
    return out_dir

if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default=None, help="Path to the input image")
    parser.add_argument("--dir", type=str, default=None, help="Path to the directory containing images")
    parser.add_argument("--out_dir", type=str, default="./output", help="Path to the output directory")
    parser.add_argument("--no_intermediate", default=False, action="store_true", help="If set, skips saving intermediate results")
    args = parser.parse_args()
    if args.img is None and args.dir is None:
        print("Please provide either an image path or a directory containing images.")
        sys.exit(1)
    if args.img:
        run_inklayer_pipeline(args.img, args.out_dir, no_intermediate=args.no_intermediate)
    elif args.dir:
        sketch_images = sorted(glob.glob(os.path.join(args.dir, "*.png"))) + sorted(glob.glob(os.path.join(args.dir, "*.jpg")))
        for sketch_image in sketch_images:
            print(f"Processing {sketch_image}")
            run_inklayer_pipeline(sketch_image, args.out_dir, no_intermediate=args.no_intermediate)
        