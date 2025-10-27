# Third-party imports
import os, base64, re
from PIL import Image
import subprocess
from PIL import Image, ImageDraw
from InkLayer.inpainting.inpaint_single_layer import inpaint_single_layer


# Inklayer imports
from InkLayer.detector.gdino import run_ft_dino_on_sketch
from InkLayer.segmentor.sam import run_SAM
from InkLayer.refinement.mask_cleaner import run_clean_masks_on_sketch_dir
from InkLayer.utils.processing import process_dino_output, save_norm_bboxes
from InkLayer.utils.visualization import draw_norm_bbox_on_image, color_sketch_by_masks
from InkLayer.refinement.bbox_filter import run_postprocess_boxes_on_sketch_dir
from InkLayer.refinement.refiner import run_refinement_on_sketch_dir
from InkLayer.inpainting.inpaint_ControlNet import run_inpainting_on_sketch_dir
from InkLayer.inpainting.fill_object_bg_mask import create_rgba_with_background_mask_on_dir
from InkLayer.inpainting.inpaint_single_layer import inpaint_single_layer

def run_inklayer_pipeline(input_path, out_base_dir, no_intermediate=False, inpaint=False):
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
    Inpainting step
    """
    if inpaint:
        inpainted_dir = run_inpainting_on_sketch_dir(out_dir)
        print(f"Inpainting completed. Output saved to {inpainted_dir}")
        # We need to create RGBA images with background masks -- only sketch content + necessary background region is filled
        rgba_output_dir = inpainted_dir.replace("layers", "layers_rgba")
        create_rgba_with_background_mask_on_dir(inpainted_dir, rgba_output_dir)
    else:
        print("Skipping inpainting step as 'inpaint' is set to False.")
    
    """
    Clean up intermediate results if no_intermediate is True
    """
    if no_intermediate:
        for item in os.listdir(out_dir):
            item_path = os.path.join(out_dir, item)
            # Edit this list to keep or remove specific dir/files
            if item not in ["masks_final", "complete_layers", "complete_layers_rgba", 
                            "bboxes_final.json","bboxes_final.png", "segmented_sketch_final.png", 
                            "depth_map.png", "input.png"]:
                if os.path.isdir(item_path):
                    subprocess.run(["rm", "-r", item_path])
                else:
                    os.remove(item_path)

    return out_dir
def run_inpaint_single_layer(request_data, cur_dir, out_dir):
    """
    단일 레이어 인페인팅 실행 함수

    request_data: dict 형태로 프론트에서 전달받은 JSON 데이터
      {
          "image_name": "...",
          "layer_id": "2",
          "layer_path": "static/outputs/.../layer_2.png",
          "prompt": "make it blue"
      }
    out_dir: 결과 저장 디렉토리 (ex: static/outputs/inpaint_results)
    """

    # 1. JSON 데이터 파싱
    image_name = request_data.get("image_name")
    layer_id = request_data.get("layer_id")
    layer_path = request_data.get("layer_path")
    prompt = request_data.get("prompt")


    print(f"[run_inpaint_single_layer] image_name={image_name}, layer_id={layer_id}")
    print(f"layer_path={layer_path}, prompt={prompt}")

    # 2. 전체 이미지와 해당 레이어 마스크 불러오기
    layer_rel_path = layer_path.replace("http://127.0.0.1:5000", "").lstrip("/")
    abs_layer_path = os.path.join(cur_dir, layer_rel_path)
    abs_layer_path = os.path.normpath(abs_layer_path)

    # base_dir: 해당 이미지의 상위 디렉토리 (예: static/outputs/fscoco_animals)
    base_dir = os.path.join(cur_dir, f"static/outputs/{image_name}")
    abs_image_path = os.path.join(base_dir, "input.png")
    layer_id = os.path.basename(layer_path).split("_")[-1].split(".")[0]
    abs_mask_path = os.path.join(base_dir, "masks_disjoint", f"mask_{layer_id}.png")

    print(f"abs_layer_path = {abs_layer_path}")
    print(f"abs_image_path = {abs_image_path}")
    print(f"abs_mask_path  = {abs_mask_path}")

    # 3. 마스크 영역을 박스 형태로 확장하여 별도 저장
    expanded_mask_path = os.path.join(out_dir, f"mask_expanded_{layer_id}.png")
    mask = Image.open(abs_mask_path).convert("L")

    bbox = mask.getbbox()
    if bbox:
        # 여유 10px
        x0, y0, x1, y1 = bbox
        x0, y0 = max(0, x0 - 10), max(0, y0 - 10)
        x1, y1 = min(mask.width, x1 + 10), min(mask.height, y1 + 10)
        expanded = Image.new("L", mask.size, 0)
        draw = ImageDraw.Draw(expanded)
        draw.rectangle([x0, y0, x1, y1], fill=255)
        expanded.save(expanded_mask_path)
    else:
        mask.save(expanded_mask_path)

    print(f"expanded mask saved at: {expanded_mask_path}")

    # 4. 실제 inpainting 실행
    output_path = inpaint_single_layer(
        image_path=abs_image_path,
        mask_path=expanded_mask_path,
        output_dir=out_dir,
        prompt=prompt,
        layer_id=layer_id,
    )

    # 5. 결과 반환 (RGBA 경로)
    return output_path
