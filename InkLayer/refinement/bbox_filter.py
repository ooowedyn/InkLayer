import numpy as np
import json
from typing import Dict
import os
import glob

from InkLayer.utils.visualization import draw_boxes
from InkLayer.refinement.nms_sketch import sketch_nms


def process_json_with_sketch_NMS(
    sketch_path: str, masks_dir: str, input_data: Dict, iou_threshold: float = 0.2
) -> Dict:
    """
    Process input JSON data and apply NMS to bounding boxes.
    Returns filtered data in the same format.
    """
    keep_indices = sketch_nms(
        sketch_path,
        np.array(input_data["bboxes"]),
        np.array(input_data["scores"]),
        masks_dir,
        iou_threshold,
    )

    filtered_data = {
        "bboxes": [input_data["bboxes"][i] for i in keep_indices],
        "scores": [input_data["scores"][i] for i in keep_indices],
        "kept_indices": [int(i) for i in keep_indices],
        "threshold": iou_threshold,
    }

    return filtered_data


def run_postprocess_boxes_on_sketch_dir(sketch_dir, sketch_iou_thresh=0.5):
    if not os.path.exists(sketch_dir):
        print("no sketch dir")
        return
    mmdet_json = glob.glob(f"{sketch_dir}/mmdet_out/*.json")
    json_path = None
    if len(mmdet_json) > 0:
        json_path = mmdet_json[0]
    else:
        json_path = f"{sketch_dir}/bboxes.json"
    with open(json_path, "r") as f:
        input_data = json.load(f)

    filtered_data = process_json_with_sketch_NMS(
        sketch_path=f"{sketch_dir}/input.png",
        masks_dir=f"{sketch_dir}/masks_cleaned",
        input_data=input_data,
        iou_threshold=sketch_iou_thresh,
    )
    print(f"Got filtered data with {len(filtered_data['bboxes'])} boxes")
    out_file_name = f"bboxes_final"
    out_path = f"{sketch_dir}/{out_file_name}.json"
    with open(out_path, "w") as f:
        json.dump(filtered_data, f, indent=4)
    print(f"Output saved to {out_path}")

    image_out_path = f"{sketch_dir}/{out_file_name}.png"
    sketch_path = f"{sketch_dir}/input.png"
    draw_boxes(
        sketch_path,
        filtered_data["bboxes"],
        filtered_data["scores"],
        output_path=image_out_path,
    )

    print(f"Output saved to {image_out_path}")
    return out_path

