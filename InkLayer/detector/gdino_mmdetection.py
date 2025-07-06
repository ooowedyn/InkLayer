import os
import glob
import argparse
import sys
from PIL import Image
import json
from datetime import datetime

# Make sure you have mmdetection installed and configured properly
from mmdet.apis import DetInferencer
from InkLayer.utils.paths import get_model_path
gdino_config_path = get_model_path("mmdetection_config.py")
gdino_weights_path = get_model_path("inklayer_gdino_mmdetection.pth")
device = "cuda:0"
def_score_threshold = 0.2


def run_inf_wrapper(init_args, call_args):
    inferencer = DetInferencer(**init_args)
    chunked_size = call_args.pop('chunked_size')
    inferencer.model.test_cfg.chunked_size = chunked_size
    results_dict = inferencer(**call_args)
    return results_dict

def run_ft_dino_inference_on_image(
    image_path,
    nouns,
    mmdet_out_base_dir,
    out_dir,
    model_config=gdino_config_path,
    weights=gdino_weights_path,
    score_threshold=def_score_threshold,
):
    print(
        f"Running inference with image {image_path} using model {model_config} and weights {weights}"
    )

    image_name = os.path.basename(image_path).split(".")[0]
    prompts_str = " . ".join(nouns)
    call_args = {
        "inputs": image_path,
        "out_dir": mmdet_out_base_dir,
        "texts": prompts_str,
        "pred_score_thr": score_threshold,
        "batch_size": 1,
        "show": False,
        "no_save_vis": False,
        "no_save_pred": False,
        "print_result": False,
        "custom_entities": False,
        "tokens_positive": None,
        "chunked_size": -1,
    }

    init_args = {
        "model": model_config,
        "weights": weights,
        "device": device,
        "palette": "none",
    }

    results_dict = run_inf_wrapper(init_args=init_args, call_args=call_args)
    visualization = results_dict["visualization"][0]
    predictions = results_dict["predictions"][0]  # only one image
    labels_idxs = predictions["labels"]
    scores = predictions["scores"]
    bboxes = predictions["bboxes"]
    assert len(labels_idxs) == len(scores) == len(bboxes)
    """
    { 
        // normalize boxes
        "bboxes": [
            [x1, y1, x2, y2]
            ...
        ],
        "labels": [
            "dog",
            ...
        ],
    }
    """
    
    out_dict = {"bboxes": [], "labels": [], "scores": []}
    image = Image.open(image_path)
    img_w, img_h = image.size
    # print("Label idxs", labels_idxs)
    for i, (label_idx, score, bbox) in enumerate(zip(labels_idxs, scores, bboxes)):
        if score < score_threshold:
            continue
        print("Label idx", label_idx, "noun", nouns)
        if label_idx >= len(nouns):
            label = "unknown"
        else:
            label = nouns[label_idx]
        norm_bbox = [bbox[0] / img_w, bbox[1] / img_h, bbox[2] / img_w, bbox[3] / img_h]
        out_dict["bboxes"].append(norm_bbox)
        out_dict["labels"].append(label)
        out_dict["scores"].append(score)
    os.makedirs(out_dir, exist_ok=True)
    image.save(os.path.join(out_dir, f"input_image.png"))
    vis_img_pil = Image.fromarray(visualization)
    pred_path = os.path.join(out_dir, f"pred.png")
    vis_img_pil.save(pred_path)
    json_out_path = os.path.join(out_dir, f"{image_name}.json")
    out_dict["model_info"] = {
        "model_config": model_config,
        "weights": weights,
        "device": device,
        "score_threshold": score_threshold,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(json_out_path, "w") as f:
        json.dump(out_dict, f, indent=4)
    print(f"Saved json to {json_out_path}")
    print(f"Saved image to {pred_path}")

    return out_dict
