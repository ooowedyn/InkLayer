import torch
import json
import numpy as np


def process_boxes_ours(out_dict, input_pil):
    if "bboxes" in out_dict:
        norm_boxes = out_dict["bboxes"]  # (x1, y1, x2, y2)
    else:
        norm_boxes = out_dict
    # We need (cx, cy, w, h) format for processing
    res_boxes = []
    for norm_box in norm_boxes:
        x1, y1, x2, y2 = norm_box
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        res_boxes.append([cx, cy, w, h])
    size = input_pil.size
    H, W = size[1], size[0]
    boxes_filt = torch.tensor(res_boxes).float()

    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    return boxes_filt

def process_dino_output(out_dict, input_pil):  
    res_boxes_tensor = process_boxes_ours(out_dict, input_pil)
    return res_boxes_tensor, out_dict["labels"]


def save_norm_bboxes(bboxes_list, scores_list, input_pil, out_path, labels=None):
    # save normalized bboxes into a json 
    norm_bboxes = [
        [
            bbox[0] / input_pil.size[0],
            bbox[1] / input_pil.size[1],
            bbox[2] / input_pil.size[0],
            bbox[3] / input_pil.size[1],
        ]
        for bbox in bboxes_list
    ]
    res_obj = {
        "bboxes": norm_bboxes,
        "scores": scores_list,
    }
    if labels is not None:
        res_obj["labels"] = labels
    with open(out_path, "w") as f:
        json.dump(res_obj, f, indent=4)


def cxcywh_to_xyxy(boxes):
    boxes = np.array(boxes)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x_min = cx - w / 2
    y_min = cy - h / 2
    x_max = cx + w / 2
    y_max = cy + h / 2
    return np.stack([x_min, y_min, x_max, y_max], axis=-1)
