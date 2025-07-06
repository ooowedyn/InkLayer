import os
import sys
import cv2
import numpy as np
import torch
from segment_anything import (
    build_sam,
    SamPredictor,
)
from InkLayer.utils.paths import get_model_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
default_ckpt = get_model_path("sam_vit_h_4b8939.pth")


def run_SAM(image_pil, boxes_filt, sam_checkpoint=default_ckpt):
    if not os.path.exists(sam_checkpoint):
        print(f"Checkpoint not found at {sam_checkpoint}")
        breakpoint()
    if len(boxes_filt) == 0:
        return np.array([]), np.array([])
    
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    open_cv_image = np.array(image_pil)
    image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes_filt, image.shape[:2]
    ).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )
    
    # masks is cuda tensor, we want a list
    masks_np = []
    for mask in masks:
        masks_np.append(mask.cpu().numpy()[0])
    del predictor
    return masks_np
