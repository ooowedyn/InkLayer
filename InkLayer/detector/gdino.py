from groundingdino.util.inference import load_model, load_image, predict
import os 
import InkLayer 
from InkLayer.utils.processing import cxcywh_to_xyxy
from InkLayer.utils.paths import get_model_path

inklayer_root = os.path.dirname(InkLayer.__file__)
gdino_config_path = get_model_path("GroundingDINO_SwinT_OGC.py")
weights_path = get_model_path("inklayer_gdino.pth")
model = load_model(gdino_config_path, weights_path)

def run_ft_dino_on_sketch(sketch_path):
    image_source, image = load_image(sketch_path)
    
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption="object",
        box_threshold=0.2, # You can adjust this threshold based on your needs
        text_threshold=0
    )

    normalized_bboxes = boxes.tolist() # cxcywh
    normalized_bboxes = cxcywh_to_xyxy(normalized_bboxes) # xyxy
    normalized_bboxes = normalized_bboxes.tolist()
    out_dict = { 
                "bboxes": normalized_bboxes,
                "scores": logits.tolist(),
                "labels": phrases}
    return out_dict
    