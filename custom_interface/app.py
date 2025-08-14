from flask import Flask, jsonify, render_template, request
import glob
import os
import sys
import base64
from PIL import Image
import io
from datetime import datetime
import torch
import gc

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir)
from InkLayer.runner import run_inklayer_pipeline

project_dir = os.path.abspath(f"{cur_dir}/../../")
sys.path.append(project_dir)

app = Flask(__name__)


@app.route("/")
def index():
    torch.cuda.empty_cache()
    gc.collect()
    return render_template(
        "index.html",
    )



upload_dir = os.path.abspath(f"{cur_dir}/static/uploads/")
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)
output_dir = os.path.abspath(f"{cur_dir}/static/outputs/")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def disk_path_to_url(disk_path):
    return disk_path.replace(cur_dir, "")


def get_sketch_layers_urls(dir_path):
    abs_path_urls = glob.glob(f"{dir_path}/complete_layers_rgba/layer_*.png")
    sorted_urls = sorted(
        abs_path_urls, key=lambda x: int(x.split("layer_")[1].split(".")[0])
    )
    abs_path_urls = [url.replace(cur_dir, "") for url in sorted_urls]
    # reverse it
    abs_path_urls = abs_path_urls[::-1]
    print(f"abs_path_urls: {abs_path_urls}")
    return abs_path_urls


def retrieve_input_image_path(image_name):
    # Look for the image with any common extension
    possible_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
    image_path = []
    
    # Search in upload directory
    for ext in possible_extensions:
        image_path.extend(glob.glob(f"{upload_dir}/{image_name}.{ext}"))

    
    if not image_path:
        print(f"Could not find image {image_name} at {upload_dir}")
        return None
    return image_path[0]


@app.route("/segment-sketch", methods=["POST"])
def segment_sketch():
    data = request.get_json()
    image_name = data.get("imageName")
    print(f"Segmenting sketch for {image_name}")
    if not image_name:
        return jsonify({"error": "No image name provided"}), 400

    image_path = retrieve_input_image_path(image_name)
    if not image_path:
        return jsonify({"error": f"Image not found: {image_name}"}), 404

    print(f"Processing image at path: {image_path}")

    try:
        log_dir = run_inklayer_pipeline(image_path, out_base_dir=output_dir, 
                                        no_intermediate=False, inpaint=True)
        assert os.path.exists(log_dir), f"result dir {log_dir} does not exist"

        layers_urls = get_sketch_layers_urls(f"{log_dir}")
        result = {
            "message": f"Segmentation completed for {image_name}",
            "layers": layers_urls,
        }

        torch.cuda.empty_cache()
        gc.collect()
        return jsonify(result)
        
    except Exception as e:
        print(f"Error during segmentation: {str(e)}")
        return jsonify({"error": f"Segmentation failed: {str(e)}"}), 500


@app.route("/upload-image", methods=["POST"])
def upload_image():
    print(f"Uploading image")
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Ensure the filename has a proper extension
        filename = file.filename
        if not any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']):
            if '.' not in filename:
                filename += '.png'
            else:
                filename = filename.rsplit('.', 1)[0] + '.png'

        disk_path = os.path.join(upload_dir, filename)
        file_path = f"static/uploads/{filename}"
        
        # Save the file
        file.save(disk_path)
        print(f"File uploaded to {disk_path}")

        try:
            with Image.open(disk_path) as img:
                # Convert to RGB if necessary (for PNG with transparency)
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        background.paste(img, mask=img.split()[-1])  
                    else:
                        background.paste(img, mask=img.split()[-1]) 
                    img = background
                    img.save(disk_path)
                print(f"Image verified and processed: {img.size}, mode: {img.mode}")
        except Exception as img_error:
            print(f"Error processing image: {str(img_error)}")
            return jsonify({"error": f"Invalid image file: {str(img_error)}"}), 400

        torch.cuda.empty_cache()
        gc.collect()

        return jsonify({
            "message": "File uploaded successfully", 
            "file_path": file_path,
            "filename": filename
        })
        
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500



@app.route("/save-canvas-drawing", methods=["POST"])
def save_canvas_drawing():
    """
    Alternative endpoint specifically for saving canvas drawings as base64 data
    """
    try:
        data = request.get_json()
        image_data = data.get("imageData")
        filename = data.get("filename", f"canvas_drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        
        if image_data.startswith('data:image/png;base64,'):
            image_data = image_data.split(',', 1)[1]
        
        image_bytes = base64.b64decode(image_data)
        
        if not filename.endswith('.png'):
            filename += '.png'
        
        file_path = os.path.join(upload_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(image_bytes)
        
        print(f"Canvas drawing saved to {file_path}")
        
        return jsonify({
            "message": "Canvas drawing saved successfully",
            "filename": filename,
            "file_path": f"static/uploads/{filename}"
        })
        
    except Exception as e:
        print(f"Error saving canvas drawing: {str(e)}")
        return jsonify({"error": f"Failed to save canvas drawing: {str(e)}"}), 500


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)