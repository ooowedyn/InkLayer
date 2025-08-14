<p align="center">

  <div align="center">
    <img  src="docs/teaser.png">
  </div>

  <h1 align="center">Instance Segmentation of Scene Sketches Using Natural Image Priors</h1>
  <p align="center">
    <a href="https://mia-tang.com/" target="_blank"><strong>Mia Tang</strong></a>
    ·
    <a href="https://yael-vinker.github.io/website/" target="_blank"><strong>Yael Vinker</strong></a>
    ·
    <a href="https://nauhcnay.github.io/" target="_blank"><strong>Chuan Yan</strong></a>
    ·
    <a href="https://lllyasviel.github.io/Style2PaintsResearch/lvmin" target="_blank"><strong>Lvmin Zhang</strong></a>
    ·
    <a href="https://graphics.stanford.edu/~maneesh/" target="_blank"><strong>Maneesh Agrawala</strong></a>
  </p>
  <h2 align="center">SIGGRAPH 2025</h2>

  <div align="center">
    <img src="docs/thankful_handshake.png">
    We introduce <b><i>InkLayer</i></b>, a method for instance segmentation of raster scene sketches. It effectively handles diverse types of sketches, accommodating variations in stroke style and complexity.
  </div>

  <p align="center">
  <br>
    <a href="https://inklayer.github.io/" target="_blank"><strong>🌐 Project Page</strong></a>
    |
    <a href="https://arxiv.org/abs/2502.09608" target="_blank"><strong>📄 arXiv</strong></a>
  </p>
</p>


## 📚 Table of Contents
- [🔖 Release Status](#-release-status)
- [🛠️ Installation](#️-installation)
- [🏃‍♀️ Running Inference](#️-running-inference)
- [📎 Notes](#-notes)
- [🖊️ Citation](#-citation)

## 🔖 Release Status

- ✅ Benchmark dataset and dataset viewer: <a href="https://www.inkscenes-dataset.com/" target="_blank">🔗 Visit Our Viewer!</a>
- ✅ Segmentation inference code and weights
- ✅ Sketch layering code & sketch editing interface
- &#9744; Hugginface demo


## 🛠️ Installation

Please clone this repository with submodules!
```bash
# Clone the repository with submodules
git clone --recurse-submodules git@github.com:miatang13/InkLayer.git
cd InkLayer
```

## Environment Setup
```bash
conda create -n inklayer python=3.10
conda activate inklayer
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

<details>
<summary><strong>Check your CUDA installation</strong></summary>
Make sure you have CUDA set up correctly!

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
```

Run the sanity checks above. If you see the correct versions (something like `2.5.1` and `12.4`), you are good to go! 👏
</details>


### Install dependencies
You should see `GroundingDINO`, `segment-anything`, and `Depth_Anything_V2` in `InkLayer/third_party/` folder. If you do not, it means you did not clone the repository with submodules 😔. Run the following command to clone the submodules:
```bash
git submodule update --init --recursive
``` 

Okay, now let's set up the dependencies! 
```bash
(cd ./InkLayer/third_party/GroundingDINO && pip install -e . )
(cd ./InkLayer/third_party/segment-anything && pip install -e . )
```

### Download Weights
```bash
bash models/download_ckpts.sh
```

In this script we download the following weights:

| Model                   | File Name                    | Download Link |
|------------------------|------------------------------|---------------|
| Segment Anything       | `sam_vit_h_4b8939.pth`       | [Download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| Depth Anything V2 Base | `depth_anything_v2_vitb.pth` | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth) |
| InkLayer GroundingDINO | `inklayer_gdino.pth`         | [Download](https://huggingface.co/miatang13/InkLayer/resolve/main/inklayer_gdino.pth) |




<details>
<summary>A note on the weights</summary>

In our download script, we include `models/inklayer_gdino.pth`, which is our fine-tuned version of GroundingDINO for sketch detection, following the official GroundingDINO architecture and format. While we originally fine-tuned GroundingDINO using the [mmdetection](https://github.com/open-mmlab/mmdetection) framework, we converted the resulting weights to the original GroundingDINO format to simplify integration, as setting up mmdetection can be a bit more involved. If you prefer to use the mmdetection version directly, you can find it on [huggingface](https://huggingface.co/miatang13/InkLayer/tree/main) at `inklayer_gdino_mmdetection.pth`. The conversion was done using the script provided in this GitHub issue: [mmdetection issue](https://github.com/open-mmlab/mmdetection/issues/11200).

We observe very similar performance between the two versions of the weights, with slightly worse box IoU for the official GroundingDINO format, but higher AR and AP. We recommend using this version if your task is not highly sensitive to box precision. To replicate exact evaluation results reported in the paper, please use the mmdetection version of the weights. Feel free to reach out to miatang@stanford dot edu if you need help with the mmdetection version.

</details>

### Install InkLayer
Now you can install InkLayer. At the root directory, run
```bash 
pip install -e .
pip install scikit-image # for basic segmentation (required)
pip install diffusers accelerate # for inpainting (skip if not needed)
pip install flask # for sketch editing interface (skip if not needed)
```
Now you should be able to import InkLayer anywhere in your Python scripts! 🎉


## 🏃‍♀️ Running Inference

### Segmentation Inference
You can run inference on a single image using the following command:
```bash
python main.py --img {PATH_TO_YOUR_IMAGE}
```
A sample command using our test sketch is
```bash
python main.py --img data/bunny_cook_sketch.png 
``` 
To run inference on an entire directory of images:
```bash
python main.py --dir {PATH_TO_IMAGE_DIRECTORY}
```
By default, all outputs will be saved to `./output/{IMAGE_NAME}/`. You can specify a different output directory using the `--out_dir` argument:

If you would like to skip saving intermediate outputs (e.g., intermediate masks, visualizations), you can add the `--no_intermediate` flag:

```bash
python main.py --img data/bunny_cook_sketch.png --no_intermediate
```

The final segmented sketch is visualized at `./{OUT_DIR}/{IMAGE_NAME}/segmented_sketch_final.png` and the masks are at `./{OUT_DIR}/{IMAGE_NAME}/masks_final/`. 

<details>
<summary><strong>Common Questions</strong></summary>

1. Computation Time: The inference time depends on the size of the input image, and how complex the sketch is. It can take a little while if your sketch is on the extreme complicated side due to our non-optimized sketch NMS process.

2. libpng Warning: If you see `libpng warning: iCCP: known incorrect sRGB profile` when running the inference, it means your input image has an incorrect sRGB profile. You can fix it following the instructions in this [StackOverflow post](https://stackoverflow.com/questions/22745076/libpng-warning-iccp-known-incorrect-srgb-profile).

</details>

### Layer Completion
To run the layer completion step, you just add a `--inpaint` flag to calling the `main.py` script. The output will be saved to `{OUT_DIR}/complete_layers` or `complete_layers_rgba` if you want the layers with transparent background.  

<details>
<summary><strong>Running on existing segmented sketch</strong></summary>
To run the layer completion step on an existing segmentation output directory, simply run <code>python InkLayer/inpaint_ControlNet.py --dir {PATH_TO_SEG_OUTPUT_DIR}</code>. Our default code uses the stable diffusion inpaintng with controlnet conditioned for best performance. 
</details>

<details>
<summary><strong>⚠️ A note on completion quality</strong></summary>
The performance heavily depends on the inpainting model (and is generally highly stochastic), so we encourage you to experiment with different models and seeds. Also feel free to play around with the inpainting parameters and prompt.
</details>


## 🏃‍♀️ Running Sketch Editing Interface
We also provide a sketch editing interface that allow you to upload / draw a sketch and edit it interactively.

To start the interface, run the following command:
```bash
python -m custom_interface
```
This will start a Flask server on `http://localhost:5000/`. You can open this URL in your web browser to access the interface.

## 📎 Notes
For reference, here are the  commit hash for the submodules that we used in our experiments:
```bash
GroundingDINO: 856dde20aee659246248e20734ef9ba5214f5e44
segment-anything: 3f6d89896768f04ded863803775069855c5360b6
Depth_Anything_V2: e5a2732d3ea2cddc081d7bfd708fc0bf09f812f1
```

## 🖊️ Citation
If you find our work useful, please consider citing our paper:
```bibtex
@inproceedings{tang2025inklayer,
  author = {Tang, Mia and Vinker, Yael and Yan, Chuan and Zhang, Lvmin and Agrawala, Maneesh},
  title = {Instance Segmentation of Scene Sketches Using Natural Image Priors},
  year = {2025},
  isbn = {9798400715402},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3721238.3730606},
  doi = {10.1145/3721238.3730606},
  articleno = {96},
  numpages = {10},
  keywords = {Sketch Segmentation, Sketch Understanding, Sketch Editing},
  series = {SIGGRAPH Conference Papers '25}
}
```

If you have any questions, please feel free to reach out to us at miatang@stanford dot edu. We will respond as soon as we can! Thank you for your interest in our work. Happy sketching! 😊