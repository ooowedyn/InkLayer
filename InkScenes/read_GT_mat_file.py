import numpy as np
from scipy.io import loadmat
import matplotlib.colors as mcolors
import argparse
from PIL import Image

def generate_pastel_colors(n_colors):
    """Generate pastel colors with interleaved hue ordering for better contrast"""
    # Generate evenly spaced hues
    hues = [x / n_colors for x in range(n_colors)]

    # Interleave hues to maximize contrast
    def interleave(lst):
        result = []
        queue = [lst]
        while queue:
            current = queue.pop(0)
            if len(current) <= 1:
                result += current
            else:
                queue.append(current[::2])
                queue.append(current[1::2])
        return result

    reordered_hues = interleave(hues)

    # Convert to RGB (pastel style with fixed S and V)
    colors = [
        mcolors.hsv_to_rgb([h, 0.7, 0.88]) for h in reordered_hues
    ]

    # Convert to 0-255 RGB
    colors = [
        (int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors
    ]
    
    return colors


def visualize_label_matrix(mat_path, mat_type, out_path):
    """
    mat_path: path to the .mat file
    mat_type: type of the matrix to visualize (e.g., "INSTANCE_GT" or "CLASS_GT")
    out_path: path to the output visualization image
    """
    
    # Load the .mat file
    label_matrix = loadmat(mat_path)[mat_type]

    # Get number of unique labels (including background)
    unique_labels = np.unique(label_matrix)
    num_labels = len(unique_labels)

    # Generate colors (white for background, random for others)
    colors = [(255, 255, 255)] + generate_pastel_colors(num_labels - 1)

    # Create RGB image
    height, width = label_matrix.shape
    rgb_image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Fill in colors for each label
    for idx, label in enumerate(unique_labels):
        if label == 0:  # background
            continue
        mask = label_matrix == label
        rgb_image[mask] = colors[idx]

    annotated_sketch = Image.fromarray(rgb_image)
    annotated_sketch.save(out_path)
    return annotated_sketch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize instance ground truth from .mat file")
    parser.add_argument("--mat_file", type=str, required=True, help="Path to the .mat file")
    parser.add_argument("--type", type=str, default="INSTANCE_GT", help="Type of matrix to visualize (e.g., INSTANCE_GT or CLASS_GT)")
    parser.add_argument("--output", type=str, default="./vis_mat.png", help="Path to the output visualization image")
    args = parser.parse_args()

    # Basic visualization
    labels = visualize_label_matrix(args.mat_file, args.type, args.output)