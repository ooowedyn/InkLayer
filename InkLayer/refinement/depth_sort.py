import numpy as np
import cv2
import torch
from scipy.spatial import KDTree
import os
import matplotlib.pyplot as plt
import glob
from PIL import Image

import InkLayer
proj_dir = os.path.dirname(InkLayer.__file__)
from InkLayer.third_party.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
from InkLayer.refinement.utils import sketch_to_01binary, get_binned_frequent

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
depth_model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}

encoder = "vitb"  # or 'vits', 'vitb', 'vitg'
depth_model = DepthAnythingV2(**depth_model_configs[encoder])
depth_model.load_state_dict(
    torch.load(f"{proj_dir}/../models/depth_anything_v2_{encoder}.pth", map_location="cpu")
)
depth_model = depth_model.to(DEVICE).eval()

def get_depth_map(sketch_path):
    sketch = cv2.imread(sketch_path)
    depth = depth_model.infer_image(sketch)  # HxW raw depth map in numpy
    return depth


def sparse_sketch_sample(binary_edge_map):
    radius = binary_edge_map.shape[0] * 0.01
    # Get all edge coordinates
    edge_points = np.column_stack(np.where(binary_edge_map > 0))
    # number of points we will sample on the edge map
    num_points = int(0.05 * len(edge_points))
    # Convert edge points to a KDTree for efficient distance queries
    tree = KDTree(edge_points)
    sampled_points = []
    remaining_indices = set(range(len(edge_points)))
    while remaining_indices:
        # Take the first available point
        current_index = next(iter(remaining_indices))
        current_point = edge_points[current_index]
        sampled_points.append(tuple(current_point))
        # Find all points within the radius of the current point
        indices_to_exclude = tree.query_ball_point(current_point, radius)
        # Remove these indices from the remaining set
        remaining_indices.difference_update(indices_to_exclude)
    return sampled_points



def get_mask_depth_score(mask, edge_points, depth_map):
    depth_values = []
    depth_value_points = []

    # 1. First get depth from edge points that lie within mask
    for point in edge_points:
        y, x = point
        if mask[y, x]:
            depth_values.append(depth_map[y, x])
            depth_value_points.append([y, x])

    if not depth_values:
        return float("inf")  # Return infinity if no points in mask

    res = get_binned_frequent(depth_values)
    return res

def compute_mask_overlap(mask1, mask2):
    """
    Compute overlap metrics between two binary masks.
    
    Returns:
        intersection_area: Number of overlapping pixels
        mask1_area: Total pixels in mask1
        mask2_area: Total pixels in mask2
        overlap_ratio_1: intersection / mask1_area
        overlap_ratio_2: intersection / mask2_area
    """
    intersection = np.logical_and(mask1, mask2)
    intersection_area = np.sum(intersection)
    
    mask1_area = np.sum(mask1)
    mask2_area = np.sum(mask2)
    
    overlap_ratio_1 = intersection_area / (mask1_area + 1e-6)
    overlap_ratio_2 = intersection_area / (mask2_area + 1e-6)
    
    return intersection_area, mask1_area, mask2_area, overlap_ratio_1, overlap_ratio_2

def build_containment_graph_fast(
    bboxes,                # iterable of (x1,y1,x2,y2); pixels or [0,1]
    image_size,            # (H, W)
):
    """
    Returns graph[i, j] = True iff box i strictly contains box j (directional).
    No IoU-as-containment. Small pixel slack, relative area check, center-in check.
    """
    import numpy as np

    if bboxes is None or len(bboxes) == 0:
        return np.zeros((0, 0), dtype=bool)

    H, W = int(image_size[0]), int(image_size[1])

    b = np.asarray(bboxes, dtype=float)  # (N,4)
    if b.ndim != 2 or b.shape[1] != 4:
        raise ValueError(f"bboxes must be (N,4); got {b.shape}")

    # If boxes look normalized, convert to pixels
    if np.max(b) <= 1.0 + 1e-6:
        b[:, [0, 2]] *= W
        b[:, [1, 3]] *= H

    # normalize coords so x1<=x2, y1<=y2
    x1 = np.minimum(b[:, 0], b[:, 2])
    x2 = np.maximum(b[:, 0], b[:, 2])
    y1 = np.minimum(b[:, 1], b[:, 3])
    y2 = np.maximum(b[:, 1], b[:, 3])
    b  = np.stack([x1, y1, x2, y2], axis=1)

    # --- strict containment tests ---
    # Use a small pixel slack; DO NOT derive from (H,W) tuple directly
    eps_px = float(max(1.0, 0.002 * max(H, W)))   # ~0.2% of max dim, at least 1px
    min_area_gap = 0.02                           # inner must be >=2% smaller

    w  = np.clip(b[:, 2] - b[:, 0], 0, None)
    h  = np.clip(b[:, 3] - b[:, 1], 0, None)
    areas = w * h
    cx = (b[:, 0] + b[:, 2]) * 0.5
    cy = (b[:, 1] + b[:, 3]) * 0.5

    b1, b2 = b[:, None, :], b[None, :, :]  # (N,1,4) vs (1,N,4)

    # corner-in-box (with slack)
    contained = (
        (b1[..., 0] - eps_px <= b2[..., 0]) &   # left
        (b1[..., 1] - eps_px <= b2[..., 1]) &   # top
        (b1[..., 2] + eps_px >= b2[..., 2]) &   # right
        (b1[..., 3] + eps_px >= b2[..., 3])     # bottom
    )

    # relative size filter (avoid near-duplicates)
    area_ratio_ok = (areas[:, None] * (1.0 - min_area_gap)) > areas[None, :]
    contained &= area_ratio_ok

    # center-in check to strengthen directionality
    cx_in = (b1[..., 0] - eps_px <= cx[None, :]) & (cx[None, :] <= b1[..., 2] + eps_px)
    cy_in = (b1[..., 1] - eps_px <= cy[None, :]) & (cy[None, :] <= b1[..., 3] + eps_px)
    contained &= (cx_in & cy_in)

    np.fill_diagonal(contained, False)
    return contained.astype(bool)

def compute_major_overlap_matrix(
    masks,
    bboxes=None,          # list of (x1,y1,x2,y2) in *pixels*; computed if None
    thr=0.6,              # major-overlap threshold
    dilate_px=1,          # dilate strokes to be robust to thin lines / small gaps
):
    """
    major_overlap[i, j] = True iff intersection(m_i, m_j) / min(|m_i|, |m_j|) >= thr
    """
    # prepare masks
    M = [m.astype(np.uint8) for m in masks]
    H, W = M[0].shape

    # optional dilation to catch 1px strokes / tiny misalignments
    if dilate_px and dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        M = [cv2.dilate(m, kernel) for m in M]

    # areas
    areas = np.array([int(m.sum()) for m in M], dtype=np.int64)

    # bboxes if missing
    if bboxes is None:
        bboxes = []
        for m in M:
            ys, xs = np.where(m > 0)
            if len(ys) == 0:
                bboxes.append((0, 0, 0, 0))
            else:
                bboxes.append((int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)))
    b = np.asarray(bboxes, dtype=int)
    N = len(M)

    major = np.zeros((N, N), dtype=bool)

    for i in range(N):
        x1i, y1i, x2i, y2i = b[i]
        if areas[i] == 0 or x2i <= x1i or y2i <= y1i:
            continue
        for j in range(i+1, N):
            x1j, y1j, x2j, y2j = b[j]
            if areas[j] == 0 or x2j <= x1j or y2j <= y1j:
                continue

            # bbox intersection
            xi1, yi1 = max(x1i, x1j), max(y1i, y1j)
            xi2, yi2 = min(x2i, x2j), min(y2i, y2j)
            if xi2 <= xi1 or yi2 <= yi1:
                continue

            sub_i = M[i][yi1:yi2, xi1:xi2]
            sub_j = M[j][yi1:yi2, xi1:xi2]
            inter = int(np.count_nonzero(sub_i & sub_j))
            if inter == 0:
                continue

            denom = float(min(areas[i], areas[j]))
            r = inter / denom if denom > 0 else 0.0
            if r >= thr:
                major[i, j] = major[j, i] = True

    return major



def sort_sketch_masks(masks, bboxes, sketch_path, depth_sketch=None):
    # Validate inputs
    assert os.path.exists(sketch_path), f"Sketch path {sketch_path} does not exist."
    if depth_sketch is None:
        depth_sketch = get_depth_map(sketch_path)

    # Get image dimensions
    image = cv2.imread(sketch_path)
    h, w = image.shape[:2]

    # Convert sketch to binary and sample points
    binary_sketch = sketch_to_01binary(image)
    sampled_points = sparse_sketch_sample(binary_sketch)

    # Scale bboxes if normalized
    if np.all(np.array(bboxes) <= 1.0):
        bboxes = [box * np.array([w, h, w, h]) for box in bboxes]

    # Get initial depth scores
    depth_scores = []
    for mask in masks:
        score = get_mask_depth_score(mask, sampled_points, depth_sketch)
        depth_scores.append(score)

    # Build containment relationships
    containment = build_containment_graph_fast(bboxes, image_size=(h, w))

    sketch_masks = [mask & binary_sketch.astype(bool) for mask in masks]
    overlap = compute_major_overlap_matrix(sketch_masks, bboxes=bboxes, dilate_px=1)

    # Get initial sorting based on depth scores
    final_order = list(np.argsort(depth_scores)[::-1])
    
    for _ in range(3):
        for i in range(len(final_order)):
            for j in range(i + 1, len(final_order)):
                a = final_order[i]  
                b = final_order[j] 

                # Gate by pixel overlap:
                if not overlap[a, b]:
                    continue

                # If the later box is the container, move it earlier
                # (container first when they overlap)
                if containment[a, b]:
                    final_order[i], final_order[j] = final_order[j], final_order[i]

    return final_order, depth_scores, containment


def create_depth_points_figure(sketch_path, masks_dir, save_path, point_size=15):
    """
    Create a single panel figure showing depth map with sampled points overlay.
    Points are colored by which mask they belong to.
    
    Args:
        sketch_path: Path to the input sketch image
        masks_dir: Path to directory containing mask files
        save_path: Path to save the figure
        point_size: Size of the plotted points
    """
    
    # Read all mask files from directory
    mask_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    mask_files = []
    for ext in mask_extensions:
        mask_files.extend(glob.glob(os.path.join(masks_dir, ext)))
    
    # Sort mask files for consistent ordering
    mask_files.sort()
    
    if not mask_files:
        raise ValueError(f"No mask files found in directory: {masks_dir}")
    
    print(f"Found {len(mask_files)} mask files:")
    for i, mask_file in enumerate(mask_files):
        print(f"  {i+1}: {os.path.basename(mask_file)}")
    
    # Load all masks
    masks = []
    for mask_file in mask_files:
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not load mask {mask_file}")
            continue
        # Convert to binary (assuming masks are binary or can be thresholded)
        mask_binary = (mask > 127).astype(bool)
        masks.append(mask_binary)
    
    if not masks:
        raise ValueError("No valid masks could be loaded")
    
    # Get depth and sampled points using your existing functions
    sketch = cv2.imread(sketch_path)
    depth_map = get_depth_map(sketch_path)
    binary_sketch = sketch_to_01binary(sketch)
    sampled_points = sparse_sketch_sample(binary_sketch)
    
    # Normalize depth for visualization
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    # Create a figure of just this depth map and save it
    plt.figure(figsize=(8, 8))

    # Show depth map
    plt.imshow(depth_normalized, cmap='viridis')
    #turn off axis
    plt.axis('off')
    plt.savefig("./depth_map.png", dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"Depth map saved to: ./depth_map.png")
    plt.close()  # Close the figure to free memory

    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Show depth map
    plt.imshow(depth_normalized, cmap='viridis')
    
    # Generate distinct colors for each mask
    if len(masks) == 2:
        # Hard code for figure
        colors = ['red', 'yellow']
    elif len(masks) < 10:
        colors = plt.cm.Set1(np.linspace(0, 1, 4))  # or use 'tab10', 'Set3', etc.
    else:
        colors = plt.cm.Set1(np.linspace(0, 1, len(masks)))  # More than 10 masks, use all colors

    # For each mask, find points that fall within it and plot with distinct color
    total_points_plotted = 0
    mask_point_counts = []
    
    for mask_idx, mask in enumerate(masks):
        mask_points = []
        
        for point in sampled_points:
            y, x = point
            if mask[y, x]:  # Point falls within this mask
                mask_points.append([x, y])  # Note: switched to x,y for plotting
        
        if len(mask_points) > 0:
            mask_points = np.array(mask_points)
            mask_name = os.path.splitext(os.path.basename(mask_files[mask_idx]))[0]
            plt.scatter(mask_points[:, 0], mask_points[:, 1], 
                       c=[colors[mask_idx]], s=point_size, 
                       alpha=1,
                       label=f'{mask_name} ({len(mask_points)} pts)')
            total_points_plotted += len(mask_points)
        
        mask_point_counts.append(len(mask_points))
    
    plt.axis('off')
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"Figure saved to: {save_path}")
    
    plt.close()  # Close the figure to free memory
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"Total sampled points: {len(sampled_points)}")
    print(f"Points plotted (within masks): {total_points_plotted}")
    for i, count in enumerate(mask_point_counts):
        mask_name = os.path.splitext(os.path.basename(mask_files[i]))[0]
        print(f"  {mask_name}: {count} points")
    print(f"Points outside all masks: {len(sampled_points) - total_points_plotted}")


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser(description="Create depth points figure from sketch.")
    parser.add_argument("--sketch", type=str, required=True, help="Path to sketch")
    parser.add_argument("--masks_dir", type=str, required=True, help="Directory containing mask files")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the figure")
    args = parser.parse_args()
    print(f"args: {args}")

    create_depth_points_figure(args.sketch, args.masks_dir, args.save_path)
