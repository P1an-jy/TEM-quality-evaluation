"""
00_generate_minimal_sim_stem.py
Minimal simulated STEM image generation for AtomSegNet localization accuracy experiment.

Generates 256x256 simulated STEM-like images with a square atomic lattice.
Each atomic column is a 2D Gaussian blob. Ground truth coordinates are saved
alongside for computing prediction errors (Δx, Δy, σx, σy).
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================
OUTPUT_ROOT = "quality_experiment"
IMAGE_SIZE = 256
LATTICE_SPACING = 16
BORDER = 24
GAUSSIAN_SIGMA = 2.0
PEAK_INTENSITY = 1.0
NUM_IMAGES = 10
OFFSET_RANGE = (-2.0, 2.0)
SEED = 42


# ============================================================
# Utility
# ============================================================

def make_output_dirs() -> dict:
    """Create output directory tree and return paths keyed by name."""
    dirs = {
        "clean":    os.path.join(OUTPUT_ROOT, "data", "clean"),
        "gt":       os.path.join(OUTPUT_ROOT, "data", "gt"),
        "preview":  os.path.join(OUTPUT_ROOT, "outputs", "preview"),
    }
    for p in dirs.values():
        os.makedirs(p, exist_ok=True)
    return dirs


def normalize01(arr: np.ndarray) -> np.ndarray:
    """Linearly rescale array values into [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# ============================================================
# Lattice generation
# ============================================================

def generate_lattice_points(offset_x: float = 0.0, offset_y: float = 0.0) -> np.ndarray:
    """
    Return (N, 2) array of atom centre coordinates for a square lattice.

    Columns: [x, y]  where  x = column coordinate, y = row coordinate.
    Lattice points are placed starting from BORDER and spaced by LATTICE_SPACING,
    stopping before IMAGE_SIZE - BORDER.  A random sub-pixel offset is applied
    globally to the whole lattice.
    """
    start = BORDER + 0.0                # nominal start
    stop  = IMAGE_SIZE - BORDER + 1e-9  # inclusive for floating equality

    cols = np.arange(start, stop, LATTICE_SPACING)  # x coordinates
    rows = np.arange(start, stop, LATTICE_SPACING)  # y coordinates

    xx_g, yy_g = np.meshgrid(cols, rows)            # xx_g: columns vary across columns; yy_g: rows vary across rows

    # Apply the random offset
    xx_g += offset_x
    yy_g += offset_y

    # Shape (num_rows * num_cols, 2)
    coords = np.column_stack([xx_g.ravel(), yy_g.ravel()])
    return coords


# ============================================================
# Rendering
# ============================================================

def gaussian_2d_kernel(sigma: float, half_size: int = 0) -> np.ndarray:
    """Return a 2D Gaussian kernel.  half_size is auto-set to ceil(3*sigma) if 0."""
    if half_size == 0:
        half_size = int(np.ceil(3 * sigma))
    xs = np.arange(-half_size, half_size + 1, dtype=np.float64)
    ys = np.arange(-half_size, half_size + 1, dtype=np.float64)
    xv, yv = np.meshgrid(xs, ys)
    kernel = np.exp(-(xv ** 2 + yv ** 2) / (2 * sigma * sigma))
    return kernel


def render_gaussian_atoms(coords: np.ndarray,
                          image_size: int,
                          sigma: float = GAUSSIAN_SIGMA,
                          peak: float = PEAK_INTENSITY) -> np.ndarray:
    """
    Render atom positions as 2D Gaussians onto a blank canvas.

    Parameters
    ----------
    coords : (N, 2) array of (x, y) centres.
    image_size : side length of the square output image.
    sigma : Gaussian sigma in pixels.
    peak  : peak intensity of a single atom.

    Returns
    -------
    image : np.ndarray of shape (image_size, image_size), dtype float64.
    """
    kernel = gaussian_2d_kernel(sigma)
    kh     = kernel.shape[0] // 2          # kernel half-size

    canvas = np.zeros((image_size, image_size), dtype=np.float64)

    for xc, yc in coords:
        # Integer pixel centre (nearest grid point)
        xi, yi = int(round(xc)), int(round(yc))

        # Determine placement on the canvas with clamping
        x0_src = 0
        x1_src = kernel.shape[1]
        y0_src = 0
        y1_src = kernel.shape[0]

        x0_dst = xi - kh
        x1_dst = xi + kh + 1
        y0_dst = yi - kh
        y1_dst = yi + kh + 1

        # Clip to image bounds (both source and destination)
        if x0_dst < 0:
            x0_src -= x0_dst
            x0_dst = 0
        if y0_dst < 0:
            y0_src -= y0_dst
            y0_dst = 0
        if x1_dst > image_size:
            x1_src -= (x1_dst - image_size)
            x1_dst = image_size
        if y1_dst > image_size:
            y1_src -= (y1_dst - image_size)
            y1_dst = image_size

        canvas[y0_dst:y1_dst, x0_dst:x1_dst] += kernel[y0_src:y1_src, x0_src:x1_src] * peak

    return canvas


# ============================================================
# I/O helpers
# ============================================================

def save_image(arr: np.ndarray, path: str):
    """Save a float64 image in [0, 1] as 8-bit grayscale PNG (0-255)."""
    img_uint8 = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(img_uint8, mode="L").save(path)


def save_gt_csv(coords: np.ndarray, path: str):
    """
    Save ground-truth atom coordinates as CSV.

    Columns: atom_id, x_gt, y_gt
    x_gt = column coordinate, y_gt = row coordinate.
    """
    n = coords.shape[0]
    df = pd.DataFrame({
        "atom_id": np.arange(n),
        "x_gt":    coords[:, 0],
        "y_gt":    coords[:, 1],
    })
    df.to_csv(path, index=False)


def save_overlay_preview(image: np.ndarray, coords: np.ndarray, path: str):
    """
    Save a preview PNG: clean image with red circles marking ground-truth atom positions.
    No GUI window is shown.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, origin="upper")
    # coords[:, 0] = x (column), coords[:, 1] = y (row)
    ax.scatter(coords[:, 0], coords[:, 1],
               s=12, facecolors="none", edgecolors="red", linewidths=0.6, alpha=0.8)
    ax.set_title("Overlay preview")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    rng = np.random.default_rng(SEED)

    # 1. Create directories
    dirs = make_output_dirs()

    # Prepare metadata rows
    metadata_rows = []

    for img_idx in range(NUM_IMAGES):
        # Random sub-pixel offset for this image
        offset_x = rng.uniform(*OFFSET_RANGE)
        offset_y = rng.uniform(*OFFSET_RANGE)

        # Generate lattice points with offset
        coords = generate_lattice_points(offset_x, offset_y)

        # Render Gaussian atoms
        raw = render_gaussian_atoms(coords, IMAGE_SIZE, GAUSSIAN_SIGMA, PEAK_INTENSITY)

        # Normalize to [0, 1]
        clean = normalize01(raw)

        # Build file paths
        img_id = f"img_{img_idx:03d}"
        clean_path   = os.path.join(dirs["clean"],   f"{img_id}.png")
        gt_path      = os.path.join(dirs["gt"],      f"{img_id}_gt.csv")
        preview_path = os.path.join(dirs["preview"], f"{img_id}_overlay.png")

        # Save
        save_image(clean, clean_path)
        save_gt_csv(coords, gt_path)
        save_overlay_preview(clean, coords, preview_path)

        # Collect metadata
        metadata_rows.append({
            "image_id":        img_id,
            "image_path":      clean_path,
            "gt_path":         gt_path,
            "offset_x":        round(offset_x, 4),
            "offset_y":        round(offset_y, 4),
            "lattice_spacing": LATTICE_SPACING,
            "gaussian_sigma":  GAUSSIAN_SIGMA,
            "image_size":      IMAGE_SIZE,
        })

    # Save metadata CSV
    meta_df = pd.DataFrame(metadata_rows)
    meta_path = os.path.join(OUTPUT_ROOT, "data", "metadata.csv")
    meta_df.to_csv(meta_path, index=False)

    print(f"generated {NUM_IMAGES} clean simulated STEM-like images")
    print(f"output root: {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
