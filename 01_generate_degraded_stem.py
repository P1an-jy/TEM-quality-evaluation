"""
01_generate_degraded_stem.py
基于 data/clean/ 中的 simulated STEM 图像，生成多质量退化版本，
用于 AtomSegNet 定位精度单因素实验。

三种退化模式：Poisson 噪声、背景叠加、Gaussian 模糊。
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
SEED = 42
DATA_ROOT = "quality_experiment"

CLEAN_DIR    = os.path.join(DATA_ROOT, "data", "clean")
GT_DIR       = os.path.join(DATA_ROOT, "data", "gt")
DEGRADED_DIR = os.path.join(DATA_ROOT, "data", "degraded")
PREVIEW_DIR  = os.path.join(DATA_ROOT, "outputs", "degraded_preview")

POISSON_COUNTS = [1000, 500, 200, 100, 50]
BLUR_SIGMAS    = [0.5, 1.0, 1.5, 2.0]

# (level_name, bg_type, alpha)   linear / nonlinear 各取两个 alpha
BACKGROUND_CONFIGS = [
    ("B_linear_010",  "linear",    0.10),
    ("B_linear_030",  "linear",    0.30),
    ("B_nonlin_010",  "nonlinear", 0.10),
    ("B_nonlin_030",  "nonlinear", 0.30),
]


# ============================================================
# Utility
# ============================================================

def normalize01(arr: np.ndarray) -> np.ndarray:
    """Linearly rescale to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def make_output_dirs():
    """Create all degraded-output and preview directories."""
    for c in POISSON_COUNTS:
        os.makedirs(os.path.join(DEGRADED_DIR, "poisson", f"P{c}"), exist_ok=True)
    for lvl, _, _ in BACKGROUND_CONFIGS:
        os.makedirs(os.path.join(DEGRADED_DIR, "background", lvl), exist_ok=True)
    for s in BLUR_SIGMAS:
        os.makedirs(os.path.join(DEGRADED_DIR, "blur", f"S{s}"), exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)


# ============================================================
# I/O helpers
# ============================================================

def load_clean_images() -> dict:
    """Return dict {img_id: float64 array in [0,1]} for all clean PNGs."""
    images = {}
    for fname in sorted(os.listdir(CLEAN_DIR)):
        if fname.lower().endswith(".png"):
            img_id = os.path.splitext(fname)[0]
            arr = np.array(Image.open(os.path.join(CLEAN_DIR, fname)), dtype=np.float64)
            images[img_id] = arr / 255.0
    return images


def load_gt(img_id: str) -> np.ndarray:
    """Load GT coordinates for img_id, return (N,2) array [x_gt, y_gt]."""
    gt_path = os.path.join(GT_DIR, f"{img_id}_gt.csv")
    df = pd.read_csv(gt_path)
    return df[["x_gt", "y_gt"]].values


def save_image(arr: np.ndarray, path: str):
    """Save float [0,1] image as 8-bit grayscale PNG."""
    uint8 = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(uint8, mode="L").save(path)


# ============================================================
# Degradation functions
# ============================================================

def degrade_poisson(image: np.ndarray, electron_count: int, rng: np.random.Generator) -> np.ndarray:
    """Poisson shot noise.  image in [0,1], count = mean electrons per pixel."""
    scaled = image * electron_count
    noisy  = rng.poisson(scaled) / electron_count
    return normalize01(noisy)


def generate_background(shape: tuple, bg_type: str, rng: np.random.Generator) -> np.ndarray:
    """
    Return a background image (float64, [0,1]).

    bg_type == 'linear'   →  a*x + b*y  plane, a,b ∈ [-0.3, 0.3]
    bg_type == 'nonlinear' → Gaussian-filtered white noise, sigma=20
    """
    h, w = shape
    yv, xv = np.meshgrid(np.arange(h, dtype=np.float64),
                         np.arange(w, dtype=np.float64),
                         indexing="ij")

    if bg_type == "linear":
        a = rng.uniform(-0.3, 0.3)
        b = rng.uniform(-0.3, 0.3)
        bg = a * (xv / (w - 1)) + b * (yv / (h - 1))
    else:   # nonlinear
        noise = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float64)
        bg = gaussian_filter(noise, sigma=20.0)

    return normalize01(bg)


def degrade_background(image: np.ndarray, bg_type: str, alpha: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Generate background, add to image, re-normalize."""
    bg = generate_background(image.shape, bg_type, rng)
    result = image + alpha * bg
    return normalize01(result)


def degrade_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur then re-normalize."""
    blurred = gaussian_filter(image, sigma=sigma)
    return normalize01(blurred)


# ============================================================
# Preview
# ============================================================

def save_preview(image: np.ndarray, gt_coords: np.ndarray, path: str, title: str = ""):
    """Save overlay preview: degraded image + red GT markers."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, origin="upper")
    ax.scatter(gt_coords[:, 0], gt_coords[:, 1],
               s=10, facecolors="none", edgecolors="red",
               linewidths=0.5, alpha=0.7)
    if title:
        ax.set_title(title, fontsize=8)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    rng = np.random.default_rng(SEED)
    make_output_dirs()

    clean_images = load_clean_images()
    img_ids = sorted(clean_images.keys())
    print(f"loaded {len(img_ids)} clean images")

    metadata_rows = []
    preview_pool = {"poisson": [], "background": [], "blur": []}

    # --------------------------------------------------------
    # (1) Poisson noise
    # --------------------------------------------------------
    for count in POISSON_COUNTS:
        for img_id in img_ids:
            clean = clean_images[img_id]
            degraded = degrade_poisson(clean, count, rng)

            rel_dir = os.path.join("poisson", f"P{count}")
            out_path = os.path.join(DEGRADED_DIR, rel_dir, f"{img_id}.png")
            save_image(degraded, out_path)

            gt_path = os.path.join(GT_DIR, f"{img_id}_gt.csv")
            metadata_rows.append({
                "image_id":         img_id,
                "degradation_type": "poisson",
                "level":            f"P{count}",
                "param_value":      count,
                "image_path":       out_path,
                "gt_path":          gt_path,
            })
            preview_pool["poisson"].append((degraded, img_id, f"Poisson e⁻={count}"))

    # --------------------------------------------------------
    # (2) Background
    # --------------------------------------------------------
    for level_name, bg_type, alpha in BACKGROUND_CONFIGS:
        for img_id in img_ids:
            clean = clean_images[img_id]
            degraded = degrade_background(clean, bg_type, alpha, rng)

            rel_dir = os.path.join("background", level_name)
            out_path = os.path.join(DEGRADED_DIR, rel_dir, f"{img_id}.png")
            save_image(degraded, out_path)

            gt_path = os.path.join(GT_DIR, f"{img_id}_gt.csv")
            metadata_rows.append({
                "image_id":         img_id,
                "degradation_type": "background",
                "level":            level_name,
                "param_value":      f"{bg_type}_α{alpha}",
                "image_path":       out_path,
                "gt_path":          gt_path,
            })
            preview_pool["background"].append((degraded, img_id,
                                               f"BG {bg_type} α={alpha}"))

    # --------------------------------------------------------
    # (3) Blur
    # --------------------------------------------------------
    for sigma in BLUR_SIGMAS:
        for img_id in img_ids:
            clean = clean_images[img_id]
            degraded = degrade_blur(clean, sigma)

            rel_dir = os.path.join("blur", f"S{sigma}")
            out_path = os.path.join(DEGRADED_DIR, rel_dir, f"{img_id}.png")
            save_image(degraded, out_path)

            gt_path = os.path.join(GT_DIR, f"{img_id}_gt.csv")
            metadata_rows.append({
                "image_id":         img_id,
                "degradation_type": "blur",
                "level":            f"S{sigma}",
                "param_value":      sigma,
                "image_path":       out_path,
                "gt_path":          gt_path,
            })
            preview_pool["blur"].append((degraded, img_id, f"Blur σ={sigma}"))

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------
    meta_df = pd.DataFrame(metadata_rows)
    meta_path = os.path.join(DATA_ROOT, "data", "degraded_metadata.csv")
    meta_df.to_csv(meta_path, index=False)

    # --------------------------------------------------------
    # Preview: 每种退化随机选 3 张
    # --------------------------------------------------------
    preview_rng = np.random.default_rng(SEED + 1)
    for dtype in ["poisson", "background", "blur"]:
        candidates = preview_pool[dtype]
        n_pick = min(3, len(candidates))
        chosen = preview_rng.choice(len(candidates), size=n_pick, replace=False)
        for j, idx in enumerate(chosen):
            degraded, img_id, title = candidates[idx]
            gt_coords = load_gt(img_id)
            p_path = os.path.join(PREVIEW_DIR, f"{dtype}_{j + 1:02d}.png")
            save_preview(degraded, gt_coords, p_path,
                         title=f"{title}  [{img_id}]")

    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------
    n_poisson    = len(POISSON_COUNTS) * len(img_ids)
    n_background = len(BACKGROUND_CONFIGS) * len(img_ids)
    n_blur       = len(BLUR_SIGMAS) * len(img_ids)

    print(f"generated {n_poisson} poisson images")
    print(f"generated {n_background} background images")
    print(f"generated {n_blur} blur images")
    print(f"total degraded images: {n_poisson + n_background + n_blur}")
    print(f"output root: {DATA_ROOT}/")


if __name__ == "__main__":
    main()
