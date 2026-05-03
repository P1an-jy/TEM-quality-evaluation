"""
02_run_atomsegnet_batch.py
Batch inference with AtomSegNet gaussianMask+ model on degraded STEM images.
Outputs: gaussian map (.npy) and predicted atom centroids (.csv).
"""

import argparse
import os
import sys
import types
import warnings
import numpy as np
import pandas as pd
from PIL import Image

# ---- paths ----
DATA_ROOT = "quality_experiment"
METADATA_PATH = os.path.join(DATA_ROOT, "data", "degraded_metadata.csv")
OUT_GMAPS = os.path.join(DATA_ROOT, "outputs", "gaussian_maps")
OUT_COORDS = os.path.join(DATA_ROOT, "outputs", "pred_coords")


# ============================================================
# helpers
# ============================================================

def setup_atomsegnet_imports(atomsegnet_root: str):
    """Insert AtomSegNet path and mock PyQt5 (not needed for batch inference)."""
    # Mock PyQt5 before utils.py tries to import it at module level
    _pyqt5 = types.ModuleType("PyQt5")
    _qtgui = types.ModuleType("PyQt5.QtGui")
    _qtgui.QImage = type("QImage", (), {})
    _qtgui.QPixmap = type("QPixmap", (), {})
    _pyqt5.QtGui = _qtgui
    sys.modules["PyQt5"] = _pyqt5
    sys.modules["PyQt5.QtGui"] = _qtgui

    atomsegnet_root = os.path.abspath(atomsegnet_root)
    # Only add the root, NOT utils/ — otherwise 'import utils' finds
    # the utils/ directory itself rather than the utils package.
    sys.path.insert(0, atomsegnet_root)


def load_image(path: str) -> np.ndarray:
    """Load grayscale PNG as uint8 numpy array (H, W)."""
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    return np.array(img)


def run_model(model_path: str, image: np.ndarray) -> np.ndarray:
    """
    Run AtomSegNet model on a single uint8 image.
    Returns gaussian map (float, [0,1]) as (H, W) numpy array.
    """
    from utils.utils import load_model as atomseg_load_model
    return atomseg_load_model(model_path, image, cuda=False)


def extract_centroids(gmap: np.ndarray) -> np.ndarray:
    """
    Extract atom centroids from a gaussian map.

    Steps: Otsu threshold → connected-components → filter area<2 →
           centroid extraction.

    Returns (N,2) array with columns [x_pred, y_pred] (x=col, y=row).
    """
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops

    # Otsu binarization
    thresh = threshold_otsu(gmap)
    binary = gmap > thresh

    # Connected components
    labeled = label(binary)

    # Extract centroids, filter tiny regions
    centroids = []
    for region in regionprops(labeled):
        if region.area >= 2:
            ry, cx = region.centroid       # regionprops returns (row, col) = (y, x)
            centroids.append([cx, ry])      # store as (x_pred, y_pred)

    if not centroids:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(centroids, dtype=np.float64)


def save_coords(coords: np.ndarray, path: str):
    """Save predicted coordinates as CSV (atom_id, x_pred, y_pred)."""
    n = coords.shape[0]
    df = pd.DataFrame({
        "atom_id":  np.arange(n),
        "x_pred":   coords[:, 0],
        "y_pred":   coords[:, 1],
    })
    df.to_csv(path, index=False)


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Batch AtomSegNet inference on degraded STEM images")
    parser.add_argument("--atomsegnet_root", type=str, required=True,
                        help="Path to AtomSegNet-master directory")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N images (0 = all)")
    args = parser.parse_args()

    atomsegnet_root = os.path.abspath(args.atomsegnet_root)
    setup_atomsegnet_imports(atomsegnet_root)

    model_path = os.path.join(atomsegnet_root, "model_weights", "gaussianMask+.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    print(f"Using AtomSegNet from: {atomsegnet_root}")

    os.makedirs(OUT_GMAPS, exist_ok=True)
    os.makedirs(OUT_COORDS, exist_ok=True)

    # Load metadata (drop potential unnamed index column)
    meta = pd.read_csv(METADATA_PATH)
    meta = meta.loc[:, ~meta.columns.str.contains("^Unnamed")]

    if args.limit > 0:
        meta = meta.head(args.limit)

    n_total = len(meta)
    n_ok = 0

    for idx, row in meta.iterrows():
        img_id = row["image_id"]
        img_path = row["image_path"]
        level = row.get("level", "")
        dtype = row.get("degradation_type", "")

        # Build a unique output id: include level to avoid overwriting
        out_id = f"{img_id}_{dtype}_{level}"

        print(f"[{idx+1}/{n_total}] processing {out_id} ...", end=" ", flush=True)

        try:
            # 1) load
            image = load_image(img_path)

            # 2) run model
            gmap = run_model(model_path, image)

            # 3) save gaussian map
            gmap_path = os.path.join(OUT_GMAPS, f"{out_id}.npy")
            np.save(gmap_path, gmap.astype(np.float32))

            # 4) extract centroids
            coords = extract_centroids(gmap)

            # 5) save predictions
            coord_path = os.path.join(OUT_COORDS, f"{out_id}_pred.csv")
            save_coords(coords, coord_path)

            print(f"OK ({len(coords)} atoms)")
            n_ok += 1

        except Exception as e:
            msg = str(e).replace("\n", " ")
            print(f"FAILED: {msg}")
            warnings.warn(f"skipped {out_id}: {e}")

    print(f"\nFinished processing {n_ok}/{n_total} images")


if __name__ == "__main__":
    main()
