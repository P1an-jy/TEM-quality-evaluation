"""
03_evaluate_precision.py
Match predicted atom coordinates to ground truth via Hungarian algorithm.
Debiasing is done ONCE per degradation level (not per image), so σ reflects
true localization precision across all atoms in that experimental condition.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
DATA_ROOT = "quality_experiment"
LATTICE_SPACING = 16
MATCH_RADIUS = 0.4 * LATTICE_SPACING   # 6.4 pixels

METADATA_PATH = os.path.join(DATA_ROOT, "data", "degraded_metadata.csv")
PRED_DIR      = os.path.join(DATA_ROOT, "outputs", "pred_coords")
MATCHED_DIR   = os.path.join(DATA_ROOT, "outputs", "matched_coords")
RESULTS_DIR   = os.path.join(DATA_ROOT, "results")
PLOTS_DIR     = os.path.join(RESULTS_DIR, "plots")

os.makedirs(MATCHED_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================
# Utilities
# ============================================================

def parse_numeric_param(param_value):
    """Extract a sortable numeric value from param_value string."""
    if isinstance(param_value, (int, float, np.integer, np.floating)):
        return float(param_value)
    s = str(param_value)
    m = re.search(r"[\d.]+", s)
    if m:
        return float(m.group())
    return float("nan")


def load_coords(path, x_col="x_gt", y_col="y_gt"):
    """Load CSV and return (N,2) float64 array [x, y]."""
    df = pd.read_csv(path)
    return df[[x_col, y_col]].to_numpy(dtype=np.float64)


def get_pred_path(out_id):
    """Construct predicted-coords CSV path from output id."""
    return os.path.join(PRED_DIR, f"{out_id}_pred.csv")


# ============================================================
# Matching
# ============================================================

def match_points(gt, pred, match_radius=MATCH_RADIUS):
    """Global optimal one-to-one matching via Hungarian algorithm."""
    n_gt, n_pred = len(gt), len(pred)
    if n_gt == 0 or n_pred == 0:
        return [], set(range(n_gt)), set(range(n_pred))

    dist = cdist(gt, pred)
    n_max = max(n_gt, n_pred)
    cost = np.full((n_max, n_max), fill_value=1e9, dtype=np.float64)
    cost[:n_gt, :n_pred] = dist

    row_ind, col_ind = linear_sum_assignment(cost)

    matched = []
    for r, c in zip(row_ind, col_ind):
        if r < n_gt and c < n_pred and dist[r, c] <= match_radius:
            matched.append((r, c, dist[r, c]))

    matched_gt   = {m[0] for m in matched}
    matched_pred = {m[1] for m in matched}
    unmatched_gt   = set(range(n_gt)) - matched_gt
    unmatched_pred = set(range(n_pred)) - matched_pred

    return matched, unmatched_gt, unmatched_pred


# ============================================================
# Per-image evaluation  (raw dx/dy, NO per-image debiasing)
# ============================================================

def evaluate_one_image(gt_path, pred_path, match_radius=MATCH_RADIUS):
    """
    Match atoms, return matched-detail DataFrame (raw dx, dy) and
    per-image summary dict.  Debiasing is deferred to the group level.
    """
    gt   = load_coords(gt_path,   "x_gt",   "y_gt")
    pred = load_coords(pred_path, "x_pred", "y_pred")

    n_gt   = len(gt)
    n_pred = len(pred)

    matched, unmatched_gt, unmatched_pred = match_points(gt, pred, match_radius)
    n_matched = len(matched)

    rows = []
    for gt_idx, pred_idx, dist in matched:
        dx = pred[pred_idx, 0] - gt[gt_idx, 0]
        dy = pred[pred_idx, 1] - gt[gt_idx, 1]
        rows.append({
            "gt_atom_id":   gt_idx,
            "pred_atom_id": pred_idx,
            "x_gt":    gt[gt_idx, 0],
            "y_gt":    gt[gt_idx, 1],
            "x_pred":  pred[pred_idx, 0],
            "y_pred":  pred[pred_idx, 1],
            "dx":      dx,
            "dy":      dy,
            "distance": dist,
        })
    matched_df = pd.DataFrame(rows)

    if n_matched > 0:
        dx_arr   = matched_df["dx"].to_numpy(dtype=np.float64)
        dy_arr   = matched_df["dy"].to_numpy(dtype=np.float64)
        dist_arr = matched_df["distance"].to_numpy(dtype=np.float64)

        stats = {
            "n_gt":           n_gt,
            "n_pred":         n_pred,
            "n_matched":      n_matched,
            "detection_rate":  n_matched / n_gt if n_gt > 0 else 0.0,
            "false_positive":  n_pred - n_matched,
            "bias_x":         float(np.mean(dx_arr)),
            "bias_y":         float(np.mean(dy_arr)),
            "sigma_x":        float(np.std(dx_arr, ddof=1)) if len(dx_arr) > 1 else 0.0,
            "sigma_y":        float(np.std(dy_arr, ddof=1)) if len(dy_arr) > 1 else 0.0,
            "mean_dx":        float(np.mean(dx_arr)),
            "mean_dy":        float(np.mean(dy_arr)),
            "mean_abs_error": float(np.mean(dist_arr)),
            "median_error":   float(np.median(dist_arr)),
        }
    else:
        stats = {
            "n_gt": n_gt, "n_pred": n_pred, "n_matched": 0,
            "detection_rate": 0.0, "false_positive": n_pred,
            "bias_x": 0.0, "bias_y": 0.0,
            "sigma_x": 0.0, "sigma_y": 0.0,
            "mean_dx": 0.0, "mean_dy": 0.0,
            "mean_abs_error": 0.0, "median_error": 0.0,
        }
    return matched_df, stats


# ============================================================
# Group-level summary  (ONE debiasing per degradation level)
# ============================================================

def summarize_results(per_image_df, matched_dfs):
    """
    Group by (degradation_type, param_value).  Pool all raw dx/dy across
    images in the group, apply a SINGLE de-biasing, then compute σ.

    Returns
    -------
    summary_df : DataFrame with one row per degradation level.
    level_dx_dy : dict  (dtype, str(pv)) → (dx_debiased, dy_debiased)
                  used by the histogram plot.
    """
    groups = per_image_df.groupby(["degradation_type", "param_value"], sort=False)
    rows = []
    level_dx_dy = {}

    for (dtype, pv), grp in groups:
        n_images = len(grp)
        total_gt      = int(grp["n_gt"].sum())
        total_pred    = int(grp["n_pred"].sum())
        total_matched = int(grp["n_matched"].sum())

        mean_det_rate = grp["detection_rate"].mean()
        mean_fp       = grp["false_positive"].mean()

        # ---- pool raw dx/dy from every matched atom in this group ----
        all_dx_raw = []
        all_dy_raw = []
        all_dist   = []
        for _, prow in grp.iterrows():
            out_id = prow["image_id"]
            mdf = matched_dfs[out_id]
            if len(mdf) > 0:
                all_dx_raw.append(mdf["dx"].to_numpy(dtype=np.float64))
                all_dy_raw.append(mdf["dy"].to_numpy(dtype=np.float64))
                all_dist.append(mdf["distance"].to_numpy(dtype=np.float64))

        if all_dx_raw:
            dx_pool_raw = np.concatenate(all_dx_raw)
            dy_pool_raw = np.concatenate(all_dy_raw)
            dist_pool   = np.concatenate(all_dist)

            # ---- ONE de-biasing at the degradation-level ----
            bias_x_level = np.mean(dx_pool_raw)
            bias_y_level = np.mean(dy_pool_raw)

            dx_pool = dx_pool_raw - bias_x_level
            dy_pool = dy_pool_raw - bias_y_level

            sigma_x = float(np.std(dx_pool, ddof=1))
            sigma_y = float(np.std(dy_pool, ddof=1))
            mae     = float(np.mean(dist_pool))
            med_err = float(np.median(dist_pool))
        else:
            sigma_x = sigma_y = mae = med_err = 0.0
            dx_pool = dy_pool = np.array([])
            bias_x_level = bias_y_level = 0.0
            dx_pool_raw = dy_pool_raw = np.array([])

        rows.append({
            "degradation_type":     dtype,
            "param_value":          pv,
            "n_images":             n_images,
            "total_gt":             total_gt,
            "total_pred":           total_pred,
            "total_matched":        total_matched,
            "mean_detection_rate":  round(mean_det_rate, 6),
            "mean_false_positive":  round(mean_fp, 2),
            "level_bias_x":         round(float(bias_x_level), 4),
            "level_bias_y":         round(float(bias_y_level), 4),
            "sigma_x":              round(sigma_x, 4),
            "sigma_y":              round(sigma_y, 4),
            "mean_abs_error":       round(mae, 4),
            "median_error":         round(med_err, 4),
        })
        level_dx_dy[(dtype, str(pv))] = (dx_pool, dy_pool)

    summary_df = pd.DataFrame(rows)
    summary_df["_sort_key"] = summary_df["param_value"].apply(parse_numeric_param)
    summary_df = summary_df.sort_values("_sort_key").drop(columns=["_sort_key"])

    col_order = ["degradation_type", "param_value", "n_images",
                 "total_gt", "total_pred", "total_matched",
                 "mean_detection_rate", "mean_false_positive",
                 "level_bias_x", "level_bias_y",
                 "sigma_x", "sigma_y", "mean_abs_error", "median_error"]
    summary_df = summary_df[col_order]
    return summary_df, level_dx_dy


# ============================================================
# Plotting
# ============================================================

def _dtype_label(dtype):
    return {"poisson": "Poisson noise", "background": "Background",
            "blur": "Gaussian blur"}.get(dtype, dtype)


def plot_summary_line(summary_df, metric, ylabel, filename,
                      use_log=False, ylim=None):
    """
    Plot a single summary-level metric (one value per degradation level)
    from summary_df.  One subplot per degradation_type.  No error bars.
    """
    dtypes = summary_df["degradation_type"].unique()
    n_dtypes = len(dtypes)
    fig, axes = plt.subplots(1, n_dtypes, figsize=(5 * n_dtypes, 4.2), squeeze=False)
    axes = axes[0]

    for ax, dtype in zip(axes, dtypes):
        sub = summary_df[summary_df["degradation_type"] == dtype].copy()
        sub = sub.sort_values("param_value", key=lambda s: s.map(parse_numeric_param))

        x_num = [parse_numeric_param(v) for v in sub["param_value"]]
        y_vals = sub[metric].to_numpy(dtype=np.float64)
        labels = sub["param_value"].astype(str).tolist()

        ax.plot(x_num, y_vals, "o-", color="steelblue", markersize=7, linewidth=1.5)
        ax.set_title(_dtype_label(dtype), fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("param value", fontsize=8)
        ax.grid(True, alpha=0.3)

        if use_log and len(x_num) > 1:
            ax.set_xscale("log")
        else:
            ax.set_xticks(x_num)
            ax.set_xticklabels(labels, fontsize=8, rotation=30)

        if ylim is not None:
            ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close(fig)


def plot_grouped_metric(per_image_df, metric, ylabel, filename,
                        use_log=False, ylim=None):
    """
    Group per-image metrics by (degradation_type, param_value), compute
    mean ± std, and plot with error bars.  One subplot per degradation_type.
    """
    dtypes = per_image_df["degradation_type"].unique()
    n_dtypes = len(dtypes)
    fig, axes = plt.subplots(1, n_dtypes, figsize=(5 * n_dtypes, 4.2), squeeze=False)
    axes = axes[0]

    for ax, dtype in zip(axes, dtypes):
        grp = (per_image_df[per_image_df["degradation_type"] == dtype]
               .groupby("param_value")[metric])
        means = grp.mean()
        stds  = grp.std(ddof=1)

        keys_sorted = sorted(means.index, key=lambda v: parse_numeric_param(v))
        x_num = [parse_numeric_param(k) for k in keys_sorted]
        y_m   = [means[k] for k in keys_sorted]
        y_s   = [stds[k] for k in keys_sorted]
        labels = [str(k) for k in keys_sorted]

        ax.errorbar(x_num, y_m, yerr=y_s, fmt="o-", color="steelblue",
                    markersize=7, linewidth=1.5, capsize=4, capthick=1.2)
        ax.set_title(_dtype_label(dtype), fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("param value", fontsize=8)
        ax.grid(True, alpha=0.3)

        if use_log and len(x_num) > 1:
            ax.set_xscale("log")
        else:
            ax.set_xticks(x_num)
            ax.set_xticklabels(labels, fontsize=8, rotation=30)

        if ylim is not None:
            ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close(fig)


def plot_hist_grid(level_dx_dy):
    """
    3×3 histogram grid using **level-debiased** dx/dy from summary stage.
    Rows = degradation_type, cols = low/med/high param.
    """
    level_map = {
        "poisson":    ["50", "200", "1000"],
        "background": ["linear_α0.1", "linear_α0.3", "nonlinear_α0.3"],
        "blur":       ["0.5", "1.0", "2.0"],
    }
    dtypes = ["poisson", "background", "blur"]
    n_rows = len(dtypes)
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.2))
    bins = np.linspace(-3, 3, 41)

    for r, dtype in enumerate(dtypes):
        for c, pv_key in enumerate(level_map[dtype]):
            ax = axes[r, c]
            key = (dtype, pv_key)
            if key in level_dx_dy:
                dx_pool, dy_pool = level_dx_dy[key]
                if len(dx_pool) > 0:
                    sx = np.std(dx_pool, ddof=1)
                    sy = np.std(dy_pool, ddof=1)
                    ax.hist(dx_pool, bins=bins, alpha=0.5, color="steelblue",
                            label=f"dx (σ={sx:.3f})")
                    ax.hist(dy_pool, bins=bins, alpha=0.5, color="darkorange",
                            label=f"dy (σ={sy:.3f})")
                    ax.legend(fontsize=6, loc="upper right")
            ax.set_title(f"{_dtype_label(dtype)}  {pv_key}", fontsize=8)
            ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
            if r == n_rows - 1:
                ax.set_xlabel("error (pixels)", fontsize=7)
            if c == 0:
                ax.set_ylabel("count", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "dx_dy_hist_examples.png"), dpi=150)
    plt.close(fig)


def plot_results(per_image_df, summary_df, level_dx_dy):
    """Generate all figures."""
    # ---- summary-level metrics (sigma computed from level-debiased pool) ----
    plot_summary_line(summary_df, "sigma_x",
                      "σx (pixels, level-debiased)",
                      "sigma_x_vs_degradation_grouped.png")
    plot_summary_line(summary_df, "sigma_y",
                      "σy (pixels, level-debiased)",
                      "sigma_y_vs_degradation_grouped.png")

    # ---- per-image grouped metrics ----
    plot_grouped_metric(per_image_df, "mean_abs_error",
                        "Mean abs error (pixels)",
                        "mean_error_vs_degradation_grouped.png")
    plot_grouped_metric(per_image_df, "detection_rate",
                        "Detection rate",
                        "detection_rate_vs_degradation_grouped.png",
                        ylim=(0, 1.05))
    plot_grouped_metric(per_image_df, "bias_x",
                        "bias_x (pixels)",
                        "bias_x_vs_degradation_grouped.png")
    plot_grouped_metric(per_image_df, "bias_y",
                        "bias_y (pixels)",
                        "bias_y_vs_degradation_grouped.png")

    # ---- histogram grid (level-debiased dx/dy) ----
    plot_hist_grid(level_dx_dy)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("AtomSegNet Precision Evaluation  (level-debiased σ)")
    print("=" * 60)

    meta = pd.read_csv(METADATA_PATH)
    meta = meta.loc[:, ~meta.columns.str.contains("^Unnamed")]
    print(f"Loaded metadata: {len(meta)} rows")

    meta["out_id"] = meta.apply(
        lambda r: f"{r['image_id']}_{r['degradation_type']}_{r['level']}", axis=1)

    per_image_rows = []
    matched_dfs = {}
    n_processed = 0
    n_warn = 0
    type_stats = {}

    for _, row in meta.iterrows():
        out_id   = row["out_id"]
        gt_path  = row["gt_path"]
        dtype    = row["degradation_type"]
        pv       = row["param_value"]

        pred_path = get_pred_path(out_id)

        if not os.path.isfile(gt_path):
            warnings.warn(f"GT file not found: {gt_path}")
            n_warn += 1
            continue
        if not os.path.isfile(pred_path):
            warnings.warn(f"Pred file not found: {pred_path}")
            n_warn += 1
            continue

        matched_df, stats = evaluate_one_image(gt_path, pred_path, MATCH_RADIUS)
        if matched_df is None:
            n_warn += 1
            continue

        matched_path = os.path.join(MATCHED_DIR, f"{out_id}_matched.csv")
        matched_df.to_csv(matched_path, index=False)

        stats["image_id"] = out_id
        stats["degradation_type"] = dtype
        stats["param_value"] = pv
        per_image_rows.append(stats)
        matched_dfs[out_id] = matched_df

        if dtype not in type_stats:
            type_stats[dtype] = {"count": 0}
        type_stats[dtype]["count"] += 1

        n_processed += 1

    # ---- Save per-image precision (raw stats, no per-image debiasing) ----
    per_image_df = pd.DataFrame(per_image_rows)
    col_order = ["image_id", "degradation_type", "param_value",
                 "n_gt", "n_pred", "n_matched", "detection_rate",
                 "false_positive", "bias_x", "bias_y",
                 "sigma_x", "sigma_y",
                 "mean_dx", "mean_dy", "mean_abs_error", "median_error"]
    per_image_df = per_image_df[col_order]
    per_image_path = os.path.join(RESULTS_DIR, "per_image_precision.csv")
    per_image_df.to_csv(per_image_path, index=False)

    # ---- Summarize (ONE debiasing per degradation level) ----
    summary_df, level_dx_dy = summarize_results(per_image_df, matched_dfs)
    summary_path = os.path.join(RESULTS_DIR, "precision_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # ---- Plots ----
    plot_results(per_image_df, summary_df, level_dx_dy)

    # ---- Terminal report ----
    print(f"\nTotal images processed: {n_processed}")
    if n_warn:
        print(f"Warnings (missing files): {n_warn}")

    print("\nPer degradation type (level-debiased σ):")
    for _, srow in summary_df.iterrows():
        print(f"  {srow['degradation_type']:12s} {str(srow['param_value']):20s}  "
              f"σx={srow['sigma_x']:.4f}  σy={srow['sigma_y']:.4f}")

    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"  - per_image_precision.csv")
    print(f"  - precision_summary.csv")
    print(f"  - plots/  ({len(os.listdir(PLOTS_DIR))} files)")


if __name__ == "__main__":
    main()
