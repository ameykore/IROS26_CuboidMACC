# analyze_overlap_stats.py
# To run
# python analyze_overlap_stats.py ^
# --src_csv "C:\Users\biorobotics\Desktop\MACC\cuboidMACC-DARS24\src\out\random_10x10x4_15blocks_1000s\combined_200_20260120\summary_20260120_180213.csv" ^
# --updated_csv "C:\Users\biorobotics\Desktop\MACC\cuboidMACC-DARS24\updated_src\out\random_10x10x4_15blocks_1000s\combined_200_20260120\summary_20260120_180133.csv" ^
# --out_dir "C:\Users\biorobotics\Desktop\MACC\cuboidMACC-DARS24"

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Helpers (unchanged)
# ----------------------------

NUM_FIELDS = {
    "time_limit",
    "high_level_s",
    "interface_s",
    "task_alloc_s",
    "mapf_s",
    "total_s",
    "makespan",
    "index",
}

def normalize_path(p):
    if p is None:
        return None
    return os.path.normpath(str(p).strip()).replace("\\", "/")

def build_run_key(file_path):
    p = normalize_path(file_path)
    base = os.path.basename(p) if p else ""
    if not base:
        return None
    parent = os.path.basename(os.path.dirname(p))
    return f"{parent}/{base}"

def is_success(row):
    return (row.get("hl_status") == "FOUND") and (row.get("ll_status") == "FOUND")

def read_summary_csv(path):
    path = normalize_path(path)
    df = pd.read_csv(path)

    if "file" in df.columns:
        df["file"] = df["file"].apply(normalize_path)

    for k in NUM_FIELDS:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")

    df["low_level_s"] = df["mapf_s"] if "mapf_s" in df.columns else np.nan
    df["low_level_total_s"] = (
        df["task_alloc_s"] + df["mapf_s"]
        if {"task_alloc_s", "mapf_s"} <= set(df.columns)
        else np.nan
    )

    if "file" in df.columns:
        df["run_key"] = df["file"].apply(build_run_key)
    else:
        df["run_key"] = None

    if "index" in df.columns:
        df["run_key"] = df["run_key"].fillna(
            df["index"].apply(lambda x: f"INDEX_ONLY/{int(x)}" if pd.notna(x) else None)
        )

    df = df[df["run_key"].notna()].copy()
    df = df.drop_duplicates(subset=["run_key"], keep="last")

    return {r["run_key"]: r.to_dict() for _, r in df.iterrows()}

# ----------------------------
# Plot utilities
# ----------------------------

def savefig_png(out_dir, name):
    path = os.path.join(out_dir, f"{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path

def plot_delta_vs_baseline(baseline, delta, title, x_label, y_label, out_dir, name, log_x=False):
    """
    Scatter: X = Year3 baseline time, Y = (Year3 - Year4) improvement in seconds.
    If log_x=True, uses log scale on x (only if x>0 exists).
    """
    x = np.asarray(baseline, dtype=float)
    y = np.asarray(delta, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if x.size == 0:
        print(f"SKIP {name}: no finite data")
        return

    plt.figure()
    plt.scatter(x, y, s=25)
    plt.axhline(0.0, linestyle="--")

    if log_x:
        # Only apply log if we have positive x values
        if np.any(x > 0):
            plt.xscale("log")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    savefig_png(out_dir, name)

def plot_relative_bar(values, title, y_label, out_dir, name):
    """
    Screenshot-style bar plot:
    y = (Year3 - Year4) / Year3 * 100
    Index order preserved (overlap ordering)
    """
    vals = np.asarray(values, dtype=float)
    x = np.arange(len(vals))

    plt.figure(figsize=(14, 5))
    plt.bar(x, vals)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("World State Index")
    plt.ylabel(y_label)
    plt.title(title)
    savefig_png(out_dir, name)

def safe_percent(delta, baseline):
    """
    Returns 100 * delta / baseline, but masks invalid baseline (<=0 or nan).
    """
    d = np.asarray(delta, dtype=float)
    b = np.asarray(baseline, dtype=float)
    out = np.full_like(d, np.nan, dtype=float)
    m = np.isfinite(d) & np.isfinite(b) & (b > 0)
    out[m] = 100.0 * d[m] / b[m]
    return out

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_csv", required=True)
    ap.add_argument("--updated_csv", required=True)
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    src = read_summary_csv(args.src_csv)
    upd = read_summary_csv(args.updated_csv)

    overlap = sorted(
        k for k in src
        if k in upd and is_success(src[k]) and is_success(upd[k])
    )

    if not overlap:
        print("No overlap successes.")
        return

    plot_dir = os.path.join(
        normalize_path(args.out_dir),
        f"plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(plot_dir, exist_ok=True)

    def arr(d, key):
        return np.array([d[k].get(key, np.nan) for k in overlap], dtype=float)

    # Baselines (Year 3) and Year 4
    hl3, hl4 = arr(src, "high_level_s"), arr(upd, "high_level_s")
    it3, it4 = arr(src, "interface_s"),  arr(upd, "interface_s")
    # Low-level: use TOTAL low-level time (task_alloc + MAPF), consistent with your writeups
    ll3, ll4 = arr(src, "low_level_total_s"), arr(upd, "low_level_total_s")
    tot3, tot4 = arr(src, "total_s"), arr(upd, "total_s")

    # ✅ Deltas: Year3 − Year4 (positive = improvement)
    d_hl  = hl3  - hl4
    d_it  = it3  - it4
    d_ll  = ll3  - ll4
    d_tot = tot3 - tot4

    # ----------------------------
    # SCATTERS (log-x versions)  [keeps your existing behavior]
    # ----------------------------
    plot_delta_vs_baseline(
        tot3, d_tot,
        "Δ Total time vs Year Three baseline (log-x)",
        "Year Three Total Planning Time (s)",
        "Improvement (Year3 − Year4) in Total Time (s)",
        plot_dir,
        "scatter_total_logx",
        log_x=True
    )

    plot_delta_vs_baseline(
        hl3, d_hl,
        "Δ Assembly Ordering time vs Year Three baseline (log-x)",
        "Year Three Assembly Ordering Time (s)",
        "Improvement (Year3 − Year4) in Assembly Ordering Time (s)",
        plot_dir,
        "scatter_high_level_logx",
        log_x=True
    )

    plot_delta_vs_baseline(
        it3, d_it,
        "Δ Interface time vs Year Three baseline (log-x)",
        "Year Three Interface Time (s)",
        "Improvement (Year3 − Year4) in Interface Time (s)",
        plot_dir,
        "scatter_interface_logx",
        log_x=True
    )

    plot_delta_vs_baseline(
        ll3, d_ll,
        "Δ Low-level total time vs Year Three baseline (log-x)",
        "Year Three Low-level Total Time (s)",
        "Improvement (Year3 − Year4) in Low-level Total Time (s)",
        plot_dir,
        "scatter_low_level_total_logx",
        log_x=True
    )

    # ----------------------------
    # SCATTERS (linear-x versions)  ✅ NEW
    # ----------------------------
    plot_delta_vs_baseline(
        tot3, d_tot,
        "Δ Total time vs Year Three baseline",
        "Year Three Total Planning Time (s)",
        "Improvement (Year3 − Year4) in Total Time (s)",
        plot_dir,
        "scatter_total_linearx",
        log_x=False
    )

    plot_delta_vs_baseline(
        hl3, d_hl,
        "Δ Assembly Ordering time vs Year Three baseline",
        "Year Three Assembly Ordering Time (s)",
        "Improvement (Year3 − Year4) in Assembly Ordering Time (s)",
        plot_dir,
        "scatter_high_level_linearx",
        log_x=False
    )

    plot_delta_vs_baseline(
        it3, d_it,
        "Δ Interface time vs Year Three baseline",
        "Year Three Interface Time (s)",
        "Improvement (Year3 − Year4) in Interface Time (s)",
        plot_dir,
        "scatter_interface_linearx",
        log_x=False
    )

    plot_delta_vs_baseline(
        ll3, d_ll,
        "Δ Low-level total time vs Year Three baseline",
        "Year Three Low-level Total Time (s)",
        "Improvement (Year3 − Year4) in Low-level Total Time (s)",
        plot_dir,
        "scatter_low_level_total_linearx",
        log_x=False
    )

    # ----------------------------
    # Screenshot-style relative bars (percent)  [kept + extended]
    # ----------------------------
    rel_tot = safe_percent(d_tot, tot3)
    plot_relative_bar(
        rel_tot,
        "Relative Total Time Improvement per Instance",
        "Percentage Difference in Total Time (%)",
        plot_dir,
        "bar_relative_total_time"
    )

    rel_hl = safe_percent(d_hl, hl3)
    plot_relative_bar(
        rel_hl,
        "Relative Assembly Ordering Time Improvement per Instance",
        "Percentage Difference in Assembly Ordering Time (%)",
        plot_dir,
        "bar_relative_high_level"
    )

    rel_it = safe_percent(d_it, it3)
    plot_relative_bar(
        rel_it,
        "Relative Interface Time Improvement per Instance",
        "Percentage Difference in Interface Time (%)",
        plot_dir,
        "bar_relative_interface"
    )

    rel_ll = safe_percent(d_ll, ll3)
    plot_relative_bar(
        rel_ll,
        "Relative Low-level Total Time Improvement per Instance",
        "Percentage Difference in Low-level Total Time (%)",
        plot_dir,
        "bar_relative_low_level_total"
    )

    print(f"Wrote plots to: {plot_dir}")

if __name__ == "__main__":
    main()
