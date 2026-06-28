# summarize_year4_updated.py
#   python summarize_year4_updated.py ^
#     --csv "C:\Users\biorobotics\Desktop\MACC\cuboidMACC-DARS24\updated_src\out\random_10x10x4_15blocks_1000s\combined_200_20260120\summary_20260120_180133.csv" ^
#     --out_dir "C:\Users\biorobotics\Desktop\MACC\cuboidMACC-DARS24\updated_src\out\random_10x10x4_15blocks_1000s\combined_200_20260120"

import argparse
import os
from datetime import datetime
import numpy as np
import pandas as pd


def normalize_path(p):
    if p is None:
        return None
    return os.path.normpath(str(p).strip()).replace("\\", "/")


def build_run_key(file_path):
    """
    Unique stable key using file column (prevents struct/random index collisions).
    Uses parent folder + basename.
    """
    p = normalize_path(file_path)
    base = os.path.basename(p) if p else ""
    if not base:
        return None
    parent = os.path.basename(os.path.dirname(p))
    return f"{parent}/{base}"


def safe_stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "min": np.nan, "median": np.nan, "mean": np.nan, "max": np.nan}
    return {
        "n": int(x.size),
        "min": float(np.min(x)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "max": float(np.max(x)),
    }


def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "nan"
    return f"{v:.6g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to UPDATED (Year Four) summary_*.csv")
    ap.add_argument("--out_dir", default=".", help="Where to write the clean summary CSV")
    args = ap.parse_args()

    csv_path = normalize_path(args.csv)
    out_dir = normalize_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Expected columns (from you)
    # index, hl_status, ll_status, time_limit, high_level_s, interface_s, task_alloc_s, mapf_s, total_s, makespan, file
    required = {"hl_status", "ll_status", "time_limit", "high_level_s", "interface_s",
                "task_alloc_s", "mapf_s", "total_s", "makespan", "file"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV missing required columns: {sorted(missing)}")

    # Normalize file + unique key
    df["file"] = df["file"].apply(normalize_path)
    df["run_key"] = df["file"].apply(build_run_key)
    df = df[df["run_key"].notna()].copy()

    # Coerce numeric columns
    num_cols = ["time_limit", "high_level_s", "interface_s", "task_alloc_s", "mapf_s", "total_s", "makespan"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep last duplicate run_key (if any)
    dup = df["run_key"].duplicated(keep="last").sum()
    if dup:
        print(f"WARN: {dup} duplicate run_keys found (kept last occurrence).")
    df = df.drop_duplicates(subset=["run_key"], keep="last").copy()

    # Derived: low-level total time includes task allocation + MAPF
    df["low_level_total_s"] = df["task_alloc_s"] + df["mapf_s"]

    # Core sets
    n_total = int(df["run_key"].nunique())

    hl_found = df["hl_status"].eq("FOUND")
    ll_found = df["ll_status"].eq("FOUND")
    success = hl_found & ll_found

    n_success = int(success.sum())
    n_hl_fail = int((~hl_found).sum())
    n_ll_fail_after_hl = int((hl_found & (~ll_found)).sum())

    # Time limit (should be constant = 1000)
    tl_vals = df["time_limit"].dropna().unique()
    time_limit = float(tl_vals[0]) if len(tl_vals) == 1 else None

    # Stage dominance counts among complete successes
    succ = df[success].copy()
    # Dominance categories:
    #   - High-level time
    #   - Low-level total time (task_alloc + mapf)
    #   - Interface time
    dom_counts = {"high_level": 0, "low_level_total": 0, "interface": 0}
    dom_n = 0

    if len(succ) > 0:
        hl = succ["high_level_s"].to_numpy(float)
        ll = succ["low_level_total_s"].to_numpy(float)
        it = succ["interface_s"].to_numpy(float)
        m = np.isfinite(hl) & np.isfinite(ll) & np.isfinite(it)
        hl, ll, it = hl[m], ll[m], it[m]
        dom_n = int(hl.size)

        if dom_n > 0:
            stack = np.vstack([hl, ll, it])  # 0 HL, 1 LL, 2 IF
            arg = np.argmax(stack, axis=0)
            dom_counts["high_level"] = int(np.sum(arg == 0))
            dom_counts["low_level_total"] = int(np.sum(arg == 1))
            dom_counts["interface"] = int(np.sum(arg == 2))

    # Runtime summary stats for complete successes
    stages = [
        ("High-Level (s)", "high_level_s"),
        ("Interface (s)", "interface_s"),
        ("Low-Level (s)", "mapf_s"),                 # low-level MAPF only (matches your earlier usage)
        ("Low-Level Total (s)", "low_level_total_s"),# task_alloc + MAPF
        ("Total (s)", "total_s"),
    ]

    print("\n== Year Four (updated_src) summary ==")
    print(f"Total runs (unique by file): {n_total}")
    print(f"Complete successes (hl=FOUND & ll=FOUND): {n_success}")
    print(f"HL failed to find abstract sequence (hl!=FOUND): {n_hl_fail}")
    print(f"Reached LL but did not solve (hl=FOUND & ll!=FOUND): {n_ll_fail_after_hl}")
    if time_limit is not None:
        print(f"Time limit from CSV: {time_limit:g} s")

    print("\n== Stage dominance (complete successes) ==")
    if dom_n == 0:
        print("No successes with finite (HL, LL_total, Interface) times to compare.")
    else:
        print(f"Compared on n={dom_n} successes (finite HL/LL_total/Interface):")
        print(f"  Low-level total longest: {dom_counts['low_level_total']}")
        print(f"  High-level longest:      {dom_counts['high_level']}")
        print(f"  Interface longest:       {dom_counts['interface']}")

    print("\n== Walltime statistics (complete successes only) ==")
    for label, key in stages:
        st = safe_stats(succ[key].to_numpy(float))
        print(f"{label:18s} -> min={fmt(st['min'])}  med={fmt(st['median'])}  mean={fmt(st['mean'])}  max={fmt(st['max'])}  (n={st['n']})")

    # Write clean summary CSV (like your screenshot: Stage, Min, Median, Mean, Max)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(out_dir, f"year4_summary_table_{ts}.csv")

    rows = []
    for label, key in stages:
        st = safe_stats(succ[key].to_numpy(float))
        rows.append([label, st["min"], st["median"], st["mean"], st["max"]])

    out_df = pd.DataFrame(rows, columns=["Stage", "Min", "Median", "Mean", "Max"])
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved summary table CSV -> {out_csv}")

    # Paragraph-ready text (numbers only; you can adjust wording)
    print("\n== Paragraph-ready numbers (Year Four only) ==")
    print(
        f"Of the {n_total} test structures, the high-level planner is unable to find an abstract action sequence for {n_hl_fail}. "
        f"Under a fixed {time_limit:g} s time limit, the low-level stage does not complete for a further {n_ll_fail_after_hl} structures. "
        f"Summary walltime statistics for the {n_success} complete successes are reported in Table X. "
        f"Among the {dom_n} successes with finite timing breakdown, the low-level stage took the longest in {dom_counts['low_level_total']} cases, "
        f"while the high-level and interface stages were the longest in {dom_counts['high_level']} and {dom_counts['interface']} cases, respectively."
    )
    # ----------------------------
    # Save paragraph-level stats CSV
    # ----------------------------

    para_csv = os.path.join(out_dir, f"year4_paragraph_stats_{ts}.csv")

    rows = [
        ("Total structures", n_total),
        ("Complete successes (HL & LL FOUND)", n_success),
        ("HL failed to find abstract sequence", n_hl_fail),
        ("LL failed after HL success", n_ll_fail_after_hl),
        ("Time limit (s)", time_limit),
        ("Successes used for stage dominance", dom_n),
        ("Low-level longest (count)", dom_counts["low_level_total"]),
        ("High-level longest (count)", dom_counts["high_level"]),
        ("Interface longest (count)", dom_counts["interface"]),
    ]

    pd.DataFrame(rows, columns=["Metric", "Value"]).to_csv(para_csv, index=False)

    print(f"Saved paragraph statistics CSV -> {para_csv}")


if __name__ == "__main__":
    main()
