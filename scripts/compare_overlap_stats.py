# compare_overlap_stats.py
import argparse
import csv
import os
import re
import statistics as S
from datetime import datetime

# ----------------------------
# Helpers
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

def to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "nan"}:
        return None
    try:
        return float(s)
    except Exception:
        return None

def to_int(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "nan"}:
        return None
    try:
        return int(float(s))
    except Exception:
        return None

def stats(xs):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(xs),
        "min": min(xs),
        "median": S.median(xs),
        "mean": S.mean(xs),
        "max": max(xs),
    }

def fmt(st):
    def f(x):
        return "nan" if x is None else f"{x:.6g}"
    return f"count={st['count']}  min={f(st['min'])}  median={f(st['median'])}  mean={f(st['mean'])}  max={f(st['max'])}"

def normalize_path(p):
    if p is None:
        return None
    # Normalize Windows paths and slashes so keys match across machines
    return os.path.normpath(str(p).strip()).replace("\\", "/")

def build_run_key(file_path):
    """
    Make a unique stable key using the file column.
    This resolves collisions like:
      struct_10x10x4_n0.out.pkl  vs random_10x10x4_15block_n0.out.pkl
    """
    p = normalize_path(file_path)
    base = os.path.basename(p) if p else ""
    # Fall back: if file column is missing, try to key by index only (not ideal)
    if not base:
        return None

    # Prefer the basename (already unique across struct vs random),
    # but also include parent folder name to be extra safe if needed.
    parent = os.path.basename(os.path.dirname(p))
    return f"{parent}/{base}"

def is_success(row):
    return (row.get("hl_status") == "FOUND") and (row.get("ll_status") == "FOUND")

def read_summary_csv(path):
    """
    Returns dict: run_key -> row_dict
    If duplicates exist (same run_key repeated), keeps the *last* occurrence and warns.
    """
    path = normalize_path(path)
    by_key = {}
    dup = 0

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Normalize core fields
            r2 = dict(r)
            r2["file"] = normalize_path(r2.get("file"))

            # Parse numeric fields
            for k in NUM_FIELDS:
                if k not in r2:
                    continue
                if k == "index":
                    r2[k] = to_int(r2[k])
                else:
                    r2[k] = to_float(r2[k])

            # Compute derived fields
            ta = r2.get("task_alloc_s")
            mp = r2.get("mapf_s")
            r2["low_level_s"] = mp  # "low-level time" == MAPF phase time
            r2["low_level_total_s"] = (ta + mp) if (ta is not None and mp is not None) else None

            key = build_run_key(r2.get("file"))
            if key is None:
                # Worst-case fallback: index-only (can collide). Still better than dropping.
                key = f"INDEX_ONLY/{r2.get('index')}"
            if key in by_key:
                dup += 1
            by_key[key] = r2

    if dup:
        print(f"WARN: {dup} duplicate run_keys in {path} (kept last occurrence).")

    return by_key

def print_block(title):
    print("\n" + title)
    print("-" * len(title))

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_csv", required=True, help="Path to SRC summary_*.csv")
    ap.add_argument("--updated_csv", required=True, help="Path to UPDATED summary_*.csv")
    ap.add_argument("--out_dir", default=".", help="Where to write output CSVs (default: current dir)")
    ap.add_argument("--write_per_run_csv", action="store_true", help="Also write per-run comparison CSV")
    args = ap.parse_args()

    src = read_summary_csv(args.src_csv)
    upd = read_summary_csv(args.updated_csv)

    # Success sets (by run_key, so no struct/random collision)
    src_success = {k for k, r in src.items() if is_success(r)}
    upd_success = {k for k, r in upd.items() if is_success(r)}

    overlap = sorted(src_success & upd_success)

    print(f"SRC success count: {len(src_success)}")
    print(f"UPDATED success count: {len(upd_success)}")
    print(f"OVERLAP (src success ∩ updated success): {len(overlap)}")
    print("\n== Check: 'src solved' also solved in updated? ==")
    denom = max(len(src_success), 1)
    print(f"src successes that are also success in updated: {len(overlap)} / {len(src_success)} ({len(overlap)/denom:.3f})")

    if not overlap:
        print("\nNo overlap successes found. Check that your CSVs are from the same experiment batch and that 'file' paths are comparable.")
        return

    # Metrics to report
    metrics = [
        "high_level_s",
        "interface_s",
        "task_alloc_s",
        "mapf_s",             # low-level (MAPF)
        "low_level_total_s",  # task_alloc + mapf
        "total_s",
        "makespan",
        "time_limit",
        "index",
    ]

    print_block("== OVERLAP-ONLY STATS (same runs for both) ==")
    for m in metrics:
        src_vals = [src[k].get(m) for k in overlap]
        upd_vals = [upd[k].get(m) for k in overlap]
        sst = stats(src_vals)
        ust = stats(upd_vals)
        print(f"\n{m}")
        print(f"  SRC     -> {fmt(sst)}")
        print(f"  UPDATED -> {fmt(ust)}")

    # Also output a DARS-style "table" CSV like summarize_results does
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = normalize_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------
    # ONLY CHANGE: CSV OUTPUT FORMAT
    # - No Set / Count columns
    # - Keep two rows per stage: SRC overlap-success, UPDATED overlap-success
    # - Put dataset label into Stage (so you don't need a Set column)
    # ----------------------------
    table_path = os.path.join(out_dir, f"overlap_table_{ts}.csv")
    with open(table_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Stage", "Min", "Median", "Mean", "Max"])

        stages = [
            ("High-Level (s)", "high_level_s"),
            ("Interface (s)", "interface_s"),
            ("Task-Alloc (s)", "task_alloc_s"),
            ("Low-Level MAPF (s)", "mapf_s"),
            ("Low-Level Total (s)", "low_level_total_s"),
            ("Total (s)", "total_s"),
        ]

        for label, key in stages:
            sst = stats([src[k].get(key) for k in overlap])
            ust = stats([upd[k].get(key) for k in overlap])

            w.writerow([f"{label} - SRC overlap-success", sst["min"], sst["median"], sst["mean"], sst["max"]])
            w.writerow([f"{label} - UPDATED overlap-success", ust["min"], ust["median"], ust["mean"], ust["max"]])

    print(f"\nWrote DARS-style overlap table: {table_path}")

    # Optional: write per-run comparison CSV (helpful for debugging)
    if args.write_per_run_csv:
        per_run_path = os.path.join(out_dir, f"overlap_per_run_{ts}.csv")
        cols = [
            "run_key",
            "src_file", "upd_file",
            "src_index", "upd_index",
            "src_hl_status", "src_ll_status",
            "upd_hl_status", "upd_ll_status",
            "src_high_level_s", "upd_high_level_s", "delta_high_level_s",
            "src_interface_s", "upd_interface_s", "delta_interface_s",
            "src_task_alloc_s", "upd_task_alloc_s", "delta_task_alloc_s",
            "src_mapf_s", "upd_mapf_s", "delta_mapf_s",
            "src_low_level_total_s", "upd_low_level_total_s", "delta_low_level_total_s",
            "src_total_s", "upd_total_s", "delta_total_s",
            "src_makespan", "upd_makespan", "delta_makespan",
        ]
        with open(per_run_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for k in overlap:
                rs = src[k]
                ru = upd[k]
                def d(a, b):
                    return (b - a) if (a is not None and b is not None) else None
                w.writerow([
                    k,
                    rs.get("file"), ru.get("file"),
                    rs.get("index"), ru.get("index"),
                    rs.get("hl_status"), rs.get("ll_status"),
                    ru.get("hl_status"), ru.get("ll_status"),
                    rs.get("high_level_s"), ru.get("high_level_s"), d(rs.get("high_level_s"), ru.get("high_level_s")),
                    rs.get("interface_s"), ru.get("interface_s"), d(rs.get("interface_s"), ru.get("interface_s")),
                    rs.get("task_alloc_s"), ru.get("task_alloc_s"), d(rs.get("task_alloc_s"), ru.get("task_alloc_s")),
                    rs.get("mapf_s"), ru.get("mapf_s"), d(rs.get("mapf_s"), ru.get("mapf_s")),
                    rs.get("low_level_total_s"), ru.get("low_level_total_s"), d(rs.get("low_level_total_s"), ru.get("low_level_total_s")),
                    rs.get("total_s"), ru.get("total_s"), d(rs.get("total_s"), ru.get("total_s")),
                    rs.get("makespan"), ru.get("makespan"), d(rs.get("makespan"), ru.get("makespan")),
                ])

        print(f"Wrote per-run overlap comparison: {per_run_path}")

if __name__ == "__main__":
    main()
