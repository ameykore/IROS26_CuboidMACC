# summarize_results.py (fixed)
import argparse, os, re, glob, dill, csv, statistics as S
from math import isnan
from datetime import datetime

def stats(xs):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {"count": len(xs), "min": min(xs), "median": S.median(xs), "mean": S.mean(xs), "max": max(xs)}

def _safe_makespan(low_out):
    """Return makespan from low_level_output whether it's dict, list, or None."""
    if low_out is None:
        return None
    if isinstance(low_out, dict):
        return low_out.get("makespan")
    if isinstance(low_out, list) and low_out:
        # take last element if it’s a dict with makespan, else None
        last = low_out[-1]
        if isinstance(last, dict):
            return last.get("makespan")
    return None

def load_record(pkl_path):
    d = dill.load(open(pkl_path, "rb"))
    m = re.search(r"_n(\d+)\.out\.pkl$", os.path.basename(pkl_path))
    idx = int(m.group(1)) if m else None

    phases = d.get("phase_durations") or {}
    # Normalize known keys; absent → 0.0 so totals still make sense.
    hl  = float(phases.get("high_level", 0.0))
    tw  = float(phases.get("time_windows", 0.0))
    ta  = float(phases.get("task_allocation", 0.0))
    mpf = float(phases.get("MAPF", 0.0))
    total = (hl + tw + ta + mpf) if phases else None

    return {
        "file": pkl_path,
        "index": idx,
        "hl_status": d.get("high_level_plan_status"),
        "ll_status": d.get("low_level_plan_status"),
        "time_limit": d.get("time_limit"),
        "high_level_s": hl,
        "interface_s": tw,
        "task_alloc_s": ta,
        "mapf_s": mpf,
        "total_s": total,
        "makespan": _safe_makespan(d.get("low_level_output")),
    }

def fmt(st):
    def f(x):
        return "nan" if x is None else f"{x:.3f}"
    return f"count={st['count']}  min={f(st['min'])}  median={f(st['median'])}  mean={f(st['mean'])}  max={f(st['max'])}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder holding result files (searched recursively).")
    ap.add_argument(
        "--pattern",
        default="*.out.pkl",
        help="Glob for result files (default: *.out.pkl to include random + struct outputs)."
    )
    ap.add_argument("--out", default=None, help="Output CSV path (default: <root>/summary_<ts>.csv)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "**", args.pattern), recursive=True))
    if not files:
        print("No result files found. Root:", args.root, "Pattern:", args.pattern)
        return

    rows = []
    for f in files:
        try:
            rows.append(load_record(f))
        except Exception as e:
            print("WARN: failed to load", f, "->", repr(e))

    # Overall stats
    HL = stats([r["high_level_s"] for r in rows])
    IF = stats([r["interface_s"]   for r in rows])
    LL = stats([r["mapf_s"]        for r in rows])
    TOT= stats([r["total_s"]       for r in rows])

    # Success-only (both levels FOUND)
    succ = [r for r in rows if r["hl_status"] == "FOUND" and r["ll_status"] == "FOUND"]
    sHL = stats([r["high_level_s"] for r in succ])
    sIF = stats([r["interface_s"]  for r in succ])
    sLL = stats([r["mapf_s"]       for r in succ])
    sTO = stats([r["total_s"]      for r in succ])

    print(f"Files: {len(rows)}  Successes: {len(succ)}  Fail/Timeout: {len(rows)-len(succ)}")
    print("\n== OVERALL (all results) ==")
    print("High-Level (s) ->", fmt(HL))
    print("Interface (s)  ->", fmt(IF))
    print("Low-Level (s)  ->", fmt(LL))
    print("Total (s)      ->", fmt(TOT))
    print("\n== SUCCESS-ONLY (hl=FOUND & ll=FOUND) ==")
    print("High-Level (s) ->", fmt(sHL))
    print("Interface (s)  ->", fmt(sIF))
    print("Low-Level (s)  ->", fmt(sLL))
    print("Total (s)      ->", fmt(sTO))

    # Write per-instance CSV and DARS-style table
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = args.out or os.path.join(args.root, f"summary_{ts}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index","hl_status","ll_status","time_limit","high_level_s","interface_s","task_alloc_s","mapf_s","total_s","makespan","file"])
        for r in rows:
            w.writerow([r["index"],r["hl_status"],r["ll_status"],r["time_limit"],
                        r["high_level_s"],r["interface_s"],r["task_alloc_s"],r["mapf_s"],r["total_s"],r["makespan"],r["file"]])
    print("Wrote per-instance CSV:", out_csv)

    table_csv = os.path.join(args.root, f"table_{ts}.csv")
    with open(table_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Stage","Set","Min","Median","Mean","Max","Count"])
        for label,st in [("High-Level (s)",HL),("Interface (s)",IF),("Low-Level (s)",LL),("Total (s)",TOT)]:
            w.writerow([label,"All",st["min"],st["median"],st["mean"],st["max"],st["count"]])
        for label,st in [("High-Level (s)",sHL),("Interface (s)",sIF),("Low-Level (s)",sLL),("Total (s)",sTO)]:
            w.writerow([label,"Success-Only",st["min"],st["median"],st["mean"],st["max"],st["count"]])
    print("Wrote DARS-style table:", table_csv)

if __name__ == "__main__":
    main()
