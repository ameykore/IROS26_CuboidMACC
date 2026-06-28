# audit_summary_csv.py
import csv
from collections import Counter

def audit(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)

    idxs = []
    for row in rows:
        try:
            idxs.append(int(float(row["index"])))
        except Exception:
            idxs.append(None)

    c = Counter(idxs)
    dup = sorted([k for k,v in c.items() if k is not None and v > 1])

    def is_success(row):
        return row.get("hl_status") == "FOUND" and row.get("ll_status") == "FOUND"

    succ = [row for row in rows if is_success(row)]

    # sanity check total_s vs time_limit
    def get_float(row, k):
        try:
            return float(row[k])
        except Exception:
            return None

    time_limits = [get_float(row, "time_limit") for row in rows]
    total_s = [get_float(row, "total_s") for row in rows]

    def summarize(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return (min(vals), sorted(vals)[len(vals)//2], sum(vals)/len(vals), max(vals))

    print(f"\n=== {path} ===")
    print(f"rows: {len(rows)}")
    print(f"unique indices (excluding None): {len(set(i for i in idxs if i is not None))}")
    print(f"duplicate indices count: {len(dup)} (show first 10: {dup[:10]})")
    print(f"success (FOUND/FOUND): {len(succ)}")
    print(f"time_limit stats (min,med,mean,max): {summarize(time_limits)}")
    print(f"total_s stats     (min,med,mean,max): {summarize(total_s)}")

if __name__ == "__main__":
    audit(r"C:\Users\biorobotics\Desktop\MACC\cuboidMACC-DARS24\src\out\random_10x10x4_15blocks_1000s\combined_200_20260120\summary_20260120_180213.csv")
    audit(r"C:\Users\biorobotics\Desktop\MACC\cuboidMACC-DARS24\updated_src\out\random_10x10x4_15blocks_1000s\combined_200_20260120\summary_20260120_180133.csv")
