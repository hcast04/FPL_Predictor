#!/usr/bin/env python3
"""Compare the F2-on and F2-off fold arms on identical evaluation code."""
import glob, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_on_real_folds as V

ROOT = Path(__file__).resolve().parent.parent

def arm(name):
    out = {}
    for p in sorted(glob.glob(str(ROOT / name / "*.pkl.gz"))):
        pay = pd.read_pickle(p, compression="gzip")
        out[str(pay["holdout_season"])] = pay["bundle"]["holdout_expected_points"]
    return out

on, off = arm("data/fold_cache_F2on"), arm("data/fold_cache_F2off")
if not on or not off:
    print("Both arms needed. Run save-on, then the F2-off notebook run, then save-off.")
    raise SystemExit(1)

rows = []
for season in sorted(set(on) & set(off)):
    for label, frames in (("F2 on (mask applied)", on), ("F2 off (raw zeros kept)", off)):
        f = frames[season]
        r = V.score(f, pd.to_numeric(f["expected_points"], errors="coerce").to_numpy(float), label)
        r["season"] = season
        rows.append(r)

df = pd.DataFrame(rows)
cols = ["MAE", "spearman", "ndcg@10", "top10_pts", "best_XI_pts"]
print("\nPer season:")
print(df.set_index(["season", "variant"])[cols].round(4).to_string())
base = df[df.variant.str.startswith("F2 off")].set_index("season")[cols]
test = df[df.variant.str.startswith("F2 on")].set_index("season")[cols]
print("\nPaired delta (F2 on - F2 off), averaged over folds:")
print((test - base).mean().round(4).to_string())
print("\nPositive top10_pts / best_XI_pts means the mask HELPS.")
