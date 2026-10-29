#!/usr/bin/env python3
"""
Gate G2: does a per-player reliability correction help?

Component gate FIRST (top-20 realised points, all three seasons), then G1.
Declared in fixes/EXPERIMENT_LEDGER.md before this was run.

Needs honest out-of-sample predictions on the season BEFORE each holdout, so it
runs an inner walk-forward. In-sample residuals would make every player look
reliable and are never used.
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, model, reliability, team   # noqa: E402

PRIOR_OF = {"2024-25": "2023-24", "2025-26": "2024-25"}   # 2023-24 has no honest prior


def main():
    feats_df = pd.read_pickle("/tmp/fpl2_features.pkl")
    names = pickle.load(open("/tmp/fpl2_featnames.pkl", "rb"))
    tf = team.team_fixtures(data.load_raw())
    cached = pickle.load(open("/tmp/fpl2_xp.pkl", "rb"))

    rows = []
    adjusted = {}
    for season in C.HOLDOUT_SEASONS:
        base = cached[season].copy()
        base["xp_base"] = base["xp_raw"] if "xp_raw" in base else base["xp"]

        rel = reliability.Reliability()
        prior_season = PRIOR_OF.get(season)
        if prior_season:
            print(f"  {season}: inner walk-forward on {prior_season} "
                  f"for honest residuals...", flush=True)
            honest = model.walk_forward(feats_df, tf, names, prior_season)
            rel.fit_prior_season(honest, pred_col="xp", actual_col="points")
        else:
            print(f"  {season}: no honest prior season available, "
                  f"within-season evidence only", flush=True)
            rel.table = {"prior": {}, "prior_n": {}}

        # apply gameweek by gameweek, using only strictly earlier weeks
        out = []
        for gw in sorted(base["gameweek"].unique()):
            wk = base[base["gameweek"].eq(gw)].copy()
            hist = base[base["gameweek"] < gw]
            wk["xp_adj"] = rel.apply(wk, wk["xp_base"].to_numpy(float),
                                     history=hist.rename(columns={"xp_base": "xp"}))
            out.append(wk)
        adj = pd.concat(out, ignore_index=True)
        adjusted[season] = adj

        t = adj[adj["fixtures"] > 0]
        b = [g.nlargest(20, "xp_base")["points"].mean()
             for _, g in t.groupby("gameweek") if len(g) >= 50]
        a = [g.nlargest(20, "xp_adj")["points"].mean()
             for _, g in t.groupby("gameweek") if len(g) >= 50]
        rows.append({"season": season, "top20_base": float(np.mean(b)),
                     "top20_adj": float(np.mean(a)),
                     "delta": float(np.mean(a) - np.mean(b))})
        print(f"    top20 base {np.mean(b):.3f} -> adjusted {np.mean(a):.3f} "
              f"({np.mean(a)-np.mean(b):+.3f})", flush=True)

    d = pd.DataFrame(rows)
    print("\n" + "=" * 62)
    print("GATE G2 — COMPONENT GATE (declared before the run)")
    print("=" * 62)
    print(d.round(3).to_string(index=False))
    wins = int((d["delta"] > 0).sum())
    passed = wins == 3
    print(f"\n  improves top-20 in all three seasons: "
          f"{'YES' if passed else 'NO'}  ({wins}/3)")
    print(f"  => component gate {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("\n  Does NOT proceed to G1. Record and do not tune alpha to rescue it.")
    else:
        pickle.dump(adjusted, open("/tmp/fpl2_xp_adj.pkl", "wb"))
        print("\n  proceeds to G1; adjusted forecasts cached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
