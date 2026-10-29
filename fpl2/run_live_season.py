#!/usr/bin/env python3
"""
Simulate 2026-27 to date with the rebuilt model, and compare against what the
user actually scored and against the global average.

Trains on 2022-26 and forecasts 2026-27 walk-forward, then runs the same
constrained simulator used for the holdout seasons: GBP 100m, 2/5/5/3, max 3
per club, one free transfer a week, no hits.

CAVEAT carried from fpl2.live: prices are current, not historical, so
budget-constrained selection is mildly optimistic.
"""
from __future__ import annotations
import pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, live, model, simulate, team  # noqa: E402

SEASON = "2026-27"
# what actually happened, from the FPL API (entry $FPL_ENTRY) and bootstrap
ACTUAL_USER = {1: 54, 2: 67, 3: 47, 4: 63, 5: 39, 6: 30}
GLOBAL_AVG  = {1: 50, 2: 81, 3: 51, 4: 69, 5: 48, 6: 58}


def main():
    t0 = time.time()
    hist = data.load_raw()
    cur = pd.read_pickle(str(C.cache("fpl2_live2627.pkl")))
    raw = pd.concat([hist, cur], ignore_index=True)
    panel = data.build_panel(raw)
    tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)
    print(f"[{time.time()-t0:5.0f}s] combined panel {len(panel):,} rows, "
          f"{len(names)} features; {SEASON} gameweeks "
          f"{sorted(panel[panel.season.eq(SEASON)].gameweek.unique())}", flush=True)

    out = model.walk_forward(feats, tf, names, SEASON)
    t = out[out.fixtures > 0]
    print(f"[{time.time()-t0:5.0f}s] forecast: ratio {t.xp.sum()/t.points.sum():.3f}"
          f"  spearman {t.xp.corr(t.points, method='spearman'):.4f}", flush=True)

    grid = data.with_blanks(panel, SEASON)
    grid = grid.merge(out[["season","gameweek","player_code","xp"]],
                      on=["season","gameweek","player_code"], how="left")
    grid["xp"] = grid["xp"].fillna(0.0)
    grid.loc[grid["fixtures"] <= 0, "xp"] = 0.0

    r = simulate.simulate_season(grid, "xp")
    gws = sorted(grid.gameweek.unique())
    print(f"\n{'gw':>3} {'model':>7} {'you':>6} {'avg':>6} {'vs you':>8} {'vs avg':>8}")
    cum_m = cum_u = cum_a = 0
    for i, gw in enumerate(gws):
        m = r.per_gameweek[i] if i < len(r.per_gameweek) else float("nan")
        u, a = ACTUAL_USER.get(gw), GLOBAL_AVG.get(gw)
        cum_m += m; cum_u += (u or 0); cum_a += (a or 0)
        print(f"{gw:>3} {m:>7.0f} {u if u else '-':>6} {a if a else '-':>6}"
              f" {m-(u or 0):>+8.0f} {m-(a or 0):>+8.0f}")
    print(f"\n{'TOT':>3} {cum_m:>7.0f} {cum_u:>6} {cum_a:>6}"
          f" {cum_m-cum_u:>+8.0f} {cum_m-cum_a:>+8.0f}")
    print(f"\n  transfers made: {r.transfers}")
    print(f"  per gameweek   : model {cum_m/len(gws):.1f}  you {cum_u/len(gws):.1f}"
          f"  average {cum_a/len(gws):.1f}")
    print("\n  NOTE: prices are current, not historical (see fpl2/live.py).")
    print("  The model also picks a fresh optimal squad at GW1; a real manager")
    print("  is locked into whatever they chose before the season started.")


if __name__ == "__main__":
    main()
