#!/usr/bin/env python3
"""
Gate G1: does the rebuilt component model beat the legacy signal?

The gate was declared in fixes/EXPERIMENT_LEDGER.md BEFORE this was run:

  PASS requires BOTH
    1. the new signal wins in all three holdout seasons, and
    2. the mean margin exceeds +148 points (the measured noise floor).

Anything else is a FAIL. The gate is not edited after seeing the result.

Usage:  .venv/bin/python fpl2/run_gate_g1.py
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, legacy, simulate   # noqa: E402

NOISE_FLOOR = 148.0
XP_CACHE = Path("/tmp/fpl2_xp.pkl")


def build_comparison_panels():
    """One panel per season carrying BOTH signals on identical rows."""
    new_xp = pickle.load(open(XP_CACHE, "rb"))
    base = data.build_panel()
    out = {}
    for season in C.HOLDOUT_SEASONS:
        grid = legacy.attach(data.with_blanks(base, season))
        nx = new_xp[season][["season", "gameweek", C.PLAYER_KEY, "xp"]].rename(
            columns={"xp": "xp_new"})
        g = grid.merge(nx, on=["season", "gameweek", C.PLAYER_KEY], how="left")
        g["xp_new"] = g["xp_new"].fillna(0.0)
        g = g.rename(columns={"xp": "xp_legacy"})
        # a blank gameweek scores nothing under either signal
        blank = g["fixtures"] <= 0
        g.loc[blank, ["xp_new", "xp_legacy"]] = 0.0
        out[season] = g
    return out


def main():
    panels = build_comparison_panels()
    rows = []
    for season, p in panels.items():
        r_new = simulate.simulate_season(p, "xp_new")
        r_old = simulate.simulate_season(p, "xp_legacy")
        if r_new is None or r_old is None:
            print(f"  {season}: simulation infeasible"); continue
        rows.append({"season": season, "new": r_new.total, "legacy": r_old.total,
                     "margin": r_new.total - r_old.total,
                     "new_tr": r_new.transfers, "legacy_tr": r_old.transfers})
        print(f"  {season}  new {r_new.total:7.0f}  legacy {r_old.total:7.0f}  "
              f"margin {r_new.total - r_old.total:+7.0f}", flush=True)
    d = pd.DataFrame(rows)
    if d.empty:
        print("no results"); return 1

    wins_all = bool((d["margin"] > 0).all())
    mean_margin = float(d["margin"].mean())
    passed = wins_all and mean_margin > NOISE_FLOOR

    print("\n" + "=" * 64)
    print("GATE G1  (declared 2026-10-20, before this run)")
    print("=" * 64)
    print(d.round(0).to_string(index=False))
    print(f"\n  1. wins in all three seasons      : {'YES' if wins_all else 'NO'}"
          f"   ({int((d['margin']>0).sum())}/3)")
    print(f"  2. mean margin exceeds {NOISE_FLOOR:.0f} points : "
          f"{'YES' if mean_margin > NOISE_FLOOR else 'NO'}"
          f"   (mean {mean_margin:+.0f})")
    print(f"\n  => G1 {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("\n  The legacy signal stays. Record the result in the ledger and")
        print("  do NOT adjust the gate -- that is how the old project got to -inf.")
    d.to_csv("/tmp/fpl2_gate_g1.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
