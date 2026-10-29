#!/usr/bin/env python3
"""
The opening squad is ~148 of the ~304-point gap. Why?

It is chosen from mean xp over gameweeks 1-5 -- the window where the model is
blindest: no within-season data, stale tiers, and the new-to-PL cold start that
runs at 0.713 for Championship arrivals.

Three questions, each a separate arm:

  1 HORIZON    does looking at 1/3/5/8/10 gameweeks change it? If the 5-GW mean
               is noisy, a longer horizon should help and a shorter one hurt.
  2 ORACLE     what would a PERFECT opening squad score? Picked on realised
               points, it is unattainable by construction -- it bounds how much
               is available here at all, so we do not chase a ceiling that is
               mostly variance.
  3 LEGACY-OPEN already measured: +148. Reported for reference.

Everything after the opening squad is held identical: same signal, same
transfer rule, same XI and captain.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, legacy, simple, simulate, team, optimise  # noqa: E402


def opening_from(panel, signal, horizon, budget=C.BUDGET):
    gws = sorted(panel["gameweek"].unique())[:horizon]
    w = panel[panel["gameweek"].isin(gws)]
    st = (w.groupby(C.PLAYER_KEY)
           .agg(**{signal: (signal, "mean"), "cost": ("cost", "first"),
                   "position": ("position", "first"), "team": ("team", "first")})
           .reset_index())
    sq = optimise.optimise_squad(st, signal, budget=budget)
    return None if sq is None or len(sq) != C.SQUAD_SIZE else sq[C.PLAYER_KEY].tolist()


def main():
    t0 = time.time()
    raw = data.load_raw(); panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    print(f"[{time.time()-t0:4.0f}s] setup\n", flush=True)

    rows = {}
    for s in C.HOLDOUT_SEASONS:
        fc = simple.build(panel, tf, s)
        g = legacy.attach(data.with_blanks(panel, s)).rename(columns={"xp": "xp_legacy"})
        g = g.merge(fc[["season", "gameweek", C.PLAYER_KEY, "xp"]],
                    on=["season", "gameweek", C.PLAYER_KEY], how="left")
        g["xp"] = g["xp"].fillna(0.0)
        blank = g["fixtures"] <= 0
        g.loc[blank, ["xp", "xp_legacy"]] = 0.0

        leg = simulate.simulate_season(g, "xp_legacy")
        print(f"--- {s}   legacy {leg.total:.0f} ---", flush=True)
        for h in (1, 3, 5, 8, 10):
            sq = opening_from(g, "xp", h)
            r = simulate.simulate_season(g, "xp", opening_squad=sq)
            if r:
                rows.setdefault(f"horizon {h}", []).append(r.total - leg.total)
                print(f"    horizon {h:>2}        {r.total:>7.0f}"
                      f"   {r.total-leg.total:>+7.0f}", flush=True)
        # oracle: opening squad picked on REALISED points over the same window
        sq = opening_from(g, "points", 5)
        r = simulate.simulate_season(g, "xp", opening_squad=sq)
        if r:
            rows.setdefault("oracle open (5gw)", []).append(r.total - leg.total)
            print(f"    oracle open       {r.total:>7.0f}"
                  f"   {r.total-leg.total:>+7.0f}   <- unattainable bound", flush=True)
        sq = opening_from(g, "xp_legacy", 5)
        r = simulate.simulate_season(g, "xp", opening_squad=sq)
        if r:
            rows.setdefault("legacy open", []).append(r.total - leg.total)
            print(f"    legacy open       {r.total:>7.0f}"
                  f"   {r.total-leg.total:>+7.0f}", flush=True)
        print(flush=True)

    print("MEAN MARGIN vs LEGACY, by opening-squad rule", flush=True)
    for k, v in rows.items():
        print(f"  {k:20s}{np.mean(v):>+9.0f}", flush=True)
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
