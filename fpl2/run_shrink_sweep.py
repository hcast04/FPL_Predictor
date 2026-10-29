#!/usr/bin/env python3
"""
Shrinkage sweep on the simple model, judged by the unmodified G1 gate.

The 2025-26 opening squad put 14.5 into Salah (123 pts) on an unregressed
2024-25 rate; legacy's opening 15 was worth 2184 season points to ours at 1683.
Empirical-Bayes shrinkage toward the position mean is the direct fix.

k=0 is the control (no shrinkage, today's simple model). The whole curve is
reported, not the best point: a dose-response is what makes a parameter result
trustworthy, and picking the best k from three noisy seasons is the multiple-
testing trap R3 warns about.

Also reported: season points of the OPENING 15 itself, which is the quantity
the mechanism actually predicts should improve.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, legacy, simple, simulate, team, optimise  # noqa: E402

GRID = (0.0, 5.0, 10.0, 20.0, 40.0, 80.0)


def opening(g, sig):
    gws = sorted(g["gameweek"].unique())[:5]
    w = g[g["gameweek"].isin(gws)]
    st = (w.groupby(C.PLAYER_KEY)
           .agg(**{sig: (sig, "mean"), "cost": ("cost", "first"),
                   "position": ("position", "first"), "team": ("team", "first")})
           .reset_index())
    sq = optimise.optimise_squad(st, sig)
    return None if sq is None or len(sq) != C.SQUAD_SIZE else sq


def main():
    t0 = time.time()
    raw = data.load_raw(); panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    season_pts = {s: panel[panel.season.eq(s)].groupby(C.PLAYER_KEY).points.sum()
                  for s in C.HOLDOUT_SEASONS}
    print(f"[{time.time()-t0:4.0f}s] setup\n", flush=True)

    leg_tot, leg_open = {}, {}
    for s in C.HOLDOUT_SEASONS:
        g = legacy.attach(data.with_blanks(panel, s)).rename(columns={"xp": "xp_legacy"})
        g.loc[g["fixtures"] <= 0, "xp_legacy"] = 0.0
        leg_tot[s] = float(simulate.simulate_season(g, "xp_legacy").total)
        sq = opening(g, "xp_legacy")
        leg_open[s] = float(sq[C.PLAYER_KEY].map(season_pts[s]).fillna(0).sum())
    print("legacy:  " + "  ".join(f"{s} {leg_tot[s]:.0f} (open15 {leg_open[s]:.0f})"
                                  for s in C.HOLDOUT_SEASONS) + "\n", flush=True)

    print(f"{'k':>6}{'season':>10}{'total':>8}{'margin':>9}{'open15_pts':>12}"
          f"{'vs_leg_open':>13}", flush=True)
    print("-" * 58, flush=True)
    summary = {}
    for k in GRID:
        simple.SHRINK_K = k
        margins, opens = [], []
        for s in C.HOLDOUT_SEASONS:
            fc = simple.build(panel, tf, s)
            g = legacy.attach(data.with_blanks(panel, s)).rename(columns={"xp": "xp_legacy"})
            g = g.merge(fc[["season", "gameweek", C.PLAYER_KEY, "xp"]],
                        on=["season", "gameweek", C.PLAYER_KEY], how="left")
            g["xp"] = g["xp"].fillna(0.0)
            g.loc[g["fixtures"] <= 0, ["xp", "xp_legacy"]] = 0.0
            r = simulate.simulate_season(g, "xp")
            sq = opening(g, "xp")
            op = float(sq[C.PLAYER_KEY].map(season_pts[s]).fillna(0).sum()) if sq is not None else np.nan
            margins.append(float(r.total) - leg_tot[s]); opens.append(op)
            print(f"{k:>6.0f}{s:>10}{r.total:>8.0f}{margins[-1]:>+9.0f}"
                  f"{op:>12.0f}{op-leg_open[s]:>+13.0f}", flush=True)
        summary[k] = (float(np.mean(margins)), float(np.mean(opens)),
                      sum(1 for m in margins if m > 0))
        print(flush=True)

    print("DOSE-RESPONSE  (mean over 3 seasons)", flush=True)
    print(f"{'k':>6}{'mean_margin':>13}{'mean_open15':>13}{'seasons_won':>13}", flush=True)
    for k, (mm, mo, w) in summary.items():
        print(f"{k:>6.0f}{mm:>+13.0f}{mo:>13.0f}{w:>10}/3", flush=True)
    mean_leg_open = np.mean([leg_open[s] for s in C.HOLDOUT_SEASONS])
    print(f"\n  legacy open15 mean = {mean_leg_open:.0f}", flush=True)
    print("  G1 PASS needs 3/3 seasons AND mean margin > +148.", flush=True)
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
