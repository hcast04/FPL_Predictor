#!/usr/bin/env python3
"""
Where do legacy's ~300 points a season actually come from?

Three signals -- a 120-feature GBM, a no-training tier lookup, and legacy --
rank almost identically (spearman 0.67-0.72) and calibrate similarly, yet
legacy wins the squad simulation by ~290-304 points. So the gap is NOT forecast
quality. It must live in the decision layers.

Decomposition, each step removing one source of difference:

  A  full           each signal picks its own opening squad, transfers, XI, captain
  B  shared opening both start from LEGACY's opening 15; gap = transfers+XI+captain
  C  shared opening each signal's own squad, but xp RESCALED to legacy's spread;
     isolates the absolute-scale effect on min_gain (0.30 in raw xp units) and
     on budget allocation

If B collapses the gap, the opening squad is the problem. If C collapses it,
the transfer threshold is mis-scaled for our signal and is firing at the wrong
rate -- a policy bug, not a model one.
"""
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, legacy, simple, simulate, team, optimise  # noqa: E402

CACHE = Path("/tmp/claude-1000/-home-mdmsystems-acer-Escritorio-pers-fpl/"
             "a909bbf0-dfbe-4cc0-a5f8-534424e1d2b9/scratchpad")


def opening(panel, signal, horizon=5, budget=C.BUDGET):
    gws = sorted(panel["gameweek"].unique())[:horizon]
    w = panel[panel["gameweek"].isin(gws)]
    strength = (w.groupby(C.PLAYER_KEY)
                 .agg(**{signal: (signal, "mean"), "cost": ("cost", "first"),
                         "position": ("position", "first"), "team": ("team", "first")})
                 .reset_index())
    sq = optimise.optimise_squad(strength, signal, budget=budget)
    return None if sq is None or len(sq) != C.SQUAD_SIZE else sq[C.PLAYER_KEY].tolist()


def main():
    t0 = time.time()
    raw = data.load_raw(); panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    print(f"[{time.time()-t0:4.0f}s] setup\n", flush=True)

    print(f"{'season':10s}{'arm':22s}{'simple':>9}{'legacy':>9}{'margin':>9}", flush=True)
    print("-" * 59, flush=True)
    agg = {}
    for s in C.HOLDOUT_SEASONS:
        fc = simple.build(panel, tf, s)
        g = legacy.attach(data.with_blanks(panel, s)).rename(columns={"xp": "xp_legacy"})
        g = g.merge(fc[["season", "gameweek", C.PLAYER_KEY, "xp"]],
                    on=["season", "gameweek", C.PLAYER_KEY], how="left")
        g["xp"] = g["xp"].fillna(0.0)
        blank = g["fixtures"] <= 0
        g.loc[blank, ["xp", "xp_legacy"]] = 0.0

        # C: rescale our xp to legacy's spread, on PLAYING rows only
        live = ~blank
        a, b = g.loc[live, "xp"], g.loc[live, "xp_legacy"]
        g["xp_scaled"] = 0.0
        if a.std() > 1e-9:
            g.loc[live, "xp_scaled"] = (a - a.mean()) / a.std() * b.std() + b.mean()
            g["xp_scaled"] = g["xp_scaled"].clip(lower=0.0)

        leg_open = opening(g, "xp_legacy")
        res = {}
        for arm, sig, sq in (("A full", "xp", None),
                             ("B shared opening", "xp", leg_open),
                             ("C rescaled xp", "xp_scaled", None)):
            r = simulate.simulate_season(g, sig, opening_squad=sq)
            rl = simulate.simulate_season(g, "xp_legacy",
                                          opening_squad=leg_open if sq else None)
            if r is None or rl is None:
                continue
            res[arm] = (float(r.total), float(rl.total))
            print(f"{s if arm=='A full' else '':10s}{arm:22s}"
                  f"{r.total:>9.0f}{rl.total:>9.0f}{r.total-rl.total:>+9.0f}", flush=True)
            agg.setdefault(arm, []).append(r.total - rl.total)
        print(flush=True)

    print("MEAN MARGIN BY ARM  (simple - legacy)", flush=True)
    for arm, v in agg.items():
        print(f"  {arm:22s}{np.mean(v):>+9.0f}", flush=True)
    print("\n  A->B isolates the OPENING SQUAD; A->C isolates ABSOLUTE SCALE", flush=True)
    print("  (which drives both budget allocation and the 0.30 transfer threshold).",
          flush=True)
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
