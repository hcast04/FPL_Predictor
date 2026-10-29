#!/usr/bin/env python3
"""
Does S1 hold up? Ownership as the signal through OUR simulator.

S1 claims model_xp 2145 vs template 1923 -- a +222 margin, "SOLID", 3/3
seasons. Its template was ownership (`selected`) fed through a budget-
constrained simulator: 100.0, 15-man squad, one free transfer, no hits.

But an UNCONSTRAINED most-owned XI scores 2073-2336 (54.6-61.5/gw), and the
real FPL average is 59.4/gw -- about 2257 over a full season. So the real
average manager performs like the unconstrained template, NOT like the
constrained one. Real managers have wildcards, chips, hits and accumulated team
value; the simulator has none of those.

If that is right, S1's +222 is a statement about the SIMULATOR, not about
beating real managers, and the honest comparison for "are we beating the
average" is against ~2257, not against 1923.

This runs ownership through `fpl2.simulate` exactly as the model signal is run,
so both face identical constraints.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, legacy, simulate   # noqa: E402

AVG_PER_GW = 59.4     # FPL global average, 2026-27 GW1-7


def main():
    t0 = time.time()
    own = (pd.read_csv("data/historical_fpl.csv.gz", compression="gzip",
                       low_memory=False,
                       usecols=["season", "gameweek", "player_code", "selected"])
             .groupby(["season", "gameweek", "player_code"], as_index=False)
             .selected.max())
    panel = data.build_panel()
    print(f"[{time.time()-t0:4.0f}s] setup\n", flush=True)

    print(f"{'season':10s}{'ownership':>11}{'legacy':>9}{'avg_equiv':>11}", flush=True)
    print("-" * 41, flush=True)
    rows = {}
    for s in C.HOLDOUT_SEASONS:
        g = legacy.attach(data.with_blanks(panel, s)).rename(columns={"xp": "xp_leg"})
        g = g.merge(own[own.season.eq(s)], on=["season", "gameweek", C.PLAYER_KEY],
                    how="left")
        g["own"] = pd.to_numeric(g["selected"], errors="coerce").fillna(0.0)
        for c in ("own", "xp_leg"):
            g.loc[g["fixtures"] <= 0, c] = 0.0
        r_own = simulate.simulate_season(g, "own")
        r_leg = simulate.simulate_season(g, "xp_leg")
        n = g.gameweek.nunique()
        rows[s] = (float(r_own.total), float(r_leg.total), AVG_PER_GW * n)
        print(f"{s:10s}{rows[s][0]:>11.0f}{rows[s][1]:>9.0f}{rows[s][2]:>11.0f}",
              flush=True)

    o = np.mean([v[0] for v in rows.values()])
    l = np.mean([v[1] for v in rows.values()])
    a = np.mean([v[2] for v in rows.values()])
    print(f"\n  mean ownership-through-simulator : {o:.0f}", flush=True)
    print(f"  mean legacy                      : {l:.0f}", flush=True)
    print(f"  FPL average equivalent (59.4/gw) : {a:.0f}", flush=True)
    print(f"\n  S1 recorded template = 1923. Reproduced here = {o:.0f}.", flush=True)
    print(f"  Gap from constrained template to the REAL average: "
          f"{a - o:+.0f} points/season.", flush=True)
    print("  That gap is what wildcards, chips, hits and accumulated team value", flush=True)
    print("  are worth -- none of which the simulator has. It applies to the", flush=True)
    print("  MODEL equally, so it is not a defect of the model.", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
