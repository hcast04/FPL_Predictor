#!/usr/bin/env python3
"""
O13: where are the OTHER ~150 points?

S11 split the ~300-point deficit: ~150 is the opening squad, ~150 remains with
an identical opening 15 and near-identical ranking. This isolates the rest.

Both signals always start from LEGACY's opening 15, so squad selection is
removed by construction. Then `min_gain` is swept:

  999  no transfers at all -- the 15 players are IDENTICAL all season for both
       signals, so ONLY XI and captain can differ. If the gap survives here it
       is XI/captain; if it collapses, it is transfers.
  0.0 .. 1.0  the real threshold. `min_gain` is an ABSOLUTE number compared
       against xp differences (O8), so if our xp is more tightly spread than
       legacy's it fires at a different rate for reasons unrelated to accuracy.

Reported per arm: margin, and the transfer COUNT for each signal -- if the two
make very different numbers of transfers at the same threshold, that is the
scale effect showing up directly.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, legacy, simple, simulate, team, optimise  # noqa: E402

GRID = (0.0, 0.15, 0.30, 0.60, 1.00, 999.0)


def main():
    t0 = time.time()
    raw = data.load_raw(); panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    print(f"[{time.time()-t0:4.0f}s] setup\n", flush=True)

    panels, opens = {}, {}
    for s in C.HOLDOUT_SEASONS:
        fc = simple.build(panel, tf, s)
        g = legacy.attach(data.with_blanks(panel, s)).rename(columns={"xp": "xp_legacy"})
        g = g.merge(fc[["season", "gameweek", C.PLAYER_KEY, "xp"]],
                    on=["season", "gameweek", C.PLAYER_KEY], how="left")
        g["xp"] = g["xp"].fillna(0.0)
        g.loc[g["fixtures"] <= 0, ["xp", "xp_legacy"]] = 0.0
        panels[s] = g
        gws = sorted(g["gameweek"].unique())[:5]
        w = g[g["gameweek"].isin(gws)]
        st = (w.groupby(C.PLAYER_KEY)
               .agg(xp_legacy=("xp_legacy", "mean"), cost=("cost", "first"),
                    position=("position", "first"), team=("team", "first")).reset_index())
        sq = optimise.optimise_squad(st, "xp_legacy")
        opens[s] = sq[C.PLAYER_KEY].tolist() if sq is not None else None

    print(f"{'min_gain':>9}{'season':>10}{'simple':>8}{'legacy':>8}{'margin':>8}"
          f"{'tr_s':>6}{'tr_l':>6}", flush=True)
    print("-" * 55, flush=True)
    summary = {}
    for mg in GRID:
        margins, trs, trl = [], [], []
        for s in C.HOLDOUT_SEASONS:
            g, sq = panels[s], opens[s]
            a = simulate.simulate_season(g, "xp", min_gain=mg, opening_squad=sq)
            b = simulate.simulate_season(g, "xp_legacy", min_gain=mg, opening_squad=sq)
            if a is None or b is None:
                continue
            margins.append(float(a.total) - float(b.total))
            trs.append(a.transfers); trl.append(b.transfers)
            lbl = f"{mg:.2f}" if mg < 900 else "none"
            print(f"{lbl:>9}{s:>10}{a.total:>8.0f}{b.total:>8.0f}"
                  f"{margins[-1]:>+8.0f}{a.transfers:>6}{b.transfers:>6}", flush=True)
        summary[mg] = (float(np.mean(margins)), float(np.mean(trs)), float(np.mean(trl)))
        print(flush=True)

    print("SUMMARY  (simple - legacy, both from legacy's opening 15)", flush=True)
    print(f"{'min_gain':>9}{'mean_margin':>13}{'mean_tr_s':>11}{'mean_tr_l':>11}", flush=True)
    for mg, (m, a, b) in summary.items():
        lbl = f"{mg:.2f}" if mg < 900 else "none"
        print(f"{lbl:>9}{m:>+13.0f}{a:>11.1f}{b:>11.1f}", flush=True)
    frozen = summary.get(999.0, (float('nan'),))[0]
    print(f"\n  min_gain=none isolates XI+CAPTAIN only (squad identical all season).", flush=True)
    print(f"  margin there = {frozen:+.0f}.  If ~0, the gap is TRANSFERS.", flush=True)
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
