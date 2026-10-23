#!/usr/bin/env python3
"""
O52c -- the opening-horizon dose-response.

`run_opt_bakeoff2.py` section B found the only non-null in the whole optimiser
race, and found it as a PAIR rather than a p-value:

    opening_horizon = 1    -159.3   1/8 seasons
    opening_horizon = 5       0.0   (the shipped default)
    opening_horizon = 10   +136.6   7/8 seasons

Monotone across three points, with the shipped value in the middle and the
extremes ~300 points apart. Neither extreme clears |t| > 2.36 on its own
(p = 0.064 and 0.076) because the MDE there is ~190 -- one squad per arm per
season leaves the between-squad variance in. A dose-response measured at many
doses is stronger evidence than either endpoint's p-value, and it also says
whether the effect saturates or keeps climbing, which decides what to ship.

`opening_horizon` is how many gameweeks of fixtures are averaged when the
opening squad is chosen. Short = chase the first fixture; long = pick players
who are good across a longer stretch. That is a real strategic parameter and
nothing on this project has ever swept it.

PATH-FREE, like section B: build the 15 at gameweek 1, hold it all season,
score XI + captain weekly. No transfers, so nothing forks.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, measure, simulate                     # noqa: E402
from run_bakeoff_score import cluster, load, HOLDOUTS                # noqa: E402

SCRATCH = Path(os.environ.get("BAKE_DIR", "/tmp/bakeoff"))
SIGNAL = os.environ.get("OPT_SIGNAL", "m_full_us")
BASE = 5
DOSES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 19]


def main():
    t0 = time.time()
    _, grids, _ = load()
    print(f"[{time.time()-t0:4.0f}s] signal={SIGNAL}, doses={DOSES}", flush=True)
    rows = []
    for s in HOLDOUTS:
        if s not in grids:
            continue
        g = grids[s]
        for h in DOSES:
            sq = simulate._opening_squad(g, SIGNAL, h, C.BUDGET, 1.0)
            if sq is None or len(sq) != C.SQUAD_SIZE:
                print(f"   !! {s} h={h}: no legal squad", flush=True); continue
            d = measure.fixed_squad_score(g, [sq[C.PLAYER_KEY].tolist()],
                                          SIGNAL, actual="points")
            d["season"] = s; d["h"] = h
            rows.append(d)
        pd.concat(rows, ignore_index=True).to_pickle(SCRATCH / "opt_horizon.pkl")
        print(f"   [{time.time()-t0:5.0f}s] {s}", flush=True)

    M = pd.concat(rows, ignore_index=True)
    M["value"] *= 38.0
    ref = M[M["h"].eq(BASE)].set_index(["season", "gameweek"])["value"]
    print("\n" + "=" * 78)
    print(f"OPENING HORIZON dose-response vs h={BASE}, path-free, signal {SIGNAL}")
    print("season-clustered over G=8, bar |t| > 2.36")
    print("=" * 78)
    print(f"{'h':>4}{'mean':>10}{'t':>8}{'p':>8}{'95% CI':>22}{'MDE':>8}  seasons")
    out = []
    for h in DOSES:
        if h == BASE:
            print(f"{h:>4}{'0.0 (reference)':>10}"); continue
        cur = M[M["h"].eq(h)].set_index(["season", "gameweek"])["value"]
        j = pd.concat({"a": cur, "r": ref}, axis=1).dropna()
        d = (j["a"] - j["r"]).to_numpy(float)
        gg = j.index.get_level_values("season").to_numpy()
        m, se, tt, pp, lo, hi = cluster(d, gg)
        w = sum(1 for _, dd in pd.Series(d).groupby(gg) if dd.mean() > 0)
        out.append({"h": h, "mean": m, "t": tt, "p": pp, "mde": 2.8 * se, "wins": w})
        print(f"{h:>4}{m:>+10.1f}{tt:>8.2f}{pp:>8.3f}"
              f"{f'[{lo:+.1f}, {hi:+.1f}]':>22}{2.8*se:>8.1f}  {w}/8", flush=True)
    pd.DataFrame(out).to_pickle(SCRATCH / "opt_horizon_res.pkl")
    print("\nabsolute season total per horizon:")
    t = M.groupby(["h", "season"])["value"].mean().unstack()
    t["MEAN"] = t.mean(axis=1)
    print(t.round(0).to_string())
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
