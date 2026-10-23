#!/usr/bin/env python3
"""
O51 / S52 phase C -- re-score the surviving arms WITH transfers.

Metric 1 fixes the squad and measures only the XI and the armband. S49 found
the two instruments disagree: the rate layer contributes nothing to picking an
XI from a given squad and does contribute to deciding WHO TO OWN. Since M26 is
entirely a story about the rate layer being absent, metric 2 is where a repair
would show up, and running only metric 1 would repeat S48's stated limitation.

K restarts per season from the SAME neutral opening squads, so the transfer
path forks from a common start and the fork averages down instead of
dominating (S38: the old instrument's within-arm restart sd was 102 points).
Season-clustered over the same 8 holdouts.

Arms are passed on the command line -- this is deliberately run AFTER phase B,
on the few arms worth the compute, not on all sixteen.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy import stats                                               # noqa: E402
from fpl2 import config as C, measure, simulate                       # noqa: E402
from run_bakeoff_score import cluster, load, REF, HOLDOUTS            # noqa: E402

SCRATCH = Path(os.environ.get("BAKE_DIR", "/tmp/bakeoff"))
K = int(os.environ.get("K2", "12"))
# Season-suffixed outputs, so the runner can be fanned out one process per
# holdout without eight processes overwriting each other's pickle. Metric 2 is
# ~7s per season-simulation and does not parallelise internally.
TAG = ("_" + HOLDOUTS[0]) if len(HOLDOUTS) == 1 else ""


def main(arms):
    t0 = time.time()
    _, grids, available = load()
    arms = [a for a in arms if a in available]
    if REF not in arms:
        arms = [REF] + arms
    print(f"[{time.time()-t0:4.0f}s] metric 2, K={K}, arms: {', '.join(arms)}\n",
          flush=True)

    rows = []
    for s in HOLDOUTS:
        if s not in grids:
            continue
        g = grids[s]
        squads = measure.neutral_squads(g, k=K, seed=1)
        for a in arms:
            r = measure.restart_score(g, squads, a, simulate.simulate_season)
            r["season"] = s; r["arm"] = a
            rows.append(r)
            print(f"   [{time.time()-t0:5.0f}s] {s} {a:<16} "
                  f"mean {r['value'].mean():7.1f}", flush=True)
        # persist after every season -- 40 minutes of compute has been lost
        # twice on this project to a reporting bug at the very end
        pd.concat(rows, ignore_index=True).to_pickle(SCRATCH / f"metric2{TAG}.pkl")

    M = pd.concat(rows, ignore_index=True)
    ref = M[M["arm"].eq(REF)].set_index(["season", "squad"])["value"]
    print("\n" + "=" * 78)
    print(f"METRIC 2 (with transfers) vs {REF} -- season-clustered, bar |t| > 2.36")
    print("=" * 78)
    print(f"{'arm':<16}{'mean':>9}{'t':>8}{'p':>8}{'95% CI':>20}  seasons")
    for a in arms:
        if a == REF:
            continue
        cur = M[M["arm"].eq(a)].set_index(["season", "squad"])["value"]
        j = pd.concat({"a": cur, "r": ref}, axis=1).dropna()
        d = (j["a"] - j["r"]).to_numpy(float)
        gg = j.index.get_level_values("season").to_numpy()
        mean, se, tt, p, lo, hi = cluster(d, gg)
        wins = sum(1 for _, dd in pd.Series(d).groupby(gg) if dd.mean() > 0)
        print(f"{a:<16}{mean:>+9.1f}{tt:>8.2f}{p:>8.3f}"
              f"{f'[{lo:+.1f}, {hi:+.1f}]':>20}  {wins}/8", flush=True)

    print("\n" + "=" * 78)
    print("METRIC 2 -- absolute season total per arm")
    print("=" * 78)
    tot = M.groupby(["arm", "season"])["value"].mean().unstack()
    tot["MEAN"] = tot.mean(axis=1)
    print(tot.round(0).sort_values("MEAN", ascending=False).to_string(), flush=True)
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
