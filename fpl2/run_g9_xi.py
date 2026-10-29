#!/usr/bin/env python3
"""
G9 decision metric: do manual Championship priors win POINTS?

Ratio and spearman are diagnostics. The ledger's decision metric is
best-legal-XI points, so this is the test that actually decides.

Gate, declared before the run:
  * best-XI points must improve in >=3 of the 4 seasons, AND
  * must not lose more than 5 pts/season in any season.
A win in 1 or 2 of 4 is the F4/F8 pattern and is rejected.

Selection is by `xp`, scoring by realised `points` -- never the same column,
so a forecast is not scored against itself.
"""
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, external, features, model, team, optimise  # noqa: E402

SEASONS = list(C.HOLDOUT_SEASONS) + ["2026-27"]


def xi_points(out):
    """Sum of best-legal-XI realised points over every gameweek of a season."""
    tot, n = 0.0, 0
    for gw, rows in out[out.fixtures > 0].groupby("gameweek"):
        v = optimise.xi_value(rows, score="xp", actual="points")
        if np.isfinite(v):
            tot += v; n += 1
    return tot, n


def main():
    t0 = time.time()
    raw = pd.concat([data.load_raw(), pd.read_pickle("/tmp/fpl2_live2627.pkl")],
                    ignore_index=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    base_ext = pd.read_pickle("/tmp/fpl2_ext_understat.pkl")
    feats, names = features.build(panel, tf)
    print(f"[{time.time()-t0:4.0f}s] setup done\n", flush=True)

    built = {}
    for arm in ("understat_only", "understat+manual"):
        ext = base_ext.copy()
        if arm == "understat+manual":
            parts = [external.with_manual(ext[ext.season.eq(s)].copy(), panel, s)
                     for s in SEASONS if not ext[ext.season.eq(s)].empty]
            ext = pd.concat(parts, ignore_index=True)
        cols = [c for c in ext.columns if c.startswith("ext_")
                and c not in ("ext_match_method", "ext_source", "ext_name",
                              "ext_league", "ext_squads")]
        f = feats.merge(ext[[C.PLAYER_KEY, "season"] + cols],
                        on=[C.PLAYER_KEY, "season"], how="left")
        f["ext_available"] = f["ext_available"].fillna(0.0)
        built[arm] = (f, names + cols)

    print(f"{'season':10s}{'gws':>5}{'before':>10}{'after':>10}{'delta':>9}"
          f"{'per_gw':>9}", flush=True)
    deltas = {}
    for s in SEASONS:
        vals = {}
        for arm, (f, n) in built.items():
            out = model.walk_forward(f, tf, n, s)
            vals[arm] = xi_points(out)
        (b, nb), (a, na) = vals["understat_only"], vals["understat+manual"]
        d = a - b
        deltas[s] = d
        print(f"{s:10s}{nb:>5}{b:>10.1f}{a:>10.1f}{d:>+9.1f}{d/max(nb,1):>+9.2f}",
              flush=True)

    wins = sum(1 for d in deltas.values() if d > 0)
    worst = min(deltas.values())
    print(f"\nGATE: improved in {wins}/4 seasons (need >=3); "
          f"worst season {worst:+.1f} pts (need > -5.0)", flush=True)
    print("  VERDICT:", "PASS" if wins >= 3 and worst > -5.0 else "FAIL", flush=True)
    pickle.dump(deltas, open("/tmp/fpl2_g9_xi.pkl", "wb"))
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
