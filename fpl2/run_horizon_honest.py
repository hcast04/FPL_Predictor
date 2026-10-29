#!/usr/bin/env python3
"""
O56 -- the opening-horizon sweep, done without the leak.

M28: `simulate._opening_squad` averages the signal over gameweeks 1..h, but
the walk-forward xp for gameweek 15 was computed from form data through
gameweek 14. At the gameweek-1 deadline that forecast cannot exist. The sweep
came back perfectly monotone to +312.8 at h=19 (t 4.52, 8/8 seasons) and that
shape is a leak signature, not a strategy (R38). Ledger F10 had already
recorded it: "Leaky version said +196/season. Leakage-free: H=2-10 all worse."

THE HONEST VERSION. `project.project(model, feats, tf, season, from_gw=1,
to_gw=k)` forecasts week k using only week-1 information: player form features
FROZEN at week 1 (strictly lagged, so legitimately known then), opponent and
venue from the PUBLISHED fixture list, and opponent STRENGTH rewound to week 1
by `_strength_as_of` -- the one place the leak would hide.

Both arms are built so the difference is ONLY the information:

  leaky_<h>   mean of the walk-forward xp over GW1..h   (reproduces M28)
  honest_<h>  mean of project(from_gw=1, to_gw=k) for k in 1..h

Reference: honest_5, the shipped `opening_horizon`.

CAVEAT, stated because it bounds the claim. The target fixture frame comes
from the realised season, so the projection knows about any DOUBLE or BLANK
in GW1..h. In June a manager knows the opponent list but not those. For h <= 4
`project.py` argues the published list is safe; beyond that the honest arm is
still mildly optimistic, in the direction of FAVOURING long horizons. If the
honest arm shows no gain, that caveat cannot have hidden one.
"""
from __future__ import annotations
import os, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import (config as C, data, features, measure, model,     # noqa: E402
                  optimise, project, rates)
from run_bakeoff_score import cluster                              # noqa: E402

SCRATCH = Path(os.environ.get("BAKE_DIR", "/tmp/bakeoff"))
HOLDOUTS = ["2018-19", "2019-20", "2020-21", "2021-22",
            "2022-23", "2023-24", "2024-25", "2025-26"]
DOSES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 19]
BASE = 5


class _WP:
    def __init__(self, est):
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.est = est
    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight); return self
    def predict(self, X): return self.est.predict(self.pre.transform(X))


def squad_from(strength: pd.DataFrame, grid: pd.DataFrame):
    """Best legal 15 on a per-player mean signal, under the real constraints."""
    pool = strength.merge(
        grid.groupby(C.PLAYER_KEY, as_index=False).agg(
            cost=("cost", "first"), position=("position", "first"),
            team=("team", "first")), on=C.PLAYER_KEY, how="inner")
    pool = pool[pool["cost"].gt(0)]
    sq = optimise.optimise_squad(pool, "sig", budget=C.BUDGET)
    return None if sq is None or len(sq) != C.SQUAD_SIZE else sq[C.PLAYER_KEY].tolist()


def run(holdout: str):
    from sklearn.linear_model import PoissonRegressor
    t0 = time.time()
    features.USE_UNDERSTAT_FEATURES = True
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    feats = pd.read_pickle(SCRATCH / "feats_us.pkl")
    names = pickle.load(open(SCRATCH / "names_us.pkl", "rb"))
    tf = pd.read_pickle(SCRATCH / "tf.pkl")
    panel = pd.read_pickle(C.cache("fpl2_panel_us.pkl"))
    feats = feats[feats["season"] <= holdout]
    tr_p = feats[feats["season"] < holdout]; tr_t = tf[tf["season"] < holdout]
    m = model.ComponentModel(feats=list(names)).fit(tr_p, tr_t)
    o = model.walk_forward(feats, tf, names, holdout)
    print(f"[{holdout}] fitted in {time.time()-t0:.0f}s", flush=True)

    grid = data.with_blanks(panel, holdout).merge(
        o[["season", "gameweek", C.PLAYER_KEY, "xp"]],
        on=["season", "gameweek", C.PLAYER_KEY], how="left")
    grid["xp"] = grid["xp"].fillna(0.0)
    grid.loc[grid["fixtures"] <= 0, "xp"] = 0.0

    gws = sorted(int(g) for g in o["gameweek"].unique())
    first = gws[0]
    # honest: every target week forecast from week `first` ONLY
    proj = {}
    for k in gws[:max(DOSES)]:
        fr = project.project(m, feats, tf, holdout, first, int(k))
        if not fr.empty:
            proj[int(k)] = fr.groupby(C.PLAYER_KEY)["xp"].mean()
    print(f"[{holdout}] projected {len(proj)} weeks from GW{first}", flush=True)

    rows = []
    for h in DOSES:
        wk = [g for g in gws[:h]]
        leaky = (grid[grid["gameweek"].isin(wk) & (grid["fixtures"] > 0)]
                 .groupby(C.PLAYER_KEY, as_index=False)["xp"].mean()
                 .rename(columns={"xp": "sig"}))
        hon = pd.concat([proj[k] for k in wk if k in proj], axis=1)
        hon = hon.mean(axis=1).rename("sig").reset_index()
        for arm, strength in (("leaky", leaky), ("honest", hon)):
            sq = squad_from(strength, grid)
            if sq is None:
                print(f"   !! {holdout} {arm}_{h}: no legal squad", flush=True); continue
            sc = measure.fixed_squad_score(grid, [sq], "xp", actual="points")
            sc["season"] = holdout; sc["arm"] = f"{arm}_{h}"
            rows.append(sc)
    out = pd.concat(rows, ignore_index=True)
    out.to_pickle(SCRATCH / f"horizon_honest_{holdout}.pkl")
    print(f"[{holdout}] WROTE {len(out)} rows in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
