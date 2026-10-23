#!/usr/bin/env python3
"""
O53 -- the rate CEILING, and whether it is suppressing exactly the players
worth captaining.

M27 fixed a ceiling that was clipping predictions to ZERO. This asks the
opposite question about the same mechanism: when the ceiling works as
designed, who does it bind on?

DIAGNOSTIC (`scratchpad/ceiling_diag.py`), holdout 2024-25:

    xG ceiling binds on 0.38% of played rows -- and those rows are:
       Mohamed Salah    32 weeks   raw 0.904 vs MID cap 0.628   -0.278 goals/90
       Erling Haaland    8 weeks   raw 1.087 vs FWD cap 0.989   -0.090 goals/90

**Salah is clipped in every single gameweek of the season.** The cap is the
99th percentile of sustained player-season rates, so it binds precisely on the
players who ARE the 99th percentile -- which is the same set you want the
armband on. That is S20's mechanism stated in one line.

The ceiling is NOT simply wrong. Holdout 2025-26 shows what it prevents: a
GLM extrapolating to **2478 xG/90** for Rio Ngumoha, a player with almost no
history. Remove it and that reaches the captain pick. So this is a
dose-response on `WINSOR_QUANTILE`, not a deletion:

    0.95  0.99 (shipped)  0.995  0.999  none

A real effect here should be an INVERTED U -- relaxing helps until
extrapolation blowups start reaching the XI -- not monotone to the boundary,
which M28 just established is the signature of a leak (R38).

CHEAP BY CONSTRUCTION: the ceiling is applied in `predict`, never in `fit`, so
each season is fitted ONCE and re-predicted per arm with the caps swapped.
"""
from __future__ import annotations
import os, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, features, model, rates, team       # noqa: E402

SCRATCH = Path(os.environ.get("BAKE_DIR", "/tmp/bakeoff"))
OUT = SCRATCH / "bake"; OUT.mkdir(parents=True, exist_ok=True)
QUANTILES = {"ceil_p95": 0.95, "ceil_p99": 0.99, "ceil_p995": 0.995,
             "ceil_p999": 0.999, "ceil_none": None}
KEEP = ["season", "gameweek", "player_code", "points", "cost",
        "position", "team", "minutes", "fixtures"]


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


def run(holdout: str):
    from sklearn.linear_model import PoissonRegressor
    t0 = time.time()
    features.USE_UNDERSTAT_FEATURES = True
    rates._reg = lambda seed=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    feats = pd.read_pickle(SCRATCH / "feats_us.pkl")
    names = pickle.load(open(SCRATCH / "names_us.pkl", "rb"))
    tf = pd.read_pickle(SCRATCH / "tf.pkl")
    feats = feats[feats["season"] <= holdout]
    for c in feats.columns:
        if feats[c].dtype == "float64":
            feats[c] = feats[c].astype("float32")

    tr_p = feats[feats["season"] < holdout]
    te_p = feats[feats["season"].eq(holdout)]
    tr_t = tf[tf["season"] < holdout]; te_t = tf[tf["season"].eq(holdout)]
    m = model.ComponentModel(feats=list(names)).fit(tr_p, tr_t)
    print(f"[{holdout}] fitted in {time.time()-t0:.0f}s", flush=True)

    # caps at each quantile, from TRAINING data only
    caps = {}
    orig_q = rates.WINSOR_QUANTILE
    for arm, q in QUANTILES.items():
        if q is None:
            caps[arm] = {}
        else:
            rates.WINSOR_QUANTILE = q
            caps[arm] = rates.rate_caps(tr_p)
    rates.WINSOR_QUANTILE = orig_q

    out = None
    for arm in QUANTILES:
        m.rates.caps = caps[arm]
        # STATE LEAK ACROSS ARMS. `refit_defensive` mutates the model and only
        # fires once the holdout has >500 rows of its own history, so the
        # early gameweeks of arm 2..N would otherwise inherit arm N-1's FINAL
        # defensive fit, where `model.walk_forward` gives them an empty one.
        # Caught by the control: max|ceil_p99 - m_full_us| was 0.0016, not 0.
        m.rates.defensive = {}
        parts = []
        for gw in sorted(te_p["gameweek"].unique()):
            hist = te_p[te_p["gameweek"] < gw]
            if len(hist) > 500:
                m.refit_defensive(hist)
            parts.append(m.predict(te_p[te_p["gameweek"].eq(gw)],
                                   te_t[te_t["gameweek"].eq(gw)]))
        o = pd.concat(parts, ignore_index=True)
        if out is None:
            out = o[KEEP].copy()
        else:
            assert o[["gameweek", "player_code"]].reset_index(drop=True).equals(
                out[["gameweek", "player_code"]].reset_index(drop=True)), "row order"
        out[arm] = o["xp"].to_numpy(float)
        print(f"   [{time.time()-t0:5.0f}s] {arm}", flush=True)

    # the shipped arm must reproduce the bake-off's m_full_us exactly
    ref = OUT / f"us_{holdout}.pkl"
    if ref.exists():
        prev = pd.read_pickle(ref)[["gameweek", "player_code", "m_full_us"]]
        j = out[["gameweek", "player_code", "ceil_p99"]].merge(
            prev, on=["gameweek", "player_code"])
        d = float(np.abs(j["ceil_p99"] - j["m_full_us"]).max())
        print(f"   control: max|ceil_p99 - m_full_us| = {d:.10f} "
              f"({'IDENTICAL' if d < 1e-9 else '*** DIFFERS ***'})", flush=True)

    p = OUT / f"ceil_{holdout}.pkl"
    out.to_pickle(p)
    print(f"[{holdout}] WROTE {p.name} {out.shape} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
