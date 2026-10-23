#!/usr/bin/env python3
"""
O57 -- discount CAREER features by tenure at the current club.

**The user's hypothesis**, formed from watching his own squad: Elliot Anderson
is recommended on the strength of 180 points at Nottingham Forest and is now
at Manchester City; Enzo Fernandez carries 9,812 minutes of Chelsea history
into the same move. Six of his fifteen players changed club.

MEASURED FIRST (8 holdouts, players with 1500+ minutes last season):

| last season's points | moved | stayed | difference |
|---|---|---|---|
| < 100 | -0.010 | -0.009 | -0.007 (p 0.80) |
| 100-150 | +0.140 | +0.174 | -0.016 (p 0.82) |
| **150+** | **+0.049** | **+0.380** | **-0.347**, t -1.91, 6/7 seasons |

High-output players who STAY beat their forecast by +0.380/gameweek; the same
calibre who MOVE beat it by +0.049. That gap is ~13 points per player per
season, and the dose-response across the three bands is monotone -- which is
the shape the mechanism predicts.

THE MECHANISM. `features.py` has three club-aware columns -- `changed_club`,
`history_at_current_club_r5`/`_r10`, `fixtures_at_current_club` -- but after
about five games at a new club `history_at_current_club_r5` reads 1.0 and
stops flagging anything, while the 21 `*_career` columns carry the old club's
output undiscounted for ever.

THE TREATMENT. Shrink every career feature toward its POSITION MEAN, with the
weight recovering as tenure accumulates:

    w = n / (n + K),  n = fixtures_at_current_club
    career_adj = w * career + (1 - w) * position_mean(career)

Position means come from TRAINING rows only. K = 0 is the shipped model and
must reproduce it bit-identically.

PASSES THE R43 FILTER: the shrink keys off tenure, which is orthogonal to xp,
so it reorders rather than rescaling -- unlike S54's ceiling and S56's
conditional bonus, which both moved the ranking by ~2 places and measured null.

PRE-REGISTERED GATE: metric 1, 8 holdouts, K=10 squads, season-clustered,
bar |t| > 2.36. BETTER needs the CI to exclude 0 AND the mean to exceed +13.6.

EXPECTED MAGNITUDE, stated before the result: ~13 pts/season per affected
player, and a squad holds one or two. So the honest ceiling here is perhaps
15-30 pts/season -- worth having, but NOT an explanation for a 79-point
deficit, which S55 showed is ~72% noise at this horizon anyway.
"""
from __future__ import annotations
import os, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, features, model, rates                 # noqa: E402

SCRATCH = Path(os.environ.get("BAKE_DIR", "/tmp/bakeoff"))
OUT = SCRATCH / "bake"; OUT.mkdir(parents=True, exist_ok=True)
KS = {"ten_K0": 0, "ten_K5": 5, "ten_K10": 10, "ten_K20": 20, "ten_K40": 40}
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


def shrink(feats, names, holdout, K):
    """Career features pulled toward the TRAINING position mean by tenure."""
    if K <= 0:
        return feats
    d = feats.copy()
    car = [c for c in names if c.endswith("_career")]
    n = pd.to_numeric(d["fixtures_at_current_club"], errors="coerce").fillna(0.0)
    w = (n / (n + K)).to_numpy(float)
    tr = d[d["season"] < holdout]
    for c in car:
        pm = tr.groupby("position")[c].mean()
        prior = d["position"].map(pm).astype(float).to_numpy()
        v = pd.to_numeric(d[c], errors="coerce").to_numpy(float)
        # a NaN career value stays NaN -- `usable`/the imputer handle it
        d[c] = np.where(np.isnan(v), v, w * v + (1.0 - w) * np.nan_to_num(prior))
    return d


def run(holdout: str):
    from sklearn.linear_model import PoissonRegressor
    t0 = time.time()
    features.USE_UNDERSTAT_FEATURES = True
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    feats = pd.read_pickle(SCRATCH / "feats_us.pkl")
    names = pickle.load(open(SCRATCH / "names_us.pkl", "rb"))
    tf = pd.read_pickle(SCRATCH / "tf.pkl")
    feats = feats[feats["season"] <= holdout]
    for c in feats.columns:
        if feats[c].dtype == "float64":
            feats[c] = feats[c].astype("float32")

    out = None
    for arm, K in KS.items():
        f = shrink(feats, names, holdout, K)
        o = model.walk_forward(f, tf, names, holdout)
        if out is None:
            out = o[KEEP].copy()
        else:
            assert o[["gameweek", "player_code"]].reset_index(drop=True).equals(
                out[["gameweek", "player_code"]].reset_index(drop=True)), "row order"
        out[arm] = o["xp"].to_numpy(float)
        print(f"   [{time.time()-t0:5.0f}s] {arm} (K={K}) mean xp {out[arm].mean():.4f}",
              flush=True)

    ref = OUT / f"us_{holdout}.pkl"
    if ref.exists():
        prev = pd.read_pickle(ref)[["gameweek", "player_code", "m_full_us"]]
        j = out[["gameweek", "player_code", "ten_K0"]].merge(
            prev, on=["gameweek", "player_code"])
        dd = float(np.abs(j["ten_K0"] - j["m_full_us"]).max())
        print(f"   control: max|ten_K0 - m_full_us| = {dd:.10f} "
              f"({'IDENTICAL' if dd < 1e-9 else '*** DIFFERS ***'})", flush=True)
    out.to_pickle(OUT / f"ten_{holdout}.pkl")
    print(f"[{holdout}] WROTE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
