#!/usr/bin/env python3
"""
O59 -- re-test F4: does training on RECENT seasons only beat training on all?

**Ledger F4**: "Recency: train on the last X gameweeks only. More data wins
MONOTONICALLY. Best X beats 'all' in 1/3 seasons. Rejected."

Worth re-running for the same reason S47 was: F4 ran at G=3 on the instrument
S37 later proved unfit, before ten seasons existed, and before M26/M27 -- when
the rate layer had NO xG or xA target on five of eight folds. "More data wins
monotonically" is exactly what a model whose attacking layer only exists in
recent seasons would produce, because more data meant more OPTA data. That
confound is now gone: Understat covers all ten seasons.

The user's framing is also sharper than F4's. The model is already
player-agnostic -- it has no identity feature, only lagged rolling stats -- so
the question is whether the RELATIONSHIPS it learns are better estimated from
recent football than from eleven years of it.

ARMS: train on the last N seasons before the holdout, N in {1, 2, 3, 5, all}.
An arm collapses to `all` where fewer seasons exist, which is reported.

Today's re-test record on findings of this vintage: S47 reversed completely,
F7, F9 and F10 confirmed. So this is genuinely open.

The dose-response is passed through `verify.dose_response`: monotone to the
boundary would make it a leak suspect (R38), which is how the opening-horizon
sweep was caught.
"""
from __future__ import annotations
import os, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, features, model, rates, verify        # noqa: E402

SCRATCH = Path(os.environ.get("BAKE_DIR", "/tmp/bakeoff"))
OUT = SCRATCH / "bake"; OUT.mkdir(parents=True, exist_ok=True)
WINDOWS = {"win_1": 1, "win_2": 2, "win_3": 3, "win_5": 5, "win_all": 99}
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
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    feats = pd.read_pickle(SCRATCH / "feats_us.pkl")
    names = pickle.load(open(SCRATCH / "names_us.pkl", "rb"))
    tf = pd.read_pickle(SCRATCH / "tf.pkl")
    feats = feats[feats["season"] <= holdout]
    for c in feats.columns:
        if feats[c].dtype == "float64":
            feats[c] = feats[c].astype("float32")

    prior = sorted(s for s in feats["season"].unique() if s < holdout)
    out = None
    for arm, n in WINDOWS.items():
        keep = prior[-n:] if n < 99 else prior
        # FEATURES stay full-history: a rolling career mean is a property of
        # the player, not of the training window. Only the rows the estimator
        # FITS ON are restricted, which is what F4 tested.
        sub = feats[feats["season"].isin(keep) | feats["season"].eq(holdout)]
        o = model.walk_forward(sub, tf, names, holdout)
        if out is None:
            out = o[KEEP].copy()
        else:
            assert o[["gameweek", "player_code"]].reset_index(drop=True).equals(
                out[["gameweek", "player_code"]].reset_index(drop=True)), "row order"
        out[arm] = o["xp"].to_numpy(float)
        print(f"   [{time.time()-t0:5.0f}s] {arm}: trained on {len(keep)} season(s) "
              f"{keep[0]}..{keep[-1]}  mean xp {out[arm].mean():.4f}", flush=True)

    ref = OUT / f"us_{holdout}.pkl"
    if ref.exists():
        prev = pd.read_pickle(ref)[["gameweek", "player_code", "m_full_us"]]
        j = out[["gameweek", "player_code", "win_all"]].merge(
            prev, on=["gameweek", "player_code"])
        try:
            d = verify.reference_identical(j["win_all"], j["m_full_us"], "win_all")
            print(f"   control: win_all == m_full_us  (max diff {d:.10f})", flush=True)
        except verify.VerificationError as e:
            print(f"   !! CONTROL FAILED: {e}", flush=True)
    out.to_pickle(OUT / f"win_{holdout}.pkl")
    print(f"[{holdout}] WROTE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
