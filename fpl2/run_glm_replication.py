#!/usr/bin/env python3
"""
Replication of the Poisson-GLM result, with the controls it demands.

FIRST RUN: swapping rates._reg from HistGradientBoosting(poisson) to a Poisson
GLM won top-20 3/3, captain-added 3/3, and moved mean-vs-legacy from -267 to
-57 (+210 over the baseline, beating legacy outright in 2024-25). Four families
were tried, so this could be one of the ~1.9 spurious winners the design
budgeted for.

The GLM is DETERMINISTIC -- PoissonRegressor has no random_state -- so it
cannot be reseeded. The baseline can. So the test is: does the GLM's single
point lie outside the baseline's seed distribution?

GATE, declared before this run:
  1. the GLM's mean-vs-legacy must beat the BEST of 4 baseline seeds, not just
     the mean -- otherwise it is inside seed noise;
  2. top-20 and captain-added must still win 3/3 against the baseline seed MEAN;
  3. `baseline_imputed` must NOT close the gap. The GLM gets median-imputed,
     standardised inputs while the GBDT routes NaN natively, and that asymmetry
     is a live alternative explanation. If a GBDT on identical preprocessing
     also jumps, the finding is PREPROCESSING, not model class.
  4. the alpha sweep must not be ragged -- a real effect should vary smoothly
     with regularisation, not peak at one arbitrary value.

Failing 3 would be the most informative outcome available and is not a
disappointment.
"""
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, legacy, model, rates, simulate, team  # noqa: E402

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

BASE = rates._reg


class _WeightedPipe:
    """Impute + scale + estimator, forwarding sample_weight to the final step."""

    def __init__(self, est):
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.est = est

    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.est.predict(self.pre.transform(X))


def gbdt(seed):
    return lambda s=0: HistGradientBoostingRegressor(
        loss="poisson", max_iter=300, learning_rate=0.06, max_depth=6,
        min_samples_leaf=40, l2_regularization=1.0, random_state=seed)


def gbdt_imputed(seed):
    return lambda s=0: _WeightedPipe(HistGradientBoostingRegressor(
        loss="poisson", max_iter=300, learning_rate=0.06, max_depth=6,
        min_samples_leaf=40, l2_regularization=1.0, random_state=seed))


def glm(alpha):
    return lambda s=0: _WeightedPipe(PoissonRegressor(alpha=alpha, max_iter=500))


ARMS = ([(f"baseline_seed{k}", gbdt(k)) for k in (0, 1, 2, 3)]
        + [(f"glm_alpha{a}", glm(a)) for a in (0.1, 1.0, 10.0)]
        + [(f"gbdt_imputed_seed{k}", gbdt_imputed(k)) for k in (0, 1)])


def metrics(out):
    t = out[out.fixtures > 0]
    top20, cap = [], []
    for gw, g in t.groupby("gameweek"):
        if len(g) < 50:
            continue
        top20.append(float(g.nlargest(20, "xp").points.sum()))
        cap.append(float(g.loc[g.xp.idxmax(), "points"]))
    return float(np.mean(top20)), float(np.mean(cap))


def main():
    t0 = time.time()
    raw = data.load_raw(); panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)
    leg = {}
    for s in C.HOLDOUT_SEASONS:
        g = legacy.attach(data.with_blanks(panel, s)).rename(columns={"xp": "xp_legacy"})
        g.loc[g["fixtures"] <= 0, "xp_legacy"] = 0.0
        leg[s] = (g, float(simulate.simulate_season(g, "xp_legacy").total))
    print(f"[{time.time()-t0:4.0f}s] setup\n", flush=True)

    res = {}
    for arm, factory in ARMS:
        rates._reg = factory
        t20s, caps, margins = [], [], []
        try:
            for s in C.HOLDOUT_SEASONS:
                out = model.walk_forward(feats, tf, names, s)
                a, b = metrics(out)
                g = leg[s][0].merge(out[["season", "gameweek", C.PLAYER_KEY, "xp"]],
                                    on=["season", "gameweek", C.PLAYER_KEY], how="left")
                g["xp"] = g["xp"].fillna(0.0)
                g.loc[g["fixtures"] <= 0, "xp"] = 0.0
                tot = float(simulate.simulate_season(g, "xp").total)
                t20s.append(a); caps.append(b); margins.append(tot - leg[s][1])
        except Exception as exc:
            print(f"  {arm:22s} FAILED: {exc}", flush=True); rates._reg = BASE; continue
        res[arm] = (float(np.mean(t20s)), float(np.mean(caps)),
                    float(np.mean(margins)), margins)
        print(f"  {arm:22s} top20 {res[arm][0]:6.2f}  cap {res[arm][1]:5.2f}  "
              f"vs_leg {res[arm][2]:+7.0f}   per-season "
              f"{[round(m) for m in margins]}", flush=True)
    rates._reg = BASE

    bl = [v for k, v in res.items() if k.startswith("baseline_seed")]
    gl = [v for k, v in res.items() if k.startswith("glm_alpha")]
    im = [v for k, v in res.items() if k.startswith("gbdt_imputed")]
    print("\nGATE", flush=True)
    if bl and gl:
        best_bl = max(v[2] for v in bl)
        bl_mean_t20 = float(np.mean([v[0] for v in bl]))
        bl_mean_cap = float(np.mean([v[1] for v in bl]))
        g1 = gl[[v[2] for v in gl].index(max(v[2] for v in gl))]
        print(f"  baseline seeds vs_leg : {[round(v[2]) for v in bl]}  "
              f"best {best_bl:+.0f}", flush=True)
        print(f"  glm alphas     vs_leg : {[round(v[2]) for v in gl]}", flush=True)
        print(f"  1. glm beats BEST baseline seed : "
              f"{'YES' if g1[2] > best_bl else 'NO'}  "
              f"({g1[2]:+.0f} vs {best_bl:+.0f})", flush=True)
        print(f"  2. top20 {g1[0]:.2f} vs baseline mean {bl_mean_t20:.2f}  "
              f"| cap {g1[1]:.2f} vs {bl_mean_cap:.2f}", flush=True)
    if im:
        print(f"  3. gbdt_imputed vs_leg : {[round(v[2]) for v in im]}  "
              f"-- if these match the GLM, the finding is PREPROCESSING", flush=True)
    if gl:
        print(f"  4. alpha dose-response : "
              f"{[(a, round(v[2])) for a, v in zip((0.1, 1.0, 10.0), gl)]}", flush=True)
    pickle.dump(res, open(C.cache("fpl2_glm_replication.pkl"), "wb"))
    print(f"\n[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
