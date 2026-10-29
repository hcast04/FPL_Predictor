#!/usr/bin/env python3
"""
Model-family sweep, run exactly as designed in the ledger on 2026-10-20.

PRIOR, from that design and unchanged: "today's evidence says the TARGET
matters far more than the estimator... Expect model family to be second-order.
If a family appears to win big, suspect the harness first."

S10 strengthens it further: a no-training four-tier lookup already ranks with
the 120-feature GBM, so the estimator is unlikely to be the binding constraint.

ONE component is swapped -- the RATES regressor (`rates._reg`) -- so any win is
attributable. Minutes, team and the scoring composition are untouched.

GATE, fixed in advance (5 families x 3 components = 15 comparisons, ~1.9
spurious 3/3 winners expected by chance at the measured 0.125 rate):
  * must win on BOTH top-20 realised points AND captain-added,
  * in ALL THREE holdout seasons,
  * and exceed the ~148-point noise floor on the G1 season total.
  A win on one metric only is recorded as NOT a result.

ridge and glm_poisson are THE CONTROL and run first: if a linear model on these
features matches the GBDT, the GBDT is not earning its complexity, and that
changes what the rest of the sweep means.
"""
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, legacy, model, rates, simulate, team  # noqa: E402

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

BASE = rates._reg


class _WeightedPipe:
    """
    Impute + scale + linear model, forwarding sample_weight to the final step.

    `rates.py` fits with `sample_weight=minutes`, and a bare sklearn Pipeline
    rejects that without a `stepname__` prefix -- which silently killed the
    ridge/glm control on the first run. Linear models cannot take NaN, so the
    imputer is required; the GBDT baseline routes NaN natively and sees the
    raw columns. That asymmetry is unavoidable and is noted rather than hidden:
    the control answers "does a linear model on these features keep up", not
    "is the preprocessing identical".
    """

    def __init__(self, est):
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.est = est

    def fit(self, X, y, sample_weight=None):
        Xt = self.pre.fit_transform(X)
        self.est.fit(Xt, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.est.predict(self.pre.transform(X))


def f_ridge(seed=0):
    return _WeightedPipe(Ridge(alpha=10.0))


def f_glm(seed=0):
    return _WeightedPipe(PoissonRegressor(alpha=1.0, max_iter=500))


def f_xgb(seed=0):
    from xgboost import XGBRegressor
    return XGBRegressor(objective="count:poisson", n_estimators=300,
                        learning_rate=0.06, max_depth=6, min_child_weight=40,
                        reg_lambda=1.0, random_state=seed, n_jobs=4,
                        verbosity=0)


def f_quantile(seed=0):
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=0.6, max_iter=300, learning_rate=0.06,
        max_depth=6, min_samples_leaf=40, l2_regularization=1.0,
        random_state=seed)


FAMILIES = [("baseline_histgb", BASE), ("ridge", f_ridge), ("glm_poisson", f_glm)]


def metrics(out):
    """top-20 realised points and captain-added, per gameweek means."""
    t = out[out.fixtures > 0]
    top20, cap = [], []
    for gw, g in t.groupby("gameweek"):
        if len(g) < 50:
            continue
        s = g.nlargest(20, "xp")
        top20.append(float(s.points.sum()))
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
    print(f"[{time.time()-t0:4.0f}s] setup; legacy totals "
          f"{ {s: round(v[1]) for s, v in leg.items()} }\n", flush=True)

    res = {}
    for fam, factory in FAMILIES:
        rates._reg = factory
        per = {}
        try:
            for s in C.HOLDOUT_SEASONS:
                out = model.walk_forward(feats, tf, names, s)
                t20, cap = metrics(out)
                g = leg[s][0].merge(out[["season", "gameweek", C.PLAYER_KEY, "xp"]],
                                    on=["season", "gameweek", C.PLAYER_KEY], how="left")
                g["xp"] = g["xp"].fillna(0.0)
                g.loc[g["fixtures"] <= 0, "xp"] = 0.0
                tot = float(simulate.simulate_season(g, "xp").total)
                per[s] = (t20, cap, tot, tot - leg[s][1])
                print(f"  {fam:16s} {s}  top20 {t20:6.2f}  cap {cap:5.2f}  "
                      f"total {tot:6.0f}  vs_leg {tot-leg[s][1]:+6.0f}", flush=True)
        except Exception as exc:
            print(f"  {fam:16s} FAILED: {exc}", flush=True)
            rates._reg = BASE
            continue
        res[fam] = per
        print(flush=True)
    rates._reg = BASE

    b = res.get("baseline_histgb")
    print("GATE  (vs baseline_histgb; needs BOTH metrics, all 3 seasons, "
          "and >148 on G1)", flush=True)
    print(f"{'family':18s}{'top20 wins':>12}{'cap wins':>10}{'mean_vs_leg':>13}"
          f"{'verdict':>10}", flush=True)
    for fam, per in res.items():
        if fam == "baseline_histgb" or b is None:
            continue
        tw = sum(1 for s in C.HOLDOUT_SEASONS if per[s][0] > b[s][0])
        cw = sum(1 for s in C.HOLDOUT_SEASONS if per[s][1] > b[s][1])
        mv = float(np.mean([per[s][3] for s in C.HOLDOUT_SEASONS]))
        ok = tw == 3 and cw == 3 and mv > 148
        print(f"{fam:18s}{tw:>9}/3{cw:>7}/3{mv:>+13.0f}"
              f"{'PASS' if ok else 'FAIL':>10}", flush=True)
    pickle.dump(res, open(C.cache("fpl2_family_sweep.pkl"), "wb"))
    print(f"\n[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
