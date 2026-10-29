#!/usr/bin/env python3
"""
O17: swap the OTHER components to GLMs, one at a time.

S12 swapped only `rates._reg` and gained ~194 pts/season. The sweep design calls
for one component at a time so a win is attributable. The remaining two:

  minutes  3 classifiers (appear / start / 60+) + 1 regressor (minutes|appear)
  team     the xGC->conceded model

NO PREDICTION IS OFFERED. The design originally said "if the tail-overfit
explanation is right, minutes should matter less (bounded 0-90) and team xGC
more" -- but that explanation was REFUTED by O18 and retracted as W13. Three
explanations have now been retracted today against zero retracted measurements,
so this runs as a measurement with no mechanism attached.

Arms: baseline / rates / minutes / team / all three.
`all` is included to test additivity: if rates alone gives +X and all three give
~+X, the other two contribute nothing and the component story is simple.

GATE, declared before the run: a component swap counts only if it improves mean
vs legacy by more than the ~148 noise floor AND wins 3/3 on both top-20 and
captain-added. Matched folds only -- O20's gate compared a 2-season mean against
a 3-season one and printed a spurious PASS.
"""
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, legacy, model, rates, simulate, team  # noqa: E402
from fpl2 import minutes as minutes_mod   # noqa: E402

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor, LogisticRegression

BASE_RATES_REG = rates._reg
BASE_MIN_CLF, BASE_MIN_REG = minutes_mod._clf, minutes_mod._reg
BASE_TEAM_EST = team.HistGradientBoostingRegressor


def _lin(est):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)


class _WeightedPipe:
    """rates fits with sample_weight=minutes; a bare Pipeline rejects it."""

    def __init__(self, est):
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.est = est

    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.est.predict(self.pre.transform(X))


def set_arm(arm):
    rates._reg = BASE_RATES_REG
    minutes_mod._clf, minutes_mod._reg = BASE_MIN_CLF, BASE_MIN_REG
    team.HistGradientBoostingRegressor = BASE_TEAM_EST
    if arm in ("rates", "all"):
        rates._reg = lambda s=0: _WeightedPipe(PoissonRegressor(alpha=1.0, max_iter=500))
    if arm in ("minutes", "all"):
        minutes_mod._clf = lambda s=0: _lin(LogisticRegression(max_iter=1000, C=1.0))
        minutes_mod._reg = lambda s=0: _lin(PoissonRegressor(alpha=1.0, max_iter=500))
    if arm in ("team", "all"):
        team.HistGradientBoostingRegressor = (
            lambda **kw: _lin(PoissonRegressor(alpha=1.0, max_iter=500)))


def metrics(out):
    t = out[out.fixtures > 0]
    t20, cap = [], []
    for gw, g in t.groupby("gameweek"):
        if len(g) < 50:
            continue
        t20.append(float(g.nlargest(20, "xp").points.sum()))
        cap.append(float(g.loc[g.xp.idxmax(), "points"]))
    return float(np.mean(t20)), float(np.mean(cap))


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
    for arm in ("baseline", "rates", "minutes", "team", "all"):
        set_arm(arm)
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
            print(f"  {arm:10s} FAILED: {type(exc).__name__}: {exc}", flush=True)
            set_arm("baseline"); continue
        res[arm] = (float(np.mean(t20s)), float(np.mean(caps)),
                    float(np.mean(margins)), margins)
        print(f"  {arm:10s} top20 {res[arm][0]:6.2f}  cap {res[arm][1]:5.2f}  "
              f"vs_leg {res[arm][2]:+7.0f}   {[round(m) for m in margins]}", flush=True)
    set_arm("baseline")

    b = res.get("baseline")
    if b:
        print(f"\nGATE  (vs baseline; >148 on mean AND 3/3 on both metrics)", flush=True)
        for arm in ("rates", "minutes", "team", "all"):
            if arm not in res:
                continue
            r = res[arm]
            tw = sum(1 for i in range(3) if r[3][i] > b[3][i])
            gain = r[2] - b[2]
            print(f"  {arm:10s} gain {gain:+7.0f}   seasons_better {tw}/3   "
                  f"top20 {r[0]:6.2f} vs {b[0]:6.2f}   cap {r[1]:5.2f} vs {b[1]:5.2f}   "
                  f"{'PASS' if gain > 148 and tw == 3 else 'FAIL'}", flush=True)
    pickle.dump(res, open(C.cache("fpl2_o17.pkl"), "wb"))
    print(f"\n[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
