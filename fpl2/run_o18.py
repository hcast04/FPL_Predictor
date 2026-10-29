#!/usr/bin/env python3
"""
O18: does the GLM actually fix the top-end over-prediction?

S12 is recorded as SOLID with an EXPLANATION attached: a GBDT fit on ~3 seasons
overfits the top of the distribution, and the top is where every expensive
decision in this game is made. That explanation is currently inferred, not
measured.

It is directly testable with no simulation at all. Legacy under-predicts its
top picks (ratio 0.81-0.97); the GBDT over-predicts (0.94-1.14). If the GLM's
top-end moves toward legacy, the mechanism is confirmed. If it does not, S12 is
real but its stated cause is wrong -- which must be recorded before anything is
built on it.

Reported at xp >= 4/5/6: calibration ratio, mean signed bias, and n. Plus the
same for legacy, as the reference the whole diagnosis is anchored on.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, legacy, model, rates, team  # noqa: E402

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor

BASE = rates._reg


class _WeightedPipe:
    def __init__(self, est):
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.est = est

    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.est.predict(self.pre.transform(X))


def main():
    t0 = time.time()
    raw = data.load_raw(); panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)
    leg = legacy.load_fold_forecasts().rename(columns={"xp": "xp_leg"})
    print(f"[{time.time()-t0:4.0f}s] setup\n", flush=True)

    arms = {}
    for arm, factory in (("baseline_gbdt", BASE),
                         ("glm_poisson", lambda s=0: _WeightedPipe(
                             PoissonRegressor(alpha=1.0, max_iter=500)))):
        rates._reg = factory
        arms[arm] = {s: model.walk_forward(feats, tf, names, s)
                     for s in C.HOLDOUT_SEASONS}
        print(f"[{time.time()-t0:4.0f}s] {arm} done", flush=True)
    rates._reg = BASE

    print("\nTOP-END CALIBRATION  (ratio 1.0 = perfect; >1 = OVER-predicting)", flush=True)
    print(f"{'season':10s}{'cut':>5}{'n_gbdt':>8}{'gbdt':>8}{'glm':>8}{'legacy':>9}"
          f"{'bias_gbdt':>11}{'bias_glm':>10}", flush=True)
    moved = []
    for s in C.HOLDOUT_SEASONS:
        a = arms["baseline_gbdt"][s]; b = arms["glm_poisson"][s]
        a = a[a.fixtures > 0][["season", "gameweek", C.PLAYER_KEY, "points", "xp"]]
        b = b[b.fixtures > 0][["season", "gameweek", C.PLAYER_KEY, "xp"]]
        m = (a.rename(columns={"xp": "xp_g"})
              .merge(b.rename(columns={"xp": "xp_m"}),
                     on=["season", "gameweek", C.PLAYER_KEY])
              .merge(leg, on=["season", "gameweek", C.PLAYER_KEY]))
        for cut in (4.0, 5.0, 6.0):
            g = m[m.xp_g >= cut]; q = m[m.xp_m >= cut]; l = m[m.xp_leg >= cut]
            if len(g) < 30 or len(q) < 30:
                continue
            rg = g.xp_g.sum() / max(g.points.sum(), 1)
            rq = q.xp_m.sum() / max(q.points.sum(), 1)
            rl = l.xp_leg.sum() / max(l.points.sum(), 1) if len(l) >= 30 else np.nan
            moved.append((abs(1 - rq) < abs(1 - rg), rg, rq))
            print(f"{s:10s}{cut:>5.1f}{len(g):>8}{rg:>8.4f}{rq:>8.4f}{rl:>9.4f}"
                  f"{(g.xp_g - g.points).mean():>11.3f}"
                  f"{(q.xp_m - q.points).mean():>10.3f}", flush=True)
        print(flush=True)

    better = sum(1 for ok, _, _ in moved if ok)
    print(f"VERDICT: GLM is closer to 1.0 in {better}/{len(moved)} top-end cells.", flush=True)
    print(f"  mean |1-ratio|  gbdt {np.mean([abs(1-r) for _, r, _ in moved]):.4f}"
          f"   glm {np.mean([abs(1-r) for _, _, r in moved]):.4f}", flush=True)
    print("\n  If the GLM is NOT closer, S12 is real but its stated cause is wrong", flush=True)
    print("  and the explanation in the ledger must be corrected.", flush=True)
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
