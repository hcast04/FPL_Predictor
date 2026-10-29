#!/usr/bin/env python3
"""
O19: WHY does the Poisson GLM beat the GBDT by ~194 pts/season?

S12 is replicated and unexplained. W13 retracted the first explanation: the GLM
over-predicts the top MORE than the GBDT, so it is not fixing calibration.
What is measured is better ORDERING at the top (top-20 realised 94.54 vs 90.08)
alongside WORSE calibration there -- independent properties.

Two tests:

1 TOP-DECILE SPEARMAN. Overall spearman is near-useless here (R16) because it
  is dominated by the hundreds of players nobody would ever field. Restricting
  to the top 10% by xp each gameweek asks the question the optimiser actually
  faces: among plausible picks, who is better ordered?

2 QUANTILE-MATCHED SIMULATION -- the decisive one. Within each gameweek, both
  signals are replaced by the SAME template of values (the baseline's sorted xp
  for that gameweek), assigned by rank. Both arms then carry an identical value
  distribution and differ ONLY in which player gets which number.

     advantage survives  -> the win is ORDERING
     advantage vanishes  -> the win was SCALE/SHAPE, and the ordering story
                            in S12's footnotes is wrong too

Declared before the run: I expect ordering to survive, because top-20 realised
points already moved and that metric is scale-free. If it vanishes I will say
so -- W13 is one retracted explanation already.
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

BASE = rates._reg
CACHE = C.cache("fpl2_o19_arms.pkl")


class _WeightedPipe:
    def __init__(self, est):
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.est = est

    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.est.predict(self.pre.transform(X))


def quantile_match(df, col, template_col):
    """Give `col` the value multiset of `template_col`, preserving col's ORDER."""
    out = df[col].to_numpy(float).copy()
    tmpl = np.sort(df[template_col].to_numpy(float))
    order = np.argsort(np.argsort(out))      # rank of each element, 0-based
    return tmpl[order]


def main():
    t0 = time.time()
    raw = data.load_raw(); panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)

    if CACHE.exists():
        arms = pickle.load(open(CACHE, "rb")); print("  (arms from cache)", flush=True)
    else:
        arms = {}
        for arm, factory in (("gbdt", BASE),
                             ("glm", lambda s=0: _WeightedPipe(
                                 PoissonRegressor(alpha=1.0, max_iter=500)))):
            rates._reg = factory
            arms[arm] = {s: model.walk_forward(feats, tf, names, s)
                         for s in C.HOLDOUT_SEASONS}
            print(f"[{time.time()-t0:4.0f}s] {arm} done", flush=True)
        rates._reg = BASE
        pickle.dump(arms, open(CACHE, "wb"))

    print("\n1. TOP-DECILE SPEARMAN  (within gameweek, top 10% by xp)", flush=True)
    print(f"{'season':10s}{'gbdt':>9}{'glm':>9}{'delta':>9}"
          f"{'all_gbdt':>11}{'all_glm':>9}", flush=True)
    for s in C.HOLDOUT_SEASONS:
        row = []
        for arm in ("gbdt", "glm"):
            t = arms[arm][s]; t = t[t.fixtures > 0]
            rs = []
            for gw, g in t.groupby("gameweek"):
                if len(g) < 50:
                    continue
                top = g.nlargest(max(10, len(g) // 10), "xp")
                if top.points.nunique() > 1:
                    rs.append(top.xp.corr(top.points, method="spearman"))
            row.append((float(np.mean(rs)),
                        float(t.xp.corr(t.points, method="spearman"))))
        print(f"{s:10s}{row[0][0]:>9.4f}{row[1][0]:>9.4f}"
              f"{row[1][0]-row[0][0]:>+9.4f}{row[0][1]:>11.4f}{row[1][1]:>9.4f}",
              flush=True)

    print("\n2. QUANTILE-MATCHED SIMULATION  (identical value distribution per GW)",
          flush=True)
    print(f"{'season':10s}{'raw_gbdt':>10}{'raw_glm':>9}{'raw_d':>8}"
          f"{'qm_gbdt':>10}{'qm_glm':>9}{'qm_d':>8}", flush=True)
    raw_d, qm_d = [], []
    for s in C.HOLDOUT_SEASONS:
        grid = data.with_blanks(panel, s)
        g = grid.merge(arms["gbdt"][s][["season", "gameweek", C.PLAYER_KEY, "xp"]]
                       .rename(columns={"xp": "xp_gbdt"}),
                       on=["season", "gameweek", C.PLAYER_KEY], how="left")
        g = g.merge(arms["glm"][s][["season", "gameweek", C.PLAYER_KEY, "xp"]]
                    .rename(columns={"xp": "xp_glm"}),
                    on=["season", "gameweek", C.PLAYER_KEY], how="left")
        for c in ("xp_gbdt", "xp_glm"):
            g[c] = g[c].fillna(0.0)
            g.loc[g["fixtures"] <= 0, c] = 0.0
        # quantile-match BOTH to the gbdt template, per gameweek
        for c in ("xp_gbdt", "xp_glm"):
            g[f"qm_{c}"] = (g.groupby("gameweek", group_keys=False)
                             .apply(lambda d, c=c: pd.Series(
                                 quantile_match(d, c, "xp_gbdt"), index=d.index)))
            g.loc[g["fixtures"] <= 0, f"qm_{c}"] = 0.0
        r = {c: float(simulate.simulate_season(g, c).total)
             for c in ("xp_gbdt", "xp_glm", "qm_xp_gbdt", "qm_xp_glm")}
        raw_d.append(r["xp_glm"] - r["xp_gbdt"])
        qm_d.append(r["qm_xp_glm"] - r["qm_xp_gbdt"])
        print(f"{s:10s}{r['xp_gbdt']:>10.0f}{r['xp_glm']:>9.0f}{raw_d[-1]:>+8.0f}"
              f"{r['qm_xp_gbdt']:>10.0f}{r['qm_xp_glm']:>9.0f}{qm_d[-1]:>+8.0f}",
              flush=True)

    print(f"\n  mean raw advantage          {np.mean(raw_d):+.0f}", flush=True)
    print(f"  mean quantile-matched       {np.mean(qm_d):+.0f}", flush=True)
    share = np.mean(qm_d) / np.mean(raw_d) if abs(np.mean(raw_d)) > 1e-9 else np.nan
    print(f"  share of the win that is ORDERING: {share:.0%}", flush=True)
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
