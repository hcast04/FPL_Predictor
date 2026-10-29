#!/usr/bin/env python3
"""
S47: do Understat features beat FPL's, measured honestly?

Arms differ ONLY in the feature source. Both see the same ten seasons, the same
fixed squads, the same scoring.

  fpl        expected_goals (Opta) -- absent before 2022-23, so six of ten
             seasons contribute nothing to the xG component
  understat  us_xg across all ten, plus shots, key passes, xg_chain and
             xg_buildup, which have NO FPL equivalent

Instrument: metric 1 (path-free, MDE 13.6), paired on (squad, gameweek,
season), and reported with SEASON-CLUSTERED standard errors (R32). With more
holdouts the cluster bar falls well below the |t| > 4.30 that made almost
everything unprovable today.
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy import stats
from fpl2 import config as C, data, features, measure, model, rates, simulate, team, understat as un  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
K = 10
PANEL = C.cache("fpl2_panel_us.pkl")
# holdouts: every season with at least one season before it. 2016-17 trains
# nothing, so it is the first training season only.
HOLDOUTS = ["2018-19", "2019-20", "2020-21", "2021-22",
            "2022-23", "2023-24", "2024-25", "2025-26"]


class _WP:
    def __init__(self, est):
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()); self.est = est
    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight); return self
    def predict(self, X): return self.est.predict(self.pre.transform(X))


def cluster(d, g):
    d = np.asarray(d, float); g = np.asarray(g)
    n = len(d); db = d.mean(); G = len(np.unique(g))
    ss = sum((d[g == k] - db).sum() ** 2 for k in np.unique(g))
    se = np.sqrt((G / (G - 1)) * ss / n ** 2)
    t = db / se if se > 0 else np.nan
    crit = stats.t.ppf(0.975, G - 1)
    return db, se, t, 2 * (1 - stats.t.cdf(abs(t), G - 1)), db - crit * se, db + crit * se, G


def main():
    t0 = time.time()
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    panel = pd.read_pickle(PANEL)
    raw = data.load_raw(extended=True)
    tf = team.team_fixtures(raw)
    squads = {s: measure.neutral_squads(data.with_blanks(panel, s), k=K, seed=1)
              for s in HOLDOUTS}
    print(f"[{time.time()-t0:5.0f}s] {K} shared squads x {len(HOLDOUTS)} holdouts", flush=True)

    scores = {}
    for arm, flag in (("fpl", False), ("understat", True)):
        features.USE_UNDERSTAT_FEATURES = flag
        feats, names = features.build(panel, tf)
        parts = []
        for s in HOLDOUTS:
            try:
                out = model.walk_forward(feats, tf, names, s)
            except Exception as e:
                print(f"    {arm} {s} FAILED: {type(e).__name__}: {e}", flush=True); continue
            g = data.with_blanks(panel, s).merge(
                out[["season", "gameweek", C.PLAYER_KEY, "xp"]],
                on=["season", "gameweek", C.PLAYER_KEY], how="left")
            g["xp"] = g["xp"].fillna(0.0); g.loc[g["fixtures"] <= 0, "xp"] = 0.0
            d = measure.fixed_squad_score(g, squads[s], "xp"); d["season"] = s
            parts.append(d)
            print(f"[{time.time()-t0:5.0f}s] {arm} {s}: {d.value.mean():.3f}/gw", flush=True)
        scores[arm] = pd.concat(parts)
        print(f"[{time.time()-t0:5.0f}s] {arm}: {len(names)} features, "
              f"mean {scores[arm].value.mean():.3f}/gw", flush=True)
    features.USE_UNDERSTAT_FEATURES = False

    ON = ("squad", "gameweek", "season")
    m = scores["understat"].merge(scores["fpl"], on=list(ON), suffixes=("_u", "_f"))
    d = (m.value_u - m.value_f) * 38
    naive = measure.per_season_equivalent(
        measure.paired(scores["understat"], scores["fpl"], on=ON))
    mu, se, t, p, lo, hi, G = cluster(d, m.season)
    print("\n" + "=" * 78)
    print("S47 -- UNDERSTAT FEATURES vs FPL FEATURES (metric 1, season-eq pts)")
    print("=" * 78)
    print(f"  n = {len(m):,} paired observations over G = {G} seasons")
    print(f"  NAIVE    mean {naive['mean']:+7.1f}  t {naive['t']:6.2f}   (ignores clustering)")
    print(f"  CLUSTERED mean {mu:+7.1f}  t {t:6.2f}  p {p:.4f}  CI [{lo:+.1f}, {hi:+.1f}]")
    print(f"  bar at G={G}: |t| > {stats.t.ppf(0.975, G-1):.2f}   "
          f"-> {'SIGNIFICANT' if abs(t) > stats.t.ppf(0.975, G-1) else 'not significant'}")
    print(f"\n  per season (understat - fpl, season-equivalent):")
    for s in HOLDOUTS:
        dd = d[m.season.eq(s)]
        if len(dd): print(f"    {s}: {dd.mean():+7.1f}")
    pickle.dump(scores, open(C.cache("fpl2_us_features.pkl"), "wb"))
    print("\ndone")


if __name__ == "__main__":
    main()
