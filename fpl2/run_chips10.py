#!/usr/bin/env python3
"""
S50: re-measure the chip value on TEN seasons, not three.

S43 reported +196.5 (clustered t 7.81, p 0.016) and is the ONLY finding of
2026-10-22 that survived proper analysis. It is also the sole basis for every
practical recommendation made to the user. But it was measured on **G = 3**
seasons, before the data was extended -- and on 2026-10-22 four findings
collapsed when re-tested properly (S12 magnitude, S32, S33, S34), two more were
withdrawn under clustering (F26, F28), and S48's headline needed qualifying by
S49. A number carrying that much weight should not be the one untested at G=8.

Metric 2 (chips change transfers and the squad, so a path-free metric cannot
see them), K restarts, paired on (squad, season), season-clustered.
At G=8 the bar is |t| > 2.36 rather than 4.30.
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy import stats
from fpl2 import chips as chips_mod  # noqa: E402
from fpl2 import config as C, data, features, measure, model, rates, simulate, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
K = 15
PANEL = C.cache("fpl2_panel_us.pkl")
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
    return db, t, 2 * (1 - stats.t.cdf(abs(t), G - 1)), db - crit * se, db + crit * se, G


def main():
    t0 = time.time()
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    panel = pd.read_pickle(PANEL)
    raw = data.load_raw(extended=True)
    tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)
    print(f"[{time.time()-t0:5.0f}s] setup", flush=True)

    rows = []
    for s in HOLDOUTS:
        o = model.walk_forward(feats, tf, names, s)
        g = data.with_blanks(panel, s).merge(
            o[["season", "gameweek", C.PLAYER_KEY, "xp"]],
            on=["season", "gameweek", C.PLAYER_KEY], how="left")
        g["xp"] = g["xp"].fillna(0.0); g.loc[g["fixtures"] <= 0, "xp"] = 0.0
        sq = measure.neutral_squads(g, k=K, seed=1)
        off = measure.restart_score(g, sq, "xp", simulate.simulate_season)
        on = measure.restart_score(g, sq, "xp", simulate.simulate_season,
                                   chip_policy=chips_mod.rolling_policy(g, "xp"))
        m = off.merge(on, on="squad", suffixes=("_off", "_on"))
        m["season"] = s
        rows.append(m)
        print(f"[{time.time()-t0:5.0f}s] {s}: no-chips {m.value_off.mean():.0f}  "
              f"chips {m.value_on.mean():.0f}  diff {(m.value_on-m.value_off).mean():+.1f}",
              flush=True)
    R = pd.concat(rows, ignore_index=True)
    d = R.value_on - R.value_off
    mu, t, p, lo, hi, G = cluster(d, R.season)
    print("\n" + "=" * 76)
    print("S50 -- CHIP VALUE ON TEN SEASONS")
    print("=" * 76)
    print(f"  n = {len(R)} restarts over G = {G} seasons, {K} per season")
    naive_se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"  NAIVE      {d.mean():+7.1f}  t {d.mean()/naive_se:6.2f}")
    print(f"  CLUSTERED  {mu:+7.1f}  t {t:6.2f}  p {p:.4f}  CI [{lo:+.1f}, {hi:+.1f}]")
    print(f"  bar at G={G}: |t| > {stats.t.ppf(0.975, G-1):.2f}  -> "
          f"{'CONFIRMED' if abs(t) > stats.t.ppf(0.975, G-1) else 'NOT significant'}")
    print(f"\n  S43 claimed +196.5 at G=3 (clustered t 7.81, p 0.016)")
    print(f"\n  per season:")
    for s in HOLDOUTS:
        dd = d[R.season.eq(s)]
        if len(dd): print(f"    {s}: {dd.mean():+7.1f}  (n={len(dd)})")
    pickle.dump(R, open(C.cache("fpl2_chips10.pkl"), "wb"))
    print("\ndone")


if __name__ == "__main__":
    main()
