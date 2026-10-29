#!/usr/bin/env python3
"""
O50: is the minimal model as good as the full one?

S48 showed that flattening the whole rate layer costs +0.3 pts/season on
metric 1 (p = 0.99). That was an ablation of a FITTED model's outputs. This
builds the minimal model properly -- the rate layer is never fitted at all --
and tests it on BOTH instruments, because metric 1 cannot see transfers and
S32 located whatever difference exists between pipelines in exactly that.

  full     the shipped model
  minimal  minutes + team/clean-sheet model; rates -> per-position constants
  strict   minutes only; clean sheets flattened too (S48's `minutes_only`)

Also reports FIT TIME, which is the practical argument: if the rate layer buys
nothing, not fitting it is free speed and far less surface area for the kind of
silent data defect M20 turned out to be.
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy import stats
from fpl2 import config as C, data, features, measure, model, rates, simulate, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
K1, K2 = 10, 12
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
    return db, t, 2 * (1 - stats.t.cdf(abs(t), G - 1)), db - crit * se, db + crit * se


def main():
    t0 = time.time()
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    panel = pd.read_pickle(PANEL)
    raw = data.load_raw(extended=True)
    tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)
    sq1 = {s: measure.neutral_squads(data.with_blanks(panel, s), k=K1, seed=1) for s in HOLDOUTS}
    sq2 = {s: measure.neutral_squads(data.with_blanks(panel, s), k=K2, seed=7) for s in HOLDOUTS}
    print(f"[{time.time()-t0:5.0f}s] setup", flush=True)

    m1, m2, fit_time = {}, {}, {}
    for mode in ("full", "minimal", "strict"):
        p1, p2, el = [], [], 0.0
        for s in HOLDOUTS:
            t1 = time.time()
            o = model.walk_forward(feats, tf, names, s, mode=mode)
            el += time.time() - t1
            g = data.with_blanks(panel, s).merge(
                o[["season", "gameweek", C.PLAYER_KEY, "xp"]],
                on=["season", "gameweek", C.PLAYER_KEY], how="left")
            g["xp"] = g["xp"].fillna(0.0); g.loc[g["fixtures"] <= 0, "xp"] = 0.0
            d1 = measure.fixed_squad_score(g, sq1[s], "xp"); d1["season"] = s; p1.append(d1)
            d2 = measure.restart_score(g, sq2[s], "xp", simulate.simulate_season)
            d2["season"] = s; p2.append(d2)
        m1[mode] = pd.concat(p1); m2[mode] = pd.concat(p2); fit_time[mode] = el
        print(f"[{time.time()-t0:5.0f}s] {mode}: metric1 {m1[mode].value.mean():.3f}/gw   "
              f"metric2 {m2[mode].value.mean():.0f}/season   fit {el:.0f}s", flush=True)

    print("\n" + "=" * 84)
    print("O50 -- MINIMAL vs FULL   (8 seasons, season-clustered, bar |t| > 2.36)")
    print("=" * 84)
    for lab, store, on, scale in (("METRIC 1 (XI+captain, path-free)", m1,
                                   ["squad", "gameweek", "season"], 38),
                                  ("METRIC 2 (with transfers)", m2,
                                   ["squad", "season"], 1)):
        print(f"\n  {lab}")
        print(f"  {'arm':<10}{'mean diff':>12}{'t':>8}{'p':>9}{'95% CI':>22}")
        for mode in ("minimal", "strict"):
            mm = store[mode].merge(store["full"], on=on, suffixes=("_a", "_f"))
            d = (mm.value_a - mm.value_f) * scale
            mu, t, pv, lo, hi = cluster(d, mm.season)
            print(f"  {mode:<10}{mu:>12.1f}{t:>8.2f}{pv:>9.4f}  [{lo:>+7.1f},{hi:>+7.1f}]")
    print(f"\n  FIT TIME over 8 seasons:")
    for mode in ("full", "minimal", "strict"):
        print(f"    {mode:<10}{fit_time[mode]:>7.0f}s"
              f"   ({fit_time['full']/max(fit_time[mode],1e-9):.1f}x vs full)")
    pickle.dump({"m1": m1, "m2": m2, "fit_time": fit_time},
                open(C.cache("fpl2_o50.pkl"), "wb"))
    print("\ndone")


if __name__ == "__main__":
    main()
