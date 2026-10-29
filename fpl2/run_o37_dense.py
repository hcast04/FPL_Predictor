#!/usr/bin/env python3
"""
Is `bench_weight` optimisable at all, or is the response chaotic?

F25 swept 5 values and got a jagged line (2024-25: -290, +44, +46, -88). Two
possibilities, and they need different answers:

  SMOOTH  -> there is a real optimum and it is worth selecting out-of-sample
  JAGGED  -> the season total is chaotic in w and any 'best w' is the arm that
             happened to fork well. Selecting on it is fitting noise.

This sweeps 16 values densely and ALSO reports a metric with far more samples
than 3 season totals: mean XI xp per gameweek, which does not depend on the
transfer path and so cannot fork.
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, model, optimise, rates, simulate, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor

GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
CACHE = C.cache("fpl2_o37_grids.pkl")


class _WP:
    def __init__(self, est):
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.est = est
    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight); return self
    def predict(self, X):
        return self.est.predict(self.pre.transform(X))


def main():
    t0 = time.time()
    if CACHE.exists():
        grids = pickle.load(open(CACHE, "rb"))
        print(f"[{time.time()-t0:4.0f}s] grids from cache", flush=True)
    else:
        cur = pd.read_pickle(C.cache("fpl2_live2627.pkl"))
        raw = pd.concat([data.load_raw(), cur], ignore_index=True)
        panel = data.build_panel(raw); tf = team.team_fixtures(raw)
        feats, names = features.build(panel, tf)
        rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
        grids = {}
        for s in C.HOLDOUT_SEASONS:
            out = model.walk_forward(feats, tf, names, s)
            g = data.with_blanks(panel, s).merge(
                out[["season", "gameweek", C.PLAYER_KEY, "xp"]],
                on=["season", "gameweek", C.PLAYER_KEY], how="left")
            g["xp"] = g["xp"].fillna(0.0)
            g.loc[g["fixtures"] <= 0, "xp"] = 0.0
            grids[s] = g
            print(f"[{time.time()-t0:4.0f}s] forecast {s}", flush=True)
        pickle.dump(grids, open(CACHE, "wb"))

    rows = []
    for s in C.HOLDOUT_SEASONS:
        g = grids[s]
        for w in GRID:
            r = simulate.simulate_season(g, "xp", bench_weight=w)
            # path-free metric: the OPENING squad's own XI xp, no transfers
            sq = simulate._opening_squad(g, "xp", 5, C.BUDGET, w)
            gws = sorted(g["gameweek"].unique())[:5]
            win = g[g["gameweek"].isin(gws) & g[C.PLAYER_KEY].isin(sq[C.PLAYER_KEY])]
            per_gw = [optimise.best_xi(win[win.gameweek.eq(x)], "xp")["xp"].sum()
                      for x in gws
                      if optimise.best_xi(win[win.gameweek.eq(x)], "xp") is not None]
            rows.append(dict(season=s, w=w, total=r.total, transfers=r.transfers,
                             open_xi_xp=float(np.mean(per_gw)),
                             bench_cost=float(C.BUDGET - sq.cost.sum()
                                              + sq.cost.sum()
                                              - optimise.best_xi(
                                                  win[win.gameweek.eq(gws[0])], "xp"
                                              ).cost.sum())))
        print(f"[{time.time()-t0:4.0f}s] swept {s}", flush=True)
    R = pd.DataFrame(rows)

    print("\n" + "="*84)
    print("SEASON TOTAL vs w  (the chaotic metric: depends on the whole transfer path)")
    print("="*84)
    piv = R.pivot_table(index="w", columns="season", values="total")
    piv["mean"] = piv.mean(axis=1)
    print(piv.round(0).to_string())

    print("\n  roughness: sum |f(w_i) - f(w_{i-1})| / (max - min), per season")
    print("  a smooth curve scores ~1.0; a jagged one scores much higher")
    for s in C.HOLDOUT_SEASONS:
        v = piv[s].to_numpy()
        rng = v.max() - v.min()
        print(f"    {s}: {np.abs(np.diff(v)).sum()/rng if rng>0 else 0:.2f}"
              f"   (range {rng:.0f} pts)")

    print("\n" + "="*84)
    print("OPENING-XI xp vs w  (path-free: cannot fork, so a real curve shows here)")
    print("="*84)
    piv2 = R.pivot_table(index="w", columns="season", values="open_xi_xp")
    piv2["mean"] = piv2.mean(axis=1)
    print(piv2.round(3).to_string())
    for s in C.HOLDOUT_SEASONS:
        v = piv2[s].to_numpy()
        rng = v.max() - v.min()
        print(f"    {s}: roughness {np.abs(np.diff(v)).sum()/rng if rng>0 else 0:.2f}"
              f"   (range {rng:.3f} xp/gw)")

    print("\n" + "="*84)
    print("THE QUESTION: does a better opening XI produce a better season?")
    print("="*84)
    for s in C.HOLDOUT_SEASONS:
        sub = R[R.season.eq(s)]
        print(f"  {s}: corr(open_xi_xp, season total) = "
              f"{sub.open_xi_xp.corr(sub.total):+.3f}  over {len(sub)} values of w")
    print(f"  pooled: {R.open_xi_xp.corr(R.total):+.3f}")
    pickle.dump(R, open(C.cache("fpl2_o37_dense.pkl"),"wb"))
    print(f"\n[{time.time()-t0:.0f}s] cached -> ~/fpl2_o37_dense.pkl")


if __name__ == "__main__":
    main()
