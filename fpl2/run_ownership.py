#!/usr/bin/env python3
"""
Does adding crowd ownership fix the under-prediction of owned players?

Measured 2026-27 GW1-7: model calibration degrades monotonically with
ownership -- <5% owned ratio 1.09 (over-rated), 25-50% 0.59, >50% **0.56**.
The model captures barely half the output of the most-owned players while
over-rating the unowned mass, and the optimiser buys on absolute xP, so it
avoids exactly the players who deliver.

LEAKAGE FIRST. Ownership is fixed before the deadline it applies to, so it is
legitimate -- but `selected`'s collection timestamp is undocumented and M17/M18
were both wrong assumptions about timestamps. The features are shift(1) lagged
and `truncation_equivalence` must show 0.0 disagreement before any result is
read.

GATE, declared before the run:
  1. the >25%-owned calibration ratio must move toward 1.0 (it is 0.56-0.59);
  2. overall spearman must not fall by more than 0.005;
  3. the decision metric must improve in >=2 of 3 holdout seasons.
Clause 3 is the weakest bar of the three deliberately: season totals carry a
148-point noise floor and three of them cannot resolve much. Clause 1 is the
mechanism, measured on ~75 player-gameweeks per season rather than 3 numbers.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, legacy, model, rates, simulate, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor

OWN_FEATS = ["owned_lag", "owned_r3", "owned_share", "owned_delta"]


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
    panel = data.build_panel()
    tf = team.team_fixtures(data.load_raw())

    print("LEAKAGE CHECK on the new ownership features", flush=True)
    for season, gw in (("2024-25", 20), ("2025-26", 12)):
        rep = features.truncation_equivalence(panel, season, gw,
                                              feats=OWN_FEATS, fixtures=tf)
        worst = float(rep["max_abs_diff"].max())
        print(f"  {season} GW{gw}: worst disagreement {worst:.3g} "
              f"({'CLEAN' if worst < 1e-9 else 'LEAKS'})", flush=True)
        if worst >= 1e-9:
            print("  ABORTING -- a leaking feature must not be measured.", flush=True)
            return

    feats_df, names = features.build(panel, tf)
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    without = [n for n in names if n not in OWN_FEATS]
    print(f"\n[{time.time()-t0:4.0f}s] {len(names)} with ownership, "
          f"{len(without)} without\n", flush=True)

    print(f"{'season':10s}{'arm':10s}{'total':>8}{'spearman':>10}"
          f"{'r_<5%':>8}{'r_25-50':>9}{'r_>50%':>8}", flush=True)
    print("-" * 55, flush=True)
    res = {}
    for s in C.HOLDOUT_SEASONS:
        g0 = legacy.attach(data.with_blanks(panel, s)).rename(columns={"xp": "_l"})
        own_col = panel[panel.season.eq(s)].set_index(
            [C.PLAYER_KEY, "gameweek"])["selected"]
        for arm, nm in (("own", names), ("base", without)):
            out = model.walk_forward(feats_df, tf, nm, s)
            t = out[out.fixtures > 0].copy()
            t["own"] = pd.to_numeric(
                t.set_index([C.PLAYER_KEY, "gameweek"]).index.map(own_col),
                errors="coerce").astype(float)
            # `selected` counts MANAGERS owning the player. Each manager owns
            # SQUAD_SIZE players, so the manager count is sum(selected)/15 --
            # dividing by the raw sum makes every percentage 15x too small and
            # collapses every band into "<5%", which is what the first version
            # did. Same error class as W18: scoring against the wrong
            # population.
            n_mgr = t.groupby("gameweek")["own"].transform("sum") / C.SQUAD_SIZE
            t["pct"] = 100.0 * t["own"] / n_mgr.replace(0, np.nan)
            b = pd.cut(t.pct, [-.01, 5, 25, 50, 100],
                       labels=["<5", "5-25", "25-50", ">50"])
            r = t.groupby(b, observed=True).apply(
                lambda x: x.xp.sum() / max(x.points.sum(), 1e-9), include_groups=False)
            grid = g0.merge(out[["season", "gameweek", C.PLAYER_KEY, "xp"]],
                            on=["season", "gameweek", C.PLAYER_KEY], how="left")
            grid["xp"] = grid["xp"].fillna(0.0)
            grid.loc[grid["fixtures"] <= 0, "xp"] = 0.0
            tot = float(simulate.simulate_season(grid, "xp").total)
            sp = t.xp.corr(t.points, method="spearman")
            res[(s, arm)] = (tot, sp, r)
            print(f"{s if arm=='own' else '':10s}{arm:10s}{tot:>8.0f}{sp:>10.4f}"
                  f"{r.get('<5', np.nan):>8.3f}{r.get('25-50', np.nan):>9.3f}"
                  f"{r.get('>50', np.nan):>8.3f}", flush=True)
        print(flush=True)

    wins = sum(1 for s in C.HOLDOUT_SEASONS if res[(s,"own")][0] > res[(s,"base")][0])
    dsp = np.mean([res[(s,"own")][1] - res[(s,"base")][1] for s in C.HOLDOUT_SEASONS])
    print("GATE", flush=True)
    print(f"  3. decision metric improves {wins}/3   "
          f"{'PASS' if wins >= 2 else 'FAIL'}", flush=True)
    print(f"  2. spearman change {dsp:+.4f}   "
          f"{'PASS' if dsp > -0.005 else 'FAIL'}", flush=True)
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
