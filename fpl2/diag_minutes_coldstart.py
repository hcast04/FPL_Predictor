#!/usr/bin/env python3
"""Does the minutes model systematically under-predict established starters early?"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, model, rates, team   # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor


class _WP:
    def __init__(self, est):
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.est = est

    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight); return self

    def predict(self, X):
        return self.est.predict(self.pre.transform(X))


def main():
    cur = pd.read_pickle(C.cache("fpl2_live2627.pkl"))
    raw = pd.concat([data.load_raw(), cur], ignore_index=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))

    print(f"{'season':10s}{'group':22s}{'n':>6}{'pred_min':>10}{'act_min':>9}"
          f"{'ratio':>8}{'pred_p60':>10}{'act_60+':>9}", flush=True)
    for s in list(C.HOLDOUT_SEASONS) + ["2026-27"]:
        out = model.walk_forward(feats, tf, names, s)
        t = out[out.fixtures > 0].copy()
        # a player's CURRENT-SEASON start rate, strictly before each gameweek
        t = t.sort_values([C.PLAYER_KEY, "gameweek"])
        g = t.groupby(C.PLAYER_KEY)
        t["started_so_far"] = g["starts"].transform(lambda x: x.shift(1).cumsum())
        t["games_so_far"] = g["starts"].transform(lambda x: x.shift(1).notna().cumsum())
        t["start_rate"] = t.started_so_far / t.games_so_far.replace(0, np.nan)
        early = t[(t.gameweek.between(3, 10)) & (t.games_so_far >= 2)]
        for lbl, sel in (("ever-present (100%)", early.start_rate.ge(0.999)),
                         ("regular (>=75%)", early.start_rate.between(0.75, 0.999)),
                         ("rotated (<75%)", early.start_rate.lt(0.75))):
            x = early[sel]
            if len(x) < 30:
                continue
            print(f"{s:10s}{lbl:22s}{len(x):>6}{x.expected_minutes.mean():>10.1f}"
                  f"{x.minutes.mean():>9.1f}"
                  f"{x.expected_minutes.mean()/max(x.minutes.mean(),1e-9):>8.3f}"
                  f"{x.p_sixty.mean():>10.3f}{x.minutes.ge(60).mean():>9.3f}", flush=True)
        print(flush=True)
    print("  ratio < 1 means the model UNDER-predicts minutes for that group.", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
