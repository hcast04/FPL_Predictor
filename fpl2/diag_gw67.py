#!/usr/bin/env python3
"""Why did the model score 30 and 32 in GW6-7 when the average scored 58-59?"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, model, rates, simulate, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor

SEASON = "2026-27"


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
    out = model.walk_forward(feats, tf, names, SEASON)

    grid = data.with_blanks(panel, SEASON).merge(
        out[["season", "gameweek", C.PLAYER_KEY, "xp"]],
        on=["season", "gameweek", C.PLAYER_KEY], how="left")
    grid["xp"] = grid["xp"].fillna(0.0)
    grid.loc[grid["fixtures"] <= 0, "xp"] = 0.0
    r = simulate.simulate_season(grid, "xp")

    nm = panel[panel.season.eq(SEASON)].groupby(C.PLAYER_KEY).player_name.last()
    for gw, xi, cap, val in r.squad_log:
        if gw not in (5, 6, 7):
            continue
        w = grid[grid.gameweek.eq(gw)].set_index(C.PLAYER_KEY)
        print(f"\n=== GW{gw}  model XI scored {val:.0f} ===")
        rows = []
        for code in xi:
            if code not in w.index:
                continue
            rr = w.loc[code]
            rows.append({"player": str(nm.get(code, code))[:20],
                         "xp": float(rr.xp), "pts": float(rr.points),
                         "mins": float(rr.minutes),
                         "cap": "C" if code == cap else ""})
        d = pd.DataFrame(rows).sort_values("xp", ascending=False)
        print(d.to_string(index=False))
        print(f"  blanks (0 min): {int((d.mins == 0).sum())}/11   "
              f"returns (>=5 pts): {int((d.pts >= 5).sum())}/11   "
              f"captain got {d[d.cap.eq('C')].pts.iloc[0]:.0f}")
    print("\nDONE")


if __name__ == "__main__":
    main()
