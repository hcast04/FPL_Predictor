#!/usr/bin/env python3
"""
O27: why does the minutes fix gain +219 in 2025-26 and lose 33 in 2024-25?

The mechanism (S19/S20) says: the model under-predicts minutes for ever-present
starters, which costs most where one such player dominates. 2024-25 had Salah
on 344, **108 clear** of second. 2025-26's top scorer led by 4. So the fix
should pay off MOST in 2024-25 -- and it is the season that loses.

Either the mechanism is wrong, or something else drives 2025-26.

Decomposition, K=0 vs K=4 on the same seasons:
  1 whose xP moved, and did the dominant player gain as predicted?
  2 did the CAPTAIN change, and what did those changes cost or earn?
  3 did the XI change beyond the captain?
A mechanism that is right should show the dominant player gaining xP in
2024-25 and being captained more often. If he gains xP and the season still
loses, the link from xP to points is what is broken, not the minutes model.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, model, rates, simulate, team  # noqa: E402
from fpl2 import minutes as minutes_mod   # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor

SEASONS = ["2024-25", "2025-26"]


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
    cur = pd.read_pickle(C.cache("fpl2_live2627.pkl"))
    raw = pd.concat([data.load_raw(), cur], ignore_index=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    print(f"[{time.time()-t0:4.0f}s] setup\n", flush=True)

    for s in SEASONS:
        nm = panel[panel.season.eq(s)].groupby(C.PLAYER_KEY).player_name.last()
        season_pts = panel[panel.season.eq(s)].groupby(C.PLAYER_KEY).points.sum()
        star = season_pts.idxmax()
        arms = {}
        for k in (0.0, 4.0):
            minutes_mod.RECENCY_K = k
            out = model.walk_forward(feats, tf, names, s)
            grid = data.with_blanks(panel, s).merge(
                out[["season", "gameweek", C.PLAYER_KEY, "xp", "expected_minutes"]],
                on=["season", "gameweek", C.PLAYER_KEY], how="left")
            grid["xp"] = grid["xp"].fillna(0.0)
            grid.loc[grid["fixtures"] <= 0, "xp"] = 0.0
            arms[k] = (out, grid, simulate.simulate_season(grid, "xp"))
        minutes_mod.RECENCY_K = 0.0

        a, b = arms[0.0], arms[4.0]
        print(f"=== {s}   K=0 {a[2].total:.0f} -> K=4 {b[2].total:.0f} "
              f"({b[2].total - a[2].total:+.0f}) ===", flush=True)
        print(f"  dominant player: {str(nm.get(star))[:22]} ({season_pts.max():.0f} pts)",
              flush=True)

        # 1. xP change for the dominant player, and the biggest movers
        ja = a[0][a[0].fixtures > 0].groupby(C.PLAYER_KEY).xp.mean()
        jb = b[0][b[0].fixtures > 0].groupby(C.PLAYER_KEY).xp.mean()
        d = (jb - ja).dropna()
        print(f"  star xP {ja.get(star, np.nan):.3f} -> {jb.get(star, np.nan):.3f} "
              f"({d.get(star, np.nan):+.3f})", flush=True)
        up = d.nlargest(4); dn = d.nsmallest(4)
        print("  biggest xP gainers: " + ", ".join(
            f"{str(nm.get(i))[:14]} {v:+.2f}" for i, v in up.items()), flush=True)
        print("  biggest xP losers : " + ", ".join(
            f"{str(nm.get(i))[:14]} {v:+.2f}" for i, v in dn.items()), flush=True)

        # 2. captain changes
        ca = {gw: cap for gw, _, cap, _ in a[2].squad_log}
        cb = {gw: cap for gw, _, cap, _ in b[2].squad_log}
        pts = panel[panel.season.eq(s)].set_index([C.PLAYER_KEY, "gameweek"]).points
        changed, delta = 0, 0.0
        for gw in sorted(set(ca) & set(cb)):
            if ca[gw] != cb[gw]:
                changed += 1
                delta += float(pts.get((cb[gw], gw), 0)) - float(pts.get((ca[gw], gw), 0))
        print(f"  captain changed in {changed}/{len(set(ca) & set(cb))} gameweeks, "
              f"worth {delta:+.0f} pts (doubled: {2*delta:+.0f})", flush=True)
        sa = sum(1 for gw, c in ca.items() if c == star)
        sb = sum(1 for gw, c in cb.items() if c == star)
        print(f"  star captained {sa} -> {sb} gameweeks", flush=True)

        # 3. XI overlap
        xa = {gw: set(xi) for gw, xi, _, _ in a[2].squad_log}
        xb = {gw: set(xi) for gw, xi, _, _ in b[2].squad_log}
        ov = [len(xa[g] & xb[g]) for g in sorted(set(xa) & set(xb))]
        print(f"  mean XI overlap {np.mean(ov):.1f}/11   transfers "
              f"{a[2].transfers} -> {b[2].transfers}\n", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
