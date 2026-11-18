#!/usr/bin/env python3
"""
S66 — does a better-shaped free-hit penalty actually win points?

S65 measured the blank penalty as CONVEX (about 3 points per blank below
three, about ten each above it) while `chips.BLANK_WEIGHT` applies a flat 2.0
from the first blank. That says the shipped shape is wrong. It does NOT say
that changing it wins, and S27 is the standing warning: season totals are
chaotic in this kind of knob, so the only way to know is to run the metric.

Arms, all identical except the free-hit penalty
`w * max(0, blanks - threshold)`:

    A  w=2,  thr=0   SHIPPED -- flat from the first blank (S50's +119.8)
    B  w=0,  thr=0   no penalty at all; the raw benefit already carries some
    C  w=8,  thr=2   hinge: nothing until three blanks, then steep
    D  w=10, thr=3   hinge: nothing until four blanks, then steeper

Metric 2 (chips change the squad and the transfer path, so a path-free metric
cannot see them), K restarts per season, paired on (squad, season),
season-clustered on G-1 df. Arm A is the REFERENCE and must reproduce
`rolling_policy()` with no overrides bit-for-bit -- if it does not, the
harness is not comparing what this docstring claims.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy import stats                                            # noqa: E402
from fpl2 import chips as chips_mod                                # noqa: E402
from fpl2 import (config as C, data, features, measure, model,     # noqa: E402
                  rates, simulate, team)
from sklearn.pipeline import make_pipeline                         # noqa: E402
from sklearn.impute import SimpleImputer                           # noqa: E402
from sklearn.preprocessing import StandardScaler                   # noqa: E402
from sklearn.linear_model import PoissonRegressor                  # noqa: E402

K = 10
PANEL = C.cache("fpl2_panel_us.pkl")
HOLDOUTS = ["2018-19", "2019-20", "2020-21", "2021-22",
            "2022-23", "2023-24", "2024-25", "2025-26"]
ARMS = [("A_shipped", 2.0, 0), ("B_none", 0.0, 0),
        ("C_hinge8@2", 8.0, 2), ("D_hinge10@3", 10.0, 3)]


class _WP:
    def __init__(self, est):
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.est = est
    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight); return self
    def predict(self, X): return self.est.predict(self.pre.transform(X))


def cluster(d, g, scale=1.0):
    d = np.asarray(d, float) * scale; g = np.asarray(g)
    ok = np.isfinite(d); d, g = d[ok], g[ok]
    n, G = len(d), len(np.unique(g)); mu = d.mean()
    ss = sum((d[g == k] - mu).sum() ** 2 for k in np.unique(g))
    se = np.sqrt((G / (G - 1)) * ss / n ** 2)
    t = mu / se if se > 0 else np.nan
    return {"mean": mu, "se": se, "t": t,
            "p": 2 * (1 - stats.t.cdf(abs(t), G - 1)), "G": G,
            "mde": 2.8 * se, "wins": sum(1 for k in np.unique(g) if d[g == k].mean() > 0)}


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
        if not sq:
            print(f"  {s}: no squads, skipped", flush=True); continue

        # NEGATIVE CONTROL: the reference arm must BE the shipped policy.
        ref = measure.restart_score(
            g, sq, "xp", simulate.simulate_season,
            chip_policy=chips_mod.rolling_policy(g, "xp"))
        a = measure.restart_score(
            g, sq, "xp", simulate.simulate_season,
            chip_policy=chips_mod.rolling_policy(g, "xp", blank_weight=2.0,
                                                 blank_threshold=0))
        gap = float((ref.set_index("squad")["value"]
                     - a.set_index("squad")["value"]).abs().max())
        print(f"  {s}: control |A - shipped| = {gap:.3e} "
              f"{'OK' if gap < 1e-9 else '<<< MISMATCH'}", flush=True)

        vals = {"A_shipped": a}
        for name, w, thr in ARMS[1:]:
            vals[name] = measure.restart_score(
                g, sq, "xp", simulate.simulate_season,
                chip_policy=chips_mod.rolling_policy(
                    g, "xp", blank_weight=w, blank_threshold=thr))
        m = vals["A_shipped"].rename(columns={"value": "A_shipped"})
        for name in vals:
            if name != "A_shipped":
                m = m.merge(vals[name].rename(columns={"value": name}), on="squad")
        m["season"] = s; m["control_gap"] = gap
        rows.append(m)
        msg = "  ".join(f"{n} {m[n].mean():.0f}" for n, _, _ in ARMS)
        print(f"  {s}: {msg}   [{time.time()-t0:.0f}s]", flush=True)

    R = pd.concat(rows, ignore_index=True)
    R.to_pickle(C.cache("fpl2_s66_freehit_policy.pkl"))
    print("\n" + "=" * 78)
    print("S66 -- FREE-HIT PENALTY SHAPE, vs the shipped flat w=2")
    print("=" * 78)
    print(f"  n = {len(R)} restarts over G = {R.season.nunique()} seasons, K = {K}")
    print(f"  max control gap across seasons: {R.control_gap.max():.3e}")
    print(f'\n{"arm":<14}{"mean":>9}{"diff":>9}{"se":>7}{"t":>7}{"p":>8}{"MDE":>7}{"seasons":>9}')
    print(f'{"A_shipped":<14}{R.A_shipped.mean():>9.0f}{"ref":>9}')
    for name, w, thr in ARMS[1:]:
        d = R[name] - R["A_shipped"]
        r = cluster(d, R.season)
        print(f'{name:<14}{R[name].mean():>9.0f}{r["mean"]:>+9.1f}{r["se"]:>7.1f}'
              f'{r["t"]:>7.2f}{r["p"]:>8.4f}{r["mde"]:>7.1f}{str(r["wins"])+"/"+str(r["G"]):>9}')
    bar = stats.t.ppf(0.975, R.season.nunique() - 1)
    print(f'\n  bar at G={R.season.nunique()}: |t| > {bar:.2f}')
    print(f'  any arm resolvable? '
          f'{any(abs(cluster(R[n]-R.A_shipped, R.season)["t"]) > bar for n,_,_ in ARMS[1:])}')
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
