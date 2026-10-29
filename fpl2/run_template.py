#!/usr/bin/env python3
"""
S51: does weighting OWNERSHIP in squad selection improve things?

The user's objection, and it is correct: chips are +120/season = ~3.2 pts/gw,
so over his seven gameweeks they explain ~22 of a 79-point deficit. The rest is
captain picks and a squad that does not hit. Our model does not fix either --
it captained B.Fernandes in GW4-6 for 1,1,2, the SAME error he made and the
same one legacy made. What beat all three was the crowd, which captained
Haaland every week.

Every optimiser here maximises expected POINTS with no ownership term. S26
measured the consequence: 12.7% mean ownership and a weekly sd of 25.4 against
the crowd's 11.9. FPL is a RANKING game against 11M managers; equal mean with
double the variance is strictly worse for rank. This has never been tested.

ARMS: objective = xp + LAMBDA * ownership_share, swept. lambda=0 is current.

REPORTED, because season total is the WRONG objective for a ranking game:
  total          the points objective (what we have always measured)
  beat_rate      share of gameweeks beating that season's crowd average
                 -- a rank-like measure
  weekly sd      the variance that costs rank
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy import stats
from fpl2 import config as C, data, features, measure, model, optimise, rates, simulate, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
K = 12
LAMBDAS = (0.0, 2.0, 5.0, 10.0, 20.0)
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
    print(f"[{time.time()-t0:5.0f}s] setup", flush=True)

    rows = []
    for s in HOLDOUTS:
        o = model.walk_forward(feats, tf, names, s)
        g = data.with_blanks(panel, s).merge(
            o[["season", "gameweek", C.PLAYER_KEY, "xp"]],
            on=["season", "gameweek", C.PLAYER_KEY], how="left")
        g["xp"] = g["xp"].fillna(0.0); g.loc[g["fixtures"] <= 0, "xp"] = 0.0
        # ownership as a share of MANAGERS: each picks 15, so 15 x share-of-picks
        tot = g.groupby("gameweek")["selected"].transform("sum")
        g["own"] = (15.0 * g["selected"] / tot.replace(0, np.nan)).fillna(0.0)
        # the crowd's weekly average, approximated by the template XI's score
        crowd = {}
        for gw, wk in g[g.fixtures > 0].groupby("gameweek"):
            top = wk.nlargest(15, "own")
            xi = optimise.best_xi(top, "own")
            crowd[gw] = (float(xi["points"].sum()) + float(xi.nlargest(1, "own")["points"].iloc[0])
                         if xi is not None else np.nan)
        gws = sorted(g.gameweek.unique())[:5]
        pool0 = (g[g.gameweek.isin(gws)].groupby(C.PLAYER_KEY)
                 .agg(xp=("xp", "mean"), own=("own", "mean"), cost=("cost", "first"),
                      position=("position", "first"), team=("team", "first")).reset_index())
        pool0 = pool0[pool0.cost.gt(0)]
        for lam in LAMBDAS:
            for seed in range(K):
                rng = np.random.default_rng(5000 + seed)
                p = pool0.copy()
                p["sig"] = (p.xp + lam * p.own) * (1.0 + 0.10 * rng.standard_normal(len(p)))
                sq = optimise.optimise_squad(p, "sig", budget=C.BUDGET)
                if sq is None: continue
                r = simulate.simulate_season(g, "xp", opening_squad=sq[C.PLAYER_KEY].tolist())
                if r is None: continue
                wk = np.array(r.per_gameweek, dtype=float)
                cg = np.array([crowd.get(x, np.nan) for x in sorted(g.gameweek.unique())[:len(wk)]])
                ok = np.isfinite(cg)
                rows.append(dict(season=s, lam=lam, squad=seed, total=r.total,
                                 beat=float((wk[ok] > cg[ok]).mean()) if ok.any() else np.nan,
                                 sd=float(wk.std(ddof=1)),
                                 own=float(sq[C.PLAYER_KEY].map(
                                     pool0.set_index(C.PLAYER_KEY)["own"]).mean())))
        print(f"[{time.time()-t0:5.0f}s] {s} done", flush=True)
    R = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("S51 -- OWNERSHIP-WEIGHTED SQUAD SELECTION (8 seasons, clustered)")
    print("=" * 80)
    piv = R.groupby("lam").agg(total=("total", "mean"), beat=("beat", "mean"),
                               sd=("sd", "mean"), own=("own", "mean"))
    piv["own"] = (piv.own * 100).round(1)
    print(piv.round(3).to_string())
    print(f"\n  vs lambda=0:")
    print(f"  {'lambda':>7}{'d total':>10}{'t':>7}{'p':>8}{'d beat%':>10}{'t':>7}{'p':>8}{'d sd':>8}")
    base = R[R.lam.eq(0.0)].set_index(["season", "squad"])
    for lam in LAMBDAS[1:]:
        arm = R[R.lam.eq(lam)].set_index(["season", "squad"])
        j = base.join(arm, lsuffix="_b", rsuffix="_a", how="inner").reset_index()
        dt, tt, pt, _, _ = cluster(j.total_a - j.total_b, j.season)
        db_, tb, pb, _, _ = cluster((j.beat_a - j.beat_b) * 100, j.season)
        print(f"  {lam:>7.1f}{dt:>10.1f}{tt:>7.2f}{pt:>8.3f}"
              f"{db_:>10.2f}{tb:>7.2f}{pb:>8.3f}{(j.sd_a-j.sd_b).mean():>8.2f}")
    print(f"\n  bar at G=8: |t| > 2.36.  beat% = share of gameweeks above the crowd.")
    pickle.dump(R, open(C.cache("fpl2_template.pkl"), "wb"))
    print("\ndone")


if __name__ == "__main__":
    main()
