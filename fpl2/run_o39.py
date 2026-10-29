#!/usr/bin/env python3
"""
O39: multi-week forward projection, against the gate pre-registered 2026-10-22.

  1. leak test -- PRECONDITION, run separately, PASSED (max diff 0.0 vs an
     oracle control that moved 13.04)
  2. forecast quality: spearman vs realised points for k=1,2,3 x 3 seasons,
     new projection vs the flat one. 9 cells, reported whole.
  3. decision metric, THE gate: MPC H=4 on the new projection must beat GREEDY
     by >= +100 pts/season, 3/3 to ship.
  4. reported, not gated: new projection vs flat projection, both under MPC.
  5. R26 control: scramble the opponent map; the gain on (2) must vanish.
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, model, mpc, project, rates, simulate, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor

KS = (1, 2, 3)
SAMPLE_GWS = range(6, 35, 2)


class _WP:
    def __init__(self, est):
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()); self.est = est
    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight); return self
    def predict(self, X): return self.est.predict(self.pre.transform(X))


def flat_projection(base_xp, base_fx, tgt_fx):
    """What mpc.honest_frames does: this week's xp per fixture x target count."""
    per = base_xp / np.clip(base_fx, 1, None)
    return per * np.clip(tgt_fx, 0, 2)


def main():
    t0 = time.time()
    cur = pd.read_pickle(C.cache("fpl2_live2627.pkl"))
    raw = pd.concat([data.load_raw(), cur], ignore_index=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    grids = pickle.load(open(C.cache("fpl2_o37_grids.pkl"), "rb"))
    print(f"[{time.time()-t0:5.0f}s] setup", flush=True)

    qual, fitted = [], {}
    for s in C.HOLDOUT_SEASONS:
        m = model.ComponentModel(feats=names).fit(feats[feats.season < s],
                                                  tf[tf.season < s])
        fitted[s] = m
        print(f"[{time.time()-t0:5.0f}s] fitted {s}", flush=True)
        sea = feats[feats.season.eq(s)]
        for g0 in SAMPLE_GWS:
            b = sea[sea.gameweek.eq(g0)]
            if b.empty:
                continue
            bp = m.predict(b, tf[tf.season.eq(s) & tf.gameweek.eq(g0)])
            bxp = bp.set_index(C.PLAYER_KEY)["xp"]
            bfx = bp.set_index(C.PLAYER_KEY)["fixtures"]
            for k in KS:
                g1 = g0 + k
                act = sea[sea.gameweek.eq(g1)]
                if act.empty:
                    continue
                pr = project.project(m, feats, tf, s, g0, g1)
                if pr.empty:
                    continue
                a = act.set_index(C.PLAYER_KEY)
                idx = pr.set_index(C.PLAYER_KEY).index.intersection(a.index)
                if len(idx) < 50:
                    continue
                y = pd.to_numeric(a.loc[idx, "points"], errors="coerce")
                new = pr.set_index(C.PLAYER_KEY).loc[idx, "xp"]
                fx1 = pd.to_numeric(a.loc[idx, "fixtures"], errors="coerce").fillna(0)
                old = pd.Series(flat_projection(
                    bxp.reindex(idx).fillna(0).to_numpy(float),
                    bfx.reindex(idx).fillna(1).to_numpy(float),
                    fx1.to_numpy(float)), index=idx)
                qual.append(dict(season=s, k=k, gw=g0,
                                 new=float(new.corr(y, method="spearman")),
                                 flat=float(old.corr(y, method="spearman"))))
        print(f"[{time.time()-t0:5.0f}s] quality {s}", flush=True)
    Q = pd.DataFrame(qual)

    print("\n" + "="*74)
    print("GATE 2 -- FORECAST QUALITY (spearman vs realised points)")
    print("="*74)
    agg = Q.groupby(["season", "k"]).agg(new=("new", "mean"), flat=("flat", "mean"),
                                         n=("gw", "size"))
    agg["delta"] = (agg["new"] - agg["flat"]).round(4)
    print(agg.round(4).to_string())
    cells = agg["delta"] > 0
    print(f"\n  cells where the new projection wins: {int(cells.sum())}/{len(cells)}")
    print(f"  -> gate 2 {'PASS' if cells.all() else 'FAIL'} (needs all 9)")

    print("\n" + "="*74)
    print("GATE 5 -- NEGATIVE CONTROL: scrambled opponent map")
    print("="*74)
    rng = np.random.default_rng(3)
    tf_bad = tf.copy()
    msk = tf_bad.season.isin(C.HOLDOUT_SEASONS)
    tf_bad.loc[msk, "opponent"] = rng.permutation(tf_bad.loc[msk, "opponent"].to_numpy())
    ctl = []
    for s in C.HOLDOUT_SEASONS:
        m = fitted[s]; sea = feats[feats.season.eq(s)]
        for g0 in list(SAMPLE_GWS)[::2]:
            act = sea[sea.gameweek.eq(g0 + 2)]
            if act.empty: continue
            pr = project.project(m, feats, tf_bad, s, g0, g0 + 2)
            if pr.empty: continue
            a = act.set_index(C.PLAYER_KEY)
            idx = pr.set_index(C.PLAYER_KEY).index.intersection(a.index)
            if len(idx) < 50: continue
            y = pd.to_numeric(a.loc[idx, "points"], errors="coerce")
            ctl.append(float(pr.set_index(C.PLAYER_KEY).loc[idx, "xp"]
                             .corr(y, method="spearman")))
    real = Q[Q.k.eq(2)]["new"].mean()
    print(f"  real opponent map : spearman {real:.4f}")
    print(f"  scrambled opponents: spearman {np.mean(ctl):.4f}")
    print(f"  delta {real-np.mean(ctl):+.4f}  -> control "
          f"{'PASSES (opponent info is real)' if real > np.mean(ctl) else 'FAILS'}")

    print("\n" + "="*74)
    print("GATE 3 -- DECISION METRIC (MPC H=4 on the new projection)")
    print("="*74)
    rows = []
    for s in C.HOLDOUT_SEASONS:
        g = grids[s]
        base = simulate.simulate_season(g, "xp").total
        flat = simulate.simulate_season(g, "xp", policy="mpc", mpc_horizon=4).total
        m = fitted[s]
        def ff(gw, H, _m=m, _s=s):
            fr = project.frames(_m, feats, tf, _s, gw, H)
            out = []
            for f in fr:
                gg = g[g.gameweek.eq(int(f["gameweek"].iloc[0]))][
                    [C.PLAYER_KEY, "cost", "position", "team", "fixtures"]]
                j = gg.merge(f[[C.PLAYER_KEY, "xp_plan"]], on=C.PLAYER_KEY, how="left")
                j["xp_plan"] = j["xp_plan"].fillna(0.0)
                out.append(j)
            return out
        new = simulate.simulate_season(g, "xp", policy="mpc", mpc_horizon=4,
                                       mpc_frames=ff).total
        rows.append(dict(season=s, greedy=base, mpc_flat=flat, mpc_new=new))
        print(f"[{time.time()-t0:5.0f}s] {s}: greedy {base:.0f}  mpc_flat {flat:.0f}"
              f"  mpc_new {new:.0f}  (vs greedy {new-base:+.0f},"
              f" vs flat {new-flat:+.0f})", flush=True)
    R = pd.DataFrame(rows)
    d = (R.mpc_new - R.greedy).to_numpy()
    npos = int((d > 0).sum())
    verdict = ("SHIP" if d.mean() >= 100 and npos == 3 else
               "REAL BUT NOT ESTABLISHED" if d.mean() >= 100 and npos == 2 else "REFUTED")
    print(f"\n  vs GREEDY : mean {d.mean():+7.1f}  positive {npos}/3  -> {verdict}")
    d2 = (R.mpc_new - R.mpc_flat).to_numpy()
    print(f"  vs FLAT   : mean {d2.mean():+7.1f}  positive {int((d2>0).sum())}/3"
          f"   [reported, NOT the gate]")
    pickle.dump({"quality": Q, "decision": R}, open(C.cache("fpl2_o39.pkl"),"wb"))
    print(f"\n[{time.time()-t0:.0f}s] cached -> ~/fpl2_o39.pkl")


if __name__ == "__main__":
    main()
