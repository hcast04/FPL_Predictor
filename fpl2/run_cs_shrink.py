#!/usr/bin/env python3
"""
Fix S18: shrink predicted goals-conceded so clean sheets stop being over-stated
for the best defences.

Diagnosis (S18, `run_cs_calib.py`): clean-sheet probability is well calibrated
in the middle (0.294 -> 0.302) and badly over-stated at the top (0.542 ->
0.371). Converting back via lambda = -ln(CS), we predict 0.61 for the best
defences against a true 0.99, and 2.20 for the worst against a true 1.94. The
lambda predictions are OVER-DISPERSED; the Poisson form is fine (var/mean
1.078). So shrink lambda toward the training mean.

GATE, declared before the run:
  1. the strong-defence bucket (predicted > 0.45) ratio must move toward 1.0
     -- it is 1.46 unshrunk;
  2. the well-calibrated middle bucket (0.25-0.35, currently 0.97) must not be
     pushed further from 1.0 than it starts;
  3. mean absolute calibration error across ALL buckets must improve;
  4. the decision metric must not degrade -- mean season total vs legacy must
     not fall by more than the 148 noise floor.

A shrink that fixes the top by breaking the middle is NOT a fix. Clause 2 is
there because the obvious failure mode of a global shrink is flattening
everything toward the mean, which would improve the tail and wreck the centre.
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

GRID = (0.0, 0.10, 0.20, 0.30, 0.50)
BUCKETS = [0, .15, .25, .35, .45, 1.0]


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
    leg = {}
    for s in C.HOLDOUT_SEASONS:
        g = legacy.attach(data.with_blanks(panel, s)).rename(columns={"xp": "xp_legacy"})
        g.loc[g["fixtures"] <= 0, "xp_legacy"] = 0.0
        leg[s] = (g, float(simulate.simulate_season(g, "xp_legacy").total))
    print(f"[{time.time()-t0:4.0f}s] setup\n", flush=True)

    res = {}
    for k in GRID:
        team.CONCEDED_SHRINK = k
        rows, margins, sp = [], [], []
        for s in list(C.HOLDOUT_SEASONS) + ["2026-27"]:
            out = model.walk_forward(feats, tf, names, s)
            t = out[(out.fixtures > 0) & (out.minutes >= 60)]
            rows.append(pd.DataFrame({"p": t.clean_sheet_prob.to_numpy(float),
                                      "real": (pd.to_numeric(t.clean_sheets,
                                               errors="coerce") > 0).astype(float)}))
            a = out[out.fixtures > 0]
            sp.append(a.xp.corr(a.points, method="spearman"))
            if s in leg:
                g = leg[s][0].merge(out[["season", "gameweek", C.PLAYER_KEY, "xp"]],
                                    on=["season", "gameweek", C.PLAYER_KEY], how="left")
                g["xp"] = g["xp"].fillna(0.0)
                g.loc[g["fixtures"] <= 0, "xp"] = 0.0
                margins.append(float(simulate.simulate_season(g, "xp").total) - leg[s][1])
        d = pd.concat(rows, ignore_index=True)
        d["b"] = pd.cut(d.p, BUCKETS)
        gb = d.groupby("b", observed=True).agg(pred=("p", "mean"),
                                               real=("real", "mean"), n=("p", "size"))
        gb["ratio"] = gb.pred / gb.real.replace(0, np.nan)
        mae = float((gb.ratio - 1.0).abs().mean())
        res[k] = (gb, mae, float(np.mean(margins)), float(np.mean(sp)))
        top = gb.ratio.iloc[-1]; mid = gb.ratio.iloc[2]
        print(f"shrink={k:.2f}  top_bucket {top:.3f}  mid_bucket {mid:.3f}  "
              f"mean|ratio-1| {mae:.3f}  vs_legacy {np.mean(margins):+.0f}  "
              f"spearman {np.mean(sp):.4f}", flush=True)
    team.CONCEDED_SHRINK = 0.0

    print("\nGATE", flush=True)
    base = res[0.0]
    for k in GRID[1:]:
        gb, mae, marg, _ = res[k]
        c1 = abs(gb.ratio.iloc[-1] - 1) < abs(base[0].ratio.iloc[-1] - 1)
        c2 = abs(gb.ratio.iloc[2] - 1) <= abs(base[0].ratio.iloc[2] - 1) + 1e-9
        c3 = mae < base[1]
        c4 = marg > base[2] - 148
        print(f"  shrink={k:.2f}  top->1 {'Y' if c1 else 'n'}  mid-held "
              f"{'Y' if c2 else 'n'}  mae-better {'Y' if c3 else 'n'}  "
              f"decision-ok {'Y' if c4 else 'n'}   "
              f"{'PASS' if all((c1, c2, c3, c4)) else 'FAIL'}", flush=True)
    print("\nFULL BUCKET TABLE, best candidate vs unshrunk", flush=True)
    print("unshrunk:\n" + base[0].round(3).to_string(), flush=True)
    best = max((k for k in GRID[1:]), key=lambda k: -res[k][1])
    print(f"\nshrink={best}:\n" + res[best][0].round(3).to_string(), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
