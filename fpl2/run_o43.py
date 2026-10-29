#!/usr/bin/env python3
"""O43 gate 2: what does restoring the coverage mask cost or gain?"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, model, rates, simulate, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor

BUGGY = {"2023-24": 2143, "2024-25": 2420, "2025-26": 2149}

class _WP:
    def __init__(self, est):
        self.pre=make_pipeline(SimpleImputer(strategy="median"),StandardScaler()); self.est=est
    def fit(self,X,y,sample_weight=None):
        self.est.fit(self.pre.fit_transform(X),y,sample_weight=sample_weight); return self
    def predict(self,X): return self.est.predict(self.pre.transform(X))

def main():
    t0=time.time()
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    cur = pd.read_pickle(C.cache("fpl2_live2627.pkl"))
    raw = pd.concat([data.load_raw(), cur], ignore_index=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)
    print(f"[{time.time()-t0:4.0f}s] {len(names)} features", flush=True)
    res={}
    for s in list(C.HOLDOUT_SEASONS)+["2026-27"]:
        out = model.walk_forward(feats, tf, names, s)
        g = data.with_blanks(panel, s).merge(
            out[["season","gameweek",C.PLAYER_KEY,"xp"]],
            on=["season","gameweek",C.PLAYER_KEY], how="left")
        g["xp"]=g["xp"].fillna(0.0); g.loc[g["fixtures"]<=0,"xp"]=0.0
        r = simulate.simulate_season(g,"xp")
        t = out[out.fixtures>0]
        res[s]=dict(total=r.total, ratio=float(t.xp.sum()/t.points.sum()),
                    spearman=float(t.xp.corr(t.points,method="spearman")))
        if s=="2026-27":
            pickle.dump(out, open(C.cache("fpl2_s20_out_fixed.pkl"),"wb"))
        print(f"[{time.time()-t0:4.0f}s] {s}: total {r.total:.0f}  "
              f"calib {res[s]['ratio']:.3f}  spearman {res[s]['spearman']:.4f}", flush=True)

    print("\n"+"="*70); print("GATE 2 -- POINTS EFFECT (reported, not the ship criterion)")
    print("="*70)
    d=[]
    for s,b in BUGGY.items():
        got=res[s]["total"]; d.append(got-b)
        print(f"  {s}: buggy {b}  fixed {got:.0f}   {got-b:+.0f}")
    d=np.array(d)
    print(f"\n  mean {d.mean():+.1f}  positive {int((d>0).sum())}/3")
    print(f"  noise floor 150/season")
    if d.mean() < -150:
        print("  -> GATE 3 ALARM: costs more than the floor. Recorded UNEXPLAINED;")
        print("     wildcard rebuild BLOCKED until understood.")
    else:
        print("  -> within the floor. Correctness criterion governs; fix SHIPS.")
    print("\n  calibration (xp/actual, 1.000 is perfect):")
    for s in res: print(f"    {s}: {res[s]['ratio']:.3f}   spearman {res[s]['spearman']:.4f}")
    pickle.dump(res, open(C.cache("fpl2_o43.pkl"),"wb"))
    print(f"\n[{time.time()-t0:.0f}s] done")

if __name__ == "__main__":
    main()
