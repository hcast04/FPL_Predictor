#!/usr/bin/env python3
"""
Does the extra data actually buy a fourth holdout season?

Two questions, and they are separate:
  1. does 2022-23 now RUN as a holdout (it has training data for the first time)
  2. do the three EXISTING holdouts change, now that the minutes model and team
     strength have two more seasons of history
If (2) moves them materially, the extra data is not free and every prior
number is on a different footing.
"""
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
class _WP:
    def __init__(self,est):
        self.pre=make_pipeline(SimpleImputer(strategy="median"),StandardScaler()); self.est=est
    def fit(self,X,y,sample_weight=None):
        self.est.fit(self.pre.fit_transform(X),y,sample_weight=sample_weight); return self
    def predict(self,X): return self.est.predict(self.pre.transform(X))

def run(extended, seasons):
    rates._reg=lambda s=0:_WP(PoissonRegressor(alpha=1.0,max_iter=500))
    raw=data.load_raw(extended=extended)
    panel=data.build_panel(raw); tf=team.team_fixtures(raw)
    feats,names=features.build(panel,tf)
    out={}
    for s in seasons:
        try:
            o=model.walk_forward(feats,tf,names,s)
            g=data.with_blanks(panel,s).merge(o[["season","gameweek",C.PLAYER_KEY,"xp"]],
                on=["season","gameweek",C.PLAYER_KEY],how="left")
            g["xp"]=g["xp"].fillna(0.0); g.loc[g["fixtures"]<=0,"xp"]=0.0
            t=o[o.fixtures>0]
            r=simulate.simulate_season(g,"xp")
            out[s]=dict(total=r.total,sp=float(t.xp.corr(t.points,method="spearman")),
                        calib=float(t.xp.sum()/t.points.sum()),n_feat=len(names))
        except Exception as e:
            out[s]=dict(error=f"{type(e).__name__}: {e}")
    return out

def main():
    t0=time.time()
    print("=== BASELINE (4 seasons, 3 holdouts) ===",flush=True)
    base=run(False,list(C.HOLDOUT_SEASONS))
    for s,v in base.items(): print(f"  {s}: {v}",flush=True)
    print(f"\n[{time.time()-t0:.0f}s] === EXTENDED (6 seasons, 4 holdouts) ===",flush=True)
    ext=run(True,["2022-23"]+list(C.HOLDOUT_SEASONS))
    for s,v in ext.items(): print(f"  {s}: {v}",flush=True)
    print("\n"+"="*72); print("DOES THE EXTRA DATA MOVE THE EXISTING HOLDOUTS?"); print("="*72)
    print(f"{'season':<10}{'base total':>12}{'ext total':>12}{'delta':>9}"
          f"{'base sp':>10}{'ext sp':>9}")
    for s in C.HOLDOUT_SEASONS:
        b,e=base.get(s,{}),ext.get(s,{})
        if "error" in b or "error" in e: print(f"  {s}: error"); continue
        print(f"{s:<10}{b['total']:>12.0f}{e['total']:>12.0f}{e['total']-b['total']:>+9.0f}"
              f"{b['sp']:>10.4f}{e['sp']:>9.4f}")
    n=ext.get("2022-23",{})
    print(f"\n  NEW HOLDOUT 2022-23: {n}")
    print(f"\n  -> G goes 3 -> 4, significance bar |t| 4.30 -> 3.18 (R32)")
    pickle.dump({"base":base,"ext":ext},open(C.cache("fpl2_extended.pkl"),"wb"))
    print("\ndone")

if __name__=="__main__": main()
