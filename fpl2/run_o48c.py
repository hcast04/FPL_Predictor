#!/usr/bin/env python3
"""
O48c: the configuration that has never actually been run --
  TRAINING TARGETS and CONVERSION FACTORS correctly masked (absence != 0)
  ROLLING FEATURES zero-filled (absence encoded as "nothing recorded")

Reference points already measured:
  NaN everywhere   (M20 as shipped)        2116.3
  zeros everywhere (pre-M20)               2237.3   == 2143/2420/2149
  legacy                                   2215.3
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

def main():
    t0=time.time()
    rates._reg=lambda s=0:_WP(PoissonRegressor(alpha=1.0,max_iter=500))
    rates.MASK_TARGETS=True; rates.MASK_CONVERSION=True
    features.ZERO_FILL_ABSENT_EXPECTED=True
    raw=data.load_raw(); panel=data.build_panel(raw); tf=team.team_fixtures(raw)
    feats,names=features.build(panel,tf)
    cf=rates.conversion_factors(feats[feats.season<"2026-27"])
    print(f"  sanity -- MID xg->goals {cf['MID']['xg_per90']:.3f} "
          f"(honest 1.018, inflated 1.122)",flush=True)
    rows=[]
    for s in C.HOLDOUT_SEASONS:
        out=model.walk_forward(feats,tf,names,s)
        g=data.with_blanks(panel,s).merge(out[["season","gameweek",C.PLAYER_KEY,"xp"]],
            on=["season","gameweek",C.PLAYER_KEY],how="left")
        g["xp"]=g["xp"].fillna(0.0); g.loc[g["fixtures"]<=0,"xp"]=0.0
        t=out[out.fixtures>0]
        rows.append(dict(season=s,total=simulate.simulate_season(g,"xp").total,
            calib=float(t.xp.sum()/t.points.sum()),
            sp=float(t.xp.corr(t.points,method="spearman"))))
        print(f"[{time.time()-t0:5.0f}s] {s}: {rows[-1]['total']:.0f}"
              f"  calib {rows[-1]['calib']:.3f}  sp {rows[-1]['sp']:.4f}",flush=True)
    features.ZERO_FILL_ABSENT_EXPECTED=False
    R=pd.DataFrame(rows); m=R.total.mean()
    REF={"NaN everywhere (M20 shipped)":2116.3,"zeros everywhere (pre-M20)":2237.3,
         "legacy":2215.3}
    print("\n"+"="*64); print("O48c -- masked targets + zeroed rolling features"); print("="*64)
    print(R.round(3).to_string(index=False))
    print(f"\n  mean {m:.1f}   calib {R.calib.mean():.3f}   spearman {R.sp.mean():.4f}")
    for k,v in REF.items():
        print(f"    vs {k:<32}{m-v:+8.1f}")
    pickle.dump(R,open(C.cache("fpl2_o48c.pkl"),"wb"))
    print("\ndone")

if __name__=="__main__": main()
