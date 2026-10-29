#!/usr/bin/env python3
"""O47: how conservative should the transfer rule be? Gate in the ledger."""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, legacy, model, rates, simulate, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
GRID=(0.30,0.5,1.0,1.5,2.0)
class _WP:
    def __init__(self,est):
        self.pre=make_pipeline(SimpleImputer(strategy="median"),StandardScaler()); self.est=est
    def fit(self,X,y,sample_weight=None):
        self.est.fit(self.pre.fit_transform(X),y,sample_weight=sample_weight); return self
    def predict(self,X): return self.est.predict(self.pre.transform(X))

def main():
    t0=time.time()
    rates._reg=lambda s=0:_WP(PoissonRegressor(alpha=1.0,max_iter=500))
    raw=data.load_raw(); panel=data.build_panel(raw); tf=team.team_fixtures(raw)
    feats,names=features.build(panel,tf)
    rows=[]
    for s in C.HOLDOUT_SEASONS:
        g0=data.with_blanks(panel,s)
        out=model.walk_forward(feats,tf,names,s)
        g=g0.merge(out[["season","gameweek",C.PLAYER_KEY,"xp"]],
                   on=["season","gameweek",C.PLAYER_KEY],how="left")
        g["xp"]=g["xp"].fillna(0.0); g.loc[g["fixtures"]<=0,"xp"]=0.0
        lg=legacy.attach(g0)[["season","gameweek",C.PLAYER_KEY,"xp"]].rename(columns={"xp":"xp_leg"})
        g=g.merge(lg,on=["season","gameweek",C.PLAYER_KEY],how="left")
        g["xp_leg"]=g["xp_leg"].fillna(0.0); g.loc[g["fixtures"]<=0,"xp_leg"]=0.0
        for mg in GRID:
            r=simulate.simulate_season(g,"xp",min_gain=mg)
            rows.append(dict(season=s,arm="model",min_gain=mg,total=r.total,tr=r.transfers))
            rl=simulate.simulate_season(g,"xp_leg",min_gain=mg)
            rows.append(dict(season=s,arm="legacy",min_gain=mg,total=rl.total,tr=rl.transfers))
        rn=simulate.simulate_season(g,"xp",min_gain=1e9)
        rows.append(dict(season=s,arm="model",min_gain=np.inf,total=rn.total,tr=rn.transfers))
        rnl=simulate.simulate_season(g,"xp_leg",min_gain=1e9)
        rows.append(dict(season=s,arm="legacy",min_gain=np.inf,total=rnl.total,tr=rnl.transfers))
        print(f"[{time.time()-t0:5.0f}s] {s} done",flush=True)
    R=pd.DataFrame(rows)
    for arm in ("model","legacy"):
        sub=R[R.arm.eq(arm)]
        piv=sub.pivot_table(index="min_gain",columns="season",values="total")
        trv=sub.pivot_table(index="min_gain",columns="season",values="tr")
        piv["mean"]=piv.mean(axis=1); piv["transfers"]=trv.mean(axis=1)
        print(f"\n{'='*66}\n{arm.upper()} -- season total by min_gain\n{'='*66}")
        print(piv.round(0).to_string())
        base=piv.loc[0.30,"mean"]
        print(f"\n  vs min_gain=0.30 (mean {base:.0f}):")
        for mg in list(GRID)+[np.inf]:
            if mg==0.30: continue
            d=np.array([piv.loc[mg,s]-piv.loc[0.30,s] for s in C.HOLDOUT_SEASONS])
            lab="NEVER" if mg==np.inf else f"{mg}"
            v=("SHIP" if d.mean()>=100 and (d>0).sum()==3 else
               "real-not-established" if d.mean()>=100 and (d>0).sum()==2 else "fail")
            print(f"    {lab:>7}: {['%+.0f'%x for x in d]}  mean {d.mean():+7.1f}"
                  f"  positive {int((d>0).sum())}/3  {v}")
    pickle.dump(R,open(C.cache("fpl2_o47.pkl"),"wb"))
    print("\ndone")

if __name__=="__main__": main()
