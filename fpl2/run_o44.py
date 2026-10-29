#!/usr/bin/env python3
"""O44: does a fixture-aware wildcard beat a single-week one? Gate above."""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, model, optimise, project, rates, simulate, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
FROM=7; TO=12; BUDGET=100.0

class _WP:
    def __init__(self,est):
        self.pre=make_pipeline(SimpleImputer(strategy="median"),StandardScaler()); self.est=est
    def fit(self,X,y,sample_weight=None):
        self.est.fit(self.pre.fit_transform(X),y,sample_weight=sample_weight); return self
    def predict(self,X): return self.est.predict(self.pre.transform(X))

def squad_from(pool_xp, grid, budget=BUDGET):
    w=grid[grid.gameweek.eq(FROM+1)]
    pool=(w.groupby(C.PLAYER_KEY).agg(cost=("cost","first"),position=("position","first"),
          team=("team","first")).reset_index())
    pool=pool.merge(pool_xp,on=C.PLAYER_KEY,how="inner")
    pool=pool[pool.cost.gt(0)]
    sq=optimise.optimise_squad(pool,"sig",budget=budget)
    return None if sq is None else sq[C.PLAYER_KEY].tolist()

def main():
    t0=time.time()
    rates._reg=lambda s=0:_WP(PoissonRegressor(alpha=1.0,max_iter=500))
    raw=data.load_raw(); panel=data.build_panel(raw); tf=team.team_fixtures(raw)
    feats,names=features.build(panel,tf)
    rows=[]
    for s in C.HOLDOUT_SEASONS:
        out=model.walk_forward(feats,tf,names,s)
        grid=data.with_blanks(panel,s).merge(out[["season","gameweek",C.PLAYER_KEY,"xp"]],
            on=["season","gameweek",C.PLAYER_KEY],how="left")
        grid["xp"]=grid["xp"].fillna(0.0); grid.loc[grid["fixtures"]<=0,"xp"]=0.0
        m=model.ComponentModel(feats=names).fit(feats[feats.season<s],tf[tf.season<s])
        hist=feats[feats.season.eq(s)&(feats.gameweek<FROM)]
        if len(hist)>500: m.refit_defensive(hist)

        single=(out[out.gameweek.eq(FROM)].groupby(C.PLAYER_KEY,as_index=False)
                  .agg(sig=("xp","mean")))
        tot={}
        for g in range(FROM+1,TO+1):
            pr=project.project(m,feats,tf,s,FROM,g)
            if pr.empty: continue
            for k,v in pr.set_index(C.PLAYER_KEY)["xp"].items():
                tot[k]=tot.get(k,0.0)+float(v)
        fixture=pd.DataFrame({C.PLAYER_KEY:list(tot),"sig":[tot[k] for k in tot]})

        rng=np.random.default_rng(5); tfb=tf.copy()
        msk=tfb.season.eq(s)
        tfb.loc[msk,"opponent"]=rng.permutation(tfb.loc[msk,"opponent"].to_numpy())
        tot2={}
        for g in range(FROM+1,TO+1):
            pr=project.project(m,feats,tfb,s,FROM,g)
            if pr.empty: continue
            for k,v in pr.set_index(C.PLAYER_KEY)["xp"].items():
                tot2[k]=tot2.get(k,0.0)+float(v)
        scram=pd.DataFrame({C.PLAYER_KEY:list(tot2),"sig":[tot2[k] for k in tot2]})

        res={}
        sqs={}
        for lab,sig in (("single",single),("fixture",fixture),("scram",scram)):
            sq=squad_from(sig,grid)
            sqs[lab]=set(sq) if sq else set()
            r=simulate.simulate_season(grid[grid.gameweek.ge(FROM+1)].copy(),"xp",
                                       opening_squad=sq) if sq else None
            res[lab]=None if r is None else r.total
        rows.append(dict(season=s,**res,
                         ov_fs=len(sqs["fixture"]&sqs["single"]),
                         ov_fx=len(sqs["fixture"]&sqs["scram"])))
        print(f"[{time.time()-t0:5.0f}s] {s}: single {res['single']:.0f}  "
              f"fixture {res['fixture']:.0f}  scrambled {res['scram']:.0f}  "
              f"overlap f-s {rows[-1]['ov_fs']}/15",flush=True)
    R=pd.DataFrame(rows)
    print("\n"+"="*72); print("O44 GATE"); print("="*72)
    print(R.round(0).to_string(index=False))
    d=(R.fixture-R.single).to_numpy(); c=(R.scram-R.single).to_numpy()
    npos=int((d>0).sum())
    v=("SHIP" if d.mean()>=100 and npos==3 else
       "REAL BUT NOT ESTABLISHED" if d.mean()>=100 and npos==2 else "REFUTED")
    print(f"\n  fixture - single : {['%+.0f'%x for x in d]}  mean {d.mean():+.1f}"
          f"  positive {npos}/3  -> {v}")
    print(f"  CONTROL scrambled: {['%+.0f'%x for x in c]}  mean {c.mean():+.1f}"
          f"  positive {int((c>0).sum())}/3")
    print(f"  -> if scrambled ~= fixture, the gain is 5-gw AVERAGING, not fixtures")
    pickle.dump(R,open(C.cache("fpl2_o44.pkl"),"wb"))
    print(f"\n[{time.time()-t0:.0f}s] done")

if __name__=="__main__": main()
