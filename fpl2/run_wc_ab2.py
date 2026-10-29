#!/usr/bin/env python3
"""Legacy vs model wildcard, RE-RUN on the fixed panel (M20/F31).

The fix changed OUR model and not legacy -- legacy is a cached notebook
forecast, and actuals (points, minutes) were never masked so they are
unchanged. The earlier A/B was internally valid (both arms shared the bug);
this one says where the two stand now.
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, legacy, model, optimise, rates, simulate, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
START=8; BUDGET=100.0

class _WP:
    def __init__(self,est):
        self.pre=make_pipeline(SimpleImputer(strategy="median"),StandardScaler()); self.est=est
    def fit(self,X,y,sample_weight=None):
        self.est.fit(self.pre.fit_transform(X),y,sample_weight=sample_weight); return self
    def predict(self,X): return self.est.predict(self.pre.transform(X))

def build_at(g,signal,gw=START,budget=BUDGET,horizon=5):
    gws=[x for x in sorted(g.gameweek.unique()) if gw<=x<gw+horizon]
    w=g[g.gameweek.isin(gws)]
    pool=(w.groupby(C.PLAYER_KEY).agg(**{signal:(signal,"mean"),"cost":("cost","first"),
          "position":("position","first"),"team":("team","first")}).reset_index())
    pool=pool[pool.cost.gt(0)]
    sq=optimise.optimise_squad(pool,signal,budget=budget)
    return None if sq is None else sq[C.PLAYER_KEY].tolist()

def play(g,squad,signal="xp",gw=START):
    r=simulate.simulate_season(g[g.gameweek.ge(gw)].copy(),signal,opening_squad=squad)
    return None if r is None else r.total

def main():
    t0=time.time()
    rates._reg=lambda s=0:_WP(PoissonRegressor(alpha=1.0,max_iter=500))
    raw=data.load_raw(); panel=data.build_panel(raw); tf=team.team_fixtures(raw)
    feats,names=features.build(panel,tf)
    rows=[]
    for s in C.HOLDOUT_SEASONS:
        out=model.walk_forward(feats,tf,names,s)
        g=data.with_blanks(panel,s).merge(out[["season","gameweek",C.PLAYER_KEY,"xp"]],
            on=["season","gameweek",C.PLAYER_KEY],how="left")
        g["xp"]=g["xp"].fillna(0.0); g.loc[g["fixtures"]<=0,"xp"]=0.0
        lg=legacy.attach(data.with_blanks(panel,s))[["season","gameweek",C.PLAYER_KEY,"xp"]]\
            .rename(columns={"xp":"xp_leg"})
        g=g.merge(lg,on=["season","gameweek",C.PLAYER_KEY],how="left")
        g["xp_leg"]=g["xp_leg"].fillna(0.0); g.loc[g["fixtures"]<=0,"xp_leg"]=0.0
        sm=build_at(g,"xp"); sl=build_at(g,"xp_leg")
        tm_=play(g,sm); tl=play(g,sl)
        tm2=play(g,sm,"xp"); tl2=play(g,sl,"xp_leg")
        rows.append(dict(season=s,model=tm_,legacy=tl,model_e2e=tm2,legacy_e2e=tl2,
                         overlap=len(set(sm)&set(sl))))
        print(f"[{time.time()-t0:4.0f}s] {s}: model {tm_:.0f}  legacy {tl:.0f}  "
              f"e2e {tm2:.0f}/{tl2:.0f}  overlap {len(set(sm)&set(sl))}/15",flush=True)
    R=pd.DataFrame(rows)
    print("\n"+"="*70); print("WILDCARD AT GW8, FIXED PANEL"); print("="*70)
    print(R.round(0).to_string(index=False))
    d=(R.model-R.legacy).to_numpy(); d2=(R.model_e2e-R.legacy_e2e).to_numpy()
    print(f"\n  model - legacy (model transfers): {['%+.0f'%x for x in d]}"
          f"  mean {d.mean():+.1f}  positive {int((d>0).sum())}/3")
    print(f"  end-to-end, own signals:          {['%+.0f'%x for x in d2]}"
          f"  mean {d2.mean():+.1f}  positive {int((d2>0).sum())}/3")
    print(f"\n  BEFORE the fix it was: -45/-54/-30 mean -43.0 (0/3)")
    pickle.dump(R,open(C.cache("fpl2_wc_ab2.pkl"),"wb"))
    print(f"\n[{time.time()-t0:.0f}s] done")

if __name__=="__main__": main()
