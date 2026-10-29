#!/usr/bin/env python3
"""
Where does each available signal actually stand, on the FIXED panel (M20)?

The question is decision-relevant: which signal should be followed for the rest
of 2026-27. Every arm runs through the same simulator, same budget, full season.
"""
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
AVG=59.4*38

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
    try:
        from fpl2 import simple as simple_mod
        HAVE_SIMPLE=True
    except Exception as e:
        HAVE_SIMPLE=False; print("simple.py unavailable:",e)
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
        r=dict(season=s,
               model=simulate.simulate_season(g,"xp").total,
               legacy=simulate.simulate_season(g,"xp_leg").total)
        if HAVE_SIMPLE:
            try:
                sp=simple_mod.score(panel,s) if hasattr(simple_mod,"score") else None
                if sp is not None:
                    g=g.merge(sp,on=["season","gameweek",C.PLAYER_KEY],how="left")
                    g["xp_simple"]=g["xp_simple"].fillna(0.0)
                    g.loc[g["fixtures"]<=0,"xp_simple"]=0.0
                    r["simple"]=simulate.simulate_season(g,"xp_simple").total
            except Exception as e:
                print(f"  simple arm failed for {s}: {e}")
        rows.append(r)
        print(f"[{time.time()-t0:5.0f}s] {s}: " +
              "  ".join(f"{k} {v:.0f}" for k,v in r.items() if k!="season"),flush=True)
    R=pd.DataFrame(rows)
    print("\n"+"="*66); print("FULL-SEASON STANDING ON THE FIXED PANEL"); print("="*66)
    print(R.round(0).to_string(index=False))
    print(f"\n{'arm':<12}{'mean':>8}{'vs legacy':>12}{'vs FPL avg':>12}")
    for c in [c for c in R.columns if c!="season"]:
        m=R[c].mean()
        print(f"{c:<12}{m:>8.0f}{m-R.legacy.mean():>+12.0f}{m-AVG:>+12.0f}")
    print(f"{'FPL average':<12}{AVG:>8.0f}{AVG-R.legacy.mean():>+12.0f}{0:>+12.0f}")
    print(f"\n  (FPL average = 59.4/gw x 38, a 2026-27-derived PROXY)")
    print(f"  (noise floor: 150/season; 3-season-mean sd 39-49)")
    pickle.dump(R,open(C.cache("fpl2_standing.pkl"),"wb"))
    print(f"\n[{time.time()-t0:.0f}s] done")

if __name__=="__main__": main()
