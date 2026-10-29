#!/usr/bin/env python3
"""
O48b: is the -162 really the FEATURE encoding of absent expected-stats?

Targets and conversion stay correctly masked throughout (that half measured
+41). Only the feature encoding changes.
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
    rates.MASK_TARGETS=True; rates.MASK_CONVERSION=True      # correct, unchanged
    raw=data.load_raw(); panel=data.build_panel(raw); tf=team.team_fixtures(raw)
    rows=[]
    for zf in (False,True):
        features.ZERO_FILL_ABSENT_EXPECTED=zf
        feats,names=features.build(panel,tf)
        for s in C.HOLDOUT_SEASONS:
            out=model.walk_forward(feats,tf,names,s)
            g=data.with_blanks(panel,s).merge(
                out[["season","gameweek",C.PLAYER_KEY,"xp"]],
                on=["season","gameweek",C.PLAYER_KEY],how="left")
            g["xp"]=g["xp"].fillna(0.0); g.loc[g["fixtures"]<=0,"xp"]=0.0
            t=out[out.fixtures>0]
            rows.append(dict(zero_fill=zf,season=s,
                total=simulate.simulate_season(g,"xp").total,
                calib=float(t.xp.sum()/t.points.sum()),
                sp=float(t.xp.corr(t.points,method="spearman"))))
        print(f"[{time.time()-t0:5.0f}s] zero_fill={zf}: " +
              "  ".join(f"{r['season']} {r['total']:.0f}" for r in rows[-3:]),flush=True)
    features.ZERO_FILL_ABSENT_EXPECTED=False
    R=pd.DataFrame(rows)
    piv=R.pivot_table(index="zero_fill",columns="season",values="total")
    piv["mean"]=piv.mean(axis=1)
    piv["calib"]=R.groupby("zero_fill").calib.mean()
    piv["spearman"]=R.groupby("zero_fill").sp.mean()
    print("\n"+"="*70); print("O48b -- FEATURE ENCODING OF ABSENT EXPECTED-STATS"); print("="*70)
    print(piv.round(3).to_string())
    d=np.array([piv.loc[True,s]-piv.loc[False,s] for s in C.HOLDOUT_SEASONS])
    print(f"\n  zero-fill - NaN: {['%+.0f'%x for x in d]}  mean {d.mean():+7.1f}"
          f"  positive {int((d>0).sum())}/3")
    print(f"  pre-M20 reference 2237;  legacy 2215")
    print(f"  best arm here: {piv['mean'].max():.0f} "
          f"({'zero-fill' if piv['mean'].idxmax() else 'NaN'})"
          f"  vs legacy {piv['mean'].max()-2215:+.0f}")
    pickle.dump(R,open(C.cache("fpl2_o48b.pkl"),"wb"))
    print("\ndone")

if __name__=="__main__": main()
