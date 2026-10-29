#!/usr/bin/env python3
"""O48: which half of M20 cost the 121 points? Gate in the ledger."""
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
ARMS={"both_old":(False,False),"both_new":(True,True),
      "targets_only":(True,False),"conversion_only":(False,True)}
LEGACY_MEAN=2215.3
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
    for arm,(mt,mc) in ARMS.items():
        rates.MASK_TARGETS=mt; rates.MASK_CONVERSION=mc
        for s in C.HOLDOUT_SEASONS:
            out=model.walk_forward(feats,tf,names,s)
            g=data.with_blanks(panel,s).merge(
                out[["season","gameweek",C.PLAYER_KEY,"xp"]],
                on=["season","gameweek",C.PLAYER_KEY],how="left")
            g["xp"]=g["xp"].fillna(0.0); g.loc[g["fixtures"]<=0,"xp"]=0.0
            t=out[out.fixtures>0]
            rows.append(dict(arm=arm,season=s,
                total=simulate.simulate_season(g,"xp").total,
                calib=float(t.xp.sum()/t.points.sum()),
                sp=float(t.xp.corr(t.points,method="spearman"))))
        print(f"[{time.time()-t0:5.0f}s] {arm}: " +
              "  ".join(f"{r['season']} {r['total']:.0f}" for r in rows[-3:]),flush=True)
    rates.MASK_TARGETS=True; rates.MASK_CONVERSION=True
    R=pd.DataFrame(rows)
    piv=R.pivot_table(index="arm",columns="season",values="total")
    piv["mean"]=piv.mean(axis=1)
    piv["calib"]=R.groupby("arm").calib.mean()
    piv["spearman"]=R.groupby("arm").sp.mean()
    print("\n"+"="*74); print("O48 -- WHICH HALF OF M20 COST THE POINTS?"); print("="*74)
    print(piv.round(3).to_string())
    base=piv.loc["both_new","mean"]; old=piv.loc["both_old","mean"]
    print(f"\n  both_old (pre-M20)  {old:.0f}")
    print(f"  both_new (current)  {base:.0f}   gap to recover: {old-base:+.0f}")
    for arm in ("targets_only","conversion_only"):
        d=np.array([piv.loc[arm,s]-piv.loc["both_new",s] for s in C.HOLDOUT_SEASONS])
        print(f"  {arm:<16} {piv.loc[arm,'mean']:.0f}   recovers "
              f"{['%+.0f'%x for x in d]}  mean {d.mean():+7.1f}  "
              f"positive {int((d>0).sum())}/3")
    print(f"\n  legacy mean {LEGACY_MEAN:.0f}; each arm vs legacy:")
    for arm in piv.index:
        print(f"    {arm:<16}{piv.loc[arm,'mean']-LEGACY_MEAN:>+8.0f}")
    pickle.dump(R,open(C.cache("fpl2_o48.pkl"),"wb"))
    print("\ndone")

if __name__=="__main__": main()
