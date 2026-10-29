#!/usr/bin/env python3
"""
XGBoost for the rate component, measured on the CALIBRATED instrument.

S12 recorded "the GBDT is actively hurting; a Poisson GLM beats it by ~194
pts/season" -- but that was the OLD instrument (MDE 123), and the GBDT there
was sklearn's HistGradientBoostingRegressor, not XGBoost. Re-tested here with
metric 1 (MDE 13.6), three arms on identical fixed squads.
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, measure, model, rates, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import xgboost as xgb
K=12
class _WP:
    def __init__(self,est):
        self.pre=make_pipeline(SimpleImputer(strategy="median"),StandardScaler()); self.est=est
    def fit(self,X,y,sample_weight=None):
        self.est.fit(self.pre.fit_transform(X),y,sample_weight=sample_weight); return self
    def predict(self,X): return self.est.predict(self.pre.transform(X))

def glm(seed=0):  return _WP(PoissonRegressor(alpha=1.0,max_iter=500))
def hgb(seed=0):  return HistGradientBoostingRegressor(loss="poisson",max_iter=300,
                      learning_rate=0.06,max_depth=6,min_samples_leaf=40,
                      l2_regularization=1.0,random_state=seed)
def xgb_reg(seed=0): return xgb.XGBRegressor(objective="count:poisson",
                      n_estimators=400,learning_rate=0.05,max_depth=5,
                      min_child_weight=20,subsample=0.8,colsample_bytree=0.8,
                      reg_lambda=2.0,random_state=seed,n_jobs=4,verbosity=0)
def xgb_shallow(seed=0): return xgb.XGBRegressor(objective="count:poisson",
                      n_estimators=600,learning_rate=0.03,max_depth=3,
                      min_child_weight=40,subsample=0.8,colsample_bytree=0.6,
                      reg_lambda=5.0,random_state=seed,n_jobs=4,verbosity=0)
ARMS={"glm":glm,"hgb":hgb,"xgb":xgb_reg,"xgb_shallow":xgb_shallow}

def main():
    t0=time.time()
    raw=data.load_raw(); panel=data.build_panel(raw); tf=team.team_fixtures(raw)
    feats,names=features.build(panel,tf)
    squads={s:measure.neutral_squads(data.with_blanks(panel,s),k=K,seed=1)
            for s in C.HOLDOUT_SEASONS}
    print(f"[{time.time()-t0:5.0f}s] {K} shared squads/season",flush=True)
    scores={}
    for arm,fn in ARMS.items():
        rates._reg=fn
        parts=[]
        for s in C.HOLDOUT_SEASONS:
            out=model.walk_forward(feats,tf,names,s)
            g=data.with_blanks(panel,s).merge(
                out[["season","gameweek",C.PLAYER_KEY,"xp"]],
                on=["season","gameweek",C.PLAYER_KEY],how="left")
            g["xp"]=g["xp"].fillna(0.0); g.loc[g["fixtures"]<=0,"xp"]=0.0
            d=measure.fixed_squad_score(g,squads[s],"xp"); d["season"]=s
            parts.append(d)
        scores[arm]=pd.concat(parts)
        print(f"[{time.time()-t0:5.0f}s] {arm}: {scores[arm].value.mean():.3f}/gw",flush=True)
    ON=("squad","gameweek","season")
    print("\n"+"="*80); print("RATE MODEL FAMILY, on the calibrated instrument (MDE 13.6)")
    print("="*80)
    print(f"{'arm':<14}{'mean/gw':>10}{'vs GLM (season-eq)':>22}{'95% CI':>22}{'t':>8}")
    for arm in ARMS:
        if arm=="glm":
            print(f"{arm:<14}{scores[arm].value.mean():>10.3f}{'(reference)':>22}")
            continue
        r=measure.paired(scores[arm],scores["glm"],on=ON)
        s=measure.per_season_equivalent(r)
        print(f"{arm:<14}{scores[arm].value.mean():>10.3f}{s['mean']:>22.1f}"
              f"  [{s['ci_lo']:>+7.1f},{s['ci_hi']:>+7.1f}]{r['t']:>8.2f}")
    print(f"\n  S12 recorded the GBDT at -194/season on the OLD instrument (MDE 123).")
    pickle.dump(scores,open(C.cache("fpl2_xgb.pkl"),"wb"))
    print("\ndone")

if __name__=="__main__": main()
