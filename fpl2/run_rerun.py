#!/usr/bin/env python3
"""
Re-run the FORECAST-change experiments through the calibrated instrument.

Instrument: metric 1, path-free XI+captain selection from 12 fixed neutral
squads x 38 gameweeks x 3 seasons, paired. Calibrated MDE = 13.6
season-equivalent points, against the old instrument's 123.

Each arm was REFUTED by the old instrument. The question is whether they were
refuted because they do not work, or because the old instrument could not see
them. Positive mean = the arm BEATS the current shipped configuration.
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, measure, minutes as minutes_mod, model, rates, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
K=12
class _WP:
    def __init__(self,est):
        self.pre=make_pipeline(SimpleImputer(strategy="median"),StandardScaler()); self.est=est
    def fit(self,X,y,sample_weight=None):
        self.est.fit(self.pre.fit_transform(X),y,sample_weight=sample_weight); return self
    def predict(self,X): return self.est.predict(self.pre.transform(X))

def set_arm(name):
    """Configure the modules for one arm. Returns a label."""
    features.MARKET=False; features.ZERO_FILL_ABSENT_EXPECTED=False
    minutes_mod.RECENCY_K=4.0; minutes_mod.RECENCY_SPREAD_FLOOR=1.0
    rates.MASK_TARGETS=True; rates.MASK_CONVERSION=True
    if name=="base": pass
    elif name=="F28_market":        features.MARKET=True
    elif name=="F29_spread0.25":    minutes_mod.RECENCY_SPREAD_FLOOR=0.25
    elif name=="F29_spread0.10":    minutes_mod.RECENCY_SPREAD_FLOOR=0.10
    elif name=="M20_zeros":         features.ZERO_FILL_ABSENT_EXPECTED=True
    elif name=="recency_off":       minutes_mod.RECENCY_K=0.0
    elif name=="recency_K2":        minutes_mod.RECENCY_K=2.0
    elif name=="recency_K8":        minutes_mod.RECENCY_K=8.0
    else: raise ValueError(name)
    return name

ARMS=["base","F28_market","F29_spread0.25","F29_spread0.10","M20_zeros",
      "recency_off","recency_K2","recency_K8"]

def main():
    t0=time.time()
    rates._reg=lambda s=0:_WP(PoissonRegressor(alpha=1.0,max_iter=500))
    raw=data.load_raw(); panel=data.build_panel(raw); tf=team.team_fixtures(raw)
    # squads are built ONCE from a neutral signal and reused by every arm,
    # so no arm is measured on a squad that favours it
    base_feats,_=features.build(panel,tf)
    squads={}
    for s in C.HOLDOUT_SEASONS:
        g=data.with_blanks(panel,s)
        squads[s]=measure.neutral_squads(g,k=K,seed=1)
    print(f"[{time.time()-t0:5.0f}s] {K} shared squads per season",flush=True)

    scores={}
    for arm in ARMS:
        set_arm(arm)
        feats,names=features.build(panel,tf)
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
        pickle.dump(scores,open(C.cache("fpl2_rerun_scores.pkl"),"wb"))
        print(f"[{time.time()-t0:5.0f}s] {arm}: mean/gw {scores[arm].value.mean():.3f}"
              f"  (scores persisted)",flush=True)
    set_arm("base")

    ON=("squad","gameweek","season")
    print("\n"+"="*86)
    print("REFUTED EXPERIMENTS, RE-RUN  (positive = beats the shipped config)")
    print("="*86)
    print(f"{'arm':<18}{'n':>6}{'mean/gw':>10}{'season-eq':>12}{'95% CI':>22}{'t':>8}{'verdict':>12}")
    rows=[]
    for arm in ARMS[1:]:
        r=measure.paired(scores[arm],scores["base"],on=ON)
        s=measure.per_season_equivalent(r)
        sig = abs(r["t"])>2
        verdict=("BETTER" if sig and r["mean"]>0 else
                 "WORSE" if sig and r["mean"]<0 else "no effect")
        print(f"{arm:<18}{r['n']:>6}{r['mean']:>10.3f}{s['mean']:>12.1f}"
              f"  [{s['ci_lo']:>+7.1f},{s['ci_hi']:>+7.1f}]{r['t']:>8.2f}{verdict:>12}")
        rows.append({"arm":arm,**s,"verdict":verdict})
    print(f"\n  instrument MDE: 13.6 season-equivalent pts (old instrument: 123)")
    print(f"  'no effect' here means the effect is SMALLER than ~14 pts/season,")
    print(f"  which the old instrument could never have established.")
    pickle.dump({"scores":scores,"summary":pd.DataFrame(rows)},
                open(C.cache("fpl2_rerun.pkl"),"wb"))
    print("\ndone")

if __name__=="__main__": main()
