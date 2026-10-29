#!/usr/bin/env python3
"""
F28 at higher power -- the ONLY undecided result left.

On metric 1 with K=12 squads it came back +6.1 season-equivalent, t=1.68,
CI [-1.0, +13.2]: mostly positive, not resolvable. It matters more than its
size suggests, because S29 isolated near-future minutes and availability as the
entire remaining headroom, and F28's transfer-market signal is the only proxy
for that which has any history to test on.

K=40 squads instead of 12 shrinks the standard error by sqrt(40/12) = 1.8x,
which should resolve it either way. Seasons are run in PARALLEL here -- the
earlier scripts looped them sequentially and wasted two thirds of the machine.
"""
from __future__ import annotations
import sys, time, pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, measure, model, rates, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
K=40

def _wp(est):
    class W:
        def __init__(s): s.pre=make_pipeline(SimpleImputer(strategy="median"),StandardScaler()); s.est=est
        def fit(s,X,y,sample_weight=None):
            s.est.fit(s.pre.fit_transform(X),y,sample_weight=sample_weight); return s
        def predict(s,X): return s.est.predict(s.pre.transform(X))
    return W()

def one_season(args):
    season, market = args
    from fpl2 import config as C, data, features, measure, model, rates, team
    rates._reg=lambda s=0:_wp(PoissonRegressor(alpha=1.0,max_iter=500))
    features.MARKET=market
    raw=data.load_raw(); panel=data.build_panel(raw); tf=team.team_fixtures(raw)
    feats,names=features.build(panel,tf)
    g0=data.with_blanks(panel,season)
    squads=measure.neutral_squads(g0,k=K,seed=1)      # seed fixed -> same for both arms
    out=model.walk_forward(feats,tf,names,season)
    g=g0.merge(out[["season","gameweek",C.PLAYER_KEY,"xp"]],
               on=["season","gameweek",C.PLAYER_KEY],how="left")
    g["xp"]=g["xp"].fillna(0.0); g.loc[g["fixtures"]<=0,"xp"]=0.0
    d=measure.fixed_squad_score(g,squads,"xp"); d["season"]=season
    return season, market, d

def main():
    t0=time.time()
    jobs=[(s,m) for m in (False,True) for s in C.HOLDOUT_SEASONS]
    parts={False:[],True:[]}
    with ProcessPoolExecutor(max_workers=6) as ex:
        for season,market,d in ex.map(one_season,jobs):
            parts[market].append(d)
            print(f"[{time.time()-t0:5.0f}s] {season} market={market}: "
                  f"{d.value.mean():.3f}/gw",flush=True)
    base=pd.concat(parts[False]); mkt=pd.concat(parts[True])
    ON=("squad","gameweek","season")
    r=measure.paired(mkt,base,on=ON); s=measure.per_season_equivalent(r)
    print("\n"+"="*74); print(f"F28 AT HIGHER POWER  (K={K} squads, n={r['n']})"); print("="*74)
    print(f"  market - base: {s['mean']:+.1f} season-equivalent")
    print(f"  95% CI       : [{s['ci_lo']:+.1f}, {s['ci_hi']:+.1f}]")
    print(f"  t            : {r['t']:.2f}")
    print(f"  MDE          : {s['mde']:.1f}")
    sig=abs(r["t"])>2
    print(f"  -> {'BETTER' if sig and r['mean']>0 else 'WORSE' if sig else 'still no effect'}")
    print(f"\n  at K=12 it was +6.1, t=1.68, CI [-1.0, +13.2]")
    pickle.dump({"base":base,"market":mkt},open(C.cache("fpl2_f28power.pkl"),"wb"))
    print("\ndone")

if __name__=="__main__": main()
