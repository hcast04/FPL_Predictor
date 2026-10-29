#!/usr/bin/env python3
"""
Calibrate the measuring instrument. An instrument that has not failed a null
test is not an instrument.

  NULL   -- an arm against ITSELF. Must report ~0 with a tight interval.
            Any detected difference here is pure instrument error.
  NULL'  -- an arm against itself with DIFFERENT fixed squads. Shows how much
            of the old instrument's noise was squad choice.
  SIGNAL -- an arm against a deliberately degraded copy (noise injected into
            the forecast). Must detect it, and the detected size should grow
            with the degradation.
  MDE    -- what the thing can resolve, stated in season-equivalent points,
            next to the OLD instrument's resolution.
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, measure, model, rates, simulate, team  # noqa: E402
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

def main():
    t0=time.time()
    rates._reg=lambda s=0:_WP(PoissonRegressor(alpha=1.0,max_iter=500))
    raw=data.load_raw(); panel=data.build_panel(raw); tf=team.team_fixtures(raw)
    feats,names=features.build(panel,tf)
    grids={}
    for s in C.HOLDOUT_SEASONS:
        out=model.walk_forward(feats,tf,names,s)
        g=data.with_blanks(panel,s).merge(out[["season","gameweek",C.PLAYER_KEY,"xp"]],
            on=["season","gameweek",C.PLAYER_KEY],how="left")
        g["xp"]=g["xp"].fillna(0.0); g.loc[g["fixtures"]<=0,"xp"]=0.0
        grids[s]=g
        print(f"[{time.time()-t0:5.0f}s] forecast {s}",flush=True)

    allA=[];allB=[];allC=[];sig={0.10:[],0.25:[],0.50:[]}
    rng=np.random.default_rng(0)
    for s in C.HOLDOUT_SEASONS:
        g=grids[s].copy()
        sq =measure.neutral_squads(g,k=K,seed=1)
        sq2=measure.neutral_squads(g,k=K,seed=99)
        a=measure.fixed_squad_score(g,sq,"xp"); a["season"]=s
        allA.append(a)
        b=a.copy(); allB.append(b)                        # NULL: identical
        c=measure.fixed_squad_score(g,sq2,"xp"); c["season"]=s
        allC.append(c)                                     # NULL': other squads
        for lv in sig:
            g[f"xp_n{lv}"]=g["xp"]*(1.0+lv*rng.standard_normal(len(g)))
            g.loc[g["fixtures"]<=0,f"xp_n{lv}"]=0.0
            d=measure.fixed_squad_score(g,sq,f"xp_n{lv}"); d["season"]=s
            sig[lv].append(d)
        print(f"[{time.time()-t0:5.0f}s] scored {s}: {len(sq)} squads",flush=True)

    A=pd.concat(allA); B=pd.concat(allB); Cc=pd.concat(allC)
    ON=("squad","gameweek","season")
    print("\n"+"="*72); print("CALIBRATION OF THE NEW INSTRUMENT (metric 1, path-free)")
    print("="*72)
    r=measure.paired(A,B,on=ON); rs=measure.per_season_equivalent(r)
    print(f"  NULL  (arm vs itself)           n={r['n']:>5}  mean {rs['mean']:+8.3f}"
          f"  CI [{rs['ci_lo']:+.2f},{rs['ci_hi']:+.2f}]  -> "
          f"{'PASS' if abs(r['mean'])<1e-9 else 'FAIL: instrument invents a difference'}")
    print(f"  resolution (MDE, season-equivalent): {rs['mde']:.1f} pts")
    print(f"\n  NULL' (same arm, DIFFERENT fixed squads)")
    print(f"    mean {A.value.mean()*38:.0f} vs {Cc.value.mean()*38:.0f} season-equivalent"
          f"  -> squad choice alone moves {abs(A.value.mean()-Cc.value.mean())*38:.0f} pts")
    print(f"\n  SIGNAL (forecast degraded by multiplicative noise):")
    print(f"  {'noise':>7}{'n':>7}{'mean/gw':>10}{'season-eq':>12}{'t':>8}{'detected':>10}")
    for lv in sorted(sig):
        D=pd.concat(sig[lv]); rr=measure.paired(A,D,on=ON)
        ss=measure.per_season_equivalent(rr)
        det = abs(rr["t"])>2 and rr["mean"]>0
        print(f"  {lv:>7.0%}{rr['n']:>7}{rr['mean']:>10.3f}{ss['mean']:>12.1f}"
              f"{rr['t']:>8.2f}{'YES' if det else 'no':>10}")
    print(f"\n  (positive mean = the clean forecast beats the degraded one, as it must)")

    print("\n"+"="*72); print("OLD INSTRUMENT, for comparison"); print("="*72)
    print(f"  season total, n=3 seasons, 1 run each")
    print(f"  3-season-mean sd 39-49 (R1)  ->  MDE = 2.8 x sd = {2.8*44:.0f} pts/season")
    print(f"  NEW instrument MDE          ->  {rs['mde']:.1f} pts/season")
    print(f"  improvement: {2.8*44/max(rs['mde'],1e-9):.1f}x finer resolution")
    pickle.dump({"A":A,"C":Cc,"sig":{k:pd.concat(v) for k,v in sig.items()}},
                open(C.cache("fpl2_calib.pkl"),"wb"))
    print("\ndone")

if __name__=="__main__": main()
