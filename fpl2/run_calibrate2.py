#!/usr/bin/env python3
"""
Calibrate METRIC 2: season totals WITH transfers, averaged over K restarts.

Metric 1 is path-free by construction and therefore blind to exactly the thing
S32 says costs us most -- the transfer policy (-83.3 end to end vs legacy,
against -19.7 on the opening squad alone). Metric 2 keeps transfers in and
attacks the variance a different way: K opening squads instead of one, paired
on squad id so the draw cancels.

Same battery as metric 1, including the lesson from it: the informative NULL is
the same arm measured through an INDEPENDENT draw of the nuisance variable, not
an arm against a copy of itself (which is zero by construction and proves
nothing). Only NULL' is run here.
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
K=20
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

    rng=np.random.default_rng(0)
    A=[];Cc=[];sig={0.10:[],0.25:[],0.50:[]}
    for s in C.HOLDOUT_SEASONS:
        g=grids[s].copy()
        sq =measure.neutral_squads(g,k=K,seed=1)
        sq2=measure.neutral_squads(g,k=K,seed=99)
        a=measure.restart_score(g,sq,"xp",simulate.simulate_season); a["season"]=s
        A.append(a)
        c=measure.restart_score(g,sq2,"xp",simulate.simulate_season); c["season"]=s
        Cc.append(c)
        for lv in sig:
            col=f"xp_n{lv}"
            g[col]=g["xp"]*(1.0+lv*rng.standard_normal(len(g)))
            g.loc[g["fixtures"]<=0,col]=0.0
            d=measure.restart_score(g,sq,col,simulate.simulate_season); d["season"]=s
            sig[lv].append(d)
        print(f"[{time.time()-t0:5.0f}s] {s}: {len(sq)} restarts scored",flush=True)

    A=pd.concat(A); Cc=pd.concat(Cc); ON=("squad","season")
    print("\n"+"="*72); print("CALIBRATION OF METRIC 2 (with transfers, K restarts)")
    print("="*72)
    print(f"  NULL' (same arm, DIFFERENT restart squads)")
    print(f"    {A.value.mean():.0f} vs {Cc.value.mean():.0f}"
          f"   -> restart choice alone moves {abs(A.value.mean()-Cc.value.mean()):.0f} pts")
    print(f"    within-arm sd across restarts: {A.value.std(ddof=1):.0f} pts"
          f"  (one draw of the OLD instrument was a single sample of this)")
    print(f"\n  SIGNAL (forecast degraded):")
    print(f"  {'noise':>7}{'n':>6}{'mean':>10}{'95% CI':>22}{'t':>8}{'MDE':>9}{'detected':>10}")
    mdes=[]
    for lv in sorted(sig):
        D=pd.concat(sig[lv]); r=measure.paired(A,D,on=ON)
        det=abs(r["t"])>2 and r["mean"]>0; mdes.append(r["mde"])
        print(f"  {lv:>7.0%}{r['n']:>6}{r['mean']:>10.1f}"
              f"  [{r['ci_lo']:>+7.1f},{r['ci_hi']:>+7.1f}]{r['t']:>8.2f}"
              f"{r['mde']:>9.1f}{'YES' if det else 'no':>10}")
    print(f"\n  METRIC 2 MDE  {min(mdes):.1f} pts/season")
    print(f"  metric 1 MDE   13.6")
    print(f"  OLD instrument 123.0")
    print(f"\n  metric 2 sees TRANSFERS, which metric 1 cannot. If its MDE is")
    print(f"  usable, F25/F32/F26/F33 become re-testable.")
    pickle.dump({"A":A,"C":Cc,"sig":{k:pd.concat(v) for k,v in sig.items()}},
                open(C.cache("fpl2_calib2.pkl"),"wb"))
    print("\ndone")

if __name__=="__main__": main()
