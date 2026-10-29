#!/usr/bin/env python3
"""
Re-measure the two load-bearing numbers on metric 2 (MDE 36.7).

Both came from the OLD instrument, which metric 2's calibration showed has a
within-arm restart sd of 102 points -- so both were single draws from a
distribution wide enough to produce them by chance:

  "model 2116 vs legacy 2215"  -> a 99-point gap, ~1 sd of one draw
  "chips are worth +87/season" (S25)

Three retractions today (S33, S34, S12's magnitude) all had this shape.

TWO COMPARISONS for model vs legacy, because they answer different questions:
  shared -- both signals handed the SAME neutral squads. Isolates "given a
            squad, which signal plays it better". Strongest pairing.
  own    -- each signal builds its OWN squad from its own xp, CRN on seed.
            This is the comparison the 2116-vs-2215 claim actually made.
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import chips as chips_mod  # noqa: E402
from fpl2 import config as C, data, features, legacy, measure, model, optimise, rates, simulate, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
OUT=C.cache("fpl2_standing2.pkl")
K=20
class _WP:
    def __init__(self,est):
        self.pre=make_pipeline(SimpleImputer(strategy="median"),StandardScaler()); self.est=est
    def fit(self,X,y,sample_weight=None):
        self.est.fit(self.pre.fit_transform(X),y,sample_weight=sample_weight); return self
    def predict(self,X): return self.est.predict(self.pre.transform(X))

def report(name,arms,base,mde=36.7):
    print(f"\n{'='*84}\n{name}   (metric 2, MDE {mde})\n{'='*84}")
    print(f"{'arm':<24}{'n':>5}{'mean':>10}{'95% CI':>22}{'t':>8}{'verdict':>14}")
    rows=[]
    for k,v in arms.items():
        if k==base:
            print(f"{k:<24}{len(v):>5}{v.value.mean():>10.1f}{'(reference)':>22}"); continue
        r=measure.paired(v,arms[base],on=("squad","season"))
        sig=abs(r["t"])>2
        vd="BETTER" if sig and r["mean"]>0 else "WORSE" if sig and r["mean"]<0 else "no effect"
        print(f"{k:<24}{r['n']:>5}{r['mean']:>10.1f}  [{r['ci_lo']:>+7.1f},{r['ci_hi']:>+7.1f}]"
              f"{r['t']:>8.2f}{vd:>14}")
        rows.append(dict(test=name,arm=k,**{x:r[x] for x in ("n","mean","ci_lo","ci_hi","t")},verdict=vd))
    return rows

def main():
    t0=time.time(); res={}; summary=[]
    rates._reg=lambda s=0:_WP(PoissonRegressor(alpha=1.0,max_iter=500))
    raw=data.load_raw(); panel=data.build_panel(raw); tf=team.team_fixtures(raw)
    feats,names=features.build(panel,tf)
    grids={}
    for s in C.HOLDOUT_SEASONS:
        g0=data.with_blanks(panel,s)
        out=model.walk_forward(feats,tf,names,s)
        g=g0.merge(out[["season","gameweek",C.PLAYER_KEY,"xp"]],
                   on=["season","gameweek",C.PLAYER_KEY],how="left")
        g["xp"]=g["xp"].fillna(0.0); g.loc[g["fixtures"]<=0,"xp"]=0.0
        lg=legacy.attach(g0)[["season","gameweek",C.PLAYER_KEY,"xp"]].rename(columns={"xp":"xp_leg"})
        g=g.merge(lg,on=["season","gameweek",C.PLAYER_KEY],how="left")
        g["xp_leg"]=g["xp_leg"].fillna(0.0); g.loc[g["fixtures"]<=0,"xp_leg"]=0.0
        grids[s]=g
        print(f"[{time.time()-t0:5.0f}s] {s} ready",flush=True)
    shared={s:measure.neutral_squads(grids[s],k=K,seed=1) for s in C.HOLDOUT_SEASONS}

    # ---- 1. model vs legacy, SHARED squads ---------------------------------
    arms={}
    for lab,sig in (("model","xp"),("legacy","xp_leg")):
        parts=[]
        for s in C.HOLDOUT_SEASONS:
            d=measure.restart_score(grids[s],shared[s],sig,simulate.simulate_season)
            d["season"]=s; parts.append(d)
        arms[lab]=pd.concat(parts)
        print(f"[{time.time()-t0:5.0f}s] shared {lab}",flush=True)
    res["shared"]=arms; pickle.dump(res,open(OUT,"wb"))
    summary+=report("MODEL vs LEGACY -- shared squads",arms,"legacy")

    # ---- 2. model vs legacy, each builds its OWN squad (CRN on seed) -------
    arms={}
    for lab,sig in (("model","xp"),("legacy","xp_leg")):
        parts=[]
        for s in C.HOLDOUT_SEASONS:
            g=grids[s]; rows=[]
            gws=sorted(g.gameweek.unique())[:5]
            base=(g[g.gameweek.isin(gws)].groupby(C.PLAYER_KEY)
                  .agg(sig0=(sig,"mean"),cost=("cost","first"),
                       position=("position","first"),team=("team","first")).reset_index())
            base=base[base.cost.gt(0)]
            for seed in range(K):
                rng=np.random.default_rng(3000+seed)      # SAME per seed, both arms
                p=base.copy()
                p["s"]=p.sig0*(1.0+0.15*rng.standard_normal(len(p)))
                sq=optimise.optimise_squad(p,"s",budget=C.BUDGET)
                if sq is None: continue
                r=simulate.simulate_season(g,sig,opening_squad=sq[C.PLAYER_KEY].tolist())
                if r: rows.append({"squad":seed,"value":float(r.total),"season":s})
            parts.append(pd.DataFrame(rows))
        arms[lab]=pd.concat(parts)
        print(f"[{time.time()-t0:5.0f}s] own-squad {lab}",flush=True)
    res["own"]=arms; pickle.dump(res,open(OUT,"wb"))
    summary+=report("MODEL vs LEGACY -- each builds its own squad",arms,"legacy")

    # ---- 3. chips (S25's +87) ----------------------------------------------
    arms={}
    for lab in ("no_chips","chips"):
        parts=[]
        for s in C.HOLDOUT_SEASONS:
            g=grids[s]
            kw={} if lab=="no_chips" else {"chip_policy":chips_mod.rolling_policy(g,"xp")}
            d=measure.restart_score(g,shared[s],"xp",simulate.simulate_season,**kw)
            d["season"]=s; parts.append(d)
        arms[lab]=pd.concat(parts)
        print(f"[{time.time()-t0:5.0f}s] {lab}",flush=True)
    res["chips"]=arms
    summary+=report("CHIPS -- S25 claimed +87/season",arms,"no_chips")
    pickle.dump({"res":res,"summary":pd.DataFrame(summary)},open(OUT,"wb"))
    print(f"\n[{time.time()-t0:.0f}s] done")

if __name__=="__main__": main()
