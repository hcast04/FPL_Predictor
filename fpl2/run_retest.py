#!/usr/bin/env python3
"""
Re-test the four REFUTED squad/policy experiments on metric 2 (MDE 36.7),
which sees transfers. The old instrument (MDE 123) could not resolve any of
them: F33's best arm was +59, F32's control +85, both inside its noise.

PAIRING DIFFERS BY TEST, and this matters:
  F26, F33  -- the policy is the treatment, so every arm can share the SAME
               opening squads. Pair on squad id. Maximum variance reduction.
  F25, F32  -- the SQUAD is the treatment, so squads cannot be shared. Use
               common random numbers instead: the same jitter seed perturbs
               both arms' selection signal, and arms are paired on SEED. The
               squads differ (they must) but the random draw behind them does
               not, which is what cancels.

Results are persisted after EVERY test -- a reporting bug binned 30 minutes of
compute earlier today.
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, measure, model, optimise, project, rates, simulate, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
OUT=C.cache("fpl2_retest.pkl")
START=8
class _WP:
    def __init__(self,est):
        self.pre=make_pipeline(SimpleImputer(strategy="median"),StandardScaler()); self.est=est
    def fit(self,X,y,sample_weight=None):
        self.est.fit(self.pre.fit_transform(X),y,sample_weight=sample_weight); return self
    def predict(self,X): return self.est.predict(self.pre.transform(X))

def save(d):
    pickle.dump(d,open(OUT,"wb"))

def report(name, arms, base_key, mde=36.7):
    print(f"\n{'='*82}\n{name}   (metric 2, MDE {mde})\n{'='*82}")
    print(f"{'arm':<22}{'n':>5}{'mean':>10}{'95% CI':>22}{'t':>8}{'verdict':>14}")
    rows=[]
    for k,v in arms.items():
        if k==base_key:
            print(f"{k:<22}{len(v):>5}{v.value.mean():>10.1f}{'(reference)':>22}")
            continue
        r=measure.paired(v,arms[base_key],on=("squad","season"))
        sig=abs(r["t"])>2
        verdict=("BETTER" if sig and r["mean"]>0 else
                 "WORSE" if sig and r["mean"]<0 else "no effect")
        print(f"{k:<22}{r['n']:>5}{r['mean']:>10.1f}"
              f"  [{r['ci_lo']:>+7.1f},{r['ci_hi']:>+7.1f}]{r['t']:>8.2f}{verdict:>14}")
        rows.append(dict(test=name,arm=k,**{x:r[x] for x in ("n","mean","ci_lo","ci_hi","t")},
                         verdict=verdict))
    return rows

def main():
    t0=time.time(); res={}; summary=[]
    rates._reg=lambda s=0:_WP(PoissonRegressor(alpha=1.0,max_iter=500))
    raw=data.load_raw(); panel=data.build_panel(raw); tf=team.team_fixtures(raw)
    feats,names=features.build(panel,tf)
    grids={}; fitted={}
    for s in C.HOLDOUT_SEASONS:
        out=model.walk_forward(feats,tf,names,s)
        g=data.with_blanks(panel,s).merge(out[["season","gameweek",C.PLAYER_KEY,"xp"]],
            on=["season","gameweek",C.PLAYER_KEY],how="left")
        g["xp"]=g["xp"].fillna(0.0); g.loc[g["fixtures"]<=0,"xp"]=0.0
        grids[s]=g
        print(f"[{time.time()-t0:5.0f}s] forecast {s}",flush=True)

    # ---------- F33: transfer threshold (shared squads, pair on squad) -------
    K=20
    arms={}
    for mg in (0.30,1.0,1.5,2.0):
        parts=[]
        for s in C.HOLDOUT_SEASONS:
            sq=measure.neutral_squads(grids[s],k=K,seed=1)
            d=measure.restart_score(grids[s],sq,"xp",simulate.simulate_season,min_gain=mg)
            d["season"]=s; parts.append(d)
        arms[f"min_gain={mg}"]=pd.concat(parts)
        print(f"[{time.time()-t0:5.0f}s] F33 mg={mg}",flush=True)
    res["F33"]=arms; save(res)
    summary+=report("F33  transfer threshold",arms,"min_gain=0.3")

    # ---------- F25: bench_weight (squad IS the treatment -> CRN on seed) ----
    K=12
    arms={}
    for w in (1.0,0.5,0.25,0.0):
        parts=[]
        for s in C.HOLDOUT_SEASONS:
            g=grids[s]; rows=[]
            for seed in range(K):
                rng=np.random.default_rng(1000+seed)        # SAME per seed, all arms
                gws=[x for x in sorted(g.gameweek.unique()) if START<=x<START+5]
                pool=(g[g.gameweek.isin(gws)].groupby(C.PLAYER_KEY)
                      .agg(xp=("xp","mean"),cost=("cost","first"),
                           position=("position","first"),team=("team","first")).reset_index())
                pool=pool[pool.cost.gt(0)].copy()
                pool["sig"]=pool.xp*(1.0+0.15*rng.standard_normal(len(pool)))
                sq=optimise.optimise_squad(pool,"sig",budget=C.BUDGET,bench_weight=w)
                if sq is None: continue
                r=simulate.simulate_season(g[g.gameweek.ge(START)].copy(),"xp",
                                           opening_squad=sq[C.PLAYER_KEY].tolist())
                if r: rows.append({"squad":seed,"value":float(r.total),"season":s})
            parts.append(pd.DataFrame(rows))
        arms[f"bench_w={w}"]=pd.concat(parts)
        print(f"[{time.time()-t0:5.0f}s] F25 w={w}",flush=True)
    res["F25"]=arms; save(res)
    summary+=report("F25  bench_weight (CRN on seed)",arms,"bench_w=1.0")

    pickle.dump({"res":res,"summary":pd.DataFrame(summary)},open(OUT,"wb"))
    print(f"\n[{time.time()-t0:.0f}s] part 1 done (F33, F25)")

if __name__=="__main__": main()
