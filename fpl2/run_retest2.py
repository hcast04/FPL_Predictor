#!/usr/bin/env python3
"""
F32 and F26 on metric 2 (MDE 36.7). Cheap test first, expensive second, with
results persisted after each.

F32  fixture-aware squad build. The SQUAD is the treatment, so arms cannot
     share squads -- common random numbers on the seed instead (same jitter
     perturbs each arm's selection signal, paired on seed). The scrambled arm
     is carried over because on the old instrument it BEAT the real fixture map
     (+85.3 at 3/3 against +15.0 at 1/3, S33); if that survives here it is a
     real and very odd result, and if it vanishes it was noise.

F26  receding-horizon MPC. The POLICY is the treatment, so every arm shares
     identical opening squads and pairs on squad id -- the strongest pairing
     available. Expensive: each MPC season solves 38 MILPs, ~58s at H=4, so K
     is reduced to 8 and the horizon sweep to {2, 4}.
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
OUT=C.cache("fpl2_retest2.pkl")
START=8; TO=12
class _WP:
    def __init__(self,est):
        self.pre=make_pipeline(SimpleImputer(strategy="median"),StandardScaler()); self.est=est
    def fit(self,X,y,sample_weight=None):
        self.est.fit(self.pre.fit_transform(X),y,sample_weight=sample_weight); return self
    def predict(self,X): return self.est.predict(self.pre.transform(X))

def report(name,arms,base,mde=36.7):
    print(f"\n{'='*84}\n{name}   (metric 2, MDE {mde})\n{'='*84}")
    print(f"{'arm':<22}{'n':>5}{'mean':>10}{'95% CI':>22}{'t':>8}{'verdict':>14}")
    rows=[]
    for k,v in arms.items():
        if k==base:
            print(f"{k:<22}{len(v):>5}{v.value.mean():>10.1f}{'(reference)':>22}"); continue
        r=measure.paired(v,arms[base],on=("squad","season"))
        sig=abs(r["t"])>2
        vd="BETTER" if sig and r["mean"]>0 else "WORSE" if sig and r["mean"]<0 else "no effect"
        print(f"{k:<22}{r['n']:>5}{r['mean']:>10.1f}  [{r['ci_lo']:>+7.1f},{r['ci_hi']:>+7.1f}]"
              f"{r['t']:>8.2f}{vd:>14}")
        rows.append(dict(test=name,arm=k,**{x:r[x] for x in ("n","mean","ci_lo","ci_hi","t")},verdict=vd))
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
        m=model.ComponentModel(feats=names).fit(feats[feats.season<s],tf[tf.season<s])
        hist=feats[feats.season.eq(s)&(feats.gameweek<START-1)]
        if len(hist)>500: m.refit_defensive(hist)
        fitted[s]=m
        print(f"[{time.time()-t0:5.0f}s] {s} ready",flush=True)

    # ---------------- F32: fixture-aware squad, CRN on seed -----------------
    K=12
    proj={}; scram={}
    for s in C.HOLDOUT_SEASONS:
        for lab,tfx in (("real",tf),("scram",None)):
            t_=tf if tfx is not None else None
            if t_ is None:
                rng=np.random.default_rng(5); t_=tf.copy()
                msk=t_.season.eq(s)
                t_.loc[msk,"opponent"]=rng.permutation(t_.loc[msk,"opponent"].to_numpy())
            tot={}
            for g_ in range(START,TO+1):
                pr=project.project(fitted[s],feats,t_,s,START-1,g_)
                if pr.empty: continue
                for k_,v_ in pr.set_index(C.PLAYER_KEY)["xp"].items():
                    tot[k_]=tot.get(k_,0.0)+float(v_)
            (proj if lab=="real" else scram)[s]=pd.Series(tot)
        print(f"[{time.time()-t0:5.0f}s] F32 projections {s}",flush=True)

    arms={}
    for lab in ("single","fixture","scrambled"):
        parts=[]
        for s in C.HOLDOUT_SEASONS:
            g=grids[s]; rows=[]
            w=g[g.gameweek.eq(START)]
            base=(g[g.gameweek.isin(range(START,TO+1))].groupby(C.PLAYER_KEY)
                  .agg(cost=("cost","first"),position=("position","first"),
                       team=("team","first")).reset_index())
            if lab=="single":
                sig0=g[g.gameweek.eq(START)].set_index(C.PLAYER_KEY)["xp"]*5
            elif lab=="fixture": sig0=proj[s]
            else: sig0=scram[s]
            base["sig0"]=base[C.PLAYER_KEY].map(sig0).fillna(0.0)
            base=base[base.cost.gt(0)]
            for seed in range(K):
                rng=np.random.default_rng(2000+seed)       # SAME per seed, all arms
                p=base.copy()
                p["sig"]=p.sig0*(1.0+0.15*rng.standard_normal(len(p)))
                sq=optimise.optimise_squad(p,"sig",budget=C.BUDGET)
                if sq is None: continue
                r=simulate.simulate_season(g[g.gameweek.ge(START)].copy(),"xp",
                                           opening_squad=sq[C.PLAYER_KEY].tolist())
                if r: rows.append({"squad":seed,"value":float(r.total),"season":s})
            parts.append(pd.DataFrame(rows))
        arms[lab]=pd.concat(parts)
        print(f"[{time.time()-t0:5.0f}s] F32 {lab}",flush=True)
    res["F32"]=arms; pickle.dump(res,open(OUT,"wb"))
    summary+=report("F32  fixture-aware squad (CRN on seed)",arms,"single")

    # ---------------- F26: MPC policy, shared squads ------------------------
    K2=8
    arms={}
    shared={s:measure.neutral_squads(grids[s],k=K2,seed=1) for s in C.HOLDOUT_SEASONS}
    for lab,kw in (("greedy",{}),("mpc_H2",{"policy":"mpc","mpc_horizon":2}),
                   ("mpc_H4",{"policy":"mpc","mpc_horizon":4})):
        parts=[]
        for s in C.HOLDOUT_SEASONS:
            d=measure.restart_score(grids[s],shared[s],"xp",simulate.simulate_season,**kw)
            d["season"]=s; parts.append(d)
            print(f"[{time.time()-t0:5.0f}s] F26 {lab} {s}",flush=True)
        arms[lab]=pd.concat(parts)
    res["F26"]=arms
    summary+=report("F26  receding-horizon MPC (shared squads)",arms,"greedy")
    pickle.dump({"res":res,"summary":pd.DataFrame(summary)},open(OUT,"wb"))
    print(f"\n[{time.time()-t0:.0f}s] done")

if __name__=="__main__": main()
