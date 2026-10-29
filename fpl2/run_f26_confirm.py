#!/usr/bin/env python3
"""
Confirm F26. The first rescued finding of the day deserves the hardest look.

The re-test gave mpc_H2 +78.5 (t=3.14) where the ORIGINAL F26 gave -64.3.
That is a SIGN FLIP, and two things changed at once:
  1. the instrument (1 draw -> 8 paired restarts)
  2. the squad construction (model-built opening squad -> shared NEUTRAL squads)
Only (1) is supposed to be the improvement. If the flip depends on (2), it is
an artefact of the new test design, not a rescued finding.

So both regimes are run:
  neutral  -- squads from a neutral signal, shared by every policy (the
              re-test's design; strongest pairing)
  model_xp -- squads built from the model's own xp with CRN jitter, which is
              what the original F26 did. The squad does not depend on the
              POLICY, so greedy and MPC still share it within each restart and
              the pairing survives.

K = 20 restarts, up from 8. Seasons run in parallel: one worker per season,
each fitting the model once rather than refitting per job.
"""
from __future__ import annotations
import sys, time, pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C  # noqa: E402
K=20

def one_season(season):
    import numpy as np, pandas as pd
    from fpl2 import data, features, measure, model, optimise, rates, simulate, team
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import PoissonRegressor
    class W:
        def __init__(s,e): s.pre=make_pipeline(SimpleImputer(strategy="median"),StandardScaler()); s.est=e
        def fit(s,X,y,sample_weight=None):
            s.est.fit(s.pre.fit_transform(X),y,sample_weight=sample_weight); return s
        def predict(s,X): return s.est.predict(s.pre.transform(X))
    rates._reg=lambda sd=0: W(PoissonRegressor(alpha=1.0,max_iter=500))
    raw=data.load_raw(); panel=data.build_panel(raw); tf=team.team_fixtures(raw)
    feats,names=features.build(panel,tf)
    out=model.walk_forward(feats,tf,names,season)
    g=data.with_blanks(panel,season).merge(
        out[["season","gameweek",C.PLAYER_KEY,"xp"]],
        on=["season","gameweek",C.PLAYER_KEY],how="left")
    g["xp"]=g["xp"].fillna(0.0); g.loc[g["fixtures"]<=0,"xp"]=0.0

    regimes={"neutral":measure.neutral_squads(g,k=K,seed=1)}
    # model_xp: squads from the model's own forecast, CRN-jittered per restart.
    # Independent of policy, so greedy and MPC still share each one.
    gws=sorted(g.gameweek.unique())[:5]
    base=(g[g.gameweek.isin(gws)].groupby(C.PLAYER_KEY)
          .agg(xp=("xp","mean"),cost=("cost","first"),
               position=("position","first"),team=("team","first")).reset_index())
    base=base[base.cost.gt(0)]
    mx=[]
    for seed in range(K):
        rng=np.random.default_rng(4000+seed)
        p=base.copy(); p["s"]=p.xp*(1.0+0.15*rng.standard_normal(len(p)))
        sq=optimise.optimise_squad(p,"s",budget=C.BUDGET)
        if sq is not None: mx.append(sq[C.PLAYER_KEY].tolist())
    regimes["model_xp"]=mx

    rows=[]
    for rg,squads in regimes.items():
        pols=[("greedy",{}),("mpc_H2",{"policy":"mpc","mpc_horizon":2})]
        if rg=="neutral": pols.append(("mpc_H4",{"policy":"mpc","mpc_horizon":4}))
        for pol,kw in pols:
            for i,sq in enumerate(squads):
                r=simulate.simulate_season(g,"xp",opening_squad=list(sq),**kw)
                if r: rows.append({"season":season,"regime":rg,"policy":pol,
                                   "squad":i,"value":float(r.total)})
    return pd.DataFrame(rows)

def main():
    t0=time.time()
    with ProcessPoolExecutor(max_workers=3) as ex:
        parts=list(ex.map(one_season,list(C.HOLDOUT_SEASONS)))
    R=pd.concat(parts,ignore_index=True)
    pickle.dump(R,open(C.cache("fpl2_f26_confirm.pkl"),"wb"))
    from fpl2 import measure
    print(f"[{time.time()-t0:.0f}s] all seasons done, n={len(R)}\n")
    for rg in ("neutral","model_xp"):
        sub=R[R.regime.eq(rg)]
        if sub.empty: continue
        g=sub[sub.policy.eq("greedy")][["season","squad","value"]]
        print(f"{'='*80}\nF26 CONFIRM -- {rg} squads, K={K}   (metric 2, MDE 36.7)\n{'='*80}")
        print(f"  greedy reference mean {g.value.mean():.1f}")
        for pol in ("mpc_H2","mpc_H4"):
            m=sub[sub.policy.eq(pol)][["season","squad","value"]]
            if m.empty: continue
            r=measure.paired(m,g,on=("squad","season"))
            sig=abs(r["t"])>2
            vd="BETTER" if sig and r["mean"]>0 else "WORSE" if sig and r["mean"]<0 else "no effect"
            print(f"  {pol:<8} n={r['n']:>3}  {r['mean']:+8.1f}"
                  f"  [{r['ci_lo']:+7.1f},{r['ci_hi']:+7.1f}]  t={r['t']:>5.2f}  {vd}")
        print(f"  per season (mpc_H2 - greedy):")
        for s in C.HOLDOUT_SEASONS:
            a=sub[(sub.policy.eq("mpc_H2"))&(sub.season.eq(s))].set_index("squad").value
            b=sub[(sub.policy.eq("greedy"))&(sub.season.eq(s))].set_index("squad").value
            d=(a-b).dropna()
            if len(d): print(f"    {s}: {d.mean():+7.1f}  (n={len(d)})")
        print()
    print("  original F26 on the old instrument: -64.3 (0/3)")
    print("  re-test at K=8, neutral squads:     +78.5 (t=3.14)")
    print("\ndone")

if __name__=="__main__": main()
