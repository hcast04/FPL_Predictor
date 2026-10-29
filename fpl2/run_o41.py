#!/usr/bin/env python3
"""
O41: variance-aware minutes recency, against the gate pre-registered 2026-10-22.
Gate 1 (leak) PASSED as a precondition: minutes_sd_r5 max_abs_diff 0.0.
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, minutes as minutes_mod, model, rates, simulate, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor

GRID = (1.0, 0.5, 0.25, 0.1)
LEDGER = {"2023-24": 2143, "2024-25": 2420, "2025-26": 2149}
HAAL, BRUNO = 223094, 141746


class _WP:
    def __init__(self, est):
        self.pre=make_pipeline(SimpleImputer(strategy="median"),StandardScaler()); self.est=est
    def fit(self,X,y,sample_weight=None):
        self.est.fit(self.pre.fit_transform(X),y,sample_weight=sample_weight); return self
    def predict(self,X): return self.est.predict(self.pre.transform(X))


def main():
    t0=time.time()
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    cur = pd.read_pickle(C.cache("fpl2_live2627.pkl"))
    raw = pd.concat([data.load_raw(), cur], ignore_index=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)
    print(f"[{time.time()-t0:5.0f}s] setup, {len(names)} model features", flush=True)

    scram = feats.copy()
    rng = np.random.default_rng(13)
    scram["minutes_sd_r5"] = (scram.groupby(["gameweek","position"])["minutes_sd_r5"]
                              .transform(lambda v: rng.permutation(v.to_numpy())))

    rows, live = [], {}
    for F in GRID:
        minutes_mod.RECENCY_SPREAD_FLOOR = F
        for s in list(C.HOLDOUT_SEASONS) + ["2026-27"]:
            out = model.walk_forward(feats, tf, names, s)
            if s == "2026-27":
                live[F] = out
                continue
            g = data.with_blanks(panel, s).merge(
                out[["season","gameweek",C.PLAYER_KEY,"xp"]],
                on=["season","gameweek",C.PLAYER_KEY], how="left")
            g["xp"]=g["xp"].fillna(0.0); g.loc[g["fixtures"]<=0,"xp"]=0.0
            rows.append(dict(F=F, arm="real", season=s,
                             total=simulate.simulate_season(g,"xp").total))
        print(f"[{time.time()-t0:5.0f}s] F={F} done", flush=True)

    # control arm at the strongest dose only
    minutes_mod.RECENCY_SPREAD_FLOOR = 0.1
    for s in C.HOLDOUT_SEASONS:
        out = model.walk_forward(scram, tf, names, s)
        g = data.with_blanks(panel, s).merge(
            out[["season","gameweek",C.PLAYER_KEY,"xp"]],
            on=["season","gameweek",C.PLAYER_KEY], how="left")
        g["xp"]=g["xp"].fillna(0.0); g.loc[g["fixtures"]<=0,"xp"]=0.0
        rows.append(dict(F=0.1, arm="shuffled", season=s,
                         total=simulate.simulate_season(g,"xp").total))
    R=pd.DataFrame(rows)
    minutes_mod.RECENCY_SPREAD_FLOOR = 1.0

    print("\n"+"="*72); print("GATE 2 -- CORRECTNESS: F=1.0 must reproduce the ledger"); print("="*72)
    ok=True
    for s,exp in LEDGER.items():
        got=float(R[(R.F==1.0)&(R.arm=="real")&(R.season==s)]["total"].iloc[0])
        m=abs(got-exp)<0.5; ok&=m
        print(f"  {s}: {got:7.0f}  ledger {exp}  {'MATCH' if m else 'MISMATCH!!'}")
    print(f"  -> {'PASS' if ok else 'FAIL'}")

    print("\n"+"="*72); print("GATE 3 -- MECHANISM: Haaland minutes + the live captaincy"); print("="*72)
    for F in GRID:
        o=live[F]; h=o[o.player_code.eq(HAAL)&o.fixtures.gt(0)]
        b=o[o.player_code.eq(BRUNO)&o.fixtures.gt(0)]
        who="HAALAND" if h.xp.mean()>b.xp.mean() else "BRUNO"
        print(f"  F={F:<5} Haaland mins {h.expected_minutes.mean():5.1f} (actual 88.4)"
              f"  xp {h.xp.mean():5.3f}  | Bruno xp {b.xp.mean():5.3f}  -> {who}")

    print("\n"+"="*72); print("GATE 4 -- DECISION (vs shipped F=1.0)"); print("="*72)
    piv=R[R.arm=="real"].pivot_table(index="F",columns="season",values="total")
    print(piv.round(0).to_string())
    base=piv.loc[1.0]
    for F in GRID:
        if F==1.0: continue
        d=(piv.loc[F]-base).to_numpy()
        npos=int((d>0).sum())
        v=("SHIP" if d.mean()>=100 and npos==3 else
           "REAL BUT NOT ESTABLISHED" if d.mean()>=100 and npos==2 else "REFUTED")
        print(f"  F={F:<5} {['%+.0f'%x for x in d]}  mean {d.mean():+7.1f}"
              f"  positive {npos}/3  -> {v}")
    sh=R[R.arm=="shuffled"].set_index("season")["total"]
    ds=np.array([sh[s]-base[s] for s in C.HOLDOUT_SEASONS])
    print(f"\n  GATE 5 CONTROL (shuffled sd, F=0.1): {['%+.0f'%x for x in ds]}"
          f"  mean {ds.mean():+.1f}  positive {int((ds>0).sum())}/3")
    print(f"  -> control {'PASSES' if not (ds.mean()>=100 and (ds>0).sum()>=2) else 'FAILS'}")
    pickle.dump({"sweep":R,"live":{k:v[v.player_code.isin([HAAL,BRUNO])] for k,v in live.items()}},
                open(C.cache("fpl2_o41.pkl"),"wb"))
    print(f"\n[{time.time()-t0:.0f}s] cached -> ~/fpl2_o41.pkl")


if __name__ == "__main__":
    main()
