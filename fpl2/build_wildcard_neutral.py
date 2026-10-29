#!/usr/bin/env python3
"""
Third wildcard variant: five-gameweek projection, OPPONENT NEUTRALISED.

STATUS: built on an UN-GATED hypothesis (O46). S33 measured the scrambled
control at +85.3 (3/3) against the single-week build while the real fixture map
managed only +15.0 (1/3) -- so the averaging helps and the fixture information
does not. Averaging the projection over many random opponent draws is the
live equivalent of that control. It has NOT been gated as a candidate and is
offered as an option, not a recommendation.
"""
from __future__ import annotations
import json, sys, time, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, model, optimise, project, rates, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
API="https://fantasy.premierleague.com/api"; H={"User-Agent":"Mozilla/5.0"}
SEASON="2026-27"; FROM=7; TO=12; BUDGET=99.0; MIN_CHANCE=75; DRAWS=10
def get(u): return json.load(urllib.request.urlopen(urllib.request.Request(u,headers=H),timeout=60))

class _WP:
    def __init__(self,est):
        self.pre=make_pipeline(SimpleImputer(strategy="median"),StandardScaler()); self.est=est
    def fit(self,X,y,sample_weight=None):
        self.est.fit(self.pre.fit_transform(X),y,sample_weight=sample_weight); return self
    def predict(self,X): return self.est.predict(self.pre.transform(X))

def main():
    t0=time.time()
    rates._reg=lambda s=0:_WP(PoissonRegressor(alpha=1.0,max_iter=500))
    cur=pd.read_pickle(C.cache("fpl2_live2627.pkl"))
    raw=pd.concat([data.load_raw(),cur],ignore_index=True)
    panel=data.build_panel(raw); tf=team.team_fixtures(raw)
    feats,names=features.build(panel,tf)
    boot=get(f"{API}/bootstrap-static/"); fxs=get(f"{API}/fixtures/")
    sn={t["id"]:t["short_name"] for t in boot["teams"]}
    rows=[]
    for f in fxs:
        e=f.get("event")
        if e and FROM<e<=TO:
            for h in (True,False):
                rows.append({"season":SEASON,"gameweek":e,"fixture_id":f["id"],
                    "team":sn[f["team_h"] if h else f["team_a"]],
                    "opponent":sn[f["team_a"] if h else f["team_h"]],
                    "was_home":h,"is_home":float(h),
                    "scored":np.nan,"conceded":np.nan,"xgc":np.nan})
    fut=pd.DataFrame(rows)
    for c in tf.columns:
        if c not in fut.columns: fut[c]=np.nan
    fut["t"]=fut["gameweek"]
    tf2=pd.concat([tf,fut[tf.columns]],ignore_index=True)
    m=model.ComponentModel(feats=names).fit(feats[feats.season<SEASON],tf[tf.season<SEASON])
    hist=feats[feats.season.eq(SEASON)&(feats.gameweek<FROM)]
    if len(hist)>500: m.refit_defensive(hist)
    print(f"[{time.time()-t0:4.0f}s] fitted; averaging {DRAWS} opponent draws",flush=True)

    acc={}
    for d in range(DRAWS):
        rng=np.random.default_rng(100+d); t3=tf2.copy()
        msk=t3.season.eq(SEASON)&t3.gameweek.gt(FROM)
        t3.loc[msk,"opponent"]=rng.permutation(t3.loc[msk,"opponent"].to_numpy())
        for g in range(FROM+1,TO+1):
            pr=project.project(m,feats,t3,SEASON,FROM,g)
            if pr.empty: continue
            for k,v in pr.set_index(C.PLAYER_KEY)["xp"].items():
                acc[k]=acc.get(k,0.0)+float(v)/DRAWS
        print(f"[{time.time()-t0:4.0f}s]   draw {d+1}/{DRAWS}",flush=True)
    proj=pd.DataFrame({"player_code":list(acc),"xp5":[acc[k] for k in acc]})

    el=pd.DataFrame([{"code":e["code"],"name":e["web_name"],"cost":e["now_cost"]/10.0,
        "owned":float(e["selected_by_percent"]),"status":e["status"],
        "chance":e["chance_of_playing_next_round"],"team_id":e["team"],
        "etype":e["element_type"]} for e in boot["elements"]])
    el["team"]=el.team_id.map(sn); el["position"]=el.etype.map({1:"GK",2:"DEF",3:"MID",4:"FWD"})
    p=el.merge(proj,left_on="code",right_on="player_code",how="inner")
    av=p[(p.status.eq("a"))&(p.chance.isna()|(p.chance>=MIN_CHANCE))].copy()
    av["xp"]=av.xp5/(TO-FROM)
    sq=optimise.optimise_squad(av,"xp5",budget=BUDGET)
    xi=optimise.best_xi(sq,"xp5"); cap=optimise.pick_captain(xi,"xp5")
    print(f"\n=== OPPONENT-NEUTRAL WILDCARD (UN-GATED, O46) £{BUDGET:.1f}m ===")
    print(f"{'player':<18}{'pos':<5}{'team':<6}{'£m':>6}{'owned%':>8}{'xp/gw':>8}{'XI':>5}")
    for _,r in sq.sort_values(["position","xp5"],ascending=[True,False]).iterrows():
        mark="(C)" if r.code==cap.code else ("yes" if r.code in set(xi.code) else "-")
        print(f"{r['name'][:17]:<18}{r.position:<5}{r.team:<6}{r.cost:>6.1f}{r.owned:>8.1f}"
              f"{r.xp:>8.2f}{mark:>5}")
    print(f"\n  spend £{sq.cost.sum():.1f}m  captain {cap['name']}"
          f"  mean ownership squad {sq.owned.mean():.1f}%  XI {xi.owned.mean():.1f}%")
    fxp=pd.read_pickle(C.cache("fpl2_wc_fx_pool.pkl"))
    sqf=optimise.optimise_squad(fxp,"xp5",budget=BUDGET)
    print(f"\n  vs the fixture-aware squad: {15-len(set(sq.code)&set(sqf.code))}/15 changed")
    print(f"    OUT: {', '.join(sorted(el[el.code.isin(set(sqf.code)-set(sq.code))].name))}")
    print(f"    IN : {', '.join(sorted(el[el.code.isin(set(sq.code)-set(sqf.code))].name))}")
    pd.to_pickle(av,C.cache("fpl2_wc_neutral_pool.pkl"))

if __name__=="__main__": main()
