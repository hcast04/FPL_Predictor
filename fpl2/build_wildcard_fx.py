#!/usr/bin/env python3
"""
Wildcard built on the GW8-12 FIXTURE RUN, not on GW7's single week.

The first build used GW7 xp directly, justified by S29 (opponent identity
carries no ranking value one to three weeks out) plus the absence of any blank
or double in GW8-12. The user objected that Calvert-Lewin has an easy run, and
he is right that the justification does not cover this case: S29 measured a
FORECAST's aggregate ranking, whereas a wildcard commits a squad for five
gameweeks against five specific opponents. GW7's xp encodes GW7's opponent and
nothing else.

So this version projects every player across GW8-12 with `project.project`,
which substitutes each future week's real opponent and venue while freezing
every player-form feature at GW7 -- the construction whose leak test passes at
0.000000000000. Future fixtures are appended to the team-fixture table from
the published list, with lagged strength rewound to GW7.
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
SEASON="2026-27"; FROM=7; TO=12; BUDGET=99.0; MIN_CHANCE=75
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
    # append GW8-12 to the team-fixture table. Only identity is taken from the
    # published list; every lagged strength column is left NaN and refilled
    # from GW7 by project._strength_as_of.
    rows=[]
    for f in fxs:
        e=f.get("event")
        if e and FROM < e <= TO:
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
    print(f"[{time.time()-t0:4.0f}s] appended {len(fut)} future team-fixture rows "
          f"(GW{FROM+1}-{TO})",flush=True)

    m=model.ComponentModel(feats=names).fit(feats[feats.season<SEASON],
                                            tf[tf.season<SEASON])
    hist=feats[feats.season.eq(SEASON)&(feats.gameweek<FROM)]
    if len(hist)>500: m.refit_defensive(hist)
    print(f"[{time.time()-t0:4.0f}s] model fitted",flush=True)

    tot={}
    for g in range(FROM+1,TO+1):
        pr=project.project(m,feats,tf2,SEASON,FROM,g)
        if pr.empty: continue
        s=pr.set_index(C.PLAYER_KEY)["xp"]
        for k,v in s.items(): tot[k]=tot.get(k,0.0)+float(v)
        print(f"[{time.time()-t0:4.0f}s] projected GW{g}: {len(pr)} players, "
              f"mean xp {pr.xp.mean():.3f}",flush=True)
    proj=pd.DataFrame({"player_code":list(tot),"xp5":[tot[k] for k in tot]})

    el=pd.DataFrame([{"code":e["code"],"name":e["web_name"],"cost":e["now_cost"]/10.0,
        "owned":float(e["selected_by_percent"]),"status":e["status"],
        "chance":e["chance_of_playing_next_round"],"team_id":e["team"],
        "etype":e["element_type"]} for e in boot["elements"]])
    el["team"]=el.team_id.map(sn)
    el["position"]=el.etype.map({1:"GK",2:"DEF",3:"MID",4:"FWD"})
    p=el.merge(proj,left_on="code",right_on="player_code",how="inner")
    av=p[(p.status.eq("a"))&(p.chance.isna()|(p.chance>=MIN_CHANCE))].copy()
    av["xp"]=av.xp5/ (TO-FROM)      # per gameweek, for readability
    print(f"\n  pool {len(av)} available, projected over GW{FROM+1}-{TO}\n")

    sq=optimise.optimise_squad(av,"xp5",budget=BUDGET)
    xi=optimise.best_xi(sq,"xp5"); cap=optimise.pick_captain(xi,"xp5")
    s=sq.sort_values(["position","xp5"],ascending=[True,False])
    print("=== FIXTURE-AWARE WILDCARD (GW8-12 projection, £%.1fm) ===" % BUDGET)
    print(f"{'player':<18}{'pos':<5}{'team':<6}{'£m':>6}{'owned%':>8}{'xp/gw':>8}{'5gw':>8}{'XI':>5}")
    for _,r in s.iterrows():
        mark="(C)" if r.code==cap.code else ("yes" if r.code in set(xi.code) else "-")
        print(f"{r['name'][:17]:<18}{r.position:<5}{r.team:<6}{r.cost:>6.1f}{r.owned:>8.1f}"
              f"{r.xp:>8.2f}{r.xp5:>8.2f}{mark:>5}")
    print(f"\n  spend £{sq.cost.sum():.1f}m   5-gw XI xp {xi.xp5.sum():.1f}"
          f"   captain {cap['name']}")
    print(f"  mean ownership: squad {sq.owned.mean():.1f}%  XI {xi.owned.mean():.1f}%")

    # what changed vs the single-week build
    old=pd.read_pickle(C.cache("fpl2_s20_out_fixed.pkl"))
    oxp=old[old.gameweek.eq(7)].groupby("player_code",as_index=False).agg(xp=("xp","mean"))
    po=el.merge(oxp,left_on="code",right_on="player_code")
    avo=po[(po.status.eq("a"))&(po.chance.isna()|(po.chance>=MIN_CHANCE))]
    sqo=optimise.optimise_squad(avo,"xp",budget=BUDGET)
    print(f"\n  vs the GW7-only build: {15-len(set(sq.code)&set(sqo.code))}/15 changed")
    print(f"    OUT: {', '.join(sorted(el[el.code.isin(set(sqo.code)-set(sq.code))].name))}")
    print(f"    IN : {', '.join(sorted(el[el.code.isin(set(sq.code)-set(sqo.code))].name))}")
    for nm in ("Calvert-Lewin","Haaland","Barry","Evanilson"):
        r=av[av.name.eq(nm)]
        if len(r):
            r=r.iloc[0]
            print(f"    {nm:<15} 5gw xp {r.xp5:>6.2f} (={r.xp:.2f}/gw)  "
                  f"in squad: {r.code in set(sq.code)}")
    pd.to_pickle(av,C.cache("fpl2_wc_fx_pool.pkl"))

if __name__=="__main__": main()
