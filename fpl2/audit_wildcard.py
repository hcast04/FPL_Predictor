#!/usr/bin/env python3
"""
Adversarial audit of the GW8 wildcard. Every attack is aimed at a bias this
project has already MEASURED, not at a general worry.
"""
from __future__ import annotations
import json, sys, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, optimise, rates, scoring  # noqa: E402

API="https://fantasy.premierleague.com/api"; HDR={"User-Agent":"Mozilla/5.0"}
BUDGET=100.1; MIN_CHANCE=75
def get(u): return json.load(urllib.request.urlopen(urllib.request.Request(u,headers=HDR),timeout=60))

out = pd.read_pickle(C.cache("fpl2_s20_out.pkl"))
boot = get(f"{API}/bootstrap-static/")
el = pd.DataFrame([{"code":e["code"],"name":e["web_name"],"cost":e["now_cost"]/10.0,
    "owned":float(e["selected_by_percent"]),"status":e["status"],
    "chance":e["chance_of_playing_next_round"],"team_id":e["team"],
    "etype":e["element_type"]} for e in boot["elements"]])
teams={t["id"]:t["short_name"] for t in boot["teams"]}
el["team"]=el.team_id.map(teams)
el["position"]=el.etype.map({1:"GK",2:"DEF",3:"MID",4:"FWD"})

def pool_at(gw):
    xp=(out[out.gameweek.eq(gw)].groupby("player_code",as_index=False).agg(xp=("xp","mean")))
    p=el.merge(xp,left_on="code",right_on="player_code",how="inner")
    return p[(p.status.eq("a"))&(p.chance.isna()|(p.chance>=MIN_CHANCE))].copy()

base=pool_at(7); sq=optimise.optimise_squad(base,"xp",budget=BUDGET)
xi=optimise.best_xi(sq,"xp")

print("="*76); print("ATTACK 1 -- STABILITY: is this squad a signal or one week of noise?")
print("="*76)
sets={}
sets[7]=set(sq.code)
for gw in (5,6):
    pp=pool_at(gw); ss=optimise.optimise_squad(pp,"xp",budget=BUDGET)
    sets[gw]=set(ss.code)
for gw in (5,6,7):
    print(f"  built on GW{gw} forecast: {len(sets[gw]&sets[7])}/15 players shared with the GW7 squad")
core=sets[5]&sets[6]&sets[7]
print(f"\n  players surviving ALL THREE builds: {len(core)}/15")
print("  " + ", ".join(sorted(el[el.code.isin(core)].name)))
churn=sorted(sets[7]-core)
print(f"\n  in the GW7 squad but NOT stable: " +
      ", ".join(sorted(el[el.code.isin(churn)].name)))
print(f"  -> {len(core)}/15 is the part you should trust; the rest is week-to-week noise")

print("\n"+"="*76); print("ATTACK 2 -- S18: clean sheets are OVER-predicted for the best defences")
print("="*76)
P=rates.RateModel.PREFIX
w7=out[out.gameweek.eq(7)].set_index("player_code")
d=xi.merge(w7[["clean_sheet_prob","p_sixty","expected_minutes"]],left_on="code",
           right_index=True,how="left")
defs=d[d.position.isin(["GK","DEF"])]
print(f"  XI contains {len(defs)} GK/DEF.  mean clean_sheet_prob {defs.clean_sheet_prob.mean():.3f}")
print(f"  S18 measured: predicted 0.542 -> realised 0.371 (ratio 1.46) at the strong-defence end")
hi=defs[defs.clean_sheet_prob>0.35]
print(f"  of them, {len(hi)} sit above 0.35 where the bias bites:")
for _,r in hi.iterrows():
    print(f"    {r['name']:<16} {r.team:<5} cs_prob {r.clean_sheet_prob:.3f}"
          f"  -> S18-adjusted ~{r.clean_sheet_prob/1.46:.3f}")
cs_pts=(defs.clean_sheet_prob*defs.p_sixty*np.where(defs.position.eq("GK"),4,4)).sum()
print(f"\n  clean-sheet points in the XI's xp: {cs_pts:.2f} of {xi.xp.sum():.2f}"
      f" = {100*cs_pts/xi.xp.sum():.1f}%")
print(f"  if S18's 1.46 over-statement applies in full, that is "
      f"~{cs_pts-cs_pts/1.46:.2f} xp/gw of fiction ({100*(cs_pts-cs_pts/1.46)/xi.xp.sum():.1f}% of the XI)")

print("\n"+"="*76); print("ATTACK 3 -- F25: the objective pays for a bench that never scores")
print("="*76)
bench=sq[~sq.code.isin(xi.code)]
print(f"  bench: " + ", ".join(f"{r['name']} £{r.cost:.1f}m" for _,r in bench.iterrows()))
print(f"  bench spend £{bench.cost.sum():.1f}m vs a realistic £17.5m "
      f"-> £{bench.cost.sum()-17.5:+.1f}m stranded")
print(f"  XI spend £{xi.cost.sum():.1f}m")

print("\n"+"="*76); print("ATTACK 4 -- S26: anti-template squads lose the crowd's floor")
print("="*76)
print(f"  squad mean ownership {sq.owned.mean():.1f}%   XI {xi.owned.mean():.1f}%")
print(f"  S26 flagged our GW1-7 XI at 12.7% as structurally high-variance")
low=sq[sq.owned<5].sort_values("owned")
print(f"  sub-5%-owned players held: {len(low)}")
for _,r in low.iterrows():
    print(f"    {r['name']:<16} {r.team:<5} {r.owned:>5.1f}%  £{r.cost:.1f}m  xp {r.xp:.2f}")

print("\n"+"="*76); print("ATTACK 5 -- club concentration and correlated risk")
print("="*76)
cc=sq.team.value_counts()
print("  " + "  ".join(f"{t}:{n}" for t,n in cc.items()))
for t,n in cc[cc>=3].items():
    who=sq[sq.team.eq(t)]
    print(f"  {t} at the 3-player cap: " + ", ".join(who.name) +
          f"  (£{who.cost.sum():.1f}m, xp {who.xp.sum():.2f}) -- one bad team week hits all three")

print("\n"+"="*76); print("ATTACK 6 -- the Bruno test")
print("="*76)
BR=141746
if BR in set(sq.code):
    b=sq[sq.code.eq(BR)].iloc[0]
    act=out[(out.player_code.eq(BR))&(out.fixtures.gt(0))]
    print(f"  B.Fernandes IS in the squad at £{b.cost:.1f}m, xp {b.xp:.2f}")
    print(f"  his 2026-27 actual: {act.points.mean():.2f} pts/gw over {len(act)} gameweeks")
    print(f"  model xp for him across GW1-7: {act.xp.mean():.2f}  -> over-predicted by "
          f"{act.xp.mean()-act.points.mean():+.2f} pts/gw")
    print(f"  THIS IS THE PLAYER all three pipelines over-rated (S26). The model has")
    print(f"  not learned from it -- nothing in the pipeline updates on being wrong")
    print(f"  about a specific player. Treat his xp as the least trustworthy in the squad.")
else:
    print("  B.Fernandes not selected.")
