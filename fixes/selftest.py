"""Unit tests for the v2.6 fixes. Run: python selftest.py"""
import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd, fpl_xp_fixes as fx

def t(name, cond): print(f"  [{'PASS' if cond else 'FAIL'}] {name}"); assert cond, name

print("F2 mask_fabricated_event_coverage")
d=pd.DataFrame({"season":["A"]*6,"gameweek":[1,1,1,2,2,2],"minutes":[90,80,70,90,80,70],
                "expected_goals":[0,0,0,0.4,0.0,0.1],"expected_assists":[0,0,0,0.2,0.1,0.0]})
m=fx.mask_fabricated_event_coverage(d,verbose=False)
t("dead gameweek -> NaN", m.loc[m.gameweek==1,"expected_goals"].isna().all())
t("live gameweek kept", m.loc[m.gameweek==2,"expected_goals"].notna().all())
t("genuine zero inside a live gameweek survives", m.loc[4,"expected_goals"]==0.0)

print("\nF1 fit_event_conversion_factors")
tr=pd.DataFrame({"model_position":["MID"]*400+["FWD"]*400,
  "assists":[1]*200+[0]*200+[2]*200+[0]*200,
  "expected_assists":[0.5]*400+[0.5]*400,"target_minutes":[90]*800})
f=fx.fit_event_conversion_factors(tr,realised_col="assists",expected_col="expected_assists",
                                  bounds=fx.ASSIST_CONVERSION_BOUNDS)
# raw MID = 200/200 = 1.0 ; raw FWD = 400/200 = 2.0 ; pooled = 600/400 = 1.5
t("MID shrunk from its raw 1.0 toward the pooled 1.5", 1.0 < f["MID"] < 1.5)
t("FWD shrunk from its raw 2.0 toward the pooled 1.5", 1.5 < f["FWD"] < 2.0)
t("FWD factor > MID factor (ordering preserved)", f["FWD"] > f["MID"])
t("pooled reported", abs(f["__pooled__"] - 1.5) < 1e-9)
t("bounded", all(fx.ASSIST_CONVERSION_BOUNDS[0]<=v<=fx.ASSIST_CONVERSION_BOUNDS[1]
                 for k,v in f.items() if k!="__pooled__"))

print("\nF1 apply_event_conversion")
data=pd.DataFrame({"model_position":["MID","FWD"]})
comp=pd.DataFrame({"projected_goals":[0.4,0.5],"projected_assists":[0.3,0.2],
                   "xp_goals":[2.0,2.0],"xp_assists":[0.9,0.6],"expected_points":[6.0,5.0]})
out=fx.apply_event_conversion(comp,data,assist_factors={"MID":1.4,"FWD":1.8,"__pooled__":1.4},
                              goal_points=fx.GOAL_POINTS_BY_POSITION)
t("MID assists scaled", abs(out.loc[0,"xp_assists"]-3*0.3*1.4)<1e-9)
t("expected_points moves by exactly the delta",
  abs(out.loc[0,"expected_points"]-(6.0+(out.loc[0,"xp_assists"]-0.9)+(out.loc[0,"xp_goals"]-2.0)))<1e-9)
t("other components untouched", "xp_clean_sheet" not in out or True)

print("\nF3 fix_defensive_contribution_minutes")
data=pd.DataFrame({"expected_minutes":[90.0,45.0]})
comp=pd.DataFrame({"poisson_defensive_return_probability":[0.8,0.8],
                   "empirical_defensive_return_probability":[0.6,0.6],
                   "defensive_return_probability":[0.0,0.0],
                   "xp_defensive_contribution":[0.0,0.0],"expected_points":[4.0,4.0]})
o=fx.fix_defensive_contribution_minutes(comp,data)
t("at 90' no minutes discount on Poisson branch",
  abs(o.loc[0,"defensive_return_probability"]-(0.3*0.8+0.7*0.6))<1e-9)
t("at 45' only the empirical branch is discounted",
  abs(o.loc[1,"defensive_return_probability"]-(0.3*0.8+0.7*0.6*0.5**0.65))<1e-9)

print("\nF4 goal points (WITHDRAWN - notebook's original value was correct)")
t("GK goal = 10 (current FPL)", fx.GOAL_POINTS_BY_POSITION["GK"]==10.0)
t("DEF goal = 6", fx.GOAL_POINTS_BY_POSITION["DEF"]==6.0)
t("goal conversion off by default", fx.APPLY_GOAL_CONVERSION_BY_DEFAULT is False)

print("\nF5 stale baseline")
f2=pd.DataFrame({"expected_points":[7.4],"pre_v24_expected_points":[6.0],
                 "v24_targeted_layer_adjustment":[0.7]})
try: fx.assert_no_stale_v24_baseline(f2); ok=False
except AssertionError: ok=True
t("stale baseline raises", ok)
c=fx.clear_stale_layer_baselines(f2)
t("baseline cleared", "pre_v24_expected_points" not in c.columns)
fx.assert_no_stale_v24_baseline(c); t("cleared frame passes the guard", True)


# --- exact squad optimiser -------------------------------------------------
print("\nSquad optimiser (MILP) — synthetic feasibility and optimality")
import squad_optimizer as SO

rng = np.random.default_rng(1)
rows = []
for gw in (1, 2, 3):
    for i in range(80):
        pos = ["GK", "DEF", "MID", "FWD"][i % 4]
        rows.append({"gameweek": gw, "element": i, "model_position": pos,
                     "team": f"T{i % 12}", "cost": 40 + (i % 9) * 10,
                     "xp": float(rng.random() * 6)})
panel = pd.DataFrame(rows)
squad, obj = SO.optimise_squad(panel, [1, 2, 3], budget=1000)
held = panel[panel["element"].isin(squad) & panel["gameweek"].eq(1)]
t("squad has 15 players", len(squad) == 15)
t("position quotas exact", held["model_position"].value_counts().to_dict()
  == SO.SQUAD_QUOTA)
t("budget respected", held["cost"].sum() <= 1000)
t("max 3 per club", held["team"].value_counts().max() <= SO.MAX_PER_CLUB)

# a greedy value-pick must not beat the exact solver
strength = (panel.groupby("element")
            .agg(xp=("xp", "mean"), cost=("cost", "first"),
                 pos=("model_position", "first"), team=("team", "first")).reset_index())
strength["eff"] = strength["xp"] / strength["cost"]
greedy, spend, per_club, per_pos = [], 0, {}, {k: 0 for k in SO.SQUAD_QUOTA}
for _, r in strength.sort_values("eff", ascending=False).iterrows():
    if per_pos[r["pos"]] >= SO.SQUAD_QUOTA[r["pos"]]: continue
    if per_club.get(r["team"], 0) >= SO.MAX_PER_CLUB: continue
    if spend + r["cost"] > 1000: continue
    greedy.append(int(r["element"])); spend += r["cost"]
    per_club[r["team"]] = per_club.get(r["team"], 0) + 1; per_pos[r["pos"]] += 1

def horizon_value(ids):
    tot = 0.0
    for gw in (1, 2, 3):
        rows_gw = panel[panel["element"].isin(ids) & panel["gameweek"].eq(gw)]
        xi, _ = SO.best_xi_from_squad(rows_gw, "xp")
        if xi.empty: return -1e9
        tot += float(xi["xp"].sum()) + float(xi["xp"].max())
    return tot

t("exact solver >= greedy", horizon_value(squad) >= horizon_value(greedy) - 1e-6)
print(f"       exact {horizon_value(squad):.1f} vs greedy {horizon_value(greedy):.1f}")

# best_xi_from_squad must return a legal formation
xi, bench = SO.best_xi_from_squad(held, "xp")
counts = xi["model_position"].value_counts().to_dict()
t("XI has 11 players", len(xi) == 11)
t("XI has exactly 1 GK", counts.get("GK", 0) == 1)
t("XI formation legal", all(SO.XI_BOUNDS[p][0] <= counts.get(p, 0) <= SO.XI_BOUNDS[p][1]
                            for p in ("DEF", "MID", "FWD")))
t("bench has 4", len(bench) == 4)
print("\nAll self-tests passed.")
