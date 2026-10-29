#!/usr/bin/env python3
"""
Wildcard squad for the upcoming deadline, on the REPAIRED model.

Rebuilt rather than reusing `build_wildcard.py`, which reads a cached forecast
(`~/fpl2_s20_out.pkl`, 2026-10-22) produced before today's fixes. That cache
predates M26/M27, trained on four seasons, and carried no Understat features.

Current configuration, all measured today:
  * `USE_UNDERSTAT_FEATURES = True` (M27: S47 retracted; +47.7 xG, +7.8 xA)
  * ten seasons of training data, not four (S46)
  * live Understat coverage 99.4% (was 0.0% -- the live panel lost
    `kickoff_time` in the concat, so the fixture-date join matched nothing)
  * `opening_horizon = 1` (S57: the honest sweep is monotone toward shorter;
    the +235/season for long horizons was F10's leak)
  * per-position rate layer kept (S52: pooling it costs -17.4, p 0.002)
  * ceiling kept at p99 (S54: removing it is worth +0.8 against MDE 1.4)
  * flat bonus kept (S56: conditioning it reorders by 2.4 places and loses)

BUDGET is the real one: squad selling value plus bank, from the entry API,
not GBP100m.

KNOWN BIASES THAT STILL APPLY, stated because they bear on this squad:
  * S18 clean sheets are OVER-predicted for the best defences (0.542
    predicted -> 0.371 realised), so premium defenders are flattered
  * S20 Haaland is still UNDER-predicted: 58 actual vs 39.8 forecast over
    GW1-7 this season, a 31% shortfall, even after today's repairs
  * F25 the objective maximises over all 15, so the bench is over-bought
"""
from __future__ import annotations
import json, os, sys, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, optimise                               # noqa: E402

API = "https://fantasy.premierleague.com/api"
HDR = {"User-Agent": "Mozilla/5.0"}
ENTRY = int(os.environ.get("FPL_ENTRY") or 0)   # export FPL_ENTRY=<your team id>
MIN_CHANCE = 75          # drop anyone below this chance of playing


def get(u):
    return json.load(urllib.request.urlopen(
        urllib.request.Request(u, headers=HDR), timeout=60))


def main():
    out = pd.read_pickle(C.cache("fpl2_live_chip_out.pkl"))
    boot = get(f"{API}/bootstrap-static/")
    finished = sorted(e["id"] for e in boot["events"] if e.get("finished"))
    nxt = next((e for e in boot["events"] if e.get("is_next")), None)
    picks = get(f"{API}/entry/{ENTRY}/event/{max(finished)}/picks/")
    bank = picks["entry_history"]["bank"] / 10.0
    value = picks["entry_history"]["value"] / 10.0
    # `entry_history.value` is the PURCHASE-price valuation. A wildcard sells
    # at SELLING price, which is lower for anyone who has risen (you keep only
    # half the profit, rounded down). The public picks endpoint does not expose
    # selling prices, so the real budget is supplied by the user and the API
    # figure is printed alongside to show the gap.
    budget = float(os.environ.get("WC_BUDGET", bank + value))
    print(f"GW{nxt['id']} deadline {nxt['deadline_time']}")
    print(f"  API value £{value:.1f}m + bank £{bank:.1f}m = £{value+bank:.1f}m "
          f"(purchase prices)")
    print(f"  BUDGET USED: £{budget:.1f}m\n")

    el = pd.DataFrame([{
        "code": e["code"], "name": e["web_name"], "cost": e["now_cost"] / 10.0,
        "owned": float(e["selected_by_percent"]), "status": e["status"],
        "chance": e["chance_of_playing_next_round"], "news": e["news"],
        "team_id": e["team"], "etype": e["element_type"]} for e in boot["elements"]])
    teams = {t["id"]: t["short_name"] for t in boot["teams"]}
    el["team"] = el.team_id.map(teams)
    el["position"] = el.etype.map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})

    # THE GAMEWEEK BEING DECIDED. This read the last FINISHED gameweek, so the
    # squad was optimised against last week's opponents -- the same defect
    # `plan_gw.py` had. Van Hecke read 5.19 on a GW7 fixture and 2.42 on his
    # actual GW8 one.
    #
    # HORIZON. S57 swept this honestly and found SHORTER better: the +235/season
    # for long horizons was F10's leak. So the default is the target week
    # alone. WC_HORIZON>1 averages the projection over that many weeks, which
    # S57 measured as worse for an opening squad but is offered because a
    # mid-season wildcard has a known fixture list the GW1 decision does not.
    target = int(nxt["id"])
    H = int(os.environ.get("WC_HORIZON", "1"))
    pjp = C.cache("fpl2_live_proj.pkl")
    if pjp.exists():
        pj = pd.read_pickle(pjp)
        win = pj[(pj.gameweek >= target) & (pj.gameweek < target + H)]
    else:
        win = pd.DataFrame()
    if win.empty:
        last = int(out.gameweek.max())
        print(f"!! NO PROJECTION for GW{target}; falling back to the GW{last} "
              f"forecast, which prices GW{last}'s opponents. Run "
              f"run_live_project.py first.\n")
        xp = (out[out.gameweek.eq(last)]
              .groupby("player_code", as_index=False).agg(xp=("xp", "mean")))
    else:
        got = sorted(int(g) for g in win.gameweek.unique())
        print(f"forecast: GW{got[0]}" + (f"-{got[-1]} (mean)" if len(got) > 1 else "")
              + f", projected from GW{int(out.gameweek.max())} information\n")
        xp = win.groupby("player_code", as_index=False).agg(xp=("xp", "mean"))
    pool = el.merge(xp, left_on="code", right_on="player_code", how="inner")
    avail = pool[pool.status.eq("a") &
                 (pool.chance.isna() | (pool.chance >= MIN_CHANCE))].copy()
    print(f"pool {len(pool)} forecast -> {len(avail)} available "
          f"(status 'a', chance >= {MIN_CHANCE})")

    #  already carries player_code from the xp merge; renaming 
    # onto it would create two columns of that name and every later
    # avail[PLAYER_KEY] would return a DataFrame, not a Series.
    avail = avail.drop(columns=["code"]).reset_index(drop=True)
    assert list(avail.columns).count(C.PLAYER_KEY) == 1, "duplicate key column"
    # Three variants, so the CONCENTRATION question is priced rather than
    # argued. The MILP maximises expected points and is blind to correlation:
    # three players from one club rise and fall together, which does not change
    # the mean but does change the spread. Ledger F8 rejected a
    # fixture-concentration cap -- on the 3-season instrument S37 later showed
    # was unfit, and on a different constraint (players per MATCH, not per
    # CLUB). This simply reports what each restriction costs in xp.
    import fpl2.config as _C
    variants = {}
    for label, cap, ban in (("max 3/club (FPL limit)", 3, None),
                            ("max 2/club", 2, None),
                            ("max 3/club, no MUN", 3, "MUN")):
        orig = _C.MAX_PER_CLUB
        _C.MAX_PER_CLUB = cap
        try:
            pool_v = avail[avail.team.ne(ban)] if ban else avail
            v = optimise.optimise_squad(pool_v.reset_index(drop=True), "xp",
                                        budget=budget)
        finally:
            _C.MAX_PER_CLUB = orig
        if v is not None and len(v) == C.SQUAD_SIZE:
            vx = optimise.best_xi(v.reset_index(drop=True), "xp")
            vc = optimise.pick_captain(vx.reset_index(drop=True), "xp")
            variants[label] = (v, float(vx.xp.sum()), float(vc.xp),
                               float(v.cost.sum()))
    print("\n" + "=" * 68)
    print("WHAT CLUB CONCENTRATION COSTS")
    print("=" * 68)
    print(f"{'variant':<26}{'XI xp':>9}{'+capt':>8}{'total':>9}{'spend':>9}  clubs>=3")
    base = None
    for label, (v, xisum, capx, spend) in variants.items():
        tot = xisum + capx
        base = tot if base is None else base
        cc = v.team.value_counts()
        heavy = ", ".join(f"{t}x{n}" for t, n in cc[cc >= 3].items()) or "-"
        print(f"{label:<26}{xisum:>9.2f}{capx:>8.2f}{tot:>9.2f}"
              f"{spend:>8.1f}m  {heavy}   ({tot-base:+.2f})")

    sq = variants[os.environ.get("WC_VARIANT", "max 3/club (FPL limit)")][0]
    if sq is None:
        print("no legal squad"); return
    sq = sq.sort_values(["position", "xp"], ascending=[True, False])
    # best_xi concatenates per-position pools, so its index carries duplicates
    sq = sq.reset_index(drop=True)
    xi = optimise.best_xi(sq, "xp").reset_index(drop=True)
    cap = optimise.pick_captain(xi, "xp")
    sq["XI"] = np.where(sq[C.PLAYER_KEY].isin(xi[C.PLAYER_KEY]), "XI", "bench")
    sq.loc[sq[C.PLAYER_KEY].eq(cap[C.PLAYER_KEY]), "XI"] = "CAPTAIN"

    print("\n" + "=" * 68)
    print(f"WILDCARD SQUAD — optimiser, budget £{budget:.1f}m, horizon 1")
    print("=" * 68)
    print(sq[["name", "position", "team", "cost", "xp", "owned", "XI"]]
          .to_string(index=False,
                     formatters={"cost": "£{:.1f}m".format, "xp": "{:.2f}".format,
                                 "owned": "{:.1f}%".format}))
    print(f"\n  spend £{sq.cost.sum():.1f}m of £{budget:.1f}m "
          f"(£{budget - sq.cost.sum():.1f}m left)")
    print(f"  XI xp {xi.xp.sum():.2f} + captain {cap.xp:.2f} = "
          f"{xi.xp.sum() + cap.xp:.2f}")
    print(f"  bench xp {sq[sq.XI.eq('bench')].xp.sum():.2f}")

    cur = {p["element"] for p in picks["picks"]}
    cur_codes = {e["code"] for e in boot["elements"] if e["id"] in cur}
    keep = sq[sq[C.PLAYER_KEY].isin(cur_codes)]
    print(f"\n  keeps {len(keep)} of your current 15: "
          f"{', '.join(sorted(keep.name)) if len(keep) else '(none)'}")
    print(f"  that is {15 - len(keep)} transfers — a wildcard makes them free")


if __name__ == "__main__":
    main()
