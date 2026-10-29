#!/usr/bin/env python3
"""
Wildcard squad for GW8, built by the optimiser rather than by opinion.

Deliberately NOT "buy the player who already hauled". The squad below is what
`optimise.optimise_squad` returns on the model's own forecast under the real
constraints. Haaland is included only if the MILP puts him there.

Choices, each with its justification:

  * HORIZON. GW8-12 carry no blanks and no doubles -- every team plays exactly
    once in each -- so fixture count is uniform. S29 measured opponent identity
    as carrying no ranking value one to three weeks out (scrambling the
    opponent map left the forecast very slightly BETTER). With both of those
    true the five-week horizon ranking collapses to the current-week ranking,
    so the latest forecast is used directly instead of a projection that
    today's own evidence says would be identical.

  * AVAILABILITY. Players flagged injured or doubtful are excluded using the
    LIVE bootstrap snapshot. This is the one place that data is legitimate:
    F28 proved it cannot be GATED (no history exists), but S29 identified
    near-future availability as the entire remaining headroom, and for a
    decision taken today it is simply known.

  * BUDGET. Squad value GBP 99.5m + bank GBP 0.6m = GBP 100.1m. On a wildcard
    the whole squad sells at its current selling price, which is what team
    value already reports.

KNOWN MODEL BIASES, stated because they bear on this specific squad:
  - S20: Haaland under-predicted ~45%, irreducible (six routes closed, F29/F30)
  - S18: clean sheets OVER-predicted for the best defences (0.542 predicted ->
    0.371 realised), so premium defenders are flattered by this objective
  - F25: the objective maximises over all 15, so the bench is over-bought
"""
from __future__ import annotations
import json, sys, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, optimise  # noqa: E402

API="https://fantasy.premierleague.com/api"
HDR={"User-Agent":"Mozilla/5.0"}
BUDGET=100.1
MIN_CHANCE=75          # exclude anyone below this chance of playing GW8


def get(u): return json.load(urllib.request.urlopen(urllib.request.Request(u,headers=HDR),timeout=60))


def main():
    out = pd.read_pickle(C.cache("fpl2_s20_out.pkl"))
    boot = get(f"{API}/bootstrap-static/")
    el = pd.DataFrame([{
        "code": e["code"], "name": e["web_name"], "cost": e["now_cost"]/10.0,
        "owned": float(e["selected_by_percent"]), "status": e["status"],
        "chance": e["chance_of_playing_next_round"], "news": e["news"],
        "team_id": e["team"], "etype": e["element_type"]} for e in boot["elements"]])
    teams={t["id"]:t["short_name"] for t in boot["teams"]}
    el["team"]=el.team_id.map(teams)
    el["position"]=el.etype.map({1:"GK",2:"DEF",3:"MID",4:"FWD"})

    last = int(out.gameweek.max())
    xp = (out[out.gameweek.eq(last)].groupby("player_code", as_index=False)
             .agg(xp=("xp","mean")))
    pool = el.merge(xp, left_on="code", right_on="player_code", how="inner")

    avail = pool[(pool.status.eq("a")) &
                 (pool.chance.isna() | (pool.chance >= MIN_CHANCE))].copy()
    print(f"pool: {len(pool)} forecast players -> {len(avail)} available "
          f"(excluded {len(pool)-len(avail)} injured/doubtful)")
    print(f"forecast source: GW{last} model xp; budget £{BUDGET:.1f}m\n")

    sq = optimise.optimise_squad(avail, "xp", budget=BUDGET)
    if sq is None:
        print("INFEASIBLE"); return

    def show(s, title):
        s = s.sort_values(["position","xp"], ascending=[True,False])
        xi = optimise.best_xi(s, "xp")
        cap = optimise.pick_captain(xi, "xp")
        print(f"=== {title} ===")
        print(f"{'player':<18}{'pos':<5}{'team':<6}{'£m':>6}{'owned%':>8}{'xp':>7}{'XI':>5}")
        for _,r in s.iterrows():
            inxi = r.code in set(xi.code)
            mark = "(C)" if r.code==cap.code else ("yes" if inxi else "-")
            print(f"{r['name'][:17]:<18}{r.position:<5}{r.team:<6}{r.cost:>6.1f}"
                  f"{r.owned:>8.1f}{r.xp:>7.2f}{mark:>5}")
        print(f"\n  spend £{s.cost.sum():.1f}m   bank £{BUDGET-s.cost.sum():.1f}m"
              f"   XI xp {xi.xp.sum():.2f}   captain {cap['name']}")
        print(f"  mean ownership: squad {s.owned.mean():.1f}%  XI {xi.owned.mean():.1f}%")
        return xi, cap

    xi, cap = show(sq, f"OPTIMISER'S WILDCARD SQUAD (GW{last+1})")

    HA = 223094
    print(f"\n  Haaland selected by the optimiser: {HA in set(sq.code)}")
    if HA not in set(sq.code):
        h = avail[avail.code.eq(HA)]
        if len(h):
            h=h.iloc[0]
            print(f"    his numbers: xp {h.xp:.2f}  £{h.cost:.1f}m  "
                  f"xp/£m {h.xp/h.cost:.4f}  owned {h.owned:.1f}%")
            best=avail.nlargest(1,"xp").iloc[0]
            print(f"    top xp in pool: {best['name']} {best.xp:.2f} £{best.cost:.1f}m "
                  f"xp/£m {best.xp/best.cost:.4f}")
            forced = avail.copy()
            forced.loc[forced.code.eq(HA),"xp"] += 1000.0
            fs = optimise.optimise_squad(forced,"xp",budget=BUDGET)
            fs = avail[avail.code.isin(fs.code)]
            fxi = optimise.best_xi(fs,"xp")
            print(f"\n    FORCED-HAALAND squad XI xp {fxi.xp.sum():.2f} vs "
                  f"free {xi.xp.sum():.2f}  -> cost of forcing him "
                  f"{fxi.xp.sum()-xi.xp.sum():+.2f} xp/gw")
            print()
            show(fs, "SAME OPTIMISER, HAALAND FORCED IN")


if __name__ == "__main__":
    main()
