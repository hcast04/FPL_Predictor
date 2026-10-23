#!/usr/bin/env python3
"""
What is a wildcard actually worth right now?

Compares three options over the same horizon, under real FPL constraints:

  HOLD          keep the current 15, make zero transfers
  ONE TRANSFER  the best single free transfer (current policy)
  WILDCARD      rebuild all 15 from the full pool, budget = squad value + bank

The wildcard is only worth playing if it beats ONE TRANSFER by enough to
justify burning the chip -- a one-gameweek edge is not enough, because the
same squad could be reached over several weeks with free transfers.

Selling prices need an authenticated FPL session, which this does not have, so
current prices are used. Real sale values can be GBP 0.1-0.2m lower per player,
which makes the wildcard budget here slightly OPTIMISTIC.

Run:  .venv/bin/python fixes/price_wildcard.py
"""
from __future__ import annotations
import gzip, json, sys, urllib.request
import numpy as np, pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

TID = 3642415
QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_CLUB = 3
PT = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def api(u):
    return json.load(urllib.request.urlopen(
        urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=60))


def best_xi(rows, score):
    best, bs = None, -np.inf
    for dd in range(3, 6):
        for mm in range(2, 6):
            ff = 10 - dd - mm
            if not (1 <= ff <= 3):
                continue
            parts, ok = [], True
            for p, k in {"GK": 1, "DEF": dd, "MID": mm, "FWD": ff}.items():
                pool = rows[rows.pos.eq(p)].nlargest(k, score)
                if len(pool) < k:
                    ok = False; break
                parts.append(pool)
            if not ok:
                continue
            xi = pd.concat(parts)
            s = float(xi[score].sum())
            if s > bs:
                bs, best = s, xi
    return best


def xi_value(rows, score):
    """XI total plus the captain's score counted twice."""
    xi = best_xi(rows, score)
    if xi is None:
        return np.nan, None
    return float(xi[score].sum()) + float(xi[score].max()), xi


def optimise(pool, budget, score="xp"):
    p = pool[pool.cost.gt(0)].reset_index(drop=True)
    A, lo, hi = [], [], []
    for q, k in QUOTA.items():
        A.append((p.pos == q).astype(float).to_numpy()); lo.append(k); hi.append(k)
    A.append(p.cost.to_numpy(float)); lo.append(0.0); hi.append(budget)
    for t in p.team.unique():
        A.append((p.team == t).astype(float).to_numpy()); lo.append(0.0); hi.append(MAX_CLUB)
    res = milp(c=-p[score].to_numpy(float),
               constraints=LinearConstraint(np.array(A), lo, hi),
               integrality=np.ones(len(p)), bounds=Bounds(0, 1))
    if not res.success:
        return None
    return p.loc[np.round(res.x).astype(bool)].copy()


def main():
    boot = api("https://fantasy.premierleague.com/api/bootstrap-static/")
    el = {e["id"]: e for e in boot["elements"]}
    tm = {t["id"]: t["short_name"] for t in boot["teams"]}
    gw = next(e for e in boot["events"] if e["is_next"])["id"]

    with gzip.open("data/online_forecast_archive_v1.json.gz", "rt", encoding="utf-8") as f:
        arch = json.load(f)
    key = [k for k in arch["snapshots"] if k.endswith(f"GW{gw:02d}")]
    if not key:
        print(f"No archived forecast for GW{gw}. Run the notebook first.")
        return 1
    ver = arch["snapshots"][key[0]]["versions"][-1]
    fc = {r["player_id"]: r["expected_points"] for r in ver["player_forecasts"]}
    print(f"GW{gw} forecast: {len(fc)} players, captured {ver['captured_at_utc']}")

    pool = pd.DataFrame([
        {"pid": pid, "name": el[pid]["web_name"], "pos": PT[el[pid]["element_type"]],
         "team": tm[el[pid]["team"]], "cost": el[pid]["now_cost"] / 10.0,
         "xp": xp, "status": el[pid]["status"]}
        for pid, xp in fc.items() if pid in el])
    pool = pool[pool.status.eq("a")]

    picks = api(f"https://fantasy.premierleague.com/api/entry/{TID}/event/{gw-1}/picks/")
    squad_ids = [p["element"] for p in picks["picks"]]
    bank = picks["entry_history"]["bank"] / 10.0
    cur = pool[pool.pid.isin(squad_ids)].copy()
    missing = set(squad_ids) - set(cur.pid)
    squad_value = sum(el[i]["now_cost"] / 10.0 for i in squad_ids)
    print(f"current squad value GBP {squad_value:.1f}m   bank GBP {bank:.1f}m   "
          f"wildcard budget GBP {squad_value+bank:.1f}m")
    if missing:
        print(f"  ! {len(missing)} squad players missing from the forecast "
              f"(flagged/unavailable): {[el[i]['web_name'] for i in missing]}")

    hold, hold_xi = xi_value(cur, "xp")
    print(f"\n  HOLD          projected {hold:6.2f}")

    best_one, best_move = hold, None
    for _, o in cur.iterrows():
        cand = pool[pool.pos.eq(o["pos"]) & ~pool.pid.isin(squad_ids)]
        cand = cand[cand.cost.le(bank + o["cost"])]
        for _, i in cand.nlargest(25, "xp").iterrows():
            cnt = cur.team.value_counts()
            if i["team"] != o["team"] and cnt.get(i["team"], 0) >= MAX_CLUB:
                continue
            trial = pd.concat([cur[cur.pid != o["pid"]], i.to_frame().T],
                              ignore_index=True)
            # a Series->row concat upcasts to object, which breaks nlargest
            trial["xp"] = pd.to_numeric(trial["xp"], errors="coerce")
            trial["cost"] = pd.to_numeric(trial["cost"], errors="coerce")
            v, _ = xi_value(trial, "xp")
            if v > best_one:
                best_one, best_move = v, (o["name"], i["name"])
    print(f"  ONE TRANSFER  projected {best_one:6.2f}   "
          + (f"({best_move[0]} -> {best_move[1]})" if best_move else "(no gain)"))

    wc = optimise(pool, squad_value + bank, "xp")
    if wc is None:
        print("  WILDCARD      solver infeasible")
        return 1
    wc_v, wc_xi = xi_value(wc, "xp")
    print(f"  WILDCARD      projected {wc_v:6.2f}")
    print(f"\n  wildcard vs hold         {wc_v-hold:+6.2f}")
    print(f"  wildcard vs one transfer {wc_v-best_one:+6.2f}   <- the number that matters")

    kept = set(wc.pid) & set(squad_ids)
    print(f"\n  players kept from your current squad: {len(kept)}/15")
    print(f"  players changed                     : {15-len(kept)}")
    print("\n  WILDCARD SQUAD:")
    for p in ("GK", "DEF", "MID", "FWD"):
        for _, r in wc[wc.pos.eq(p)].sort_values("xp", ascending=False).iterrows():
            mark = "  (already owned)" if r["pid"] in kept else ""
            star = " *XI" if wc_xi is not None and r["pid"] in set(wc_xi.pid) else ""
            print(f"    {p:4s} {r['name']:16s} {r['team']:5s} "
                  f"{r['cost']:5.1f}  xp {r['xp']:5.2f}{star}{mark}")
    print("\n  NOTE: single-gameweek valuation. A wildcard should clear ONE TRANSFER")
    print("  by a wide margin over a multi-gameweek horizon, not by one week's xP.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
