#!/usr/bin/env python3
"""
Wildcard valuation over a multi-gameweek horizon.

The live pipeline reports planning_horizon: 1 -- it produces no multi-gameweek
forecast -- so one has to be constructed. This uses the same leakage-free
method as horizon_sweep_clean.py:

  hz(p, H) = xp(p, GW_next) * mean_{k in next H gameweeks} ease(opponent(p, k))

The notebook's own xP supplies player quality (it is the strongest signal we
have: it beats every dumb baseline in all three holdout seasons). The fixture
schedule and team strength to date supply the forward variation. Nothing uses
information that would not be available at this deadline.

Compares HOLD / ONE TRANSFER / WILDCARD on that horizon score.

Run:  .venv/bin/python fixes/price_wildcard_horizon.py [H ...]
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


def value(rows, score):
    xi = best_xi(rows, score)
    if xi is None:
        return np.nan
    return float(xi[score].sum()) + float(xi[score].max())


def optimise(pool, budget, score):
    p = pool[pool.cost.gt(0)].reset_index(drop=True)
    A, lo, hi = [], [], []
    for q, k in QUOTA.items():
        A.append((p.pos == q).astype(float).to_numpy()); lo.append(k); hi.append(k)
    A.append(p.cost.to_numpy(float)); lo.append(0.0); hi.append(budget)
    for t in p.team.unique():
        A.append((p.team == t).astype(float).to_numpy()); lo.append(0.0); hi.append(MAX_CLUB)
    r = milp(c=-p[score].to_numpy(float),
             constraints=LinearConstraint(np.array(A), lo, hi),
             integrality=np.ones(len(p)), bounds=Bounds(0, 1))
    return p.loc[np.round(r.x).astype(bool)].copy() if r.success else None


def main():
    Hs = [int(a) for a in sys.argv[1:]] or [1, 3, 5, 8]
    boot = api("https://fantasy.premierleague.com/api/bootstrap-static/")
    el = {e["id"]: e for e in boot["elements"]}
    tm = {t["id"]: t["short_name"] for t in boot["teams"]}
    gw0 = next(e for e in boot["events"] if e["is_next"])["id"]

    # team ease from THIS season's results so far (nothing from gw0 onward)
    conceded, played = {}, {}
    for g in range(1, gw0):
        for f in api(f"https://fantasy.premierleague.com/api/fixtures/?event={g}"):
            if not f.get("finished") and not f.get("finished_provisional"):
                continue
            h, a = tm[f["team_h"]], tm[f["team_a"]]
            hs, as_ = f.get("team_h_score"), f.get("team_a_score")
            if hs is None or as_ is None:
                continue
            conceded[h] = conceded.get(h, 0) + as_; played[h] = played.get(h, 0) + 1
            conceded[a] = conceded.get(a, 0) + hs; played[a] = played.get(a, 0) + 1
    rates = {t: conceded[t] / played[t] for t in conceded if played.get(t)}
    league = np.mean(list(rates.values())) if rates else 1.0
    ease = {t: r / league for t, r in rates.items()}

    # published schedule for the horizon
    sched = {}
    for g in range(gw0, gw0 + max(Hs) + 2):
        try:
            for f in api(f"https://fantasy.premierleague.com/api/fixtures/?event={g}"):
                sched.setdefault((g, tm[f["team_h"]]), []).append(tm[f["team_a"]])
                sched.setdefault((g, tm[f["team_a"]]), []).append(tm[f["team_h"]])
        except Exception:
            pass

    with gzip.open("data/online_forecast_archive_v1.json.gz", "rt", encoding="utf-8") as f:
        arch = json.load(f)
    ver = arch["snapshots"][f"2026-27_GW{gw0:02d}"]["versions"][-1]
    fc = {r["player_id"]: r["expected_points"] for r in ver["player_forecasts"]}

    pool = pd.DataFrame([
        {"pid": p, "name": el[p]["web_name"], "pos": PT[el[p]["element_type"]],
         "team": tm[el[p]["team"]], "cost": el[p]["now_cost"] / 10.0,
         "xp": x, "st": el[p]["status"]}
        for p, x in fc.items() if p in el])
    pool = pool[pool.st.eq("a")].reset_index(drop=True)

    picks = api(f"https://fantasy.premierleague.com/api/entry/{TID}/event/{gw0-1}/picks/")
    sq = [p["element"] for p in picks["picks"]]
    bank = picks["entry_history"]["bank"] / 10.0
    budget = sum(el[i]["now_cost"] / 10.0 for i in sq) + bank
    print(f"GW{gw0}  budget GBP {budget:.1f}m (squad value + bank)\n")
    print(f"  {'H':>2}  {'HOLD':>7} {'1 TRANSFER':>11} {'WILDCARD':>9}   "
          f"{'WC-1TR':>7}  {'kept':>5}")
    results = []
    for H in Hs:
        mult = []
        for t in pool["team"]:
            vals = []
            for k in range(gw0, gw0 + H):
                opps = sched.get((k, t), [])
                vals.extend([ease.get(o, 1.0) for o in opps] or [0.0])
            fwd = float(np.mean(vals)) if vals else 1.0
            # the notebook's xp ALREADY prices this gameweek's fixture, so the
            # horizon factor must be RELATIVE to it, not applied on top of it
            now = [ease.get(o, 1.0) for o in sched.get((gw0, t), [])]
            base = float(np.mean(now)) if now else 1.0
            mult.append(fwd / base if base > 0 else 1.0)
        pool["hz"] = pool["xp"].to_numpy(float) * np.asarray(mult)
        cur = pool[pool.pid.isin(sq)]
        hold = value(cur, "hz")
        best1 = hold
        for _, o in cur.iterrows():
            cand = pool[pool.pos.eq(o["pos"]) & ~pool.pid.isin(sq)
                        & pool.cost.le(bank + o["cost"])]
            for _, i in cand.nlargest(20, "hz").iterrows():
                cnt = cur[cur.pid != o["pid"]].team.value_counts()
                if cnt.get(i["team"], 0) >= MAX_CLUB:
                    continue
                tr = pd.concat([cur[cur.pid != o["pid"]], i.to_frame().T],
                               ignore_index=True)
                tr["hz"] = pd.to_numeric(tr.hz, errors="coerce")
                v = value(tr, "hz")
                if v > best1:
                    best1 = v
        wc = optimise(pool, budget, "hz")
        wcv = value(wc, "hz") if wc is not None else np.nan
        kept = len(set(wc.pid) & set(sq)) if wc is not None else 0
        print(f"  {H:2d}  {hold:7.2f} {best1:11.2f} {wcv:9.2f}   "
              f"{wcv-best1:+7.2f}  {kept:3d}/15")
        results.append({"H": H, "wc_minus_1tr": wcv - best1, "kept": kept})
    print("\n  WC-1TR is per-gameweek-equivalent on the horizon score, not a")
    print("  season total. Multiply by the gameweeks the squad will persist,")
    print("  and remember one free transfer a week erodes the gap over time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
