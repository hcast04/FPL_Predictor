#!/usr/bin/env python3
"""
Does the model's xP beat dumb heuristics at ACTUALLY MANAGING A TEAM?

Every previous comparison in this project used a best-legal-XI drawn from the
whole league. That benchmark costs about GBP 102m for a legal 15 and is
unaffordable in 65% of gameweeks, so it answers a question nobody can act on.

This simulates real FPL management under the real constraints:

  * GBP 100.0m budget, 15 players (2 GK / 5 DEF / 5 MID / 3 FWD)
  * at most 3 players per club
  * one free transfer per gameweek, no hits  (the user's actual policy)
  * XI and captain chosen each week from the 15 owned
  * scored on realised points, captain doubled

and swaps ONLY the ranking signal, so any difference is attributable to the
signal rather than to the optimiser around it.

  model_xp   the notebook's own out-of-sample expected_points
  owned      ownership -- "copy the template"
  price      most expensive affordable
  form3/5    trailing points per fixture
  ppm        trailing points per million
  random     negative control (seeded, 5 repeats)

Run on the notebook's cached walk-forward folds, so model_xp is genuine
out-of-sample production output, not a re-implementation.

Run:  .venv/bin/python fixes/baseline_showdown.py
"""
from __future__ import annotations
import glob
import numpy as np, pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

CACHE = "data/walk_forward_fold_cache_base_v2_4_0"
QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_B = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
BUDGET, MAX_CLUB, HORIZON = 100.0, 3, 5
POS = ["GK", "DEF", "MID", "FWD"]


def load(season_path):
    pl = pd.read_pickle(season_path, compression="gzip")
    d = pl["bundle"]["holdout_expected_points"].copy()
    d["season"] = str(pl["holdout_season"])
    d["y"] = pd.to_numeric(d.target_fpl_points, errors="coerce")
    d["cost"] = pd.to_numeric(d.price_at_collection, errors="coerce")
    d["model_xp"] = pd.to_numeric(d.expected_points, errors="coerce")
    d["owned"] = pd.to_numeric(d.selected, errors="coerce")
    d["price"] = d["cost"]
    d["form3"] = pd.to_numeric(d.points_per_fixture_last_3_gws, errors="coerce")
    d["form5"] = pd.to_numeric(d.points_per_fixture_last_5_gws, errors="coerce")
    d["ppm"] = d["form5"] / d["cost"].clip(lower=0.1)
    d = d[d.y.notna() & d.cost.notna() & d.model_position.isin(POS)]
    # one row per player-gameweek; a double gameweek sums
    g = (d.groupby(["gameweek", "player_code"], as_index=False)
           .agg(y=("y", "sum"), cost=("cost", "first"),
                pos=("model_position", "first"), team=("team_short_name", "first"),
                model_xp=("model_xp", "sum"), owned=("owned", "first"),
                price=("price", "first"), form3=("form3", "first"),
                form5=("form5", "first"), ppm=("ppm", "first")))
    return g


def pick_squad(panel, gws, score, rng=None):
    """Exact MILP: best 15 under budget / quota / club cap on mean `score`."""
    sub = panel[panel.gameweek.isin(gws)]
    agg = (sub.groupby("player_code")
              .agg(s=(score, "mean"), cost=("cost", "first"),
                   pos=("pos", "first"), team=("team", "first")).reset_index())
    agg = agg[agg.cost.gt(0)].reset_index(drop=True)
    agg["s"] = agg["s"].fillna(0.0)
    n = len(agg)
    A, lo, hi = [], [], []
    for p, q in QUOTA.items():
        A.append((agg.pos == p).astype(float).to_numpy()); lo.append(q); hi.append(q)
    A.append(agg.cost.to_numpy(float)); lo.append(0.0); hi.append(BUDGET)
    for t in agg.team.unique():
        A.append((agg.team == t).astype(float).to_numpy()); lo.append(0.0); hi.append(MAX_CLUB)
    res = milp(c=-agg.s.to_numpy(float),
               constraints=LinearConstraint(np.array(A), lo, hi),
               integrality=np.ones(n), bounds=Bounds(0, 1))
    if not res.success:
        return None
    sel = agg.loc[np.round(res.x).astype(bool), "player_code"].tolist()
    return sel if len(sel) == 15 else None


def best_xi(rows, score):
    best, bs = None, -np.inf
    for dd in range(3, 6):
        for mm in range(2, 6):
            ff = 10 - dd - mm
            if not (1 <= ff <= 3):
                continue
            need = {"GK": 1, "DEF": dd, "MID": mm, "FWD": ff}
            parts, ok = [], True
            for p, k in need.items():
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


def simulate(panel, score, rng=None):
    panel = panel.copy()
    rnd = {}
    if score == "random":
        # one draw per player, fixed for the whole season -- a random but
        # CONSISTENT preference, which is the honest negative control. Redrawing
        # every gameweek would instead simulate a manager with no memory.
        panel["random"] = [rnd.setdefault(c, rng.random()) for c in panel.player_code]
    gws = sorted(panel.gameweek.unique())
    squad = pick_squad(panel, gws[:HORIZON], score, rng)
    if squad is None:
        return None
    costs = panel.groupby("player_code")["cost"].first()
    clubs = panel.groupby("player_code")["team"].first()
    bank = BUDGET - float(costs.reindex(squad).sum())
    total = 0.0
    for gw in gws:
        wk = panel[panel.gameweek.eq(gw)]
        held = wk[wk.player_code.isin(squad)]
        xi = best_xi(held, score) if len(held) >= 11 else None
        if xi is not None:
            cap = xi.loc[xi[score].idxmax()]
            total += float(xi.y.sum()) + float(cap.y)
        # one free transfer, no hit, decided on NEXT week's signal
        nxt = panel[panel.gameweek.eq(gw + 1)]
        if nxt.empty:
            continue
        owned = nxt[nxt.player_code.isin(squad)]
        best_gain, swap = 0.30, None
        for _, o in owned.iterrows():
            cand = nxt[nxt.pos.eq(o["pos"]) & ~nxt.player_code.isin(squad)]
            cand = cand[cand.cost.le(bank + float(costs.get(o["player_code"], 4.0)))]
            if cand.empty:
                continue
            top = cand.loc[cand[score].idxmax()]
            cnt = clubs.reindex(squad).value_counts()
            if top["team"] != o["team"] and cnt.get(top["team"], 0) >= MAX_CLUB:
                continue
            gain = float(top[score]) - float(o[score])
            if gain > best_gain:
                best_gain, swap = gain, (o["player_code"], top["player_code"],
                                         float(costs.get(o["player_code"], 4.0)),
                                         float(top["cost"]))
        if swap:
            a, b, ca, cb = swap
            squad = [b if e == a else e for e in squad]
            bank += ca - cb
    return total


def main():
    strategies = ["model_xp", "owned", "price", "form3", "form5", "ppm"]
    rows = []
    for path in sorted(glob.glob(f"{CACHE}/*.pkl.gz")):
        panel = load(path)
        season = pd.read_pickle(path, compression="gzip")["holdout_season"]
        ngw = panel.gameweek.nunique()
        for s in strategies:
            tot = simulate(panel, s)
            if tot is not None:
                rows.append({"season": str(season), "strategy": s,
                             "total": tot, "gws": ngw, "per_gw": tot / ngw})
                print(f"  {season}  {s:10s} {tot:7.0f}  ({tot/ngw:5.1f}/gw)", flush=True)
        reps = []
        for seed in range(5):
            t = simulate(panel, "random", np.random.default_rng(seed))
            if t is not None:
                reps.append(t)
        if reps:
            rows.append({"season": str(season), "strategy": "random",
                         "total": float(np.mean(reps)), "gws": ngw,
                         "per_gw": float(np.mean(reps)) / ngw})
            print(f"  {season}  {'random':10s} {np.mean(reps):7.0f}  "
                  f"({np.mean(reps)/ngw:5.1f}/gw)  sd={np.std(reps):.0f}", flush=True)
    r = pd.DataFrame(rows)
    print("\n" + "=" * 76)
    print("SEASON TOTALS BY RANKING SIGNAL (same optimiser, same constraints)")
    print("=" * 76)
    piv = r.pivot_table(index="strategy", columns="season", values="total")
    piv["mean"] = piv.mean(axis=1)
    print(piv.round(0).sort_values("mean", ascending=False).to_string())
    base = piv.loc["model_xp"]
    print("\ndelta vs model_xp (negative = the dumb baseline wins):")
    d = (piv - base).drop(index="model_xp").sort_values("mean", ascending=False)
    print(d.round(0).to_string())


if __name__ == "__main__":
    main()
