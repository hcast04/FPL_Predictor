#!/usr/bin/env python3
"""
How much does a BAD STARTING SQUAD cost over a season, and can one free
transfer a week recover from it?

This is the question the one-gameweek wildcard valuation cannot answer. A
wildcard priced on a single gameweek was worth only +2.4 xP, but squad quality
compounds: a weak 15 caps every XI you can field for months, and one transfer
per week may or may not claw it back.

Starts the SAME simulator (GBP 100m, 2/5/5/3, max 3 per club, one free
transfer, no hits, model xP for every decision) from starting squads of
different quality, and measures the season total.

  optimal    MILP-best 15 on the opening horizon
  decile_k   MILP-best 15 drawn from players ranked in the k-th xP decile and
             below -- a progressively weaker but still legal and affordable squad

If the gap closes over the season, one transfer a week is enough and a
wildcard is not needed. If it persists, the starting squad is destiny and the
chip pays for itself.

Run:  .venv/bin/python fixes/squad_drag.py
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_showdown as BS   # noqa: E402

CACHE = "data/walk_forward_fold_cache_base_v2_4_0"


def handicapped_squad(panel, gws, drop_top_frac):
    """MILP-best 15 after removing the top `drop_top_frac` of players by xP."""
    sub = panel[panel.gameweek.isin(gws)]
    strength = sub.groupby("player_code")["model_xp"].mean().sort_values(ascending=False)
    cut = int(len(strength) * drop_top_frac)
    banned = set(strength.index[:cut])
    pool = panel[~panel.player_code.isin(banned)]
    return BS.pick_squad(pool, gws, "model_xp")


def simulate_from(panel, squad):
    """Season under the normal policy, but starting from a GIVEN squad."""
    gws = sorted(panel.gameweek.unique())
    costs = panel.groupby("player_code")["cost"].first()
    clubs = panel.groupby("player_code")["team"].first()
    bank = BS.BUDGET - float(costs.reindex(squad).sum())
    total = 0.0
    for gw in gws:
        wk = panel[panel.gameweek.eq(gw)]
        held = wk[wk.player_code.isin(squad)]
        xi = BS.best_xi(held, "model_xp") if len(held) >= 11 else None
        if xi is not None:
            total += float(xi.y.sum()) + float(xi.loc[xi.model_xp.idxmax()].y)
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
            top = cand.loc[cand.model_xp.idxmax()]
            cnt = clubs.reindex(squad).value_counts()
            if top["team"] != o["team"] and cnt.get(top["team"], 0) >= BS.MAX_CLUB:
                continue
            g = float(top.model_xp) - float(o["model_xp"])
            if g > best_gain:
                best_gain, swap = g, (o["player_code"], top["player_code"],
                                      float(costs.get(o["player_code"], 4.0)),
                                      float(top["cost"]))
        if swap:
            a, b, ca, cb = swap
            squad = [b if e == a else e for e in squad]
            bank += ca - cb
    return total


def main():
    rows = []
    for path in sorted(glob.glob(f"{CACHE}/*.pkl.gz")):
        panel = BS.load(path)
        season = str(pd.read_pickle(path, compression="gzip")["holdout_season"])
        gws = sorted(panel.gameweek.unique())[:BS.HORIZON]
        for label, frac in (("optimal", 0.00), ("handicap_5pct", 0.05),
                            ("handicap_15pct", 0.15), ("handicap_30pct", 0.30)):
            sq = (BS.pick_squad(panel, gws, "model_xp") if frac == 0
                  else handicapped_squad(panel, gws, frac))
            if sq is None:
                print(f"  {season} {label}: infeasible"); continue
            tot = simulate_from(panel, list(sq))
            rows.append({"season": season, "start": label, "total": tot})
            print(f"  {season}  {label:15s} {tot:7.0f}", flush=True)
    r = pd.DataFrame(rows)
    print("\n" + "=" * 68)
    print("SEASON TOTAL BY STARTING-SQUAD QUALITY (same policy throughout)")
    print("=" * 68)
    piv = r.pivot_table(index="start", columns="season", values="total")
    piv["mean"] = piv.mean(axis=1)
    order = ["optimal", "handicap_5pct", "handicap_15pct", "handicap_30pct"]
    piv = piv.reindex([o for o in order if o in piv.index])
    print(piv.round(0).to_string())
    base = piv.loc["optimal"]
    print("\ncost of starting worse (one free transfer a week to recover):")
    print((piv - base).drop(index="optimal").round(0).to_string())
    print("\n  If these gaps are large, the opening squad is destiny and a")
    print("  wildcard that resets it is worth far more than a one-week xP delta.")


if __name__ == "__main__":
    main()
