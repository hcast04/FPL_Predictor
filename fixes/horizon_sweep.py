#!/usr/bin/env python3
"""
How many gameweeks ahead should the optimiser look?

The live pipeline reports planning_horizon: 1 -- every squad and transfer
decision is made on next gameweek's xP alone. That is cheap and myopic: a
player bought for one good fixture may have four bad ones behind it, and the
cost only shows up weeks later.

This sweeps the horizon H used for DECISIONS:

  * opening squad  : maximise mean xP over the next H gameweeks
  * weekly transfer: rank candidates on mean xP over the next H gameweeks

while SCORING is always the current gameweek only -- you bank the points of the
week you are in, never a forecast. Confusing those two is the easy way to get a
spuriously good horizon result.

H = 1 reproduces current production behaviour.

Run:  .venv/bin/python fixes/horizon_sweep.py
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_showdown as BS   # noqa: E402

CACHE = "data/walk_forward_fold_cache_base_v2_4_0"
HORIZONS = [1, 2, 3, 4, 5, 6, 8]


def add_horizon_score(panel: pd.DataFrame, H: int) -> pd.DataFrame:
    """hz = mean model_xp over gameweeks [g, g+H-1] for each player."""
    p = panel.sort_values(["player_code", "gameweek"]).copy()
    if H == 1:
        p["hz"] = p["model_xp"]
        return p
    # forward-looking rolling mean, per player, over the NEXT H gameweeks
    p["hz"] = (p.groupby("player_code")["model_xp"]
                 .transform(lambda s: s[::-1].rolling(H, min_periods=1).mean()[::-1]))
    return p


def simulate_horizon(panel: pd.DataFrame, H: int) -> float:
    p = add_horizon_score(panel, H)
    gws = sorted(p.gameweek.unique())
    squad = BS.pick_squad(p, gws[:max(BS.HORIZON, H)], "hz")
    if squad is None:
        return float("nan")
    squad = list(squad)
    costs = p.groupby("player_code")["cost"].first()
    clubs = p.groupby("player_code")["team"].first()
    bank = BS.BUDGET - float(costs.reindex(squad).sum())
    total = 0.0
    for gw in gws:
        wk = p[p.gameweek.eq(gw)]
        held = wk[wk.player_code.isin(squad)]
        # XI and captain are chosen on THIS week's xP -- that is what scores
        xi = BS.best_xi(held, "model_xp") if len(held) >= 11 else None
        if xi is not None:
            total += float(xi.y.sum()) + float(xi.loc[xi.model_xp.idxmax()].y)
        nxt = p[p.gameweek.eq(gw + 1)]
        if nxt.empty:
            continue
        owned = nxt[nxt.player_code.isin(squad)]
        best_gain, swap = 0.30, None
        for _, o in owned.iterrows():
            cand = nxt[nxt.pos.eq(o["pos"]) & ~nxt.player_code.isin(squad)]
            cand = cand[cand.cost.le(bank + float(costs.get(o["player_code"], 4.0)))]
            if cand.empty:
                continue
            top = cand.loc[cand.hz.idxmax()]          # decision on the horizon
            cnt = clubs.reindex(squad).value_counts()
            if top["team"] != o["team"] and cnt.get(top["team"], 0) >= BS.MAX_CLUB:
                continue
            g = float(top.hz) - float(o["hz"])
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
        for H in HORIZONS:
            tot = simulate_horizon(panel, H)
            rows.append({"season": season, "H": H, "total": tot})
            print(f"  {season}  H={H}  {tot:7.0f}", flush=True)
    r = pd.DataFrame(rows)
    piv = r.pivot_table(index="H", columns="season", values="total")
    piv["mean"] = piv.mean(axis=1)
    print("\n" + "=" * 62)
    print("SEASON TOTAL BY DECISION HORIZON  (H=1 is current production)")
    print("=" * 62)
    print(piv.round(0).to_string())
    base = piv.loc[1]
    d = (piv - base).drop(index=1)
    print("\ndelta vs H=1:")
    print(d.round(0).to_string())
    print("\nhorizons that beat H=1 in ALL three seasons:")
    cols = [c for c in piv.columns if c != "mean"]
    ok = [int(h) for h in d.index if all(d.loc[h, c] > 0 for c in cols)]
    print("  ", ok if ok else "none")
    if ok:
        best = max(ok, key=lambda h: piv.loc[h, "mean"])
        print(f"   best consistent horizon: H={best} "
              f"({piv.loc[best,'mean']-piv.loc[1,'mean']:+.0f} pts/season)")


if __name__ == "__main__":
    main()
