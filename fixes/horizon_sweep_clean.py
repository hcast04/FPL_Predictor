#!/usr/bin/env python3
"""
Decision horizon, WITHOUT look-ahead leakage.

horizon_sweep.py scored future gameweeks with the fold cache's own
expected_points for those weeks. Those forecasts were produced walk-forward,
so the xP for gameweek g+5 already incorporates form and availability from
g..g+4. Reading them at time g is leakage, and it inflates the horizon benefit.

What a manager genuinely knows at gameweek g:
  * every player's xP for gameweek g              (the current forecast)
  * the FIXTURE SCHEDULE for all future gameweeks (published in advance)
  * team strength estimated from results up to g-1

So the leakage-free horizon score is the player's CURRENT xP, scaled by how
favourable his next H fixtures are, where fixture difficulty uses only
information available before g.

  hz(p, g, H) = xp(p, g) * mean_{k in [g, g+H-1]} ease(opponent(p, k) | data < g)

ease() is built from each team's goals conceded and scored, expanding-mean to
g-1 only. Players with no fixture in a week contribute nothing (blanks), and a
double gameweek contributes twice.

Run:  .venv/bin/python fixes/horizon_sweep_clean.py
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_showdown as BS   # noqa: E402

CACHE = "data/walk_forward_fold_cache_base_v2_4_0"
HORIZONS = [1, 2, 3, 4, 5, 6, 8, 10, 12]


def load_with_fixtures(path):
    pl = pd.read_pickle(path, compression="gzip")
    d = pl["bundle"]["holdout_expected_points"].copy()
    d["y"] = pd.to_numeric(d.target_fpl_points, errors="coerce")
    d["cost"] = pd.to_numeric(d.price_at_collection, errors="coerce")
    d["model_xp"] = pd.to_numeric(d.expected_points, errors="coerce")
    d = d[d.y.notna() & d.cost.notna() & d.model_position.isin(["GK","DEF","MID","FWD"])]
    rows = (d.groupby(["gameweek", "player_code"], as_index=False)
              .agg(y=("y","sum"), cost=("cost","first"), pos=("model_position","first"),
                   team=("team_short_name","first"), model_xp=("model_xp","sum"),
                   opp=("opponent_short_name","first")))
    # per-team, per-gameweek concede/score, then an EXPANDING mean using
    # strictly earlier gameweeks only -- no information from g or later
    tg = (d.groupby(["gameweek","team_short_name"], as_index=False)
            .agg(gf=("team_goals","first"), ga=("opponent_goals","first")))
    tg = tg.sort_values(["team_short_name","gameweek"])
    for c in ("gf","ga"):
        tg[c+"_hist"] = (tg.groupby("team_short_name")[c]
                           .transform(lambda s: s.shift().expanding().mean()))
    league_ga = tg["ga_hist"].mean()
    tg["ease"] = tg["ga_hist"] / (league_ga if league_ga and np.isfinite(league_ga) else 1.0)
    ease = {(int(r.gameweek), r.team_short_name): (r.ease if np.isfinite(r.ease) else 1.0)
            for r in tg.itertuples(index=False)}
    return rows, ease, str(pl["holdout_season"])


def add_hz(panel, ease, H):
    p = panel.copy()
    if H == 1:
        p["hz"] = p["model_xp"]
        return p
    # opponent of each team in each future gameweek (the published schedule)
    sched = {(int(r.gameweek), r.team): r.opp
             for r in panel[["gameweek","team","opp"]].drop_duplicates().itertuples(index=False)}
    gws = sorted(panel.gameweek.unique())
    out = []
    for g in gws:
        wk = p[p.gameweek.eq(g)].copy()
        future = [k for k in gws if g <= k < g + H]
        mult = []
        for t in wk["team"]:
            vals = []
            for k in future:
                o = sched.get((k, t))
                if o is None:          # blank gameweek for this club
                    vals.append(0.0)
                else:                  # ease uses only data strictly before g
                    vals.append(ease.get((min(g, k), o), 1.0))
            mult.append(float(np.mean(vals)) if vals else 1.0)
        wk["hz"] = wk["model_xp"].to_numpy(float) * np.asarray(mult)
        out.append(wk)
    return pd.concat(out, ignore_index=True)


def simulate(panel, ease, H):
    p = add_hz(panel, ease, H)
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
            top = cand.loc[cand.hz.idxmax()]
            cnt = clubs.reindex(squad).value_counts()
            if top["team"] != o["team"] and cnt.get(top["team"], 0) >= BS.MAX_CLUB:
                continue
            g_ = float(top.hz) - float(o["hz"])
            if g_ > best_gain:
                best_gain, swap = g_, (o["player_code"], top["player_code"],
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
        panel, ease, season = load_with_fixtures(path)
        for H in HORIZONS:
            t = simulate(panel, ease, H)
            rows.append({"season": season, "H": H, "total": t})
            print(f"  {season}  H={H:2d}  {t:7.0f}", flush=True)
    r = pd.DataFrame(rows)
    piv = r.pivot_table(index="H", columns="season", values="total")
    piv["mean"] = piv.mean(axis=1)
    print("\n" + "=" * 64)
    print("LEAKAGE-FREE DECISION HORIZON  (H=1 is current production)")
    print("=" * 64)
    print(piv.round(0).to_string())
    d = (piv - piv.loc[1]).drop(index=1)
    print("\ndelta vs H=1:")
    print(d.round(0).to_string())
    cols = [c for c in piv.columns if c != "mean"]
    ok = [int(h) for h in d.index if all(d.loc[h, c] > 0 for c in cols)]
    print(f"\nhorizons beating H=1 in ALL three seasons: {ok if ok else 'none'}")
    if ok:
        best = max(ok, key=lambda h: piv.loc[h, "mean"])
        print(f"  best consistent horizon: H={best} "
              f"({piv.loc[best,'mean']-piv.loc[1,'mean']:+.0f} pts/season)")


if __name__ == "__main__":
    main()
