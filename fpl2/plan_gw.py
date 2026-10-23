#!/usr/bin/env python3
"""
This gameweek's decision with no chip: XI, armband, and the best transfer.

Mirrors the SHIPPED greedy policy in `simulate.simulate_season`: one free
transfer, taken only if the gain clears `min_gain = 0.30`, no hits, armband on
argmax(xp). That policy is not a guess -- S53 measured every alternative:

  * `min_gain` is a settled null (MDE 5-10 against effects of 0-5), so the
    threshold does not matter; just take the best available swap.
  * `argmax(xp)` beat all seven captain alternatives on a repaired signal.
    Form-blending it loses at every lambda tested (O54), captaining the
    premium loses (-20.3, 1/8), the tail objective is worth nothing.

SELLING PRICE CAVEAT. The public picks endpoint does not expose per-player
selling prices, and a player who has risen sells for less than `now_cost`
(you keep half the profit). Affordability here is computed from `now_cost`
and is therefore mildly OPTIMISTIC; a swap shown as just affordable may not
be. `WC_BUDGET` for the whole squad was £99.1m against an API valuation of
£100.1m, so the haircut across 15 players is about £1.0m.
"""
from __future__ import annotations
import json, os, sys, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, optimise                               # noqa: E402

API = "https://fantasy.premierleague.com/api"
HDR = {"User-Agent": "Mozilla/5.0"}
ENTRY = int(os.environ.get("FPL_ENTRY", "3642415"))
MIN_CHANCE, MIN_GAIN = 75, 0.30


def get(u):
    return json.load(urllib.request.urlopen(
        urllib.request.Request(u, headers=HDR), timeout=60))


def main():
    out = pd.read_pickle(Path.home() / "fpl2_live_chip_out.pkl")
    boot = get(f"{API}/bootstrap-static/")
    finished = sorted(e["id"] for e in boot["events"] if e.get("finished"))
    nxt = next(e for e in boot["events"] if e.get("is_next"))
    picks = get(f"{API}/entry/{ENTRY}/event/{max(finished)}/picks/")
    bank = picks["entry_history"]["bank"] / 10.0

    el = pd.DataFrame([{
        "code": e["code"], "id": e["id"], "name": e["web_name"],
        "cost": e["now_cost"] / 10.0, "owned": float(e["selected_by_percent"]),
        "status": e["status"], "chance": e["chance_of_playing_next_round"],
        "news": e["news"], "team_id": e["team"], "etype": e["element_type"]}
        for e in boot["elements"]])
    tm = {t["id"]: t["short_name"] for t in boot["teams"]}
    el["team"] = el.team_id.map(tm)
    el["position"] = el.etype.map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})
    last = int(out.gameweek.max())
    xp = (out[out.gameweek.eq(last)].groupby("player_code", as_index=False)
          .agg(xp=("xp", "mean")))
    pool = el.merge(xp, left_on="code", right_on="player_code", how="inner")
    pool = pool.drop(columns=["code"]).reset_index(drop=True)
    avail = pool[pool.status.eq("a") &
                 (pool.chance.isna() | (pool.chance >= MIN_CHANCE))]

    # AVAILABILITY. The model's `p_appear` is learned from minutes history and
    # knows nothing about today's injury news, so a flagged player keeps his
    # full xp and walks into the XI. An earlier version of this file did
    # exactly that with Calvert-Lewin (status 'd', 50%, foot injury). The
    # transfer POOL was filtered and the squad you already own was not.
    def adjust(df):
        ch = pd.to_numeric(df["chance"], errors="coerce")
        f = np.where(df["status"].eq("a"),
                     np.where(ch.isna(), 1.0, ch / 100.0),
                     np.where(ch.isna(), 0.0, ch / 100.0))
        out = df.copy()
        out["xp_raw"] = out["xp"]
        out["xp"] = out["xp"] * f
        out["avail"] = (f * 100).round().astype(int)
        return out

    ids = [p["element"] for p in picks["picks"]]
    squad = adjust(pool[pool.id.isin(ids)]).reset_index(drop=True)
    flagged = squad[squad.avail < 100]
    if len(flagged):
        print("FLAGGED in your squad (xp scaled by chance of playing):")
        for _, r in flagged.iterrows():
            print(f"   {r['name']:<16} {r['avail']:>3}%  xp {r['xp_raw']:.2f}"
                  f" -> {r['xp']:.2f}   {(r['news'] or '')[:44]}")
        print()
    print(f"GW{nxt['id']} deadline {nxt['deadline_time']} | bank £{bank:.1f}m")
    print(f"squad resolved: {len(squad)}/15\n")

    def best(sq):
        sq = sq.reset_index(drop=True)
        xi = optimise.best_xi(sq, "xp")
        if xi is None:
            return None, None, -np.inf
        xi = xi.reset_index(drop=True)
        cap = optimise.pick_captain(xi, "xp")
        return xi, cap, float(xi.xp.sum()) + float(cap.xp)

    xi, cap, base = best(squad)
    squad["role"] = np.where(squad.id.isin(xi.id), "XI", "BENCH")
    squad.loc[squad.id.eq(cap.id), "role"] = "CAPTAIN"
    print("=" * 66); print("A. NO TRANSFER — best XI and armband from what you own")
    print("=" * 66)
    print(squad.sort_values(["role", "xp"], ascending=[True, False])
          [["name", "position", "team", "cost", "xp", "owned", "role"]]
          .to_string(index=False, formatters={
              "cost": "£{:.1f}m".format, "xp": "{:.2f}".format,
              "owned": "{:.1f}%".format}))
    print(f"\n  XI + captain = {base:.2f} xp   (captain {cap['name']} {cap.xp:.2f})")
    bn = squad[squad.role.eq("BENCH")]
    print(f"  bench: {', '.join(bn.name)}  ({bn.xp.sum():.2f} xp)")

    print("\n" + "=" * 66); print("B. BEST SINGLE TRANSFER (one free, no hit)")
    print("=" * 66)
    rows = []
    held = set(squad.id)
    for _, o in squad.iterrows():
        cand = avail[(avail.position == o.position) & (~avail.id.isin(held)) &
                     (avail.cost <= o.cost + bank + 1e-9)]
        for _, i in cand.iterrows():
            club = squad[squad.id.ne(o.id)].team.tolist() + [i.team]
            if pd.Series(club).value_counts().max() > C.MAX_PER_CLUB:
                continue
            ns = pd.concat([squad[squad.id.ne(o.id)],
                            pd.DataFrame([i])], ignore_index=True)
            _, nc, tot = best(ns)
            if np.isfinite(tot):
                rows.append({"out": o["name"], "out_xp": o.xp, "in": i["name"],
                             "in_team": i.team, "in_xp": i.xp,
                             "spend": i.cost - o.cost, "gain": tot - base,
                             "new_cap": nc["name"]})
    T = pd.DataFrame(rows)
    if T.empty:
        print("  no legal transfer found"); return
    T = T.sort_values("gain", ascending=False).head(10)
    print(T.to_string(index=False, formatters={
        "out_xp": "{:.2f}".format, "in_xp": "{:.2f}".format,
        "spend": "£{:+.1f}m".format, "gain": "{:+.2f}".format}))
    top = T.iloc[0]
    print(f"\n  best: {top['out']} -> {top['in']} ({top['in_team']}), "
          f"{top['gain']:+.2f} xp, spend £{top['spend']:+.1f}m")
    print(f"  shipped policy takes it if gain > {MIN_GAIN} -> "
          f"{'TAKE IT' if top['gain'] > MIN_GAIN else 'HOLD (below threshold)'}")

    # show the resulting XI, because "is the incoming player even starting?"
    # is the first thing anyone asks and the pre-transfer table cannot say
    o_row = squad[squad.name.eq(top["out"])].iloc[0]
    i_row = adjust(avail[avail.name.eq(top["in"])]).iloc[0]
    ns = pd.concat([squad[squad.id.ne(o_row.id)],
                    pd.DataFrame([i_row])], ignore_index=True)
    nxi, ncap, ntot = best(ns)
    ns["role"] = np.where(ns.id.isin(nxi.id), "XI", "BENCH")
    ns.loc[ns.id.eq(ncap.id), "role"] = "CAPTAIN"
    print("\n" + "=" * 66)
    print(f"C. XI AFTER THAT TRANSFER  ({ntot:.2f} xp, was {base:.2f})")
    print("=" * 66)
    print(ns.sort_values(["role", "xp"], ascending=[True, False])
          [["name", "position", "team", "cost", "xp", "avail", "role"]]
          .to_string(index=False, formatters={
              "cost": "£{:.1f}m".format, "xp": "{:.2f}".format,
              "avail": "{}%".format}))
    nb = ns[ns.role.eq("BENCH")]
    print(f"\n  captain {ncap['name']} {ncap.xp:.2f}")
    print(f"  bench: {', '.join(nb.name)}  ({nb.xp.sum():.2f} xp)")
    print(f"  incoming {top['in']} -> "
          f"{'STARTS' if top['in'] in set(nxi.name) else 'BENCHED'}")


if __name__ == "__main__":
    main()
