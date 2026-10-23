"""
Build a 2026-27 panel from the FPL API, in the same schema as the historical CSV.

The historical file stops at 2025-26, so the live season has to be assembled
from the per-gameweek `event/{gw}/live/` endpoint. The schema is matched
column-for-column so the same feature builder, model and simulator run over it
unchanged -- if the live season needed its own code path, any comparison with
the holdout seasons would be measuring the code path as much as the model.

KNOWN LIMITATION: prices are taken as CURRENT, not as they were at each
gameweek. FPL does not expose historical prices in bulk. Over seven gameweeks
drift is typically <= GBP 0.3m, but budget-constrained squad selection is
mildly affected and the simulation should be read with that in mind.
"""
from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd

from . import config as C

API = "https://fantasy.premierleague.com/api"
_HEADERS = {"User-Agent": "Mozilla/5.0"}
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

STAT_COLS = ["minutes", "goals_scored", "assists", "clean_sheets",
             "goals_conceded", "own_goals", "penalties_saved",
             "penalties_missed", "yellow_cards", "red_cards", "saves",
             "bonus", "bps", "influence", "creativity", "threat",
             "ict_index", "expected_goals", "expected_assists",
             "expected_goal_involvements", "expected_goals_conceded",
             "total_points", "starts", "defensive_contribution",
             "tackles", "recoveries", "clearances_blocks_interceptions"]


def _get(url):
    return json.load(urllib.request.urlopen(
        urllib.request.Request(url, headers=_HEADERS), timeout=60))


def build(season: str = "2026-27", through_gameweek: int | None = None) -> pd.DataFrame:
    """One row per player-fixture, matching the historical schema."""
    boot = _get(f"{API}/bootstrap-static/")
    el = {e["id"]: e for e in boot["elements"]}
    teams = {t["id"]: t["short_name"] for t in boot["teams"]}
    names = {t["id"]: t["name"] for t in boot["teams"]}

    finished = [e["id"] for e in boot["events"] if e.get("finished")]
    if through_gameweek is not None:
        finished = [g for g in finished if g <= through_gameweek]
    if not finished:
        raise RuntimeError("no finished gameweeks yet")

    rows = []
    for gw in finished:
        fixtures = {f["id"]: f for f in _get(f"{API}/fixtures/?event={gw}")}
        live = _get(f"{API}/event/{gw}/live/")
        for item in live["elements"]:
            pid = item["id"]
            meta = el.get(pid)
            if meta is None:
                continue
            played_fixtures = [b["fixture"] for b in (item.get("explain") or [])]
            if not played_fixtures:
                continue
            st = item["stats"]
            for fid in played_fixtures:
                fx = fixtures.get(fid)
                if fx is None:
                    continue
                home = fx["team_h"] == meta["team"]
                opp = fx["team_a"] if home else fx["team_h"]
                r = {"season": season, "season_start_year": 2026, "gameweek": gw,
                     "fixture_id": fid, "kickoff_time": fx.get("kickoff_time"),
                     "player_code": meta["code"], "player_id_season": pid,
                     "player_name": f'{meta["first_name"]} {meta["second_name"]}',
                     "position": POS.get(meta["element_type"]),
                     "team_id": meta["team"], "team_name": names[meta["team"]],
                     "team_short_name": teams[meta["team"]],
                     "opponent_id": opp, "opponent_name": names[opp],
                     "opponent_short_name": teams[opp],
                     "was_home": home,
                     "price_at_collection": meta["now_cost"] / 10.0,
                     "team_h_score": fx.get("team_h_score"),
                     "team_a_score": fx.get("team_a_score")}
                # a double gameweek splits the gameweek total across fixtures;
                # the API reports per-gameweek stats, so divide evenly rather
                # than double-count. Single gameweeks are unaffected.
                share = 1.0 / len(played_fixtures)
                for c in STAT_COLS:
                    v = pd.to_numeric(st.get(c), errors="coerce")
                    r[c] = (float(v) * share) if pd.notna(v) else np.nan
                hs, as_ = fx.get("team_h_score"), fx.get("team_a_score")
                r["team_goals"] = hs if home else as_
                r["opponent_goals"] = as_ if home else hs
                rows.append(r)
    d = pd.DataFrame(rows)
    d["value"] = d["price_at_collection"]
    return d


# --- contemporaneous ownership and price (ledger O36) ----------------------
#
# `bootstrap-static` reports `selected_by_percent` as of NOW. Applying it
# retroactively to an earlier gameweek is circular: managers buy players who
# have already scored, so "highly owned in GW6" computed after GW7 is partly a
# restatement of "scored well in GW6". A whole feature-engineering effort
# (ledger F23, -83 pts/season) was built on that artefact before M19 caught it.
#
# `element-summary/{id}/history` carries a per-gameweek snapshot instead:
#
#     selected        managers holding the player AT THAT gameweek's deadline
#     value           his price that week, in tenths -- the historical price
#                     the module docstring says FPL "does not expose in bulk"
#     transfers_in/out  the market's movement around that deadline
#
# `selected` is a COUNT, and the denominator grows all season (8.90M ranked
# entries at GW1, 11.06M at GW7 -- a 24% rise). Comparing raw counts across
# gameweeks would read that growth as rising ownership for everybody, so the
# share is normalised by `ranked_count` from the same gameweek's event record.

def ranked_counts(boot: dict | None = None) -> dict:
    """gameweek -> number of ranked entries, the honest ownership denominator."""
    boot = boot or _get(f"{API}/bootstrap-static/")
    return {e["id"]: e.get("ranked_count")
            for e in boot["events"] if e.get("ranked_count")}


def ownership(season: str = "2026-27", through_gameweek: int | None = None,
              workers: int = 6, progress: bool = False) -> pd.DataFrame:
    """
    Per-player, per-gameweek ownership and price AS THEY WERE AT THE DEADLINE.

    One request per player (~670), so it is slow and worth caching. Returns
    one row per (player, gameweek) with `player_code` -- the key the panel
    joins on -- not the per-season element id, which is not stable.
    """
    from concurrent.futures import ThreadPoolExecutor

    boot = _get(f"{API}/bootstrap-static/")
    el = {e["id"]: e for e in boot["elements"]}
    ranked = ranked_counts(boot)
    finished = {e["id"] for e in boot["events"] if e.get("finished")}
    if through_gameweek is not None:
        finished = {g for g in finished if g <= through_gameweek}

    def one(pid: int):
        try:
            hist = _get(f"{API}/element-summary/{pid}/")["history"]
        except Exception:
            return []
        meta = el[pid]
        out = []
        for h in hist:
            gw = h.get("round")
            if gw not in finished:
                continue
            out.append({"season": season, "gameweek": gw,
                        "player_code": meta["code"], "player_id_season": pid,
                        "player_name": f'{meta["first_name"]} {meta["second_name"]}',
                        "web_name": meta["web_name"],
                        "selected": h.get("selected"),
                        "price_at_gameweek": (h.get("value") or 0) / 10.0,
                        "transfers_in": h.get("transfers_in"),
                        "transfers_out": h.get("transfers_out")})
        return out

    rows = []
    ids = sorted(el)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, got in enumerate(ex.map(one, ids)):
            rows.extend(got)
            if progress and (i + 1) % 100 == 0:
                print(f"  ... {i + 1}/{len(ids)} players", flush=True)

    d = pd.DataFrame(rows)
    if d.empty:
        return d
    # a double gameweek repeats the player's row per fixture in `build()` but
    # NOT here: element-summary history is per fixture too, so collapse to one
    # ownership figure per (player, gameweek). The snapshot is identical across
    # a player's fixtures in the same week, so `first` is exact, not a choice.
    d = (d.sort_values(["player_code", "gameweek"])
          .groupby(["season", "gameweek", "player_code"], as_index=False)
          .agg(player_id_season=("player_id_season", "first"),
               player_name=("player_name", "first"),
               web_name=("web_name", "first"),
               selected=("selected", "first"),
               price_at_gameweek=("price_at_gameweek", "first"),
               transfers_in=("transfers_in", "sum"),
               transfers_out=("transfers_out", "sum")))
    d["ranked_entries"] = d["gameweek"].map(ranked)
    d["owned_pct"] = 100.0 * d["selected"] / d["ranked_entries"]
    return d
