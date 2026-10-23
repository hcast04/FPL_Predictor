"""
Adapt the 2020-21 and 2021-22 public FPL exports into our historical schema.

WHY. Every statistical question on this project is now bounded by having only
THREE holdout seasons. Clustered by season (R32) the bar is |t| > 4.30 on 2
degrees of freedom, and almost nothing clears it. Adding two earlier seasons
makes 2022-23 a usable holdout -- it finally has something before it -- taking
G from 3 to 4 and the bar from 4.30 to 3.18.

WHAT THESE SEASONS DO NOT HAVE, and must never be filled with zero:

    expected_goals, expected_assists   FPL added them in 2022-23
    starts                             likewise
    defensive_contribution             2025-26 only

M20 was exactly this mistake -- absent data encoded as 0.0 in three places,
teaching the model that players who generated xG generated none. These columns
are written as **NaN** and `data.build_panel`'s `min_count=1` keeps them NaN.
Anything downstream that fills them must be audited; `minutes.targets` is known
to `.fillna(0.0)` on `starts` and is handled by the caller.

JOIN KEY. `merged_gw.csv` carries `element`, a PER-SEASON id. `players_raw.csv`
carries both `id` and `code`, giving an EXACT element -> player_code map with
no name matching -- which is what produced the Nobel Mendy / Nampalys Mendy
error when this project last tried to join on names.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_extra"
SEASONS = ("2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22")
# present in our schema, genuinely absent in these exports
ABSENT = ["expected_goals", "expected_assists", "expected_goal_involvements",
          "expected_goals_conceded", "starts", "defensive_contribution",
          "tackles", "recoveries", "clearances_blocks_interceptions"]
POS = {"GK": "GK", "GKP": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}


def _read(path: Path) -> pd.DataFrame:
    """2018-19 is latin-1, everything else utf-8. Try, then fall back."""
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")


# FPL element_type -> position, for the seasons whose gw export omits `position`
ETYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _code_names() -> dict:
    """team_code -> (short_name, full_name), pooled from every file that has it."""
    out = {}
    try:
        from . import data as _d
        cur = _d.load_raw()
        for _, r in cur[["team_code", "team_short_name", "team_name"]].drop_duplicates().iterrows():
            if pd.notna(r["team_code"]):
                out[int(r["team_code"])] = (r["team_short_name"], r["team_name"])
    except Exception:
        pass
    for s in ("2019-20", "2020-21", "2021-22"):
        f = RAW_DIR / f"teams_{s}.csv"
        if f.exists() and f.stat().st_size > 100:
            try:
                t = _read(f)
                for _, r in t.iterrows():
                    out.setdefault(int(r["code"]), (r["short_name"], r["name"]))
            except Exception:
                pass
    return out


def load_season(season: str) -> pd.DataFrame:
    gw = _read(RAW_DIR / f"merged_gw_{season}.csv")
    pl = _read(RAW_DIR / f"players_raw_{season}.csv")
    # teams.csv is NOT read: it 404s for 2016-17..2018-19 and team identity
    # comes from players_raw's stable `team_code` instead.
    # 2016-17..2019-20 do not carry `position` or `team` in the gw export;
    # both come from players_raw, which is authoritative anyway.
    if "position" not in gw.columns:
        gw["position"] = gw["element"].map(pl.set_index("id")["element_type"]).map(ETYPE)
    # Team identity comes from `team_code`, which is STABLE across seasons,
    # not from teams.csv -- that file 404s for 2016-17..2018-19 and curl wrote
    # the string "404: Not Found" into it, which parsed as a 1-column frame.
    # A file existing is not a file being real.
    code = pl.set_index("id")["team_code"]
    gw["_tcode"] = gw["element"].map(code)

    code = pl.set_index("id")["code"]
    name = (pl["first_name"].fillna("") + " " + pl["second_name"].fillna("")).str.strip()
    name.index = pl["id"]
    # the gw file names the player's OWN team in full; the opponent by id

    d = pd.DataFrame({
        "season": season,
        "season_start_year": int(season[:4]),
        "gameweek": pd.to_numeric(gw["GW"] if "GW" in gw.columns
                                   else gw["round"], errors="coerce"),
        "fixture_id": pd.to_numeric(gw["fixture"], errors="coerce"),
        "kickoff_time": gw["kickoff_time"],
        "player_id_season": pd.to_numeric(gw["element"], errors="coerce"),
        "position": gw["position"].map(POS),
        "was_home": gw["was_home"],
        "price_at_collection": pd.to_numeric(gw["value"], errors="coerce") / 10.0,
        "team_h_score": pd.to_numeric(gw["team_h_score"], errors="coerce"),
        "team_a_score": pd.to_numeric(gw["team_a_score"], errors="coerce"),
    })
    d["player_code"] = d["player_id_season"].map(code)
    d["player_name"] = d["player_id_season"].map(name)
    # team_code -> short name, pooled from every source that has it. Six codes
    # (25, 38, 80, 88, 97, 110) belong to clubs relegated before 2022-23 and
    # never promoted since, so no source we hold names them. Their season
    # patterns narrow it to pairs but do not resolve them -- 25 and 88 both
    # appear in 2016-17 only -- so they are labelled `T<code>` rather than
    # guessed. The label is cosmetic: it only has to be CONSISTENT, and the
    # Understat join matches clubs by fixture-date pattern, not by name.
    names = _code_names()
    d["team_code"] = gw["_tcode"]
    d["team_short_name"] = d["team_code"].map(lambda c: names.get(c, (f"T{int(c)}",))[0]
                                              if pd.notna(c) else None)
    d["team_name"] = d["team_code"].map(
        lambda c: names.get(c, (None, f"Team {int(c)}"))[-1] if pd.notna(c) else None)
    season_team = pl.set_index("id")["team"]
    code_by_season_team = (pl[["team", "team_code"]].drop_duplicates()
                             .set_index("team")["team_code"])
    d["opponent_id"] = pd.to_numeric(gw["opponent_team"], errors="coerce")
    opp_code = d["opponent_id"].map(code_by_season_team)
    d["opponent_code"] = opp_code
    d["opponent_short_name"] = opp_code.map(
        lambda c: names.get(c, (f"T{int(c)}",))[0] if pd.notna(c) else None)
    d["opponent_name"] = opp_code.map(
        lambda c: names.get(c, (None, f"Team {int(c)}"))[-1] if pd.notna(c) else None)

    for c in ("minutes", "goals_scored", "assists", "clean_sheets",
              "goals_conceded", "own_goals", "penalties_saved",
              "penalties_missed", "yellow_cards", "red_cards", "saves",
              "bonus", "bps", "influence", "creativity", "threat",
              "ict_index", "total_points", "selected", "transfers_in",
              "transfers_out", "transfers_balance"):
        if c in gw.columns:
            d[c] = pd.to_numeric(gw[c], errors="coerce")

    # absent, not zero (M20)
    for c in ABSENT:
        d[c] = np.nan

    home = d["was_home"].astype(bool)
    d["team_goals"] = np.where(home, d["team_h_score"], d["team_a_score"])
    d["opponent_goals"] = np.where(home, d["team_a_score"], d["team_h_score"])
    return d[d["position"].notna() & d["player_code"].notna()].copy()


def load_all() -> pd.DataFrame:
    return pd.concat([load_season(s) for s in SEASONS], ignore_index=True)
