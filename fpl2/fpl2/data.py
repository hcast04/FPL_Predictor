"""
The canonical panel: exactly one row per (season, gameweek, player).

Two things this module exists to make impossible:

  * joining on a name. `web_name` is not unique -- "Palmer" is both Cole Palmer
    (CHE, MID) and an Ipswich goalkeeper. Everything keys on `player_code`,
    which is stable across seasons.
  * losing a double gameweek. A player with two fixtures in one gameweek scores
    in both. The panel sums them and records `fixtures` so nothing downstream
    has to guess.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

_POS_MAP = {"GK": "GK", "GKP": "GK", "DEF": "DEF", "MID": "MID",
            "AM": "MID", "FWD": "FWD"}

_SUM_COLS = ["total_points", "minutes", "goals_scored", "assists",
             "clean_sheets", "goals_conceded", "saves", "bonus", "bps",
             "expected_goals", "expected_assists", "starts",
             "defensive_contribution", "yellow_cards", "red_cards"]
# Market signals (O40). Carried through the panel but NOT summed across a
# double gameweek -- `selected` is a stock, not a flow, and adding a player's
# two fixtures would double his ownership. `first` is exact: the API reports
# one ownership figure per gameweek regardless of how many matches are played.
_MARKET_COLS = ["selected", "transfers_in", "transfers_out", "transfers_balance"]
_FIRST_COLS = ["position", "team_name", "team_short_name", "price_at_collection",
               "player_name"]


EXPECTED_COLS = ["expected_goals", "expected_assists",
                 "expected_goal_involvements", "expected_goals_conceded"]
COVERAGE_ZERO_THRESHOLD = 0.95
COVERAGE_MIN_MINUTES = 60


def apply_coverage_mask(d: pd.DataFrame) -> pd.DataFrame:
    """
    Mask expected-stats for gameweeks where they were never collected.

    In 2022-23 gameweeks 1-32, 100% of players with >=60 minutes have
    expected_goals == 0. That is absent data, not a measurement: a striker who
    played 90 minutes did not generate zero threat. Trained on as zeros, it
    teaches the model exactly the wrong thing.

    Detected from the data rather than hardcoded, so a future collection gap is
    caught the same way.
    """
    if "expected_goals" not in d.columns:
        return d
    played = d[pd.to_numeric(d["minutes"], errors="coerce").fillna(0)
               >= COVERAGE_MIN_MINUTES]
    if played.empty:
        return d
    zero_rate = (played.assign(_z=pd.to_numeric(played["expected_goals"],
                                                errors="coerce").fillna(0).eq(0))
                 .groupby(["season", "gameweek"])["_z"].mean())
    bad = set(zero_rate[zero_rate > COVERAGE_ZERO_THRESHOLD].index)
    if not bad:
        return d
    key = list(zip(d["season"], d["gameweek"]))
    mask = np.array([k in bad for k in key])
    for c in EXPECTED_COLS:
        if c in d.columns:
            d.loc[mask, c] = np.nan
    return d


def load_raw(extended: bool = False) -> pd.DataFrame:
    """
    The historical CSV, typed and position-normalised. One row per fixture.

    `extended=True` prepends 2020-21 and 2021-22 from the public FPL exports
    (see `legacy_seasons.py`). Those seasons carry no expected-stats and no
    `starts` -- FPL added both in 2022-23 -- so those columns arrive as NaN and
    MUST stay NaN. Their value is making 2022-23 a usable holdout, which takes
    the season-cluster count from 3 to 4 and the significance bar from
    |t| > 4.30 to 3.18 (R32). Default stays False so every prior result is
    reproducible unchanged.
    """
    d = pd.read_csv(C.HISTORICAL, compression="gzip", low_memory=False)
    if extended:
        from . import legacy_seasons as _ls
        d = pd.concat([_ls.load_all(), d], ignore_index=True, sort=False)
    for c in _SUM_COLS + ["price_at_collection"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["position"] = d["position"].map(_POS_MAP)
    d = d[d["position"].notna()].copy()
    d["kickoff"] = pd.to_datetime(d["kickoff_time"], errors="coerce", utc=True)
    return apply_coverage_mask(d)


def build_panel(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Collapse to one row per (season, gameweek, player_code).

    `fixtures` is the number of matches that player had in that gameweek:
    0 = blank (no fixture), 1 = normal, 2+ = double. Downstream code must
    branch on this rather than assuming one fixture per gameweek.
    """
    d = load_raw() if raw is None else raw
    # `min_count=1` so an all-NaN group stays NaN instead of collapsing to 0.0.
    #
    # Without it, `apply_coverage_mask`'s work is silently undone here: it NaNs
    # expected-stats for 2022-23 GW1-32 where they were never collected (45.2%
    # of 60+ minute rows in raw), and pandas' .sum() turns every one of those
    # into a genuine-looking 0.0. Absent data then re-enters as evidence --
    # polluting the xG/xA TRAINING TARGETS for ~21% of all training rows, and
    # inflating every conversion factor, because real goals get divided by an
    # expected total that includes false zeros (MID xA->assists 1.459 vs an
    # honest 1.339). Found via B.Fernandes's 3x assist over-prediction (O43).
    #
    # The identical bug was fixed once already in external.py. It recurred here
    # because `.sum()` reads as obviously correct.
    agg = {c: (c, lambda x: x.sum(min_count=1)) for c in _SUM_COLS if c in d.columns}
    agg.update({c: (c, "first") for c in _FIRST_COLS if c in d.columns})
    agg.update({c: (c, "first") for c in _MARKET_COLS if c in d.columns})
    agg["fixtures"] = ("fixture_id", "nunique")
    agg["opponents"] = ("opponent_short_name", lambda s: tuple(s))
    agg["kickoff_first"] = ("kickoff", "min")
    panel = (d.groupby(["season", "gameweek", C.PLAYER_KEY], as_index=False)
               .agg(**agg))
    # a global chronological index, so "before" is unambiguous across seasons
    seasons = sorted(panel["season"].unique())
    order = {s: i for i, s in enumerate(seasons)}
    panel["t"] = panel["season"].map(order) * 38 + panel["gameweek"]
    panel = panel.rename(columns={"price_at_collection": "cost",
                                  "team_short_name": "team",
                                  "total_points": "points"})
    return panel.sort_values(["t", C.PLAYER_KEY]).reset_index(drop=True)


def with_blanks(panel: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    Reindex a season so every player has a row in every gameweek.

    In the raw data a BLANK gameweek is a MISSING ROW, not a row with
    fixtures == 0. Code that iterates gameweeks and looks players up will
    silently skip blanks and treat them as "no information" rather than
    "scores nothing". Build the explicit grid when that distinction matters
    (squad simulation, horizon scoring); use the compact panel otherwise.
    """
    s = panel[panel["season"].eq(season)]
    gws = sorted(s["gameweek"].unique())
    players = s[C.PLAYER_KEY].unique()
    grid = pd.MultiIndex.from_product([gws, players],
                                      names=["gameweek", C.PLAYER_KEY])
    out = (s.set_index(["gameweek", C.PLAYER_KEY])
             .reindex(grid)
             .reset_index())
    out["season"] = season
    # a reindexed row is a genuine blank: no fixture, no points
    filled = out["fixtures"].isna()
    out.loc[filled, "fixtures"] = 0
    out.loc[filled, "points"] = 0.0
    out.loc[filled, "minutes"] = 0.0
    for c in ("position", "team", "cost"):
        if c in out.columns:
            out[c] = out.groupby(C.PLAYER_KEY)[c].transform(
                lambda x: x.ffill().bfill())
    return out


def usable_gameweeks(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop gameweeks too small to evaluate (pre-season, partial scrapes)."""
    n = panel.groupby(["season", "gameweek"])[C.PLAYER_KEY].transform("size")
    return panel[n >= C.MIN_PLAYERS_PER_GAMEWEEK].copy()


def sanity_report(panel: pd.DataFrame) -> dict:
    """Numbers worth eyeballing before trusting anything built on the panel."""
    dup = panel.duplicated(["season", "gameweek", C.PLAYER_KEY]).sum()
    return {
        "rows": len(panel),
        "seasons": sorted(panel["season"].unique()),
        "duplicate_player_gameweeks": int(dup),
        "double_gameweeks": int((panel["fixtures"] >= 2).sum()),
        "blank_gameweeks": int((panel["fixtures"] == 0).sum()),
        "players": int(panel[C.PLAYER_KEY].nunique()),
        "positions": panel["position"].value_counts().to_dict(),
        "mean_points": round(float(panel["points"].mean()), 4),
    }
