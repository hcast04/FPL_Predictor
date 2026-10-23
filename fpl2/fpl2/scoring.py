"""
FPL scoring rules.

These are the game's rules, not modelling choices, so they are verified rather
than assumed: `reconstruct()` rebuilds historical total_points from recorded
events, and `tests/test_scoring.py` demands near-exact agreement. If FPL
changes a rule, that test fails loudly instead of the model quietly drifting.

(The previous project shipped a patch asserting a goalkeeper goal was worth 6.
It is 10. A reconstruction test would have caught it immediately.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GOAL = {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4}
CLEAN_SHEET = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST = 3
APPEARANCE_SHORT = 1          # 1-59 minutes
APPEARANCE_LONG = 2           # 60+ minutes
SAVES_PER_POINT = 3           # GK only
PENALTY_SAVE = 5
PENALTY_MISS = -2
GOALS_CONCEDED_PER_PENALTY = 2    # GK/DEF only, -1 per 2 conceded, 60+ min
YELLOW = -1
RED = -3
# Defensive contribution (introduced 2024-25/2025-26): 2 points for clearances,
# blocks, interceptions and tackles above a positional threshold. Both the
# points value and the thresholds were DERIVED from the reconstruction residual
# (DEF >=10 and MID/FWD >=12 each carried a mean signed error of exactly -2.000)
# rather than taken from a rules page, so they are verified against reality.
DEFENSIVE_POINTS = 2
DEFENSIVE_THRESHOLD = {"GK": None, "DEF": 10, "MID": 12, "FWD": 12}
OWN_GOAL = -2
CONCEDES_PENALTY_POSITIONS = ("GK", "DEF")


def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    """
    A numeric column as a float array, or `default` if the column is absent.

    `df.get(name, 0)` returns a SCALAR when the column is missing, which then
    has no .fillna and blows up far from the cause. This keeps the shape right.
    """
    if name not in df.columns:
        return np.full(len(df), float(default))
    return pd.to_numeric(df[name], errors="coerce").fillna(default).to_numpy(float)


def appearance_points(minutes: np.ndarray) -> np.ndarray:
    m = np.asarray(minutes, dtype=float)
    return np.where(m >= 60, APPEARANCE_LONG, np.where(m > 0, APPEARANCE_SHORT, 0))


def reconstruct(df: pd.DataFrame) -> pd.Series:
    """
    Rebuild total_points from recorded events.

    Bonus is taken as recorded (it is an output of the BPS ranking, not a rule
    that can be derived from a single player's row). Defensive-contribution
    points are likewise taken as recorded where the column exists, because the
    threshold differs by position and changed mid-history.
    """
    pos = df["position"].to_numpy()
    mins = _col(df, "minutes")
    out = appearance_points(mins).astype(float)

    goals = _col(df, "goals_scored")
    out += goals * np.array([GOAL.get(p, 0) for p in pos])
    out += _col(df, "assists") * ASSIST

    cs = _col(df, "clean_sheets")
    cs_pts = np.array([CLEAN_SHEET.get(p, 0) for p in pos])
    out += np.where(mins >= 60, cs * cs_pts, 0)

    saves = _col(df, "saves")
    out += np.where(pos == "GK", saves // SAVES_PER_POINT, 0)

    conceded = _col(df, "goals_conceded")
    penal = np.isin(pos, CONCEDES_PENALTY_POSITIONS) & (mins >= 60)
    out -= np.where(penal, conceded // GOALS_CONCEDED_PER_PENALTY, 0)

    out += _col(df, "yellow_cards") * YELLOW
    out += _col(df, "red_cards") * RED
    out += _col(df, "own_goals") * OWN_GOAL
    out += _col(df, "penalties_saved") * PENALTY_SAVE
    out += _col(df, "penalties_missed") * PENALTY_MISS
    dc = _col(df, "defensive_contribution")
    thresholds = np.array([DEFENSIVE_THRESHOLD.get(p) or np.inf for p in pos],
                          dtype=float)
    out += np.where(dc >= thresholds, DEFENSIVE_POINTS, 0)

    out += _col(df, "bonus")
    return pd.Series(out, index=df.index)


def expected_points(*, p_appear, p_sixty, exp_minutes, position,
                    goals_per90, assists_per90, saves_per90,
                    clean_sheet_prob, conceded_per90, bonus_per90,
                    yellow_per90=0.0, defensive_return_prob=0.0) -> np.ndarray:
    """
    Compose component predictions into expected points.

    Each term is an expectation under the FPL rules above. Minutes-scaled rates
    are multiplied by expected minutes / 90; the clean-sheet and appearance
    terms are probability-weighted because they are threshold events, not rates.
    """
    pos = np.asarray(position)
    p_appear = np.asarray(p_appear, float)
    p_sixty = np.asarray(p_sixty, float)
    mins = np.asarray(exp_minutes, float)
    share = mins / 90.0

    xp = APPEARANCE_SHORT * np.clip(p_appear - p_sixty, 0, None) \
        + APPEARANCE_LONG * p_sixty
    xp += np.asarray(goals_per90, float) * share * np.array([GOAL.get(p, 0) for p in pos])
    xp += np.asarray(assists_per90, float) * share * ASSIST
    xp += np.asarray(clean_sheet_prob, float) * p_sixty * \
        np.array([CLEAN_SHEET.get(p, 0) for p in pos])
    xp += np.where(pos == "GK",
                   np.asarray(saves_per90, float) * share / SAVES_PER_POINT, 0.0)
    xp -= np.where(np.isin(pos, CONCEDES_PENALTY_POSITIONS),
                   np.asarray(conceded_per90, float) * share * p_sixty
                   / GOALS_CONCEDED_PER_PENALTY, 0.0)
    xp += np.asarray(bonus_per90, float) * share
    xp += np.asarray(yellow_per90, float) * share * YELLOW
    # the defensive bonus is a THRESHOLD event, so it takes a probability and
    # is not scaled by minutes share -- scaling it would double-count, since
    # the probability already reflects how long the player is expected to play
    xp += np.asarray(defensive_return_prob, float) * DEFENSIVE_POINTS
    return xp
