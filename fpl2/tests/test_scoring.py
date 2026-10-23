"""Verify the scoring rules against reality, rather than trusting them."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpl2 import data, scoring     # noqa: E402


@pytest.fixture(scope="module")
def raw():
    return data.load_raw()


def test_reconstruction_matches_recorded_points(raw):
    """Rebuilt points must match FPL's own total_points for almost every row."""
    got = scoring.reconstruct(raw)
    want = pd.to_numeric(raw["total_points"], errors="coerce")
    ok = (got - want).abs() <= 1e-9
    rate = float(ok.mean())
    assert rate > 0.97, (
        f"only {100*rate:.2f}% of rows reconstruct exactly; "
        f"mean abs error {float((got-want).abs().mean()):.4f}")


def test_goalkeeper_goal_is_worth_ten():
    """
    NEGATIVE CONTROL for a real past error: a GK goal is 10, not 6.

    Built synthetically because goalkeeper goals are too rare to appear in four
    seasons of data -- which is exactly why the wrong value survived before.
    """
    row = pd.DataFrame([{
        "position": "GK", "minutes": 90, "goals_scored": 1, "assists": 0,
        "clean_sheets": 0, "goals_conceded": 1, "saves": 3, "bonus": 3,
        "yellow_cards": 0, "red_cards": 0, "own_goals": 0,
        "penalties_saved": 0, "penalties_missed": 0}])
    pts = float(scoring.reconstruct(row).iloc[0])
    # 2 appearance + 10 goal + 1 save point + 3 bonus, no CS, 1 conceded (<2)
    assert pts == pytest.approx(2 + 10 + 1 + 3)
    assert scoring.GOAL["GK"] == 10


def test_appearance_thresholds():
    m = np.array([0, 1, 59, 60, 90])
    assert list(scoring.appearance_points(m)) == [0, 1, 1, 2, 2]


def test_expected_points_is_monotonic_in_minutes():
    base = dict(p_appear=[1.0], p_sixty=[1.0], position=["MID"],
                goals_per90=[0.5], assists_per90=[0.3], saves_per90=[0.0],
                clean_sheet_prob=[0.3], conceded_per90=[1.2], bonus_per90=[0.4])
    low = scoring.expected_points(exp_minutes=[30], **base)[0]
    high = scoring.expected_points(exp_minutes=[90], **base)[0]
    assert high > low


def test_clean_sheet_only_counts_with_sixty_minutes():
    base = dict(p_appear=[1.0], exp_minutes=[90], position=["DEF"],
                goals_per90=[0.0], assists_per90=[0.0], saves_per90=[0.0],
                clean_sheet_prob=[1.0], conceded_per90=[0.0], bonus_per90=[0.0])
    with_sixty = scoring.expected_points(p_sixty=[1.0], **base)[0]
    without = scoring.expected_points(p_sixty=[0.0], **base)[0]
    # crossing 60 minutes moves TWO terms: the clean sheet becomes payable AND
    # the appearance bonus goes 1 -> 2. Both are real.
    expected = scoring.CLEAN_SHEET["DEF"] + (
        scoring.APPEARANCE_LONG - scoring.APPEARANCE_SHORT)
    assert with_sixty - without == pytest.approx(expected)


def test_reconstruction_rate_is_reported(raw):
    """Print the exact-match rate so drift is visible, not just pass/fail."""
    got = scoring.reconstruct(raw)
    want = pd.to_numeric(raw["total_points"], errors="coerce")
    d = (got - want)
    print(f"\n  exact: {100*float((d.abs()<=1e-9).mean()):.2f}%  "
          f"mean abs err {float(d.abs().mean()):.4f}  "
          f"within 1pt {100*float((d.abs()<=1).mean()):.2f}%")
    by_pos = raw.assign(err=d.abs()).groupby("position")["err"].mean()
    print("  mean abs error by position:", by_pos.round(4).to_dict())
