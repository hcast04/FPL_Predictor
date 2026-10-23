"""
Leakage tests.

The previous codebase shipped a script called `horizon_sweep_clean.py` that
leaked three separate ways. The lesson taken: leakage must be a test, not a
claim in a docstring.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpl2 import config as C       # noqa: E402
from fpl2 import data, features    # noqa: E402


@pytest.fixture(scope="module")
def panel():
    p = data.build_panel()
    # one season keeps the test fast while still exercising cross-season lag
    return p[p["season"].isin(["2024-25", "2025-26"])].copy()


@pytest.fixture(scope="module")
def fixtures():
    from fpl2 import team
    return team.team_fixtures(data.load_raw())


def test_opponent_features_do_not_read_the_future(panel, fixtures):
    """
    The opponent join is where leakage hides: it reaches into another team's
    rows, and those rows carry that team's results. Verify the fixture-aware
    features survive truncation too.
    """
    rep = features.truncation_equivalence(panel, "2025-26", 20, fixtures=fixtures)
    fixture_aware = rep[rep["feature"].str.startswith(("opp_", "is_home",
                                                       "att_vs", "def_vs"))]
    assert len(fixture_aware) >= 8, "opponent features missing from the panel"
    worst = fixture_aware.iloc[0]
    assert worst["max_abs_diff"] < 1e-9, (
        f"fixture-aware feature '{worst['feature']}' reads the future "
        f"(max abs diff {worst['max_abs_diff']:.6g})")


def test_no_feature_reads_the_future(panel):
    """Features at GW20 must be identical whether or not GW21+ exist."""
    rep = features.truncation_equivalence(panel, "2025-26", 20)
    assert rep["n_rows"].iloc[0] > 100, "too few rows to be a real test"
    worst = rep.iloc[0]
    assert worst["max_abs_diff"] < 1e-9, (
        f"feature '{worst['feature']}' changes when future data is removed "
        f"(max abs diff {worst['max_abs_diff']:.6g}) -- it reads the future")


def test_the_leakage_test_can_actually_fail(panel):
    """
    NEGATIVE CONTROL.

    Inject a deliberately leaky feature (a FORWARD-looking rolling mean) and
    confirm truncation_equivalence flags it. Without this, a test that always
    passes proves nothing.
    """
    original = features.player_form

    def leaky(p):
        d, made = original(p)
        g = d.sort_values([C.PLAYER_KEY, "t"]).groupby(C.PLAYER_KEY, sort=False)
        # shift(-1): reads the NEXT gameweek. This is the bug we are detecting.
        d["points_future_r3"] = g["points"].transform(
            lambda s: s.shift(-1).rolling(3, min_periods=1).mean())
        return d, made + ["points_future_r3"]

    features.player_form = leaky
    try:
        rep = features.truncation_equivalence(panel, "2025-26", 20)
        row = rep[rep["feature"].eq("points_future_r3")]
        assert len(row) == 1
        assert row["max_abs_diff"].iloc[0] > 1e-6, (
            "the leakage test failed to detect a deliberately leaky feature")
    finally:
        features.player_form = original


def test_rolling_features_exclude_the_current_row(panel):
    """A 3-game mean must never include the game it is predicting."""
    d, _ = features.player_form(panel)
    d = d.sort_values([C.PLAYER_KEY, "t"])
    # pick a player with a long run of fixtures
    counts = d.groupby(C.PLAYER_KEY).size().sort_values(ascending=False)
    pid = counts.index[0]
    s = d[d[C.PLAYER_KEY].eq(pid)]
    manual = s["points"].shift(1).rolling(3, min_periods=1).mean()
    assert np.allclose(s["points_r3"].to_numpy(), manual.to_numpy(),
                       equal_nan=True)


def test_first_appearance_has_no_history(panel):
    """A player's debut row must have NaN form, not zero."""
    d, _ = features.player_form(panel)
    d = d.sort_values([C.PLAYER_KEY, "t"])
    first = d.groupby(C.PLAYER_KEY).head(1)
    assert first["points_r3"].isna().all()
    assert first["minutes_r5"].isna().all()
