"""The panel's invariants. If these break, everything downstream is wrong."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpl2 import config as C      # noqa: E402
from fpl2 import data             # noqa: E402


@pytest.fixture(scope="module")
def panel():
    return data.build_panel()


def test_one_row_per_player_gameweek(panel):
    assert panel.duplicated(["season", "gameweek", C.PLAYER_KEY]).sum() == 0


def test_doubles_are_preserved_not_collapsed(panel):
    """A double gameweek must sum both fixtures, not keep one."""
    doubles = panel[panel["fixtures"] >= 2]
    assert len(doubles) > 0, "no double gameweeks found -- suspicious"
    assert doubles["points"].mean() > panel[panel["fixtures"] == 1]["points"].mean()


def test_time_index_is_monotonic_across_seasons(panel):
    first = panel.groupby("season")["t"].min().sort_index()
    assert list(first) == sorted(first), "season ordering is not chronological"


def test_positions_are_normalised(panel):
    assert set(panel["position"].unique()) <= set(C.POSITIONS)


def test_blank_gameweeks_are_missing_rows_not_zero_rows(panel):
    """Documents the trap: a blank is an ABSENT row in the compact panel."""
    assert (panel["fixtures"] == 0).sum() == 0


def test_with_blanks_materialises_the_full_grid(panel):
    season = C.HOLDOUT_SEASONS[0]
    g = data.with_blanks(panel, season)
    n_gw = panel[panel.season.eq(season)]["gameweek"].nunique()
    n_pl = panel[panel.season.eq(season)][C.PLAYER_KEY].nunique()
    assert len(g) == n_gw * n_pl
    blanks = g[g["fixtures"] == 0]
    assert len(blanks) > 0, "a full season should contain blanks"
    assert (blanks["points"] == 0).all()
    assert blanks["position"].notna().all(), "blank rows must keep their position"
