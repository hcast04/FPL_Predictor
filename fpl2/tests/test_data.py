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


# --- M32: the two sources spell kickoff_time differently -------------------

def test_mixed_iso_kickoff_formats_all_parse():
    """
    Both kickoff spellings must survive being concatenated.

    The public exports write "2016-08-15T19:00:00Z" and
    `historical_fpl.csv.gz` writes "2022-08-05 19:00:00+00:00". Each parses at
    100% ALONE. `pd.to_datetime` infers ONE format from the first non-null
    value, so concatenated with the exports first it coerced all 113,582 rows
    of the main CSV to NaT -- and `understat.join_season` matches on fixture
    DATE, so 2022-23 through 2025-26 joined ZERO Understat rows and trained on
    NaN us_* features.

    NEGATIVE CONTROL: the same call WITHOUT format="ISO8601" must still fail,
    or this test is asserting nothing.
    """
    import pandas as pd

    exports = ["2016-08-15T19:00:00Z", "2016-08-20T14:00:00Z"]
    main_csv = ["2022-08-05 19:00:00+00:00", "2022-08-06 14:00:00+00:00"]
    mixed = pd.Series(exports + main_csv)

    good = pd.to_datetime(mixed, errors="coerce", utc=True, format="ISO8601")
    assert good.notna().all(), f"ISO8601 should parse every row, got {good.tolist()}"

    naive = pd.to_datetime(mixed, errors="coerce", utc=True)
    assert naive.isna().any(), (
        "negative control FAILED: inference now parses both spellings, so this "
        "test no longer proves format='ISO8601' is what fixes it")


def test_every_season_has_kickoffs():
    """
    No season may lose its kickoff times ON THE TEN-SEASON PATH.

    A season with no kickoff cannot join Understat (matched on fixture date)
    and the loss is silent -- the column is simply NaT and every downstream
    join returns nothing.

    This MUST use `extended=True`. The defect only exists once the public
    exports and the main CSV are concatenated; each parses at 100% alone, so
    the default `build_panel()` fixture would pass on the broken code and
    guard nothing.
    """
    ext = data.build_panel(data.load_raw(extended=True))
    got = ext.groupby("season")["kickoff_first"].apply(lambda s: s.notna().mean())
    bad = got[got < 0.99]
    assert bad.empty, f"seasons missing kickoff_first: {bad.to_dict()}"
    assert len(got) >= 10, f"expected ten+ seasons on the extended path, got {len(got)}"
