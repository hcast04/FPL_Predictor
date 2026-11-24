"""
Manager spells joined onto the panel.

The join is on CLUB NAME and DATE. Name matching is what produced the
Nobel-Mendy error on this project, so the alias table is explicit and a club
either resolves or is reported unresolved -- there is no similarity threshold.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpl2 import managers as mg       # noqa: E402

TS = pd.Timestamp


@pytest.mark.parametrize("name,want", [
    ("Man Utd", "Manchester United"),
    ("Manchester United", "Manchester United"),      # the canonical form too
    ("Spurs", "Tottenham Hotspur"),
    ("Nott'm Forest", "Nottingham Forest"),
    ("Coventry", "Coventry City"),
    ("Coventry City", "Coventry City"),              # both spellings appear
    ("Team 25", None),                               # unnamed in the exports
    ("Totally Not A Club", None),
])
def test_resolve_club(name, want):
    assert mg.resolve_club(name) == want


def test_resolve_never_guesses():
    """
    NEGATIVE CONTROL. A near-miss must return None, not the closest club.
    Fuzzy matching is what matched De Gea to David Raya (M25).
    """
    for near in ("Manchester", "United", "Man", "Tottenham Hotspurs FC",
                 "Arsenall", "Chelsey"):
        assert mg.resolve_club(near) is None, f"{near!r} should not resolve"


def _spells():
    return pd.DataFrame([
        {"manager": "A", "club": "X", "start": TS("2020-01-01"),
         "end": TS("2020-06-01"), "caretaker": False},
        {"manager": "Care", "club": "X", "start": TS("2020-06-01"),
         "end": TS("2020-07-01"), "caretaker": True},
        {"manager": "B", "club": "X", "start": TS("2020-07-01"),
         "end": pd.NaT, "caretaker": False},
    ])


def test_in_charge_picks_the_right_spell():
    s = _spells()
    assert mg.in_charge(s, "X", TS("2020-03-01"))["manager"] == "A"
    assert mg.in_charge(s, "X", TS("2020-06-15"))["manager"] == "Care"
    assert mg.in_charge(s, "X", TS("2021-01-01"))["manager"] == "B"
    assert mg.in_charge(s, "X", TS("2019-01-01")) is None   # before the first


def test_an_open_spell_runs_forever():
    """`end` is NaT for a sitting manager; that is not missing data."""
    s = _spells()
    assert mg.in_charge(s, "X", TS("2030-01-01"))["manager"] == "B"


def test_overlapping_spells_take_the_later_start():
    """
    A successor is often dated from the day the predecessor left, so both
    match. The person in the dugout is the one who started most recently.
    """
    s = _spells()
    got = mg.in_charge(s, "X", TS("2020-06-01"))
    assert got["manager"] == "Care", f"got {got['manager']}, want the successor"


def test_unknown_club_is_none_not_an_exception():
    assert mg.in_charge(_spells(), "NOT A CLUB", TS("2020-03-01")) is None


@pytest.mark.skipif(not mg.RAW.exists(),
                    reason="managers.csv absent; run fpl2/pull_managers.py")
def test_the_scraped_file_is_sane():
    d = mg.load()
    assert len(d) > 400, f"only {len(d)} spells -- the scrape looks truncated"
    assert d.start.notna().all(), "a spell with no start date"
    bad = d[d.end.notna() & d.end.lt(d.start)]
    assert bad.empty, f"{len(bad)} spells end before they start"
    # exactly one sitting manager per club among the open spells
    open_ = d[d.end.isna()]
    assert open_.club.is_unique, (
        f"clubs with two open spells: "
        f"{open_[open_.club.duplicated(keep=False)].club.tolist()}")
