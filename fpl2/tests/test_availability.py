"""
Multi-week unavailability. `chance_of_playing_next_round` is a SINGLE-WEEK
field and every tool scaled by it and stopped, so a player suspended for
three gameweeks was priced as missing one.
"""
from __future__ import annotations
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpl2 import availability as av      # noqa: E402

TODAY = dt.date(2026, 11, 18)


# --- parsing ---------------------------------------------------------------

@pytest.mark.parametrize("news,expect", [
    ("Suspended until 5 Dec", dt.date(2026, 12, 5)),
    ("Hamstring injury - Expected back 2 Jan", dt.date(2027, 1, 2)),
    ("Knock - Expected back 23 Nov", dt.date(2026, 11, 23)),
    ("Knee injury - Unknown return date", None),
    ("Ankle injury - 75% chance of playing", None),
    ("Has joined Juventus on loan for the rest of the season", None),
    ("", None),
    (None, None),
])
def test_parse_return(news, expect):
    assert av.parse_return(news, TODAY) == expect


def test_year_rolls_forward_but_only_when_needed():
    """'2 Jan' read in November is NEXT January; '5 Dec' is THIS December."""
    assert av.parse_return("Expected back 2 Jan", TODAY).year == 2027
    assert av.parse_return("Suspended until 5 Dec", TODAY).year == 2026


def test_a_past_date_is_stale_news_and_constrains_nothing():
    """'Expected back 31 Oct' read in November says nothing about the future."""
    assert av.parse_return("Thigh injury - Expected back 31 Oct", TODAY) is None


def test_gone_for_season_needs_both_status_and_text():
    assert av.gone_for_season("u", "Has joined Al Hilal permanently")
    assert av.gone_for_season("u", "Has joined Leicester City on loan for "
                                   "the rest of the season")
    # NEGATIVE CONTROL: an injury is not a departure, whatever the status
    assert not av.gone_for_season("i", "Knee injury - Unknown return date")
    assert not av.gone_for_season("a", "Has joined Juventus on loan")


# --- the multiplier --------------------------------------------------------

def test_suspension_spans_every_fixture_before_the_return_date():
    """Thiago, GW11-13 of 2026-27. This is the case that motivated the module."""
    news = "Suspended until 5 Dec"
    for wk, ko, expect in [(0, dt.date(2026, 11, 23), 0.0),   # GW11
                           (1, dt.date(2026, 11, 28), 0.0),   # GW12
                           (2, dt.date(2026, 12, 2), 0.0),    # GW13
                           (3, dt.date(2026, 12, 5), 1.0)]:   # GW14, back
        got = av.multiplier("s", 0, news, ko, TODAY, wk)
        assert got == expect, f"week {wk} ({ko}): got {got}, want {expect}"


def test_a_percentage_doubt_is_about_this_week_only():
    """
    A 75% doubt is a statement about the next round. Carrying it four weeks
    forward would under-rate a player who is simply rated to recover.
    """
    news = "Ankle injury - 75% chance of playing"
    assert av.multiplier("d", 75, news, dt.date(2026, 11, 23), TODAY, 0) == 0.75
    assert av.multiplier("d", 75, news, dt.date(2026, 11, 28), TODAY, 1) == 1.0


def test_unknown_return_lasts_the_stated_judgement_and_no_longer():
    news = "Knee injury - Unknown return date"
    for wk in range(av.UNKNOWN_WEEKS):
        assert av.multiplier("i", 0, news, dt.date(2026, 11, 23), TODAY, wk) == 0.0
    assert av.multiplier("i", 0, news, dt.date(2026, 12, 20), TODAY,
                         av.UNKNOWN_WEEKS) == 1.0


def test_a_departed_player_never_returns():
    news = "Has joined Al Hilal permanently"
    for wk in (0, 1, 5, 20):
        assert av.multiplier("u", 0, news, dt.date(2027, 3, 1), TODAY, wk) == 0.0


def test_an_available_player_is_untouched():
    for wk in (0, 1, 3):
        assert av.multiplier("a", None, "", dt.date(2026, 11, 23), TODAY, wk) == 1.0
        assert av.multiplier("a", 100, "", dt.date(2026, 11, 23), TODAY, wk) == 1.0


def test_the_old_single_week_behaviour_is_reproduced_at_week_zero():
    """
    NEGATIVE CONTROL. For the gameweek being decided this must agree with
    what plan_gw already did -- scale by chance, zero a non-'a' status with
    no chance. If it does not, the fix changes THIS week as well as the
    horizon, which is not what it claims to do.
    """
    cases = [("a", None, "", 1.0), ("a", 100, "", 1.0), ("a", 75, "x", 0.75),
             ("d", 75, "Ankle injury - 75% chance of playing", 0.75),
             ("d", 50, "Hamstring injury - 50% chance of playing", 0.50),
             ("i", 0, "Knee injury - Unknown return date", 0.0),
             ("u", 0, "Has joined Juventus on loan for the rest of the season", 0.0)]
    for status, chance, news, want in cases:
        old = (1.0 if status == "a" and chance is None
               else (float(chance) / 100.0 if chance is not None else 0.0))
        new = av.multiplier(status, chance, news, dt.date(2026, 11, 23), TODAY, 0)
        assert new == want, f"{status}/{chance}: got {new}, want {want}"
        assert new == old, f"{status}/{chance}: diverges from old week-0 rule"


def test_horizon_builds_one_row_per_player_gameweek():
    el = pd.DataFrame([
        {"player_code": 1, "team": "BRE", "status": "s", "chance": 0,
         "news": "Suspended until 5 Dec"},
        {"player_code": 2, "team": "BRE", "status": "a", "chance": None,
         "news": ""}])
    fx = pd.DataFrame([{"gameweek": 11, "team": "BRE", "kickoff": dt.date(2026, 11, 23)},
                       {"gameweek": 14, "team": "BRE", "kickoff": dt.date(2026, 12, 5)}])
    h = av.horizon(el, fx, TODAY)
    assert len(h) == 4
    got = {(r.player_code, r.gameweek): r.avail for r in h.itertuples()}
    assert got[(1, 11)] == 0.0 and got[(1, 14)] == 1.0
    assert got[(2, 11)] == 1.0 and got[(2, 14)] == 1.0
