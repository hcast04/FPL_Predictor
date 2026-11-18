"""
How long a player is unavailable for -- not just this week.

WHY THIS EXISTS. `chance_of_playing_next_round` is a SINGLE-WEEK field, and
every tool here scaled by it and stopped. At GW11 of 2026-27 Thiago (BRE) was
"Suspended until 5 Dec", which covers GW11, GW12 and GW13; the planner zeroed
him for GW11 and the projection carried him at 3.47 and 3.42 for the two weeks
he was certain to miss. The recommended transfer was therefore worth +1.66
over one week and +0.34 over four, while selling the suspended player was
worth roughly +18. Same error class as R55: a held decision priced on one week.

WHAT IS PARSED, AND WHAT IS A JUDGEMENT. The API's `news` field is free text
but its shapes are few and stable (198 players carried news on 2026-11-18):

    Suspended until 5 Dec                       <- hard date
    Hamstring injury - Expected back 2 Jan      <- hard date
    Knee injury - Unknown return date           <- NO date
    Ankle injury - 75% chance of playing        <- single-week doubt
    Has joined Juventus on loan for the rest..  <- gone

A hard date and a departure are FACTS and are treated as such. "Unknown return
date" is genuinely unknown: there is no historical news archive in the API, so
how long such absences last CANNOT be measured here, and `UNKNOWN_WEEKS` is a
stated judgement rather than a result. It is deliberately conservative for a
TRANSFER decision -- an injured player with no return date is a liability --
and the parameter is exposed so the assumption is visible and changeable.
"""
from __future__ import annotations

import datetime as _dt
import re

import pandas as pd

# How many gameweeks to treat an "Unknown return date" absence as lasting.
# A JUDGEMENT, not a measurement: see the module docstring.
UNKNOWN_WEEKS = 3

# How far a date must be in the PAST before it is read as next year rather
# than as stale news. See `parse_return`.
_ROLL_FORWARD_DAYS = 180

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"])}
_DATE = re.compile(
    r"(?:until|expected\s+back)\s+(\d{1,2})\s+"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I)
# a move away from the league: out for the rest of the season
_GONE = re.compile(r"has joined|rest of the season|not included in squad", re.I)


def parse_return(news: str, today: _dt.date) -> _dt.date | None:
    """
    The date a player is expected back, or None.

    The year is absent from the text, so it is inferred as the nearest future
    one: "Expected back 2 Jan" read in November means the following January.
    A date already in the PAST yields None -- the news is simply stale and
    says nothing about the weeks ahead.
    """
    if not news:
        return None
    m = _DATE.search(news)
    if not m:
        return None
    day, mon = int(m.group(1)), _MONTHS[m.group(2).lower()]
    try:
        d = _dt.date(today.year, mon, day)
    except ValueError:                           # 29 Feb in a non-leap year
        return None
    if d >= today:
        return d
    # The date has already passed in the current year. Either the news is
    # STALE ("Expected back 31 Oct" still showing in November) or the month
    # belongs to next year ("Expected back 2 Jan" read in November). Rolling
    # the year forward unconditionally turns the first case into an
    # eleven-month absence, so only roll when the gap is too large to be
    # staleness -- a football absence is not announced half a year ahead.
    if (today - d).days > _ROLL_FORWARD_DAYS:
        try:
            return _dt.date(today.year + 1, mon, day)
        except ValueError:
            return None
    return None


def gone_for_season(status: str, news: str) -> bool:
    """Left the league: a loan, a permanent move, or dropped from the squad."""
    return str(status) == "u" and bool(_GONE.search(news or ""))


def multiplier(status: str, chance, news: str, kickoff: _dt.date,
               today: _dt.date, weeks_ahead: int,
               unknown_weeks: int = None) -> float:
    """
    Availability for ONE fixture, as a 0-1 multiplier on expected points.

    `weeks_ahead` is 0 for the gameweek being decided, 1 for the next, and so
    on. It is what makes a doubt different from a suspension: a 75% doubt is
    about THIS week and usually resolves, while a dated absence is a fact
    about the calendar.
    """
    unknown_weeks = UNKNOWN_WEEKS if unknown_weeks is None else unknown_weeks
    news = news or ""
    if gone_for_season(status, news):
        return 0.0
    back = parse_return(news, today)
    if back is not None:
        return 0.0 if kickoff < back else 1.0
    ch = pd.to_numeric(chance, errors="coerce")
    if status == "a":
        return 1.0 if pd.isna(ch) else float(ch) / 100.0
    # flagged, with no date attached
    if pd.notna(ch) and float(ch) > 0:
        # a percentage chance is a statement about the NEXT round only
        return float(ch) / 100.0 if weeks_ahead == 0 else 1.0
    # status i / s / u with no date and no chance: unknown, assumed to last
    return 0.0 if weeks_ahead < unknown_weeks else 1.0


def horizon(elements: pd.DataFrame, fixtures: pd.DataFrame, today: _dt.date,
            unknown_weeks: int = None) -> pd.DataFrame:
    """
    One availability multiplier per (player_code, gameweek).

    `elements` needs player_code, team, status, chance, news.
    `fixtures` needs gameweek, team, kickoff (a date).
    """
    first = fixtures["gameweek"].min()
    out = []
    for _, f in fixtures.iterrows():
        wk = int(f["gameweek"]) - int(first)
        for _, e in elements[elements["team"].eq(f["team"])].iterrows():
            out.append({
                "player_code": e["player_code"], "gameweek": int(f["gameweek"]),
                "avail": multiplier(e["status"], e["chance"], e["news"],
                                    f["kickoff"], today, wk, unknown_weeks)})
    return pd.DataFrame(out)
