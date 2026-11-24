"""
Who was managing each club, in each gameweek.

The model has no manager feature and the FPL API has no manager field, so an
appointment reaches the forecast only through rolling outcome windows that
lag five to ten gameweeks. This module supplies the missing fact.

THE JOIN IS ON CLUB NAME AND DATE, and the club name is the risky half. The
panel carries `team_name` from the FPL exports ("Leicester", "Spurs",
"Man Utd") while Wikipedia writes the legal name ("Leicester City",
"Tottenham Hotspur", "Manchester United"). Name matching is what produced the
Nobel-Mendy error on this project, so the alias table below is EXPLICIT rather
than fuzzy: a club either resolves or it is reported unresolved. There is no
similarity threshold to tune and no silent near-miss.

WHAT CANNOT BE RESOLVED. Nine (season, club) pairs in 2016-17..2018-19 carry
no name at all in the public exports -- they are "Team 25", "Team 80" and so
on (S46). `style.team_map` cannot place them either; its own unmatched list
names the candidates as Hull City, Middlesbrough, Stoke City, Swansea City,
Huddersfield and Cardiff City. They stay unresolved and `attach` reports them,
because an unmatched club must be visible rather than quietly carry NaN.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from . import config as C

RAW = C.ROOT / "data" / "raw_extra" / "managers.csv"

# panel `team_name` -> the club as Wikipedia writes it. Explicit on purpose.
ALIASES = {
    "arsenal": "Arsenal", "aston villa": "Aston Villa",
    "bournemouth": "Bournemouth", "afc bournemouth": "Bournemouth",
    "brentford": "Brentford", "brighton": "Brighton & Hove Albion",
    "burnley": "Burnley", "cardiff": "Cardiff City", "chelsea": "Chelsea",
    "coventry": "Coventry City", "crystal palace": "Crystal Palace",
    "everton": "Everton", "fulham": "Fulham", "huddersfield": "Huddersfield Town",
    "hull": "Hull City", "hull city": "Hull City", "ipswich": "Ipswich Town",
    "leeds": "Leeds United", "leicester": "Leicester City",
    "liverpool": "Liverpool", "luton": "Luton Town",
    "man city": "Manchester City", "manchester city": "Manchester City",
    "man utd": "Manchester United", "manchester united": "Manchester United",
    "middlesbrough": "Middlesbrough", "newcastle": "Newcastle United",
    "norwich": "Norwich City", "nott'm forest": "Nottingham Forest",
    "nottingham forest": "Nottingham Forest", "sheffield utd": "Sheffield United",
    "sheffield united": "Sheffield United", "southampton": "Southampton",
    "stoke": "Stoke City", "stoke city": "Stoke City",
    "sunderland": "Sunderland", "swansea": "Swansea City",
    "spurs": "Tottenham Hotspur", "tottenham": "Tottenham Hotspur",
    "tottenham hotspur": "Tottenham Hotspur", "watford": "Watford",
    "west brom": "West Bromwich Albion", "west ham": "West Ham United",
    "wolves": "Wolverhampton Wanderers",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", s.lower())).strip()


def load() -> pd.DataFrame:
    """Every managerial spell, one row each."""
    d = pd.read_csv(RAW, parse_dates=["start", "end"])
    d["caretaker"] = d["caretaker"].astype(bool)
    return d.sort_values(["club", "start"]).reset_index(drop=True)


# the Wikipedia spelling is itself a valid input -- the panel uses the short
# form in some seasons ("Coventry") and the full one in others ("Coventry
# City"), and only accepting the short form silently dropped both of the
# 2026-27 promoted clubs.
_CANON = {_norm(v): v for v in set(ALIASES.values())}


def resolve_club(team_name: str) -> str | None:
    """Panel club name -> Wikipedia club name, or None."""
    n = _norm(team_name)
    if n in ALIASES:
        return ALIASES[n]
    if n in _CANON:
        return _CANON[n]
    # "Team 25" and friends: unnamed in the source, not a near miss
    return None


def in_charge(spells: pd.DataFrame, club: str, when: pd.Timestamp):
    """
    The spell covering `when` at `club`, or None.

    Where spells overlap -- a caretaker inside a gap, or a successor dated
    from the day the predecessor left -- the LATEST start wins, which is the
    person actually in the dugout.
    """
    s = spells[spells["club"].eq(club) & spells["start"].le(when)]
    s = s[s["end"].isna() | s["end"].ge(when)]
    if s.empty:
        return None
    return s.sort_values("start").iloc[-1]


def attach(panel: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add `manager`, `mgr_caretaker`, `mgr_days` and `mgr_match_no` to a panel.

    `mgr_days` is days since the appointment at the time of that gameweek's
    first kickoff; `mgr_match_no` counts that club's gameweeks under him,
    which is the quantity a tenure effect would run on.
    """
    spells = load()
    kc = ("kickoff_first" if "kickoff_first" in panel.columns
          else "kickoff_time" if "kickoff_time" in panel.columns else None)
    if kc is None:
        raise ValueError("panel carries no kickoff column to date the join")
    key = (panel[["season", "gameweek", "team", "team_name", kc]]
           .dropna(subset=[kc]).drop_duplicates(["season", "gameweek", "team"]))
    # NOT a leading underscore: `itertuples` renames those to positional
    # names and the attribute access below would silently break.
    key["kdate"] = pd.to_datetime(key[kc], errors="coerce", utc=True,
                                  format="ISO8601").dt.tz_localize(None)
    rows, unresolved = [], set()
    for r in key.itertuples():
        wiki = resolve_club(r.team_name)
        if wiki is None:
            unresolved.add((r.season, r.team, r.team_name))
            continue
        sp = in_charge(spells, wiki, r.kdate)
        if sp is None:
            continue
        rows.append({"season": r.season, "gameweek": r.gameweek, "team": r.team,
                     "manager": sp["manager"], "mgr_caretaker": bool(sp["caretaker"]),
                     "mgr_start": sp["start"],
                     "mgr_days": int((r.kdate - sp["start"]).days)})
    M = pd.DataFrame(rows)
    if M.empty:
        return panel.assign(manager=np.nan, mgr_caretaker=np.nan,
                            mgr_days=np.nan, mgr_match_no=np.nan)
    M = M.sort_values(["team", "mgr_start", "season", "gameweek"])
    M["mgr_match_no"] = M.groupby(["team", "manager", "mgr_start"]).cumcount() + 1
    if verbose:
        tot = len(key)
        print(f"  managers: {len(M):,}/{tot:,} club-gameweeks resolved "
              f"({100*len(M)/tot:.1f}%), {M.manager.nunique()} managers")
        if unresolved:
            print(f"  UNRESOLVED clubs ({len(unresolved)}): "
                  f"{sorted({u[2] for u in unresolved})}")
    return panel.merge(M.drop(columns=["mgr_start"]),
                       on=["season", "gameweek", "team"], how="left")
