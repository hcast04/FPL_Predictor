"""
Team PLAYING STYLE features, from FBref match-level data.

WHAT IS NEW HERE. Every team feature the model already carries is an OUTCOME:
`team_scored_r5`, `team_conceded_r10`, and the opponent twins. None of them
says anything about HOW a side plays. A possession team that works the ball
into the box and a counter-attacking side that crosses from deep can score at
the same rate and offer their forwards completely different chances.

S29 measured OPPONENT identity at zero for ranking -- scrambling the opponent
map left the forecast very slightly better. That is a statement about who you
play, not about how your own team plays, so it does not cover this.

THE SOURCE. `soccerdata.FBref.read_team_match_stats` over schedule, shooting
and misc. Match level, not season level: a season aggregate applied to a
mid-season fixture would read the future, which is how F10/M28 produced a
fictitious +235 pts/season.

TEAM NAMES. FBref returns `team` as NaN for this league -- the game id reads
"2024-08-16 Manchester Utd-nan". `scratchpad/fix_style_teams.py` recovers it
by pairing each fixture's two rows on reversed scoreline and reading each
side's team off the other's `opponent`. Validated at 20 teams x exactly 38
matches for all ten completed seasons.

The FBref->FPL name map is built from FIXTURE DATE OVERLAP, never from string
similarity: the Understat join matched De Gea to David Raya on a shared first
name, and coverage rose while xG correlation fell (M25).
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

RAW = C.ROOT / "data" / "raw_extra"

# per-match quantities that describe HOW a side played
METRICS = {
    "poss": "Poss",                       # possession share
    "shots": "Standard/Sh",               # shot volume
    "sot": "Standard/SoT",                # shots on target
    "sot_pct": "Standard/SoT%",           # accuracy
    "g_per_sh": "Standard/G/Sh",          # shot QUALITY, not volume
    "crosses": "Performance/Crs",         # directness / wing play
    "intercept": "Performance/Int",       # defensive engagement
    "tackles": "Performance/TklW",        # pressing intensity
    "fouls": "Performance/Fls",
}
WINDOWS = (5, 10)


def _load() -> pd.DataFrame:
    out = []
    for f in sorted(glob.glob(str(RAW / "fbref_style_*.pkl"))):
        season = re.search(r"fbref_style_(\d{4}-\d{2})", f).group(1)
        d = pd.read_pickle(f)
        if "team_fixed" not in d.columns:
            continue
        d = d[d["team_fixed"].notna() & d["Poss"].notna()].copy()
        d["season"] = season
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _norm(s: str) -> str:
    """Club name to a comparable token set."""
    s = str(s).lower()
    s = s.replace("&", "and").replace("'", "").replace(".", "")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    words = [w for w in s.split()
             if w not in {"fc", "afc", "cf", "the", "football", "club"}]
    alias = {"manchester": "man", "utd": "united", "nottingham": "nott",
             "nottm": "nott", "forest": "nott", "wolverhampton": "wolves",
             "wanderers": "", "hotspur": "", "spurs": "tottenham",
             "albion": "", "city": "city", "town": "", "united": "united",
             "brighton": "brighton", "hove": "", "and": "", "leeds": "leeds",
             "sheffield": "sheffield", "west": "west", "ham": "ham"}
    out = []
    for w in words:
        a = alias.get(w, w)
        if a:
            out.append(a)
    return " ".join(out)


def team_map(style: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """
    (season, FBref name) -> FPL short code.

    Matched on NORMALISED CLUB NAMES, not on fixture dates. Dates were tried
    first and fail twice over: `kickoff_first` is absent from 2022-23 onward
    (0% non-null), and within a season all twenty clubs play on roughly the
    same Saturdays, so the date sets barely discriminate -- that attempt
    mapped Everton to Sunderland at 0.62 overlap.

    Twenty club names a season is a small, AUDITABLE set, which is what makes
    string matching safe here and unsafe for the 3,000 player names that
    produced the De Gea / David Raya match (M25). `build` asserts twenty
    distinct clubs per completed season and raises otherwise.
    """
    out, report = {}, {}
    for season in sorted(set(style["season"]) & set(panel["season"])):
        ps = panel[panel.season.eq(season)]
        codes = (ps[["team", "team_name"]].dropna().drop_duplicates()
                 .assign(key=lambda d: d.team_name.map(_norm)))
        codes = codes[codes.team_name.ne("") & ~codes.team_name.str.match(r"Team \d+")]
        lut = {k: t for k, t in zip(codes.key, codes.team)}
        toks = {k: set(k.split()) for k in lut}
        for name in sorted(style[style.season.eq(season)].team_fixed.dropna().unique()):
            k = _norm(name); want = set(k.split())
            if k in lut:
                out[(season, name)] = (lut[k], 1.0); continue
            best, sc = None, 0.0
            for cand, ct in toks.items():
                j = len(want & ct) / max(len(want | ct), 1)
                if j > sc:
                    best, sc = cand, j
            if best is not None and sc >= 0.34:
                out[(season, name)] = (lut[best], round(sc, 3))
            else:
                report.setdefault(season, []).append((name, k, round(sc, 2)))
    if report:
        print("style.team_map UNMATCHED:", report)
    return out


def build(panel: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    One row per (season, gameweek, team) with STRICTLY LAGGED style features,
    plus the opponent's. Row t is computed from that team's matches < t.
    """
    style = _load()
    if style.empty:
        return pd.DataFrame(), []
    m = team_map(style, panel)
    style["team"] = [m.get((se, na), (None,))[0]
                     for se, na in zip(style["season"], style["team_fixed"])]
    style = style[style["team"].notna()].copy()
    for k, col in METRICS.items():
        style[k] = pd.to_numeric(style.get(col), errors="coerce")
    style = style.sort_values(["team", "date"]).reset_index(drop=True)

    g = style.groupby("team", sort=False)
    made = []
    for k in METRICS:
        for w in WINDOWS:
            name = f"sty_{k}_r{w}"
            # shift(1) FIRST: a match never sees its own style
            style[name] = g[k].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=2).mean())
            made.append(name)

    # Join on (season, TEAM, match order), not on date: `kickoff_first` is
    # 0% non-null from 2022-23 onward, so a date join silently drops four
    # seasons. Both sides are ordered by match within a team-season, so the
    # n-th FBref match is the n-th FPL gameweek that team appears in.
    style["mno"] = style.groupby(["season", "team"]).cumcount()
    key = style[["season", "team", "mno"] + made]

    p = panel.copy()
    order = (p[["season", "team", "gameweek"]].drop_duplicates()
             .sort_values(["season", "team", "gameweek"]))
    order["mno"] = order.groupby(["season", "team"]).cumcount()
    p = p.merge(order, on=["season", "team", "gameweek"], how="left")
    out = p.merge(key, on=["season", "team", "mno"], how="left")

    opp = key.rename(columns={"team": "_opp", **{c: "opp_" + c for c in made}})
    out["_opp"] = [t[0] if isinstance(t, (list, tuple)) and len(t) else None
                   for t in out["opponents"]]
    oorder = order.rename(columns={"team": "_opp", "mno": "_omno"})
    out = out.merge(oorder, on=["season", "_opp", "gameweek"], how="left")
    out = out.merge(opp.rename(columns={"mno": "_omno"}),
                    on=["season", "_opp", "_omno"], how="left")
    made += ["opp_" + c for c in made]
    cov = out[made[0]].notna().mean()
    print(f"style: {len(m)} team-seasons mapped, feature coverage "
          f"{cov*100:.1f}% of panel rows")
    return out.drop(columns=["mno", "_opp", "_omno"], errors="ignore"), made
