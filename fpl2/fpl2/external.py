"""
External priors: what a player did in another league before arriving.

The rebuilt model is blind to players with no Premier League history. They are
19.2% of the points scored in 2026-27 and the model rates them at 0.834 of
their true level -- consistently under-rated in GW1-6 across every season
measured (0.977 / 0.972 / 0.929 / 0.834), recovering only once within-season
data accrues.

That blindness cannot be calibrated away: a fixed boost cannot know whether a
given transfer window delivered good signings. It has to be INFORMED away,
with evidence about the specific players. Understat carries per-90 scoring
rates for the big-5 European leagues, which is what the legacy pipeline's
`external_*` columns are built from.

Matching is by normalised name. Names differ across sources (accents,
transliteration, middle names), so each match carries a method and the layer
records how many players it could not place -- an unmatched player must fall
back to the no-prior path rather than silently receive someone else's numbers.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from . import config as C

LEAGUES = ["ESP-La Liga", "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1",
           "ENG-Premier League"]
# FPL season -> the Understat season code for the year BEFORE it
PRIOR_SEASON = {"2022-23": "2122", "2023-24": "2223", "2024-25": "2324",
                "2025-26": "2425", "2026-27": "2526"}
MIN_PRIOR_MINUTES = 450.0      # below this an external rate is mostly noise


def _norm(name: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-zA-Z\s]", " ", s).lower()
    return re.sub(r"\s+", " ", s).strip()


def _surname_key(name: str) -> str:
    parts = _norm(name).split()
    return f"{parts[0][:1]}|{parts[-1]}" if len(parts) >= 2 else ""


def fetch(season_code: str, leagues=LEAGUES) -> pd.DataFrame:
    """Understat player-season stats for one season across several leagues."""
    import logging
    logging.disable(logging.CRITICAL)
    import soccerdata as sd
    frames = []
    for lg in leagues:
        try:
            d = sd.Understat(leagues=[lg], seasons=[season_code]
                             ).read_player_season_stats().reset_index()
            d["source_league"] = lg
            frames.append(d)
        except Exception as exc:          # a league may be unavailable
            print(f"  ! {lg} {season_code}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def to_priors(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-90 rates from an Understat player-season table."""
    if raw.empty:
        return raw
    d = raw.copy()
    for c in ("minutes", "xg", "np_xg", "xa", "shots", "key_passes",
              "goals", "assists", "matches"):
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d = d[d["minutes"] >= MIN_PRIOR_MINUTES]
    nineties = d["minutes"] / 90.0
    out = pd.DataFrame({
        "ext_name": d["player"],
        "ext_key": d["player"].map(_norm),
        "ext_surname_key": d["player"].map(_surname_key),
        "ext_league": d["source_league"],
        "ext_league_tier": d["source_league"].map(LEAGUE_TIER).fillna(DEFAULT_TIER),
        "ext_minutes": d["minutes"],
        "ext_matches": d["matches"],
        "ext_min_per_match": d["minutes"] / d["matches"].clip(lower=1),
        "ext_xg90": d["np_xg"] / nineties,
        "ext_xa90": d["xa"] / nineties,
        "ext_shots90": d["shots"] / nineties,
        "ext_kp90": d["key_passes"] / nineties,
        "ext_goals90": d["goals"] / nineties,
    })
    # one row per player: keep the season's largest minutes total
    return (out.sort_values("ext_minutes", ascending=False)
               .drop_duplicates("ext_key"))


def match(panel_players: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    """
    Attach priors to FPL players by normalised name, then surname + initial.

    Returns one row per player_code with the prior columns and a
    `ext_match_method` recording how it was found. Unmatched players keep NaN
    so downstream code can tell "no prior" from "a prior of zero".
    """
    p = panel_players.copy()
    p["key"] = p["player_name"].map(_norm)
    p["skey"] = p["player_name"].map(_surname_key)
    cols = [c for c in priors.columns if c.startswith("ext_")
            and c not in ("ext_key", "ext_surname_key", "ext_name")]

    exact = priors.drop_duplicates("ext_key").set_index("ext_key")
    m1 = p.join(exact[cols], on="key")
    m1["ext_match_method"] = np.where(m1[cols[0]].notna(), "exact", None)

    need = m1[m1[cols[0]].isna()].copy()
    if not need.empty:
        sur = (priors[priors["ext_surname_key"].ne("")]
               .drop_duplicates("ext_surname_key").set_index("ext_surname_key"))
        m2 = need.drop(columns=cols).join(sur[cols], on="skey")
        m2["ext_match_method"] = np.where(m2[cols[0]].notna(), "surname", None)
        m1 = pd.concat([m1[m1[cols[0]].notna()], m2], ignore_index=True)
    m1["ext_available"] = m1[cols[0]].notna().astype(float)
    return m1.drop(columns=["key", "skey"])


# the Understat code for the SAME season as the FPL one, for mid-season moves
SAME_SEASON = {"2022-23": "2223", "2023-24": "2324", "2024-25": "2425",
               "2025-26": "2526", "2026-27": "2627"}


def build_for_season(panel: pd.DataFrame, season: str,
                     include_same_season: bool = True) -> pd.DataFrame:
    """
    External priors for every player appearing in `season`.

    Two sources, because the January window moves players mid-season:

      * the PRIOR season, which covers summer arrivals;
      * the SAME season in the other league, which covers January arrivals --
        a player who left La Liga in January has a truncated La Liga season
        covering only the months before he moved, so those totals are genuinely
        in the past relative to his Premier League gameweeks.

    The same-season source is only applied to players with NO Premier League
    history in that season before their first appearance, so a player who was
    here all along cannot pick up a concurrent foreign season. Where both
    sources match, the one with more minutes wins.

    CAVEAT: a player who leaves the Premier League mid-season and plays abroad
    would have a same-season row that postdates his PL games. He would not be a
    new arrival, so the no-PL-history filter already excludes him -- but this is
    the edge case to re-check if the matching rule is ever widened.
    """
    code = PRIOR_SEASON.get(season)
    if code is None:
        return pd.DataFrame()
    players = (panel[panel["season"].eq(season)]
               .drop_duplicates(C.PLAYER_KEY)[[C.PLAYER_KEY, "player_name"]])

    priors = to_priors(fetch(code))
    if priors.empty:
        return pd.DataFrame()
    out = match(players, priors).assign(season=season, ext_source="prior_season")

    if include_same_season and SAME_SEASON.get(season):
        same = to_priors(fetch(SAME_SEASON[season],
                               leagues=[l for l in LEAGUES
                                        if l != "ENG-Premier League"]))
        if not same.empty:
            alt = match(players, same)
            cols = [c for c in alt.columns if c.startswith("ext_")
                    and c not in ("ext_available", "ext_match_method")]
            merged = out.merge(alt[[C.PLAYER_KEY] + cols + ["ext_available"]],
                               on=C.PLAYER_KEY, how="left",
                               suffixes=("", "_same"))
            # prefer whichever source has more minutes behind it
            take = (merged["ext_minutes_same"].fillna(-1)
                    > merged["ext_minutes"].fillna(-1))
            for c in cols:
                merged.loc[take, c] = merged.loc[take, c + "_same"]
            merged.loc[take, "ext_source"] = "same_season"
            merged["ext_available"] = (merged["ext_available"].fillna(0)
                                       .combine(merged["ext_available_same"].fillna(0),
                                                max))
            out = merged[[c for c in merged.columns if not c.endswith("_same")]]
    return out


# ---------------------------------------------------------------------------
# Manually-exported tables
#
# FBref sits behind Cloudflare bot-management, so the leagues Understat does
# not cover (Championship above all -- 466 of the 1218 new-to-PL points in
# 2026-27 GW1-7 belong to the three promoted squads) are exported by hand from
# a normal browser session into data/external_manual/.
#
# These tables are FBref "Player Standard Stats". They carry minutes, matches,
# goals and assists everywhere, and xG/npxG/xAG only for the leagues FBref
# buys advanced data for. Where a rate is absent it stays NaN rather than being
# imputed: the gradient booster routes NaN natively, and a fabricated zero
# would read as "took no shots" instead of "not measured".
# ---------------------------------------------------------------------------

MANUAL_DIR = C.ROOT / "data" / "external_manual" if hasattr(C, "ROOT") else None

# Ordinal league strength. A Championship npxG90 is not a La Liga npxG90, and
# the model is given no league identity, so without this it would read the two
# as the same quantity. This is a crude stand-in: the better version is mean
# club Elo for the league-season, which is blocked while api.clubelo.com is
# unreachable (502 on 2026-10-21). Declared before any run that uses it.
LEAGUE_TIER = {
    "ENG-Premier League": 1.0, "ESP-La Liga": 1.0, "GER-Bundesliga": 1.0,
    "ITA-Serie A": 1.0, "FRA-Ligue 1": 1.0,
    "POR-Primeira Liga": 2.0, "NED-Eredivisie": 2.0, "ENG-Championship": 2.0,
    "BEL-Pro League": 2.5, "TUR-Super Lig": 2.5,
    "SCO-Premiership": 3.0, "GRE-Super League": 3.0,
}
DEFAULT_TIER = 2.5

_FBREF_COLS = {"player": "Player", "squad": "Squad", "pos": "Pos",
               "born": "Born",
               "matches": "MP", "starts": "Starts", "minutes": "Min",
               "goals": "Gls", "assists": "Ast",
               "xg": "xG", "npxg": "npxG", "xag": "xAG",
               "shots": "Sh", "kp": "KP"}


def _read_fbref_csv(path):
    """
    One FBref player table, however it was pasted.

    Tolerates: comma or tab separation, a leading multi-row header, the rank
    column, and the header rows FBref repeats every 25 lines in the rendered
    table. Returns a frame whose columns are the raw FBref labels.
    """
    import csv as _csv
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame()

    # the header is the first line that names the Player column
    hdr_i = next((i for i, ln in enumerate(lines) if "Player" in ln), None)
    if hdr_i is None:
        raise ValueError(f"{path.name}: no header row containing 'Player'")

    sep = "\t" if lines[hdr_i].count("\t") > lines[hdr_i].count(",") else ","
    rows = list(_csv.reader(lines[hdr_i:], delimiter=sep))

    # FBref stacks a group header ("Performance", "Per 90 Minutes") above the
    # real one, so Gls/Ast/G+A/G-PK each appear TWICE -- once as a season total
    # and once as a rate. Pandas would hand back a 2-column frame for `d["Gls"]`
    # and every downstream numeric op would break, so suffix the repeats. The
    # first occurrence is the total, which is what the rates are rebuilt from.
    header, seen = [], {}
    for h in (h.strip() for h in rows[0]):
        if h in seen:
            seen[h] += 1
            header.append(f"{h}.{seen[h]}")
        else:
            seen[h] = 0
            header.append(h)

    # FBref repeats the header mid-table; drop those and any blank Player
    try:
        pcol = header.index("Player")
    except ValueError:
        raise ValueError(f"{path.name}: header has no 'Player' column")
    body = [r for r in rows[1:]
            if len(r) > pcol and r[pcol].strip() not in ("", "Player")]
    if not body:
        return pd.DataFrame()

    width = len(header)
    body = [(r + [""] * width)[:width] for r in body]
    return pd.DataFrame(body, columns=header)


def _num(s):
    """FBref numbers carry thousands separators and stray spaces."""
    return pd.to_numeric(
        pd.Series(s).astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce")


def manual_to_priors(raw: pd.DataFrame, league: str) -> pd.DataFrame:
    """Same output schema as `to_priors`, from an FBref standard-stats table."""
    if raw.empty:
        return pd.DataFrame()
    g = _FBREF_COLS
    have = {k: v for k, v in g.items() if v in raw.columns}
    for k in ("player", "minutes", "matches"):
        if k not in have:
            raise ValueError(f"{league}: table is missing {g[k]!r}")

    d = pd.DataFrame({"player": raw[have["player"]].astype(str).str.strip(),
                      "minutes": _num(raw[have["minutes"]]),
                      "matches": _num(raw[have["matches"]])})
    for k in ("goals", "assists", "xg", "npxg", "xag", "shots", "kp"):
        d[k] = _num(raw[have[k]]) if k in have else np.nan

    # A player transferred WITHIN the league has one row per club. His prior is
    # the whole season, so sum the totals before any rate is taken -- keeping
    # the larger spell instead would throw away real minutes and bias the rate
    # toward whichever half happened to be bigger. Born disambiguates players
    # who share a name (this table has two Bruno Fernandes, two Jack Robinson).
    born = (_num(raw[have["born"]]) if "born" in have
            else pd.Series(np.nan, index=d.index))
    d["_grp"] = d["player"].map(_norm) + "|" + born.fillna(-1).astype(int).astype(str)
    # min_count=1 so a column the league does not publish stays NaN. Pandas
    # sums an all-NaN group to 0.0, which would hand the model a fabricated
    # "took no shots" for every Championship player instead of "not measured".
    agg = {c: pd.NamedAgg(column=c, aggfunc=lambda x: x.sum(min_count=1))
           for c in ("minutes", "matches", "goals", "assists",
                     "xg", "npxg", "xag", "shots", "kp")}
    d["squad"] = (raw[have["squad"]].astype(str).str.strip()
                  if "squad" in have else "")
    d = (d.groupby("_grp", as_index=False)
           .agg(player=pd.NamedAgg(column="player", aggfunc="first"),
                squad=pd.NamedAgg(column="squad",
                                  aggfunc=lambda x: "|".join(sorted(set(x)))),
                **agg))

    d = d[d["minutes"] >= MIN_PRIOR_MINUTES]
    if d.empty:
        return pd.DataFrame()
    nineties = d["minutes"] / 90.0

    # npxG is the preferred shot-quality signal; fall back to xG where the
    # league has xG but not the penalty split. Both absent -> stays NaN.
    npxg = d["npxg"].where(d["npxg"].notna(), d["xg"])

    out = pd.DataFrame({
        "ext_name": d["player"],
        "ext_key": d["player"].map(_norm),
        "ext_surname_key": d["player"].map(_surname_key),
        "ext_league": league,
        "ext_squads": d["squad"],
        "ext_league_tier": LEAGUE_TIER.get(league, DEFAULT_TIER),
        "ext_minutes": d["minutes"],
        "ext_matches": d["matches"],
        "ext_min_per_match": d["minutes"] / d["matches"].clip(lower=1),
        "ext_xg90": npxg / nineties,
        "ext_xa90": d["xag"] / nineties,
        "ext_shots90": d["shots"] / nineties,
        "ext_kp90": d["kp"] / nineties,
        "ext_goals90": d["goals"] / nineties,
    })
    return out.sort_values("ext_minutes", ascending=False).drop_duplicates("ext_key")


def _clubs(s: str) -> set:
    return {_norm(x) for x in str(s).split("|") if _norm(x)}


def match_with_club(panel_players: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    """
    Name matching for hand-exported tables, corroborated by club.

    `match()` falls back to surname + first initial, which is too loose here
    and wrong in both directions. In the 2025-26 Championship table it hands
    Hull's Nobel Mendy the numbers of Watford's NAMPALYS Mendy (same initial,
    different player), while rejecting Hull's Mohamed Belloumi -- who is
    FBref's Bachir Belloumi at the same club -- because the initial differs.

    So the second tier here is surname + SAME CLUB instead, which is the
    discriminator that actually separates those two cases. It is strongest for
    promoted squads, where club is unchanged across the two tables, and that is
    where most of the missing points live. Anything weaker stays unmatched: a
    fabricated prior is worse than none, because the model cannot tell it apart
    from a measured one.
    """
    p = panel_players.copy()
    p["key"] = p["player_name"].map(_norm)
    p["sur"] = p["player_name"].map(lambda n: _norm(n).split()[-1] if _norm(n) else "")
    p["club_n"] = p["club"].map(_norm)
    cols = [c for c in priors.columns if c.startswith("ext_")
            and c not in ("ext_key", "ext_surname_key", "ext_name", "ext_squads")]

    pr = priors.copy()
    pr["sur"] = pr["ext_name"].map(lambda n: _norm(n).split()[-1] if _norm(n) else "")

    exact = pr.drop_duplicates("ext_key").set_index("ext_key")
    out = p.join(exact[cols], on="key")
    out["ext_match_method"] = np.where(out[cols[0]].notna(), "exact", None)

    need = out[out[cols[0]].isna()].copy()
    if not need.empty:
        # one row per (surname, club); a surname appearing twice at the same
        # club is ambiguous and is dropped rather than guessed
        cand = pr.assign(club_n=pr["ext_squads"].map(_clubs)).explode("club_n")
        cand = cand[cand["club_n"].notna()]
        counts = cand.groupby(["sur", "club_n"]).size()
        cand = cand.set_index(["sur", "club_n"])
        cand = cand[~cand.index.isin(counts[counts > 1].index)]

        m2 = need.drop(columns=cols).join(cand[cols], on=["sur", "club_n"])
        m2["ext_match_method"] = np.where(m2[cols[0]].notna(), "surname+club", None)
        out = pd.concat([out[out[cols[0]].notna()], m2], ignore_index=True)

    out["ext_available"] = out[cols[0]].notna().astype(float)
    return out.drop(columns=["key", "sur", "club_n"])


def _league_from_stem(stem: str, season_frag: str) -> str:
    """
    The league a filename names, regardless of how it was punctuated.

    "ENG-Championship_2025-2026", "ENG_Championship_2025-2026" and
    "eng championship 2025-2026" all mean the same league. Splitting on "_"
    would read the second as "ENG" and silently drop it to DEFAULT_TIER, so
    match on letters alone against the declared league keys.
    """
    core = stem.replace(season_frag, "").strip("_- ")
    flat = re.sub(r"[^a-z0-9]", "", core.lower())
    for key in LEAGUE_TIER:
        if re.sub(r"[^a-z0-9]", "", key.lower()) == flat:
            return key
    return core          # unknown league: keep as written, tier falls back


def load_manual(season_code: str, directory=None) -> pd.DataFrame:
    """
    Every hand-exported table for one season, concatenated.

    `season_code` is the Understat-style 4-digit code used elsewhere in this
    module ("2526"); files are named {LEAGUE}_{YYYY-YYYY}.csv, so "2526" maps
    to the filename fragment "2025-2026".
    """
    from pathlib import Path
    directory = Path(directory) if directory is not None else (
        Path(__file__).resolve().parents[2] / "data" / "external_manual")
    if not directory.is_dir():
        return pd.DataFrame()

    start = 2000 + int(season_code[:2])
    want = f"{start}-{start + 1}"
    frames = []
    for path in sorted(directory.glob("*.csv")):
        stem = path.stem
        if want not in stem:
            continue
        league = _league_from_stem(stem, want)
        try:
            got = manual_to_priors(_read_fbref_csv(path), league)
        except Exception as exc:
            print(f"  ! {path.name}: {exc}")
            continue
        # an empty frame carries no columns, and concat would drop ext_minutes
        # for every other league in the season -- a placeholder file waiting to
        # be pasted into must not break the ones already filled
        if not got.empty:
            frames.append(got)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("ext_minutes", ascending=False).drop_duplicates("ext_key")

def with_manual(out: pd.DataFrame, panel: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    Fill players Understat could not place from the hand-exported tables.

    Understat covers the big-5 only. Everyone else -- a promoted club's whole
    squad above all -- reaches the model with no prior at all. This layer runs
    strictly AFTER the Understat match and touches only rows where
    `ext_available` is 0, so a player with big-5 evidence keeps it and the
    existing path is unchanged.

    Club-corroborated matching is used here (see `match_with_club`), not the
    surname+initial fallback.
    """
    code = PRIOR_SEASON.get(season)
    if code is None:
        return out
    man = load_manual(code)
    if man.empty:
        return out

    need_codes = out.loc[out["ext_available"].fillna(0).eq(0), C.PLAYER_KEY]
    if need_codes.empty:
        return out
    players = (panel[panel["season"].eq(season)]
               .sort_values("gameweek")
               .drop_duplicates(C.PLAYER_KEY, keep="last")
               [[C.PLAYER_KEY, "player_name", "team_name"]]
               .rename(columns={"team_name": "club"}))
    players = players[players[C.PLAYER_KEY].isin(need_codes)]
    if players.empty:
        return out

    filled = match_with_club(players, man)
    filled = filled[filled["ext_available"].eq(1)]
    if filled.empty:
        return out

    cols = [c for c in out.columns if c.startswith("ext_")]
    take = filled.set_index(C.PLAYER_KEY)
    idx = out[C.PLAYER_KEY].map(lambda k: k in take.index)
    for c in cols:
        if c in take.columns:
            out.loc[idx.values, c] = out.loc[idx.values, C.PLAYER_KEY].map(take[c]).values
    out.loc[idx.values, "ext_source"] = "manual"
    out["ext_available"] = out["ext_available"].fillna(0.0)
    return out
