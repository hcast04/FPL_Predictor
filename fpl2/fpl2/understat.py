"""
Join Understat per-match xG onto FPL player-fixtures.

WHY ONLY xG. Measured on 2022-23, where FPL publishes Opta's figures and
Understat covers the same matches (n = 7,217 matched player-fixtures):

    xG   ratio 1.030   spearman 0.9910   <- the same quantity, 3% level shift
    xA   ratio 1.231   spearman 0.6903   <- a DIFFERENT quantity

That xA gap held across three independent join constructions (0.756 / 0.639 /
0.690), so it is the source and not the matching. **xA is therefore NOT taken
from Understat.** It is left absent, exactly as `starts` is, and the assist
component simply trains on Opta-era seasons only -- which is honest and is
better than feeding it a measure that ranks players differently (M20).

THE JOIN. There is no shared key, so this is name matching -- the thing that
produced the Nobel Mendy / Nampalys Mendy error on this project. Three
constraints make it safe:

  1. same DATE and same CLUB, so the candidate set is ~20 players, not ~600
  2. an assignment, not a lookup: each Understat row is consumed once, best
     match first, so two FPL players cannot claim the same source row
  3. minutes agreement enters the score, which is independent of the name

A naive any-token match reached 98.8% coverage and matched De Gea to David
Raya -- caught only because xG spearman FELL from 0.9888 to 0.9387. Coverage
is not a quality measure. `validate` reports both, per season.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

C_PLAYER_NAME = "player_name"

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_extra"
MIN_NAME_SIM = 0.45


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", " ", s.lower()).strip()


def _sim(a: str, b: str) -> float:
    r = difflib.SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    # a mononym is usually a SUBSET of the full legal name:
    # "ederson" vs "ederson santana de moraes"
    if ta and tb and (ta <= tb or tb <= ta):
        return max(r, 0.95)
    if a in b or b in a:
        return max(r, 0.90)
    return r


def team_map(us: pd.DataFrame, fpl: pd.DataFrame) -> dict:
    """
    Understat club -> FPL club, matched on FIXTURE DATES, not on names.

    Names cannot do this job. Six clubs relegated before 2022-23 have no name
    in any source we hold and are labelled `T25`, `T80`, `T88`, `T110` and so
    on (S46); name similarity matches those to nothing, which cost 20% of
    2016-17's players before this was fixed.

    A club's set of fixture dates is close to unique -- 38 dates drawn from a
    calendar of ~100 -- so the overlap identifies it with no linguistic
    assumption at all. Assignment is greedy best-first so two FPL clubs cannot
    claim the same Understat club.
    """
    ud = {t: set(g["date"].dropna()) for t, g in us.groupby("team")}
    fd = {t: set(g["date"].dropna()) for t, g in fpl.groupby("team_name")}
    pairs = []
    for u, du in ud.items():
        for f, df in fd.items():
            inter = len(du & df)
            union = len(du | df) or 1
            pairs.append((inter / union, inter, u, f))
    out, taken = {}, set()
    for j, inter, u, f in sorted(pairs, reverse=True):
        if u in out or f in taken or j < 0.5:
            continue
        out[u] = f
        taken.add(f)
    return out


def join_season(fpl: pd.DataFrame, season: str) -> pd.DataFrame:
    """One row per matched (player-fixture), carrying Understat xG."""
    path = RAW_DIR / f"understat_{season}.pkl"
    if not path.exists():
        return pd.DataFrame()
    us = pd.read_pickle(path).reset_index()
    us["date"] = pd.to_datetime(us["game"].str.slice(0, 10), errors="coerce")
    f = fpl[fpl["season"].eq(season)].copy()
    # the RAW frame carries `kickoff_time`; `build_panel` aggregates it to
    # `kickoff_first`. Accept either so this works on both.
    kc = ("kickoff_time" if "kickoff_time" in f.columns
          else "kickoff_first" if "kickoff_first" in f.columns
          else "kickoff" if "kickoff" in f.columns else None)
    if kc is None:
        return pd.DataFrame()
    f["date"] = (pd.to_datetime(f[kc], errors="coerce", utc=True)
                 .dt.tz_localize(None).dt.normalize())
    f = f[f["minutes"].gt(0)]
    if f.empty or us.empty:
        return pd.DataFrame()

    # the panel renames team_short_name -> team; accept either
    if "team_name" not in f.columns:
        f = f.assign(team_name=f["team"] if "team" in f.columns else None)
    tm = team_map(us, f)
    us["fteam"] = us["team"].map(tm)
    us["_n"] = us["player"].map(_norm)
    f["_n"] = f["player_name"].map(_norm)

    rows = []
    name_col = "player_name" if "player_name" in f.columns else C_PLAYER_NAME
    for (d, t), fg in f.groupby(["date", "team_name"]):
        ug = us[us["date"].eq(d) & us["fteam"].eq(t)]
        if ug.empty:
            continue
        pairs = []
        for fi, fr in fg.iterrows():
            for ui, ur in ug.iterrows():
                s = _sim(fr["_n"], ur["_n"])
                mn = 1.0 - min(abs(fr["minutes"] - ur["minutes"]), 30) / 30.0
                pairs.append((0.75 * s + 0.25 * mn, s, fi, ui))
        taken = set()
        for _, s, fi, ui in sorted(pairs, key=lambda x: -x[0]):
            if fi in taken or ui in taken or s < MIN_NAME_SIM:
                continue
            taken.add(fi); taken.add(ui)
            fr, ur = fg.loc[fi], ug.loc[ui]
            rows.append({"season": season, "gameweek": fr["gameweek"],
                         "player_code": fr["player_code"], "sim": s,
                         "fpl_minutes": fr["minutes"], "us_minutes": ur["minutes"],
                         "us_xg": ur["xg"], "us_xa": ur["xa"],
                         "us_shots": ur["shots"], "us_key_passes": ur["key_passes"],
                         "us_xg_chain": ur.get("xg_chain", np.nan),
                         "us_xg_buildup": ur.get("xg_buildup", np.nan),
                         "opta_xg": fr.get("expected_goals", np.nan),
                         "opta_xa": fr.get("expected_assists", np.nan)})
    return pd.DataFrame(rows)


# Columns Understat supplies. `us_xa` is carried but NOT used as a target by
# default -- see the module docstring for why. `xg_chain` and `xg_buildup` have
# no FPL equivalent at all: they credit every player in the possession that led
# to a shot, which is the closest thing to a creativity measure that is not
# assist-dependent.
US_COLS = ["us_xg", "us_xa", "us_shots", "us_key_passes",
           "us_xg_chain", "us_xg_buildup"]


def attach(panel: pd.DataFrame, seasons=None) -> pd.DataFrame:
    """
    Add Understat per-gameweek features to the panel, for every season we hold.

    One source for the FEATURES across all ten seasons, so nothing is spliced.
    FPL keeps what it alone owns -- goals, assists, clean sheets, defensive
    contributions, minutes, bonus, cards -- which are the quantities the FPL
    scoring rules are defined on and which Understat does not measure.

    A player-gameweek with no Understat row (about 1.3%, plus every blank) gets
    NaN, NOT zero. `features.usable` drops an all-NaN column per fold and
    `build_targets` keeps NaN targets out of the fit, both since M20.
    """
    have = {re.search(r"understat_(\d{4}-\d{2})", str(f)).group(1)
            for f in RAW_DIR.glob("understat_*.pkl")}
    todo = sorted(have & set(panel["season"].unique())) if seasons is None else list(seasons)
    parts = [join_season(panel, s) for s in todo]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return panel.assign(**{c: np.nan for c in US_COLS})
    j = pd.concat(parts, ignore_index=True)
    # a double gameweek has two Understat rows for one FPL gameweek: sum the
    # flows, exactly as the panel sums goals
    j = (j.groupby(["season", "gameweek", "player_code"], as_index=False)
           .agg(us_xg=("us_xg", "sum"), us_xa=("us_xa", "sum"),
                us_shots=("us_shots", "sum"), us_key_passes=("us_key_passes", "sum"),
                us_xg_chain=("us_xg_chain", "sum"),
                us_xg_buildup=("us_xg_buildup", "sum")))
    return panel.merge(j, on=["season", "gameweek", "player_code"], how="left")


def validate(fpl: pd.DataFrame, season: str, verbose: bool = True) -> dict:
    """Coverage AND quality. Coverage alone hid a De-Gea-to-Raya match."""
    m = join_season(fpl, season)
    f = fpl[fpl["season"].eq(season)]
    f = f[f["minutes"].gt(0)]
    if m.empty:
        return {"season": season, "matched": 0, "coverage": 0.0}
    b = m[m["opta_xg"].notna()]
    out = {
        "season": season, "matched": len(m), "n_fpl": len(f),
        "coverage": len(m) / max(len(f), 1),
        "mins_within_2": float((m["fpl_minutes"] - m["us_minutes"]).abs().le(2).mean()),
        "mins_off_by_5+": float((m["fpl_minutes"] - m["us_minutes"]).abs().gt(5).mean()),
        "mean_name_sim": float(m["sim"].mean()),
        "xg_spearman": (float(b["us_xg"].corr(b["opta_xg"], method="spearman"))
                        if len(b) > 100 else np.nan),
        "xg_ratio": (float(b["us_xg"].sum() / b["opta_xg"].sum())
                     if len(b) > 100 and b["opta_xg"].sum() > 0 else np.nan),
    }
    if verbose:
        print(f"  {season}: matched {out['matched']:>6,}/{out['n_fpl']:>6,} "
              f"= {100*out['coverage']:5.1f}%   mins±2 {100*out['mins_within_2']:5.1f}%"
              f"   off>5 {100*out['mins_off_by_5+']:4.1f}%   sim {out['mean_name_sim']:.3f}"
              + (f"   xG sp {out['xg_spearman']:.4f} ratio {out['xg_ratio']:.3f}"
                 if not np.isnan(out["xg_spearman"]) else "   (no Opta overlap)"))
    return out
