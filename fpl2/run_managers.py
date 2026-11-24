#!/usr/bin/env python3
"""
S67 — do a manager's sides carry an attacking or defensive signature?

The question, in the user's words: Pochettino managed Chelsea and now manages
Manchester United, so should we expect more or less attacking return from his
sides?

THE CONFOUND IS THE CLUB. A manager at a rich club scores more because the
squad is better, not because of him. Any league-wide table of "goals per game
by manager" measures the clubs they were hired by. So this reports the
descriptive table ONLY as description, and the measurement is a different
design:

  WITHIN-CLUB, BEFORE AND AFTER A CHANGE. When a club changes manager
  mid-season, the squad, the budget and the fixtures barely move. The change
  in scoring and conceding across that boundary is much closer to a manager
  effect than any cross-club comparison. Clustered on the club so that one
  volatile club cannot carry the result.

Even that is not clean -- clubs sack managers when results are bad, so the
"before" is selected for being poor and regression to the mean alone predicts
improvement. `run_managers.py` therefore reports a NEGATIVE CONTROL: the same
statistic computed at FAKE change points, placed mid-spell where no change
occurred. If the real boundaries do not beat the fake ones, there is nothing
here but mean reversion.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy import stats                                          # noqa: E402
from fpl2 import config as C, managers                           # noqa: E402

PANEL = C.cache("fpl2_panel_mgr.pkl")
MIN_GW = 20          # a manager needs this many club-gameweeks to be listed


def team_weeks(p: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (season, gameweek, club): goals for/against, xG for/against.

    The panel carries `expected_goals` but NOT `expected_goals_conceded`, so
    xG against is taken as the OPPONENT's xG for in the same match -- which is
    what it is, and is better than dropping the defensive half of the question.
    """
    d = p[p["fixtures"] > 0].copy()
    d["opp"] = d["opponents"].map(
        lambda t: t[0] if isinstance(t, (tuple, list)) and len(t) else None)
    g = (d.groupby(["season", "gameweek", "team"], as_index=False)
         .agg(gf=("goals_scored", "sum"), ga=("goals_conceded", "max"),
              xgf=("expected_goals", "sum"), opp=("opp", "first"),
              manager=("manager", "first"), mgr_days=("mgr_days", "first"),
              mgr_match_no=("mgr_match_no", "first"),
              caretaker=("mgr_caretaker", "first")))
    opp_xg = g[["season", "gameweek", "team", "xgf"]].rename(
        columns={"team": "opp", "xgf": "xga"})
    g = g.merge(opp_xg, on=["season", "gameweek", "opp"], how="left")
    return g[g["manager"].notna()]


def by_manager(tw: pd.DataFrame) -> pd.DataFrame:
    d = (tw.groupby("manager")
         .agg(gw=("gf", "size"), clubs=("team", "nunique"),
              gf=("gf", "mean"), ga=("ga", "mean"),
              xgf=("xgf", "mean"), xga=("xga", "mean")).reset_index())
    return d[d.gw >= MIN_GW].sort_values("gf", ascending=False)


def change_points(tw: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """
    Scoring and conceding either side of a managerial change at the SAME club.

    `window` gameweeks before and after, within one season so the squad is
    the same. Returns one row per change.
    """
    rows = []
    for (season, team), d in tw.sort_values(["season", "team", "gameweek"]).groupby(
            ["season", "team"]):
        d = d.reset_index(drop=True)
        chg = d.index[d["manager"].ne(d["manager"].shift())][1:]   # skip the first
        for i in chg:
            pre, post = d.iloc[max(0, i - window):i], d.iloc[i:i + window]
            if len(pre) < 3 or len(post) < 3:
                continue
            rows.append({"season": season, "team": team,
                         "out": pre["manager"].iloc[-1], "in": post["manager"].iloc[0],
                         "gw": int(d.loc[i, "gameweek"]),
                         "d_gf": post.gf.mean() - pre.gf.mean(),
                         "d_ga": post.ga.mean() - pre.ga.mean(),
                         "d_xgf": post.xgf.mean() - pre.xgf.mean(),
                         "d_xga": post.xga.mean() - pre.xga.mean(),
                         "real": True})
    return pd.DataFrame(rows)


def placebo(tw: pd.DataFrame, window: int = 6, seed: int = 0) -> pd.DataFrame:
    """
    NEGATIVE CONTROL: the same statistic at points where NOTHING happened.

    A club sacks a manager after bad results, so the `before` window is
    selected for being poor and mean reversion alone predicts a rise. Fake
    boundaries placed mid-spell carry that selection too only if it is
    selection; if the real changes do not beat these, there is no manager
    effect to speak of.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for (season, team), d in tw.sort_values(["season", "team", "gameweek"]).groupby(
            ["season", "team"]):
        d = d.reset_index(drop=True)
        if d["manager"].nunique() != 1 or len(d) < 2 * window:
            continue                       # only stable spells, so no real change
        i = int(rng.integers(window, len(d) - window + 1))
        pre, post = d.iloc[i - window:i], d.iloc[i:i + window]
        rows.append({"season": season, "team": team, "out": d["manager"].iloc[0],
                     "in": d["manager"].iloc[0], "gw": int(d.loc[i, "gameweek"]),
                     "d_gf": post.gf.mean() - pre.gf.mean(),
                     "d_ga": post.ga.mean() - pre.ga.mean(),
                     "d_xgf": post.xgf.mean() - pre.xgf.mean(),
                     "d_xga": post.xga.mean() - pre.xga.mean(),
                     "real": False})
    return pd.DataFrame(rows)


def clustered(d, g):
    d = np.asarray(d, float); g = np.asarray(g)
    ok = np.isfinite(d); d, g = d[ok], g[ok]
    n, G = len(d), len(np.unique(g)); mu = d.mean()
    ss = sum((d[g == k] - mu).sum() ** 2 for k in np.unique(g))
    se = np.sqrt((G / (G - 1)) * ss / n ** 2) if G > 1 else np.nan
    t = mu / se if se and se > 0 else np.nan
    return {"mean": mu, "se": se, "t": t, "G": G, "n": n,
            "p": 2 * (1 - stats.t.cdf(abs(t), G - 1)) if G > 1 else np.nan,
            "mde": 2.8 * se if se else np.nan}


def main():
    p = pd.read_pickle(PANEL)
    tw = team_weeks(p)
    print(f"{len(tw):,} club-gameweeks with a manager, "
          f"{tw.manager.nunique()} managers, {tw.season.nunique()} seasons\n")

    print("=" * 74)
    print("S67.1  POCHETTINO BY CLUB  (description, NOT a manager effect)")
    print("=" * 74)
    poch = tw[tw.manager.str.contains("Pochettino", na=False)]
    if len(poch):
        d = (poch.groupby(["team", "season"])
             .agg(gw=("gf", "size"), gf=("gf", "mean"), ga=("ga", "mean"),
                  xgf=("xgf", "mean"), xga=("xga", "mean")).reset_index())
        print(d.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
        lg = tw.groupby("season").agg(gf=("gf", "mean"), ga=("ga", "mean"))
        print("\n  against the league average in the same seasons:")
        for _, r in d.iterrows():
            if r.season in lg.index:
                print(f"   {r.team:<5}{r.season}  gf {r.gf:5.2f} vs {lg.loc[r.season,'gf']:5.2f}"
                      f"   ga {r.ga:5.2f} vs {lg.loc[r.season,'ga']:5.2f}")
    else:
        print("  no Pochettino rows resolved")

    print("\n" + "=" * 74)
    print(f"S67.2  BY MANAGER, {MIN_GW}+ club-gameweeks (DESCRIPTION -- confounded by club)")
    print("=" * 74)
    bm = by_manager(tw)
    print(f"  {len(bm)} managers qualify. Top and bottom 8 by goals scored:\n")
    show = pd.concat([bm.head(8), bm.tail(8)])
    print(show.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
    print("\n  This table is NOT a manager effect. It is mostly which clubs")
    print("  hired them -- the measurement is below.")

    print("\n" + "=" * 74)
    print("S67.3  WITHIN-CLUB, BEFORE vs AFTER A CHANGE (the measurement)")
    print("=" * 74)
    real = change_points(tw)
    fake = placebo(tw)
    print(f"  {len(real)} real mid-season changes, {len(fake)} placebo points\n")
    print(f'{"":<22}{"real":>9}{"se":>7}{"t":>7}{"placebo":>10}{"se":>7}{"t":>7}{"real-placebo":>14}')
    for col, lab in [("d_gf", "goals scored"), ("d_ga", "goals conceded"),
                     ("d_xgf", "xG for"), ("d_xga", "xG against")]:
        a = clustered(real[col], real.team)
        b = clustered(fake[col], fake.team)
        print(f'{lab:<22}{a["mean"]:>+9.3f}{a["se"]:>7.3f}{a["t"]:>7.2f}'
              f'{b["mean"]:>+10.3f}{b["se"]:>7.3f}{b["t"]:>7.2f}'
              f'{a["mean"]-b["mean"]:>+14.3f}')
    print(f'\n  clustered on club, G = {clustered(real.d_gf, real.team)["G"]} real / '
          f'{clustered(fake.d_gf, fake.team)["G"]} placebo')
    print("  If `real` does not beat `placebo`, a new manager improving things")
    print("  is mean reversion after a bad run, not the manager.")


if __name__ == "__main__":
    main()
