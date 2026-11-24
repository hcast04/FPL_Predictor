#!/usr/bin/env python3
"""
S68 — does a manager's PRIOR record predict his NEXT club's output?

The user's hypothesis: managers are more stable than clubs. A club's squad,
budget and league position churn; a manager's approach travels with him. If
so, manager identity carries information the club's own history does not --
and the place to look is the moment a manager ARRIVES somewhere new, when the
club's recent form is about the previous regime.

THE TEST. For every (manager, club) spell, take its first `WINDOW` gameweeks
and ask what predicts them:

    club_prior   the club's goals for / against BEFORE this spell began
    mgr_prior    the manager's goals for / against at OTHER clubs, before
                 this spell began

Both strictly lagged on `t`, the global chronological index, so neither can
see the window it is predicting. A spell with no manager history (his first
job here) contributes to the club_prior test only.

NEGATIVE CONTROL. Manager identity is SHUFFLED among spells -- each spell
keeps a real manager's prior, but the wrong one's. Everything else is
identical. If the shuffled priors predict as well as the true ones, the
apparent signal is the club's, the era's, or the window's, and not the
manager's. This is the S29 construction, which found opponent identity worth
exactly nothing once scrambled.

PROMOTED CLUBS are reported separately, per the user: a manager arriving with
a promoted side is a different population, and S9 found exactly that shape
for PLAYERS -- Championship arrivals ran at 0.713 of forecast against 0.884
for other sources, a gap invisible in the pooled number.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy import stats                                          # noqa: E402
from fpl2 import config as C                                     # noqa: E402

PANEL = C.cache("fpl2_panel_mgr.pkl")
WINDOW = 8          # gameweeks of a new spell to predict
MIN_PRIOR = 10      # a prior needs this many gameweeks to be used
SHUFFLES = 200


def team_weeks(p: pd.DataFrame) -> pd.DataFrame:
    d = p[p["fixtures"] > 0]
    g = (d.groupby(["season", "gameweek", "team"], as_index=False)
         .agg(gf=("goals_scored", "sum"), ga=("goals_conceded", "max"),
              manager=("manager", "first"), t=("t", "first"),
              caretaker=("mgr_caretaker", "first")))
    return g[g["manager"].notna()].sort_values("t").reset_index(drop=True)


def spells(tw: pd.DataFrame) -> pd.DataFrame:
    """One row per (club, manager) stint, in chronological order."""
    tw = tw.sort_values(["team", "t"]).copy()
    new = (tw["manager"].ne(tw["manager"].shift())
           | tw["team"].ne(tw["team"].shift()))
    tw["spell"] = new.cumsum()
    return tw


def build(tw: pd.DataFrame) -> pd.DataFrame:
    """Each spell's opening window, with strictly-lagged club and manager priors."""
    sp = spells(tw)
    rows = []
    for sid, d in sp.groupby("spell"):
        d = d.sort_values("t")
        start_t = d["t"].iloc[0]
        team, mgr = d["team"].iloc[0], d["manager"].iloc[0]
        win = d.head(WINDOW)
        if len(win) < 4:
            continue
        # the CLUB before this spell -- any manager
        cp = sp[(sp.team == team) & (sp.t < start_t)]
        # this MANAGER before this spell -- at any OTHER club
        mp = sp[(sp.manager == mgr) & (sp.t < start_t) & (sp.team != team)]
        rows.append({
            "spell": sid, "team": team, "manager": mgr,
            "season": d["season"].iloc[0], "start_t": start_t,
            "n": len(win), "caretaker": bool(d["caretaker"].iloc[0]),
            "y_gf": win.gf.mean(), "y_ga": win.ga.mean(),
            "club_gf": cp.gf.mean() if len(cp) >= MIN_PRIOR else np.nan,
            "club_ga": cp.ga.mean() if len(cp) >= MIN_PRIOR else np.nan,
            "club_n": len(cp),
            "mgr_gf": mp.gf.mean() if len(mp) >= MIN_PRIOR else np.nan,
            "mgr_ga": mp.ga.mean() if len(mp) >= MIN_PRIOR else np.nan,
            "mgr_n": len(mp)})
    return pd.DataFrame(rows)


def partial(y, x_keep, x_test):
    """Correlation of `x_test` with `y` after removing `x_keep` from both."""
    m = pd.DataFrame({"y": y, "k": x_keep, "x": x_test}).dropna()
    if len(m) < 12:
        return np.nan, len(m)
    ry = m.y - np.poly1d(np.polyfit(m.k, m.y, 1))(m.k)
    rx = m.x - np.poly1d(np.polyfit(m.k, m.x, 1))(m.k)
    return float(np.corrcoef(ry, rx)[0, 1]), len(m)


def main():
    p = pd.read_pickle(PANEL)
    tw = team_weeks(p)
    B = build(tw)
    B.to_pickle(C.cache("fpl2_s68_spells.pkl"))
    print(f"{len(B)} spells with an opening window of >=4 gameweeks")
    print(f"  with a club prior   : {B.club_gf.notna().sum()}")
    print(f"  with a manager prior: {B.mgr_gf.notna().sum()}  "
          f"(he had managed elsewhere first)\n")

    print("=" * 76)
    print(f"S68.1  WHAT PREDICTS THE FIRST {WINDOW} GAMEWEEKS OF A NEW SPELL")
    print("=" * 76)
    print(f'{"outcome":<16}{"predictor":<22}{"r":>8}{"n":>6}')
    for y, lab in [("y_gf", "goals scored"), ("y_ga", "goals conceded")]:
        for px, plab in [("club_", "the CLUB before"), ("mgr_", "the MANAGER elsewhere")]:
            col = px + y[2:]
            m = B[[y, col]].dropna()
            r = m[y].corr(m[col]) if len(m) > 3 else np.nan
            print(f'{lab:<16}{plab:<22}{r:>8.3f}{len(m):>6}')

    print("\n" + "=" * 76)
    print("S68.2  DOES THE MANAGER ADD ANYTHING OVER THE CLUB?  (partial r)")
    print("=" * 76)
    rng = np.random.default_rng(0)
    print(f'{"outcome":<16}{"partial r":>11}{"n":>6}{"shuffled mean":>15}{"sd":>7}{"p":>8}')
    for y, lab in [("y_gf", "goals scored"), ("y_ga", "goals conceded")]:
        ck, mk = "club_" + y[2:], "mgr_" + y[2:]
        r, n = partial(B[y], B[ck], B[mk])
        # NEGATIVE CONTROL: give each spell another manager's prior
        null = []
        sub = B[[y, ck, mk]].dropna()
        for _ in range(SHUFFLES):
            rr, _ = partial(sub[y], sub[ck], rng.permutation(sub[mk].values))
            if np.isfinite(rr):
                null.append(rr)
        null = np.array(null)
        pv = float((np.abs(null) >= abs(r)).mean()) if len(null) else np.nan
        print(f'{lab:<16}{r:>11.3f}{n:>6}{null.mean():>15.3f}{null.std():>7.3f}{pv:>8.3f}')
    print("\n  `shuffled` keeps every spell and every club prior and only moves")
    print("  WHICH manager's history is attached. If the true r sits inside")
    print("  that distribution, the signal is not the manager's.")

    print("\n" + "=" * 76)
    print("S68.3  MANAGERS ARRIVING WITH A PROMOTED CLUB")
    print("=" * 76)
    seasons = sorted(tw.season.unique())
    promoted = {}
    for i, s in enumerate(seasons[1:], 1):
        promoted[s] = set(tw[tw.season.eq(s)].team) - set(tw[tw.season.eq(seasons[i-1])].team)
    B["promoted"] = [r.team in promoted.get(r.season, set()) for r in B.itertuples()]
    opening = B[B.start_t.isin(B.groupby("team").start_t.min())]
    print(f'{"group":<28}{"spells":>8}{"gf":>8}{"ga":>8}{"has mgr prior":>15}')
    for flag, lab in [(True, "arrived with a promoted club"),
                      (False, "everyone else")]:
        g = B[B.promoted.eq(flag)]
        print(f'{lab:<28}{len(g):>8}{g.y_gf.mean():>8.2f}{g.y_ga.mean():>8.2f}'
              f'{int(g.mgr_gf.notna().sum()):>15}')
    print("\n  A promoted club scores less and concedes more because it is a")
    print("  promoted club. The question is whether its MANAGER explains any")
    print("  of it, and the partial-r test above is the one that can say.")


if __name__ == "__main__":
    main()
