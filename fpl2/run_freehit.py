#!/usr/bin/env python3
"""
S65 — when is a free hit worth playing, and does BLANK_WEIGHT do anything?

Two questions, deliberately separate.

1. WHAT A BLANK WEEK COSTS. Within-squad, over ten seasons: a squad's score
   in a blank-affected week against its OWN mean in genuinely clean weeks.
   That difference is what a free hit recovers.

   Two traps on the way to it, both of which produced a wrong answer first:
     * the compact panel OMITS blank rows, so counting `fixtures == 0` finds
       zero blanks in eleven seasons. Blanking teams are ABSENT, not zero.
     * `with_blanks` reindexes every player into every gameweek, so a player
       who left the league reads as blank for ever. 44 of 83 flagged weeks
       had NO team blanking at all. Those are squad-data artefacts and must
       be excluded by requiring that some team really did blank.

2. IS BLANK_WEIGHT INERT? It is a hand-chosen 2.0 added per blanked squad
   player. Before tuning a free parameter, ask whether it changes any
   DECISION -- S27 found season totals chaotic in exactly this kind of knob,
   so a sweep on the total would be uninterpretable noise. Counting decision
   flips is cheap and answers the prior question.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy import stats                                          # noqa: E402
from fpl2 import chips, config as C, data, measure, optimise     # noqa: E402

SEEDS, K = 0, 6
SWEEP = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)


def blank_cost(panel: pd.DataFrame) -> pd.DataFrame:
    """Question 1: what a blank week costs, within-squad."""
    rows = []
    for si, season in enumerate(s for s in sorted(panel.season.unique())
                                if s != "2026-27"):
        base = panel[panel.season.eq(season)]
        grid = data.with_blanks(base, season)
        grid["xp"] = pd.to_numeric(grid["points"], errors="coerce").fillna(0.0)
        grid.loc[grid["fixtures"] <= 0, "xp"] = 0.0
        squads = measure.neutral_squads(grid, k=K, seed=si)
        if not squads:
            continue
        teams = set(base.team.dropna().unique())
        for gw, week in grid.groupby("gameweek"):
            blank_teams = len(teams - set(base[base.gameweek.eq(gw)].team.dropna()))
            for k, squad in enumerate(squads):
                held = week[week[C.PLAYER_KEY].isin(squad)]
                xi = optimise.best_xi(held, "xp")
                if xi is None:
                    continue
                rows.append({"season": season, "gameweek": int(gw), "squad": k,
                             "blank_teams": blank_teams,
                             "my_blank": int((held["fixtures"] <= 0).sum()),
                             "pts": float(xi["xp"].sum())})
    D = pd.DataFrame(rows)
    # baseline from GENUINELY clean weeks only
    clean = D[(D.blank_teams == 0) & (D.my_blank == 0)]
    base = clean.groupby(["season", "squad"])["pts"].mean().rename("clean")
    E = D[D.blank_teams > 0].join(base, on=["season", "squad"])
    E = E[E["clean"].notna()].copy()
    E["loss"] = E["clean"] - E["pts"]
    return E


def blank_weight_is_inert(panel: pd.DataFrame) -> pd.DataFrame:
    """Question 2: does BLANK_WEIGHT change any decision?"""
    rows = []
    for si, season in enumerate(s for s in sorted(panel.season.unique())
                                if s != "2026-27"):
        base = panel[panel.season.eq(season)]
        grid = data.with_blanks(base, season)
        grid["xp"] = pd.to_numeric(grid["points"], errors="coerce").fillna(0.0)
        grid.loc[grid["fixtures"] <= 0, "xp"] = 0.0
        squads = measure.neutral_squads(grid, k=2, seed=si)
        if not squads:
            continue
        for w in SWEEP:
            for k, squad in enumerate(squads):
                pol = chips.rolling_policy(grid, "xp", blank_weight=w)
                for gw in sorted(grid.gameweek.unique()):
                    week = grid[grid.gameweek.eq(gw)]
                    avail = [c for c in chips.CHIP_WINDOWS]
                    d = pol(int(gw), list(squad), week, avail)
                    if d is not None:
                        rows.append({"season": season, "squad": k, "w": w,
                                     "gameweek": int(gw), "chip": d})
                        break        # one chip, then the squad would change
    return pd.DataFrame(rows)


def main():
    panel = pd.read_pickle(C.cache("fpl2_panel_us.pkl"))
    print("S65.1  WHAT A BLANK WEEK COSTS (within-squad, real fixture blanks)\n")
    E = blank_cost(panel)
    E["band"] = pd.cut(E.my_blank, [-1, 0, 2, 3, 5, 15],
                       labels=["0", "1-2", "3", "4-5", "6+"])
    print(f'{"your blanks":<13}{"n":>7}{"XI pts":>9}{"baseline":>10}'
          f'{"LOSS":>8}{"se":>7}{"MDE":>7}{"G":>4}')
    for b, g in E.groupby("band", observed=True):
        cl = g.groupby("season")["loss"].mean()
        se = cl.std() / np.sqrt(len(cl)) if len(cl) > 1 else np.nan
        print(f'{str(b):<13}{len(g):>7,}{g.pts.mean():>9.1f}{g["clean"].mean():>10.1f}'
              f'{g.loss.mean():>8.1f}{se:>7.1f}{2.8*se:>7.1f}{len(cl):>4}')
    for lo in (3, 4, 6):
        cl = E[E.my_blank >= lo].groupby("season")["loss"].mean()
        se = cl.std() / np.sqrt(len(cl))
        t = cl.mean() / se
        print(f'\n  {lo}+ blanks: loss {cl.mean():.1f}, se {se:.1f}, t {t:.2f}, '
              f'p {2*(1-stats.t.cdf(abs(t), len(cl)-1)):.4f}, G {len(cl)}, '
              f'{"RESOLVABLE" if abs(cl.mean()) > 2.8*se else "not resolvable"}')
    h = E[E.my_blank >= 4].groupby(["season", "gameweek"]).size().reset_index()
    print(f'\n  frequency: {len(h)} four-plus-blank weeks over {E.season.nunique()} '
          f'seasons = {len(h)/E.season.nunique():.1f}/season')
    print(f'  first half GW1-19: {len(h[h.gameweek <= 19])}   '
          f'second half: {len(h[h.gameweek > 19])}')

    print("\n\nS65.2  IS BLANK_WEIGHT INERT?\n")
    F = blank_weight_is_inert(panel)
    if F.empty:
        print("  no chip ever fired"); return
    piv = F.pivot_table(index=["season", "squad"], columns="w",
                        values="gameweek", aggfunc="first")
    same = piv.nunique(axis=1, dropna=False).le(1)
    print(f'  {len(piv)} (season, squad) runs swept over BLANK_WEIGHT {SWEEP}')
    print(f'  runs where the FIRST CHIP WEEK never moved: {same.sum()}/{len(piv)} '
          f'({100*same.mean():.0f}%)')
    chip = F.pivot_table(index=["season", "squad"], columns="w",
                         values="chip", aggfunc="first")
    samec = chip.nunique(axis=1, dropna=False).le(1)
    print(f'  runs where the CHIP CHOSEN never moved:     {samec.sum()}/{len(chip)} '
          f'({100*samec.mean():.0f}%)')
    print(f'\n  verdict: BLANK_WEIGHT is '
          f'{"INERT over this range" if same.all() and samec.all() else "LIVE -- it moves decisions"}')


if __name__ == "__main__":
    main()
