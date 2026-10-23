"""
Chip timing policy.

FPL gives EIGHT chips a season -- two each of wildcard, free hit, bench boost
and triple captain, split across GW1-19 and GW20-38 (confirmed from
bootstrap-static). The simulator ignored all of them until 2026-10-21, which
S22 measured as part of a ~287 pts/season gap between the instrument and a real
manager.

KNOWLEDGE HORIZON is the whole difficulty. A June fixture list contains no
doubles and no blanks -- every one arises later from cup progression, European
conflicts and postponements, and FPL confirms them only ~2-4 gameweeks ahead.
Policies that scanned season-long double counts were reading the future (M17)
and their results are unachievable. This policy sees the schedule only to
`gw + KNOWN`.

Measured (`fpl2/run_chip_honest.py`, 3 holdout seasons): **+87 pts/season,
2/3 seasons**, against a leaky season-long planner's +69. Adapting weekly to
the squad's actual state beats committing in advance to the busiest gameweek,
because a plan made in June cannot know the bench will be injured.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import optimise
from .simulate import CHIP_WINDOWS

SEASON_GAMEWEEKS = 38   # a season is 38 gameweeks whether or not they exist yet
KNOWN = 4          # gameweeks of fixture schedule visible at any deadline
LOOK = 0.37        # secretary-rule observation fraction of each window
DOUBLE_BONUS = 1.6  # a double gameweek is worth ~this much more to a team chip
BLANK_WEIGHT = 2.0  # points per squad player with no fixture, for the free hit


def rolling_policy(grid: pd.DataFrame, signal: str = "xp"):
    """
    A chip policy that never reads a fixture it could not have known.

    Returns a callable for `simulate.simulate_season(..., chip_policy=...)`.
    Each chip's benefit is estimated from what is visible at the deadline: the
    current squad, this week's signal, and the schedule to `gw + KNOWN`. The
    secretary rule observes the first `LOOK` of each window, then takes the
    first gameweek beating everything seen; a deadline forces a play rather
    than wasting the chip, since deferral is the expensive failure (S4, S23).
    """
    # Deadline logic must run against the SCHEDULED season, not the gameweeks
    # present in the data. A live mid-season panel stops at the current
    # gameweek, so deriving the universe from it makes the policy believe the
    # season is ending and dump every chip at once -- in the 2026-27 run
    # through GW7 it fired all four first-half chips in GW4-7 and inflated the
    # total from 358 to 449. The chip WINDOWS are fixed calendar facts; only
    # the data is partial.
    season = list(range(1, SEASON_GAMEWEEKS + 1))
    present = sorted(grid["gameweek"].unique())
    seen: dict = {}

    def policy(gw, squad, week, avail):
        visible = [g for g in present if gw <= g <= gw + KNOWN]
        vis = grid[grid["gameweek"].isin(visible)]
        held = week[week[C.PLAYER_KEY].isin(squad)]
        if held.empty:
            return None
        xi = optimise.best_xi(held, signal)
        if xi is None:
            return None
        my_dbl = float((held["fixtures"] > 1).sum())
        my_blank = float((held["fixtures"] <= 0).sum())
        mine = vis[vis[C.PLAYER_KEY].isin(squad)]

        cands = []
        for chip in avail:
            win = next((w for w in CHIP_WINDOWS[chip] if w[0] <= gw <= w[1]), None)
            if win is None:
                continue
            if chip == "3xc":
                b = float(xi[signal].max()) * (DOUBLE_BONUS if my_dbl else 1.0)
            elif chip == "bboost":
                bench = held[~held[C.PLAYER_KEY].isin(xi[C.PLAYER_KEY])]
                b = float(bench[signal].sum()) * (DOUBLE_BONUS if my_dbl else 1.0)
            elif chip == "freehit":
                pool = week[week["fixtures"] > 0]
                bx = optimise.best_xi(pool, signal)
                b = float(bx[signal].sum() - xi[signal].sum()) if bx is not None else 0.0
                b += BLANK_WEIGHT * my_blank
            else:                                    # wildcard
                allp = vis[vis["fixtures"] > 0].groupby(C.PLAYER_KEY)[signal].mean()
                mn = mine[mine["fixtures"] > 0].groupby(C.PLAYER_KEY)[signal].mean()
                b = float(allp.nlargest(C.SQUAD_SIZE).sum()
                          - mn.nlargest(C.SQUAD_SIZE).sum())
            cands.append((chip, win, b))
        if not cands:
            return None

        # one chip per gameweek, so this is scheduling: if the gameweeks left in
        # a window no longer outnumber the chips needing them, something must
        # fire now or be lost
        hi = cands[0][1][1]
        left = len([g for g in season if gw <= g <= hi])
        forced = left <= len([c for c in cands if c[1][1] == hi])

        best = None
        for chip, win, b in cands:
            span = [g for g in season if win[0] <= g <= win[1]]
            if not span:
                continue
            key = (chip, win)
            cut = span[int(len(span) * LOOK)] if len(span) > 1 else span[0]
            if gw <= cut and not forced:
                seen[key] = max(seen.get(key, 0.0), b)
                continue
            if (b > seen.get(key, 0.0) or gw >= span[-1] or forced) \
               and (best is None or b > best[1]):
                best = (chip, b)
        return best[0] if best else None

    return policy
