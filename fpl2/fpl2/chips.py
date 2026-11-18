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

# S65. `BLANK_WEIGHT` is a hand-chosen constant and it is PARTLY double
# counting: a blanked player already carries xp = 0 in the grid, so the held
# XI has already lost his points before this is added. It is not pure double
# counting, because the free-hit side of the comparison also shrinks in a
# blank week -- fewer players have fixtures -- so the raw benefit does not
# track blanks cleanly (85.8 / 81.9 / 77.9 / 80.8 / 92.1 across 0 / 1-2 / 3 /
# 4-5 / 6+ blanks on four seasons of realised points).
#
# What IS measured, within-squad over ten seasons and resolvable at G = 10,
# is the loss a blank week inflicts relative to that squad's own clean weeks:
#
#     your blanks   0-2     3      4-5     6+
#     loss          3-6    15.5   26.3   37.7     (MDE 4.5 / 7.0 / 12.0)
#
# So the penalty is CONVEX -- about 3 points per blank below three, about ten
# each above it -- and a single linear coefficient cannot express that. The
# value is left at 2.0 because the policy's +119.8 pts/season (S50) was
# measured with it; `fpl2/run_freehit.py` sweeps it honestly.
BLANK_WEIGHT = 2.0  # points per squad player with no fixture, for the free hit

# S65 measured the loss as CONVEX -- about 3 points per blank below three,
# about ten each above it -- which a single linear coefficient cannot express.
# `BLANK_THRESHOLD` makes the penalty a HINGE:
#
#     penalty = BLANK_WEIGHT * max(0, blanks - BLANK_THRESHOLD)
#
# Threshold 0 is exactly the old linear form, so the default reproduces every
# prior result unchanged and S50's +119.8 keeps its meaning. `run_freehit.py`
# sweeps both parameters.
BLANK_THRESHOLD = 0

# A real free hit is a LEGAL squad: the budget you actually have and at most
# `MAX_PER_CLUB`. `optimise.best_xi` enforces neither, so the free hit was
# priced as the best eleven IN THE GAME and came out systematically
# optimistic. Measured at 14 ms per MILP call, which is 1.1 minutes over a
# full eight-season backtest -- affordable.
FREEHIT_BUDGET = None   # None -> the squad's own value; a float overrides


def warm_up(policy, grid: pd.DataFrame, gameweeks, squad_at,
            played: dict | None = None, signal: str = "xp") -> None:
    """
    Replay earlier gameweeks so the secretary rule has its memory.

    `rolling_policy` returns a STATEFUL callable: it closes over `seen`, and
    the secretary rule works by OBSERVING the first LOOK of each window,
    recording the best benefit, then firing on the first week that beats it.
    `simulate.simulate_season` calls the policy once per gameweek in order, so
    `seen` fills up -- that is the version measured at +119.8 pts/season.

    `run_live_chip.py` called it ONCE, for the week being decided. `seen` was
    therefore empty and, past the observation cut, the firing test
    `b > seen.get(key, 0.0)` collapsed to `b > 0`: **any chip with a positive
    benefit fired immediately.** The live path was not running the policy that
    was measured. At GW10 of 2026-27 it said `bboost` on a bench worth 11.26
    against a bar of 11.94 set in GW5.

    `squad_at(gw)` must return the squad held AT THAT DEADLINE -- which is the
    PREVIOUS gameweek's picks, because FPL's `event/{gw}/picks/` reports the
    squad AFTER that week's transfers. Measuring a wildcard week against its
    own post-wildcard squad understates the benefit enormously (GW9 2026-27:
    8.89 against the new squad, 19.96 against the one actually held).

    Decisions are discarded; only `seen` matters.
    """
    played = played or {}
    for gw in sorted(gameweeks):
        squad = squad_at(gw)
        if not squad:
            continue
        week = grid[grid["gameweek"].eq(gw)]
        if week.empty:
            continue
        avail = [c for c in CHIP_WINDOWS
                 if c not in {k for k, v in played.items() if v < gw}]
        try:
            policy(int(gw), list(squad), week, avail)
        except Exception:                                      # noqa: BLE001
            continue


def _freehit_benefit(week, held, xi, signal, blank_weight,
                     blank_threshold: int = None) -> float:
    """
    What one week of ANY LEGAL squad is worth over the one you hold.

    The squad a free hit buys is bounded by your budget and by
    `MAX_PER_CLUB`; the previous version took `best_xi` over every player
    with a fixture, which respects neither. On realised points the gap is
    often zero -- the best eleven is frequently cheap -- but on model xp it
    is not, because xp and price are correlated and the unconstrained eleven
    fills up with premiums you could not all afford.
    """
    pool = week[week["fixtures"] > 0]
    if pool.empty:
        return 0.0
    mine = float(xi[signal].sum())
    budget = FREEHIT_BUDGET
    if budget is None:
        cost = pd.to_numeric(held.get("cost"), errors="coerce")
        budget = float(cost.sum()) if cost is not None and cost.notna().any() \
            else float(C.BUDGET)
    agg = pool.groupby(C.PLAYER_KEY, as_index=False).agg(
        **{signal: (signal, "max"), "cost": ("cost", "first"),
           "position": ("position", "first"), "team": ("team", "first")})
    agg = agg[pd.to_numeric(agg["cost"], errors="coerce").gt(0)]
    best = None
    if len(agg) >= C.SQUAD_SIZE:
        sq = optimise.optimise_squad(agg, signal, budget=budget)
        if sq is not None:
            best = optimise.best_xi(
                sq.merge(agg[[C.PLAYER_KEY, signal]], on=C.PLAYER_KEY,
                         suffixes=("", "_a")), signal)
    if best is None:                      # no legal squad: fall back, flagged
        best = optimise.best_xi(pool, signal)
    if best is None:
        return 0.0
    b = float(best[signal].sum()) - mine
    thr = BLANK_THRESHOLD if blank_threshold is None else int(blank_threshold)
    blanks = float((held["fixtures"] <= 0).sum())
    return b + blank_weight * max(0.0, blanks - thr)


def rolling_policy(grid: pd.DataFrame, signal: str = "xp",
                   blank_weight: float = None, blank_threshold: int = None):
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
    blank_weight = BLANK_WEIGHT if blank_weight is None else float(blank_weight)
    blank_threshold = (BLANK_THRESHOLD if blank_threshold is None
                       else int(blank_threshold))
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
                b = _freehit_benefit(week, held, xi, signal, blank_weight,
                                     blank_threshold)
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
