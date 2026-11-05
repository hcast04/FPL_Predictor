"""
The chip policy is STATEFUL, and the live path used to lose that state.

`rolling_policy` closes over `seen`: it observes the first LOOK of each
window, records the best benefit, then fires on the first week beating it.
`simulate_season` calls it once per gameweek so `seen` fills up. The live
runner called it ONCE, so `seen` was empty and past the cut the firing test
`b > seen.get(key, 0.0)` collapsed to `b > 0`.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpl2 import config as C      # noqa: E402
from fpl2 import chips           # noqa: E402

QUOTA = [("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]


def _grid(bench_xp_by_gw: dict) -> tuple[pd.DataFrame, list]:
    """
    A 15-man squad over 12 gameweeks.

    The eleven starters are pinned at a high, constant xp; the four reserves
    carry `bench_xp_by_gw[gw]` each, so the bench-boost benefit for a gameweek
    is exactly 4x that number and the test can set the bar directly.
    """
    players, squad = [], []
    i = 0
    for pos, n in QUOTA:
        for k in range(n):
            i += 1
            players.append({"code": i, "position": pos, "starter": k < (n - 1)})
            squad.append(i)
    # keep exactly 11 starters: GK1, DEF4, MID4, FWD2 -> 11
    rows = []
    for gw, bench_xp in bench_xp_by_gw.items():
        for p in players:
            rows.append({"season": "T", "gameweek": gw, C.PLAYER_KEY: p["code"],
                         "position": p["position"], "team": f"T{p['code']}",
                         "cost": 5.0, "fixtures": 1, "points": 0.0,
                         "xp": 9.0 if p["starter"] else bench_xp})
    return pd.DataFrame(rows), squad


def _ask(warm: bool, bench_by_gw: dict, decide: int):
    grid, squad = _grid(bench_by_gw)
    policy = chips.rolling_policy(grid, "xp")
    if warm:
        chips.warm_up(policy, grid, [g for g in bench_by_gw if g < decide],
                      lambda gw: squad)
    week = grid[grid["gameweek"].eq(decide)]
    return policy(decide, squad, week, ["bboost"])


def test_cold_policy_fires_on_any_positive_benefit():
    """
    NEGATIVE CONTROL for the fix.

    Without warm-up the bar is 0, so a bench far WORSE than one already seen
    still fires. If this ever stops firing, `warm_up` is no longer what makes
    the difference and the test below proves nothing.
    """
    bench = {gw: (6.0 if gw <= 8 else 1.0) for gw in range(1, 13)}
    assert _ask(warm=False, bench_by_gw=bench, decide=10) == "bboost"


def test_warmed_policy_holds_a_chip_worse_than_what_it_saw():
    """The same question, with the observation phase replayed: it must hold."""
    bench = {gw: (6.0 if gw <= 8 else 1.0) for gw in range(1, 13)}
    assert _ask(warm=True, bench_by_gw=bench, decide=10) is None


def test_warmed_policy_still_fires_when_the_week_beats_the_bar():
    """Warm-up must not simply suppress everything -- a real peak must fire."""
    bench = {gw: (1.0 if gw <= 8 else 9.0) for gw in range(1, 13)}
    assert _ask(warm=True, bench_by_gw=bench, decide=10) == "bboost"


def test_warm_up_tolerates_missing_squads():
    """A gameweek whose picks cannot be fetched must not abort the replay."""
    bench = {gw: (1.0 if gw <= 8 else 9.0) for gw in range(1, 13)}
    grid, squad = _grid(bench)
    policy = chips.rolling_policy(grid, "xp")
    chips.warm_up(policy, grid, range(1, 10),
                  lambda gw: [] if gw % 2 else squad)   # half return nothing
    week = grid[grid["gameweek"].eq(10)]
    assert policy(10, squad, week, ["bboost"]) == "bboost"
