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
from fpl2 import optimise        # noqa: E402

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


# --- S65: a free hit buys a LEGAL squad, not the best eleven in the game ---

def _market(n_cheap=40, n_premium=11):
    """A league where the best eleven is unaffordable and all one club."""
    rows = []
    for i in range(n_premium):                     # brilliant, dear, one club
        rows.append({C.PLAYER_KEY: 1000 + i, "xp": 10.0, "cost": 15.0,
                     "team": "RICH", "fixtures": 1,
                     "position": ["GK", "DEF", "DEF", "DEF", "DEF", "MID",
                                  "MID", "MID", "MID", "FWD", "FWD"][i]})
    for i in range(n_cheap):                       # ordinary, cheap, spread
        rows.append({C.PLAYER_KEY: 2000 + i, "xp": 3.0, "cost": 4.0,
                     "team": f"T{i % 12}", "fixtures": 1,
                     "position": ["GK", "DEF", "MID", "FWD"][i % 4]})
    return pd.DataFrame(rows)


def test_freehit_respects_budget_and_club_cap():
    """
    The eleven premiums cost GBP165m and are all one club. A free hit cannot
    buy them, so the benefit must come out far below what an unconstrained
    `best_xi` would claim.
    """
    week = _market()
    held = week[week[C.PLAYER_KEY] >= 2000].head(15).copy()
    xi = optimise.best_xi(held.reset_index(drop=True), "xp")
    naive = optimise.best_xi(week, "xp")
    naive_b = float(naive["xp"].sum() - xi["xp"].sum())
    real_b = chips._freehit_benefit(week, held, xi, "xp", 0.0)
    assert real_b < naive_b, (
        f"constrained benefit {real_b:.2f} should be below the unconstrained "
        f"{naive_b:.2f}")
    assert naive["cost"].sum() > C.BUDGET, "fixture is wrong: XI is affordable"
    assert naive["team"].value_counts().max() > C.MAX_PER_CLUB, (
        "fixture is wrong: the unconstrained XI already respects the club cap")


def test_freehit_benefit_counts_blanks_once_per_weight():
    """`blank_weight` is a per-blank addition and must scale exactly."""
    week = _market()
    held = week[week[C.PLAYER_KEY] >= 2000].head(15).copy()
    held.loc[held.index[:3], "fixtures"] = 0
    xi = optimise.best_xi(held.reset_index(drop=True), "xp")
    b0 = chips._freehit_benefit(week, held, xi, "xp", 0.0)
    b5 = chips._freehit_benefit(week, held, xi, "xp", 5.0)
    assert abs((b5 - b0) - 15.0) < 1e-9, (
        f"three blanks at weight 5 should add 15.0, added {b5 - b0:.3f}")


def test_rolling_policy_accepts_a_blank_weight_override():
    """The sweep in run_freehit.py depends on this being wired through."""
    bench = {gw: (1.0 if gw <= 8 else 9.0) for gw in range(1, 13)}
    grid, squad = _grid(bench)
    for w in (0.0, 8.0):
        pol = chips.rolling_policy(grid, "xp", blank_weight=w)
        assert pol(10, squad, grid[grid["gameweek"].eq(10)], ["bboost"]) == "bboost"


# --- S66: the blank penalty is a HINGE, default threshold 0 = the old form --

def _held_with_blanks(n_blank):
    week = _market()
    held = week[week[C.PLAYER_KEY] >= 2000].head(15).copy()
    if n_blank:
        held.loc[held.index[:n_blank], "fixtures"] = 0
    return week, held


def test_threshold_zero_is_exactly_the_old_linear_penalty():
    """
    NEGATIVE CONTROL for the whole hinge change. The shipped default must be
    byte-identical to the flat form S50 measured at +119.8 pts/season; if it
    is not, every prior chip result silently changes meaning.
    """
    for n in (0, 1, 3, 6):
        week, held = _held_with_blanks(n)
        xi = optimise.best_xi(held.reset_index(drop=True), "xp")
        hinge = chips._freehit_benefit(week, held, xi, "xp", 2.0, 0)
        flat = chips._freehit_benefit(week, held, xi, "xp", 0.0, 0) + 2.0 * n
        assert abs(hinge - flat) < 1e-9, (
            f"{n} blanks: hinge {hinge:.4f} != flat {flat:.4f}")


def test_hinge_pays_nothing_below_its_threshold():
    """`w * max(0, blanks - thr)` -- two blanks under a threshold of 2 is 0."""
    week, held = _held_with_blanks(2)
    xi = optimise.best_xi(held.reset_index(drop=True), "xp")
    free = chips._freehit_benefit(week, held, xi, "xp", 0.0, 0)
    assert abs(chips._freehit_benefit(week, held, xi, "xp", 8.0, 2) - free) < 1e-9
    week, held = _held_with_blanks(4)          # two ABOVE the threshold
    xi = optimise.best_xi(held.reset_index(drop=True), "xp")
    free = chips._freehit_benefit(week, held, xi, "xp", 0.0, 0)
    got = chips._freehit_benefit(week, held, xi, "xp", 8.0, 2)
    assert abs((got - free) - 16.0) < 1e-9, f"expected +16.0, got {got - free:.3f}"


def test_rolling_policy_threads_the_threshold():
    """run_freehit_policy.py's arms depend on this reaching the benefit."""
    bench = {gw: (1.0 if gw <= 8 else 9.0) for gw in range(1, 13)}
    grid, squad = _grid(bench)
    for thr in (0, 3):
        pol = chips.rolling_policy(grid, "xp", blank_weight=8.0,
                                   blank_threshold=thr)
        assert pol(10, squad, grid[grid["gameweek"].eq(10)], ["bboost"]) == "bboost"
