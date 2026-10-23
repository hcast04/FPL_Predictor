"""
The receding-horizon planner (O38).

The test that matters is `test_honest_mode_cannot_see_future_forecasts`: M17
was a policy reading a calendar it could not have, and this module invites the
same mistake at every gameweek. It carries its own negative control -- the
oracle mode, which SHOULD react to the future and would make the test vacuous
if it did not.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpl2 import config as C      # noqa: E402
from fpl2 import mpc              # noqa: E402


def panel(seed: int = 0, n: int = 90, gws: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pos = rng.choice(list(C.SQUAD_QUOTA), n, p=[0.16, 0.30, 0.34, 0.20])
    base = pd.DataFrame({
        "player_code": np.arange(n),
        "position": pos,
        "team": rng.choice([f"T{i}" for i in range(20)], n),
        "cost": rng.integers(39, 120, n) / 10.0})
    rows = []
    for gw in range(1, gws + 1):
        d = base.copy()
        d["gameweek"] = gw
        d["fixtures"] = 1
        d["xp"] = rng.gamma(2.0, 1.2, n)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def legal_squad(p: pd.DataFrame, squad, bank):
    cost = p.groupby("player_code")["cost"].first()
    pos = p.groupby("player_code")["position"].first()
    club = p.groupby("player_code")["team"].first()
    assert len(squad) == C.SQUAD_SIZE
    assert pos.reindex(squad).value_counts().to_dict() == C.SQUAD_QUOTA
    assert club.reindex(squad).value_counts().max() <= C.MAX_PER_CLUB
    assert cost.reindex(squad).sum() <= bank + 1e-6


def seed_squad(p: pd.DataFrame):
    """A legal starting 15 built cheaply, so the planner has room to move."""
    w = p[p.gameweek.eq(1)]
    sq = []
    for pos, k in C.SQUAD_QUOTA.items():
        sq += w[w.position.eq(pos)].nsmallest(k, "cost")["player_code"].tolist()
    return sq


def test_plan_returns_a_legal_squad():
    p = panel(1)
    sq = seed_squad(p)
    cost = p.groupby("player_code")["cost"].first()
    bank = C.BUDGET - float(cost.reindex(sq).sum())
    plan = mpc.plan(mpc.honest_frames(p, 1, 4), sq, bank, free_transfers=1)
    assert plan is not None
    new = [c for c in sq if c not in set(plan["sell"])] + list(plan["buy"])
    legal_squad(p, new, C.BUDGET)


def test_buys_and_sells_balance():
    p = panel(2)
    sq = seed_squad(p)
    cost = p.groupby("player_code")["cost"].first()
    bank = C.BUDGET - float(cost.reindex(sq).sum())
    plan = mpc.plan(mpc.honest_frames(p, 1, 3), sq, bank, free_transfers=1)
    assert len(plan["buy"]) == len(plan["sell"])
    assert not (set(plan["buy"]) & set(sq)), "bought a player already owned"
    assert set(plan["sell"]) <= set(sq), "sold a player not owned"


def test_honest_mode_cannot_see_future_forecasts():
    """
    THE leak test (M17). Scrambling gameweeks 2+ must not change the plan the
    honest planner commits for gameweek 1.

    NEGATIVE CONTROL is built in: the oracle mode is run on the same mutation
    and MUST change its plan. Without that, a planner that simply ignored its
    inputs would pass this test.
    """
    p = panel(3)
    sq = seed_squad(p)
    cost = p.groupby("player_code")["cost"].first()
    bank = C.BUDGET - float(cost.reindex(sq).sum())

    mutated = p.copy()
    fut = mutated["gameweek"] > 1
    rng = np.random.default_rng(99)
    mutated.loc[fut, "xp"] = rng.permutation(mutated.loc[fut, "xp"].to_numpy()) * 5.0

    def committed(frame, mode):
        pl = mpc.plan(mpc.honest_frames(frame, 1, 4, mode), sq, bank, free_transfers=1)
        return (sorted(pl["buy"]), sorted(pl["sell"]))

    assert committed(p, "honest") == committed(mutated, "honest"), \
        "honest planner reacted to a future gameweek it cannot see (M17)"
    assert committed(p, "oracle") != committed(mutated, "oracle"), \
        "oracle did not react either -- the test is vacuous, not passing"


def test_horizon_is_capped_at_the_knowable_fixture_window():
    """M17: a horizon past the published fixture list is not permitted."""
    assert mpc.HORIZON_MAX == 4
    p = panel(4, gws=12)
    sq = seed_squad(p)
    cost = p.groupby("player_code")["cost"].first()
    bank = C.BUDGET - float(cost.reindex(sq).sum())
    frames = mpc.honest_frames(p, 1, 10)          # ask for more than allowed
    plan = mpc.plan(frames, sq, bank, free_transfers=1)
    assert plan is not None                        # silently truncated, not failed


def test_hit_cost_deters_transfers():
    """NEGATIVE CONTROL: with hits free the planner must trade more."""
    p = panel(5)
    sq = seed_squad(p)
    cost = p.groupby("player_code")["cost"].first()
    bank = C.BUDGET - float(cost.reindex(sq).sum())
    orig = mpc.HIT_COST
    try:
        mpc.HIT_COST = 0.0
        free = len(mpc.plan(mpc.honest_frames(p, 1, 2), sq, bank, 1)["buy"])
        mpc.HIT_COST = 100.0
        dear = len(mpc.plan(mpc.honest_frames(p, 1, 2), sq, bank, 1)["buy"])
    finally:
        mpc.HIT_COST = orig
    assert free > dear, f"hit cost did not deter: {free} vs {dear}"


def test_hits_are_counted_from_transfers_not_from_solver_slack():
    """
    `hits` must equal max(0, transfers - free). Reading the slack variable gave
    15 hits against 2 transfers whenever HIT_COST was 0, because the constraint
    only bounds it from below.
    """
    p = panel(6)
    sq = seed_squad(p)
    cost = p.groupby("player_code")["cost"].first()
    bank = C.BUDGET - float(cost.reindex(sq).sum())
    orig = mpc.HIT_COST
    try:
        mpc.HIT_COST = 0.0
        pl = mpc.plan(mpc.honest_frames(p, 1, 2), sq, bank, free_transfers=1)
    finally:
        mpc.HIT_COST = orig
    assert pl["hits"] == max(0, len(pl["buy"]) - 1)
    assert pl["hits"] <= C.SQUAD_SIZE


def test_more_free_transfers_never_hurts():
    p = panel(7)
    sq = seed_squad(p)
    cost = p.groupby("player_code")["cost"].first()
    bank = C.BUDGET - float(cost.reindex(sq).sum())
    a = mpc.plan(mpc.honest_frames(p, 1, 3), sq, bank, free_transfers=1)
    b = mpc.plan(mpc.honest_frames(p, 1, 3), sq, bank, free_transfers=2)
    assert b["objective"] >= a["objective"] - 1e-6
