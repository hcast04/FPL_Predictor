"""The optimiser must be exactly right. These check it against brute force."""
from __future__ import annotations
import itertools, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpl2 import config as C          # noqa: E402
from fpl2 import optimise             # noqa: E402

RNG = np.random.default_rng(0)


def make_squad(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pos, k in C.SQUAD_QUOTA.items():
        for i in range(k):
            rows.append({"player_code": f"{pos}{i}", "position": pos,
                         "team": f"T{rng.integers(0, 6)}",
                         "cost": float(rng.integers(40, 130)) / 10,
                         "xp": float(rng.normal(3, 1.2)),
                         "actual": float(rng.integers(0, 16))})
    return pd.DataFrame(rows)


def brute_force_best(sq: pd.DataFrame, score: str) -> float:
    best = -np.inf
    for combo in itertools.combinations(range(len(sq)), C.XI_SIZE):
        xi = sq.iloc[list(combo)]
        c = xi["position"].value_counts()
        if c.get("GK", 0) != C.XI_BOUNDS["GK"][0]:
            continue
        ok = all(C.XI_BOUNDS[p][0] <= c.get(p, 0) <= C.XI_BOUNDS[p][1]
                 for p in ("DEF", "MID", "FWD"))
        if not ok:
            continue
        best = max(best, float(xi[score].sum()))
    return best


@pytest.mark.parametrize("seed", range(25))
def test_best_xi_matches_brute_force(seed):
    sq = make_squad(seed)
    xi = optimise.best_xi(sq, "xp")
    assert xi is not None and len(xi) == C.XI_SIZE
    assert float(xi["xp"].sum()) == pytest.approx(brute_force_best(sq, "xp"))


@pytest.mark.parametrize("seed", range(10))
def test_best_xi_is_legal(seed):
    xi = optimise.best_xi(make_squad(seed), "xp")
    c = xi["position"].value_counts()
    assert c.get("GK", 0) == 1
    for p in ("DEF", "MID", "FWD"):
        lo, hi = C.XI_BOUNDS[p]
        assert lo <= c.get(p, 0) <= hi


def test_best_xi_none_without_goalkeeper():
    sq = make_squad(1)
    assert optimise.best_xi(sq[sq["position"] != "GK"], "xp") is None


def test_xi_value_doubles_the_captain():
    sq = make_squad(3)
    xi = optimise.best_xi(sq, "xp")
    cap = optimise.pick_captain(xi, "xp")
    expected = float(xi["actual"].sum()) + float(cap["actual"])
    assert optimise.xi_value(sq, "xp", actual="actual") == pytest.approx(expected)


def test_xi_value_selects_on_score_not_on_actual():
    """NEGATIVE CONTROL: if selection leaked the actuals, this would be higher."""
    sq = make_squad(7)
    leaky = optimise.xi_value(sq, "actual", actual="actual")
    honest = optimise.xi_value(sq, "xp", actual="actual")
    assert honest <= leaky
    assert honest != leaky, "xp and actual too correlated to detect leakage"


def test_optimise_squad_respects_every_constraint():
    rng = np.random.default_rng(5)
    pool = pd.DataFrame({
        "player_code": [f"p{i}" for i in range(260)],
        "position": rng.choice(list(C.SQUAD_QUOTA), 260,
                               p=[0.12, 0.33, 0.37, 0.18]),
        "team": rng.choice([f"T{i}" for i in range(20)], 260),
        "cost": rng.integers(39, 140, 260) / 10,
        "xp": rng.normal(2.5, 1.0, 260)})
    sq = optimise.optimise_squad(pool, "xp", budget=C.BUDGET)
    assert sq is not None and len(sq) == C.SQUAD_SIZE
    assert sq["position"].value_counts().to_dict() == C.SQUAD_QUOTA
    assert sq["cost"].sum() <= C.BUDGET + 1e-6
    assert sq["team"].value_counts().max() <= C.MAX_PER_CLUB


def test_optimise_squad_beats_a_greedy_pick():
    """NEGATIVE CONTROL: the MILP must beat naive greedy-by-xp."""
    rng = np.random.default_rng(11)
    pool = pd.DataFrame({
        "player_code": [f"p{i}" for i in range(200)],
        "position": rng.choice(list(C.SQUAD_QUOTA), 200, p=[0.12, 0.33, 0.37, 0.18]),
        "team": rng.choice([f"T{i}" for i in range(20)], 200),
        "cost": rng.integers(39, 140, 200) / 10,
        "xp": rng.normal(2.5, 1.0, 200)})
    sq = optimise.optimise_squad(pool, "xp", budget=C.BUDGET)
    greedy = pd.concat([pool[pool.position.eq(p)].nlargest(k, "xp")
                        for p, k in C.SQUAD_QUOTA.items()])
    assert sq["xp"].sum() >= greedy["xp"].sum() or greedy["cost"].sum() > C.BUDGET


# --- bench_weight / XI-aware optimisation (O37) -----------------------------

def _pool(seed=21, n=240):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "player_code": [f"p{i}" for i in range(n)],
        "position": rng.choice(list(C.SQUAD_QUOTA), n, p=[0.12, 0.33, 0.37, 0.18]),
        "team": rng.choice([f"T{i}" for i in range(20)], n),
        "cost": rng.integers(39, 140, n) / 10,
        "xp": rng.normal(2.5, 1.0, n)})


def test_bench_weight_one_is_identical_to_the_old_objective():
    """
    The default MUST reproduce the pre-O37 behaviour exactly, or every result
    recorded in the ledger before 2026-10-22 silently changes meaning.
    """
    pool = _pool()
    a = optimise.optimise_squad(pool, "xp")                      # no kwarg
    b = optimise.optimise_squad(pool, "xp", bench_weight=1.0)    # explicit
    assert set(a["player_code"]) == set(b["player_code"])
    assert float(a["xp"].sum()) == pytest.approx(float(b["xp"].sum()), abs=1e-12)


@pytest.mark.parametrize("w", [0.0, 0.1, 0.25, 0.5])
def test_bench_weight_squads_are_still_legal(w):
    sq = optimise.optimise_squad(_pool(), "xp", budget=C.BUDGET, bench_weight=w)
    assert sq is not None and len(sq) == C.SQUAD_SIZE
    assert sq["position"].value_counts().to_dict() == C.SQUAD_QUOTA
    assert sq["cost"].sum() <= C.BUDGET + 1e-6
    assert sq["team"].value_counts().max() <= C.MAX_PER_CLUB


@pytest.mark.parametrize("seed", [21, 22, 23, 24])
def test_lower_bench_weight_buys_a_better_starting_eleven(seed):
    """
    THE claim, stated so it can fail. A lower bench weight must produce a squad
    whose best XI is at least as strong, since that is the only thing the
    objective now pays for.

    NEGATIVE CONTROL is `test_bench_weight_one_...` above: at w=1.0 the two
    sides are the same program, so this assertion is vacuous there -- which is
    exactly why the strict-improvement check below is separate and is allowed
    to be weak on a single seed.
    """
    pool = _pool(seed)
    hi = optimise.optimise_squad(pool, "xp", bench_weight=1.0)
    lo = optimise.optimise_squad(pool, "xp", bench_weight=0.0)
    xi_hi = optimise.best_xi(hi, "xp")["xp"].sum()
    xi_lo = optimise.best_xi(lo, "xp")["xp"].sum()
    assert xi_lo >= xi_hi - 1e-9, f"w=0 XI {xi_lo} worse than w=1 XI {xi_hi}"


def test_lower_bench_weight_strictly_improves_the_xi_somewhere():
    """
    Across seeds the fix must do SOMETHING, or the code is a no-op dressed up
    as a change. Run against the unfixed objective (both sides w=1.0) this
    fails, which is what makes it a real test.
    """
    gains = []
    for seed in range(21, 31):
        pool = _pool(seed)
        hi = optimise.optimise_squad(pool, "xp", bench_weight=1.0)
        lo = optimise.optimise_squad(pool, "xp", bench_weight=0.0)
        gains.append(optimise.best_xi(lo, "xp")["xp"].sum()
                     - optimise.best_xi(hi, "xp")["xp"].sum())
    assert max(gains) > 1e-6, f"bench_weight changed no XI at all: {gains}"


def test_bench_weight_trades_squad_total_away_for_xi_total():
    """The mechanism, not just the outcome: w=0 must give up total squad xp."""
    pool = _pool(21)
    hi = optimise.optimise_squad(pool, "xp", bench_weight=1.0)
    lo = optimise.optimise_squad(pool, "xp", bench_weight=0.0)
    assert float(lo["xp"].sum()) <= float(hi["xp"].sum()) + 1e-9
