"""
A measuring instrument with a known resolution.

WHY THIS EXISTS. Every gate on this project has been decided by one number:
the season total through the budget-constrained simulator, over three holdout
seasons. 2026-10-22 measured what that number can actually resolve:

    per-season swing from a data-ENCODING choice alone   325 pts (S36)
    per-season swing from `bench_weight` alone           338 pts (S27)
    single-season noise floor                            112-125 (R1)
    3-season-mean sd                                      39-49 (R1)

So it resolves roughly +/-120 points on a 3-season mean, with n = 3. Almost
nothing real is that big. Seven pre-registered hypotheses were refuted in one
day and four of them had a CONFIRMED mechanism and an unmeasurable effect
(R31). The instrument could not see what was being tested.

The variance has three sources, and only one of them is interesting:
  1. the thing under test                      <- wanted
  2. the opening squad each arm happens to get <- noise
  3. the 38-week transfer path forking from it <- noise, and it compounds

`fixed_squad_score` removes (2) and (3) by construction: every arm is handed
the SAME squads and makes only the XI and captain decisions, which are real
decisions but cannot fork. `restart_score` keeps transfers but averages over K
opening squads so the fork averages down instead of dominating.

NOTHING HERE IS TRUSTED UNTIL CALIBRATED. `fpl2/run_calibrate.py` runs a NULL
test (an arm against itself -- must report no difference) and a SIGNAL test (an
arm against a degraded copy -- must detect it). An instrument that has not
failed a null test is not an instrument.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import optimise


def neutral_squads(grid: pd.DataFrame, k: int = 12, seed: int = 0,
                   budget: float = C.BUDGET) -> list:
    """
    K legal squads, built WITHOUT reference to any arm under test.

    Selection uses prior-form points-per-appearance with a random perturbation,
    so the squads are realistic (not random junk, which would compress every
    arm toward the same low score) and diverse (not K copies of one optimum),
    while being identical for every arm compared. If they were built from one
    arm's forecast, that arm would be measured on home ground.
    """
    rng = np.random.default_rng(seed)
    first = grid[grid["gameweek"].eq(grid["gameweek"].min())]
    pool = (grid[grid["fixtures"] > 0]
            .groupby(C.PLAYER_KEY)
            .agg(base=("points", "mean"), cost=("cost", "first"),
                 position=("position", "first"), team=("team", "first"))
            .reset_index())
    pool = pool[pool["cost"].gt(0) & pool[C.PLAYER_KEY].isin(first[C.PLAYER_KEY])]
    out = []
    for i in range(k):
        p = pool.copy()
        # multiplicative jitter: keeps the ordering broadly sane, moves the
        # margins. 0.35 was chosen to give squads that overlap ~7-10 of 15,
        # which is the realistic spread between two managers.
        p["sig"] = p["base"] * (1.0 + 0.35 * rng.standard_normal(len(p)))
        sq = optimise.optimise_squad(p, "sig", budget=budget)
        if sq is not None and len(sq) == C.SQUAD_SIZE:
            out.append(sq[C.PLAYER_KEY].tolist())
    return out


def fixed_squad_score(grid: pd.DataFrame, squads: list, signal: str,
                      actual: str = "points") -> pd.DataFrame:
    """
    Per (squad, gameweek) realised points of the XI and captain this signal
    picks. Path-free: the squad never changes, so nothing forks.

    Returns one row per (squad_id, gameweek) so a PAIRED comparison is possible
    -- the same squad in the same gameweek under two signals is the matched
    unit, and almost all the between-squad and between-week variance cancels.
    """
    rows = []
    for si, squad in enumerate(squads):
        held_all = grid[grid[C.PLAYER_KEY].isin(squad)]
        for gw, week in held_all.groupby("gameweek", sort=True):
            v = optimise.xi_value(week, signal, actual=actual)
            if np.isfinite(v):
                rows.append({"squad": si, "gameweek": int(gw), "value": float(v)})
    return pd.DataFrame(rows)


def paired(a: pd.DataFrame, b: pd.DataFrame,
           on=("squad", "gameweek")) -> dict:
    """
    Paired difference between two arms, with the resolution stated.

    `mde` is the smallest true effect this comparison would detect 80% of the
    time at alpha=0.05, in the units of the metric. Reporting it is the whole
    point: a gate set below the MDE cannot be passed except by luck, and a gate
    set far above it wastes real effects.
    """
    m = a.merge(b, on=list(on), suffixes=("_a", "_b"))
    d = (m["value_a"] - m["value_b"]).to_numpy(float)
    n = len(d)
    if n < 2:
        return {"n": n, "mean": float("nan")}
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    t = float(d.mean() / se) if se > 0 else float("nan")
    return {"n": n, "mean": float(d.mean()), "sd": sd, "se": se, "t": t,
            "ci_lo": float(d.mean() - 1.96 * se),
            "ci_hi": float(d.mean() + 1.96 * se),
            # 2.8 = z(0.975) + z(0.80), the standard two-sided 80%-power factor
            "mde": float(2.8 * se)}


def per_season_equivalent(res: dict, gameweeks: int = 38) -> dict:
    """Scale a per-(squad, gameweek) difference to a season total."""
    return {k: (v * gameweeks if k in ("mean", "ci_lo", "ci_hi", "mde", "se")
                else v) for k, v in res.items()}


def restart_score(grid: pd.DataFrame, squads: list, signal: str,
                  simulate_fn, **kw) -> pd.DataFrame:
    """
    Season totals WITH transfers, one per opening squad.

    Keeps the transfer policy in the measurement -- it is part of the real
    game -- but averages the fork over K starts instead of taking one draw.
    """
    rows = []
    for si, squad in enumerate(squads):
        r = simulate_fn(grid, signal, opening_squad=list(squad), **kw)
        if r is not None:
            rows.append({"squad": si, "value": float(r.total)})
    return pd.DataFrame(rows)
