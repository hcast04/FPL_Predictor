"""
Squad and line-up optimisation under the real FPL constraints.

`best_xi` is EXACT, not a heuristic: for a fixed formation the best XI is the
top-k by score in each position, so iterating every legal formation covers the
whole legal space. `tests/test_optimise.py` checks this against brute force
over all C(15,11) subsets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from . import config as C


def legal_formations() -> list[tuple[int, int, int]]:
    out = []
    for d in range(C.XI_BOUNDS["DEF"][0], C.XI_BOUNDS["DEF"][1] + 1):
        for m in range(C.XI_BOUNDS["MID"][0], C.XI_BOUNDS["MID"][1] + 1):
            f = C.XI_SIZE - 1 - d - m
            if C.XI_BOUNDS["FWD"][0] <= f <= C.XI_BOUNDS["FWD"][1]:
                out.append((d, m, f))
    return out


def best_xi(rows: pd.DataFrame, score: str) -> pd.DataFrame | None:
    """Highest-scoring legal XI from `rows`. None if no legal XI exists."""
    pools = {p: rows[rows["position"].eq(p)].sort_values(score, ascending=False)
             for p in C.POSITIONS}
    if pools["GK"].empty:
        return None
    gk = pools["GK"].head(1)
    best, best_score = None, -np.inf
    for d, m, f in legal_formations():
        need = {"DEF": d, "MID": m, "FWD": f}
        if any(len(pools[p]) < k for p, k in need.items()):
            continue
        xi = pd.concat([gk] + [pools[p].head(k) for p, k in need.items()])
        s = float(xi[score].sum())
        if s > best_score:
            best_score, best = s, xi
    return best


def pick_captain(xi: pd.DataFrame, score: str) -> pd.Series:
    return xi.loc[xi[score].astype(float).idxmax()]


def xi_value(rows: pd.DataFrame, score: str, actual: str | None = None,
             captain_score: str | None = None) -> float:
    """
    Value of the best XI, captain counted twice.

    `score` decides selection; `actual` (if given) supplies the realised points.
    Keeping them separate is what stops a forecast being scored against itself.

    `captain_score` decides the ARMBAND only, leaving selection on `score`.
    Default None keeps the armband on `score`, so existing callers are
    unchanged. Separating them is what allows a captain rule to be tested
    without also perturbing squad and XI selection -- otherwise the two effects
    are confounded and neither is measurable.
    """
    xi = best_xi(rows, score)
    if xi is None:
        return float("nan")
    col = actual or score
    cap = pick_captain(xi, captain_score or score)
    return float(xi[col].sum()) + float(cap[col])


def optimise_squad(pool: pd.DataFrame, score: str, budget: float = C.BUDGET,
                   banned: set | None = None,
                   bench_weight: float = 1.0) -> pd.DataFrame | None:
    """
    Exact MILP: best legal 15 under budget, position quota and club cap.

    `bench_weight` is what the objective pays for a player who is in the squad
    but NOT in the starting XI.

      w = 1.0  (default)  maximise sum(xp) over all fifteen. This is the
                          original objective, reproduced algebraically, so
                          every result recorded before 2026-10-22 keeps its
                          meaning.
      w < 1.0             maximise sum(xp) over the best ELEVEN, paying w for
                          the four on the bench.

    Why the default is wrong but kept: FPL scores eleven players, not fifteen.
    Maximising over fifteen makes the optimiser pay full price for bench points
    it can never score -- upgrading a GBP 4.0m reserve keeper (xp 0.30) to a
    GBP 5.0m one (xp 1.68) reads as a +1.38 gain and is worth exactly zero. The
    measured effect was a GW7 bench costing GBP 21.9m against a realistic
    GBP 17.5m, a squad of mean 12.7% ownership, and Haaland never selected in
    any of GW1-7 despite ranking 1st or 2nd on xp in three of them (S26, O37).

    Why w is not 0: an unused bench is not worthless. Autosubs promote a benched
    player when a starter records no minutes, so bench xp has some real value;
    pricing it at zero would overcorrect into an all-GBP-4.0m bench.

    At w = 1.0 the two-variable program collapses to the one-variable one --
    the XI variables become free and the objective no longer distinguishes
    them -- so the original single-vector formulation is kept for that case.
    That is not an optimisation: it guarantees bit-identical behaviour for
    every existing caller rather than merely equivalent-in-principle behaviour.
    """
    p = pool[pool["cost"].gt(0)]
    if banned:
        p = p[~p[C.PLAYER_KEY].isin(banned)]
    p = p.reset_index(drop=True)
    if p.empty:
        return None
    n = len(p)
    cost = p["cost"].to_numpy(float)
    sc = p[score].to_numpy(float)

    if bench_weight >= 1.0:
        A, lo, hi = [], [], []
        for pos, k in C.SQUAD_QUOTA.items():
            A.append((p["position"] == pos).astype(float).to_numpy())
            lo.append(k); hi.append(k)
        A.append(cost); lo.append(0.0); hi.append(budget)
        for t in p["team"].dropna().unique():
            A.append((p["team"] == t).astype(float).to_numpy())
            lo.append(0.0); hi.append(C.MAX_PER_CLUB)
        res = milp(c=-sc,
                   constraints=LinearConstraint(np.array(A), lo, hi),
                   integrality=np.ones(n), bounds=Bounds(0, 1))
        if not res.success:
            return None
        return p.loc[np.round(res.x).astype(bool)].copy()

    # --- XI-aware: variables [s_0..s_{n-1}, x_0..x_{n-1}] -------------------
    w = float(bench_weight)
    z = np.zeros(n)
    A, lo, hi = [], [], []

    def row(a_s, a_x, l, h):
        A.append(np.concatenate([a_s, a_x])); lo.append(l); hi.append(h)

    for pos, k in C.SQUAD_QUOTA.items():                 # squad quota on s
        row((p["position"] == pos).astype(float).to_numpy(), z, k, k)
    row(cost, z, 0.0, budget)                            # budget on s
    for t in p["team"].dropna().unique():                # club cap on s
        row((p["team"] == t).astype(float).to_numpy(), z, 0.0, C.MAX_PER_CLUB)
    row(z, np.ones(n), C.XI_SIZE, C.XI_SIZE)             # exactly 11 start
    for pos, (blo, bhi) in C.XI_BOUNDS.items():          # legal formation on x
        row(z, (p["position"] == pos).astype(float).to_numpy(), blo, bhi)
    # x_i <= s_i, i.e. you cannot start a player you do not own
    for i in range(n):
        a_s, a_x = np.zeros(n), np.zeros(n)
        a_s[i], a_x[i] = 1.0, -1.0
        row(a_s, a_x, 0.0, 1.0)

    # pay w for being in the squad and (1-w) more for starting
    res = milp(c=-np.concatenate([w * sc, (1.0 - w) * sc]),
               constraints=LinearConstraint(np.array(A), lo, hi),
               integrality=np.ones(2 * n), bounds=Bounds(0, 1))
    if not res.success:
        return None
    return p.loc[np.round(res.x[:n]).astype(bool)].copy()
