#!/usr/bin/env python3
"""
Exact squad / line-up / captain optimiser (MILP over HiGHS).

The chip and wildcard experiments originally rebuilt squads with a two-pass
greedy knapsack. That made every wildcard look worse than it is, because a
wildcard's whole value IS the quality of the squad it builds. This replaces it
with the same formulation the notebook uses: one mixed-integer program solved by
`scipy.optimize.milp` (HiGHS), no new dependencies.

Formulation
-----------
Decision variables, for candidate players i and horizon gameweeks g:

    x[i]      player i is in the 15-man squad
    y[i,g]    player i starts in gameweek g
    c[i,g]    player i is captain in gameweek g

Maximise      sum_g  d**g * sum_i  xp[i,g] * (y[i,g] + c[i,g])

subject to
    sum_i x[i] = 15                       squad size
    sum_{i in P} x[i] = quota[P]          2 GK, 5 DEF, 5 MID, 3 FWD
    sum_i cost[i] * x[i] <= budget        budget
    sum_{i in club} x[i] <= 3             club limit
    sum_i y[i,g] = 11                     eleven starters each week
    y[i,g] <= x[i]                        can only start a squad player
    GK: sum y = 1, DEF: 3..5, MID: 2..5, FWD: 1..3     legal formation
    sum_i c[i,g] = 1                      one captain
    c[i,g] <= y[i,g]                      captain must start

The captain term appears once more in the objective because a captain scores
double: y + c gives 2x for the captain and 1x for the other ten.

Candidate pruning keeps the program small: the top `top_per_position` players by
discounted horizon xP plus the `cheap_per_position` cheapest, per position, plus
any ids passed in `must_include` (used to keep the incumbent squad comparable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, hstack, vstack

SQUAD_QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_BOUNDS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
MAX_PER_CLUB = 3
SQUAD_SIZE = 15
XI_SIZE = 11


def prune_candidates(panel: pd.DataFrame, gameweeks: list[int], *,
                     discount: float = 0.96, top_per_position: int = 32,
                     cheap_per_position: int = 12,
                     must_include: set[int] | None = None) -> pd.DataFrame:
    """Reduce the player pool to a tractable, structurally complete subset."""
    window = panel[panel["gameweek"].isin(gameweeks)].copy()
    if window.empty:
        return window
    index = {gw: i for i, gw in enumerate(gameweeks)}
    window["w"] = window["xp"] * window["gameweek"].map(index).map(
        lambda i: discount ** i
    )
    strength = (window.groupby("element")
                .agg(weighted=("w", "sum"), cost=("cost", "first"),
                     pos=("model_position", "first"), team=("team", "first"))
                .reset_index())
    strength = strength[strength["cost"] > 0]

    keep = set(must_include or set())
    for _, group in strength.groupby("pos"):
        keep |= set(group.nlargest(top_per_position, "weighted")["element"])
        keep |= set(group.nsmallest(cheap_per_position, "cost")["element"])
    return strength[strength["element"].isin(keep)].reset_index(drop=True)


def optimise_squad(panel: pd.DataFrame, gameweeks: list[int], budget: int, *,
                   discount: float = 0.96, must_include: set[int] | None = None,
                   time_limit: float = 20.0,
                   **prune_kwargs) -> tuple[list[int], float]:
    """Return (squad element ids, objective value). Empty list if infeasible."""
    gameweeks = [int(g) for g in gameweeks]
    cand = prune_candidates(panel, gameweeks, discount=discount,
                            must_include=must_include, **prune_kwargs)
    if cand.empty or len(cand) < SQUAD_SIZE:
        return [], 0.0

    n, G = len(cand), len(gameweeks)
    ids = cand["element"].to_numpy()
    pos = cand["pos"].to_numpy()
    cost = cand["cost"].to_numpy(dtype=float)
    team = cand["team"].to_numpy()

    # xp[i, g]
    wide = (panel[panel["gameweek"].isin(gameweeks) & panel["element"].isin(ids)]
            .pivot_table(index="element", columns="gameweek", values="xp",
                         aggfunc="sum")
            .reindex(index=ids, columns=gameweeks).fillna(0.0).to_numpy())

    nx, ny = n, n * G                      # x block, y block, c block
    total = nx + ny + ny
    disc = np.array([discount ** i for i in range(G)], dtype=float)
    # y and c are laid out GAMEWEEK-major: [g0 players..., g1 players..., ...],
    # matching the per-gameweek constraint blocks below. `wide` is (player,
    # gameweek), so it must be transposed before ravelling.
    obj_y = (wide * disc[None, :]).T.ravel()
    c_vec = -np.concatenate([np.zeros(nx), obj_y, obj_y])   # milp minimises

    rows, lo, hi = [], [], []

    def add(block_x=None, block_y=None, block_c=None, low=0.0, high=0.0):
        bx = block_x if block_x is not None else csc_matrix((1, nx))
        by = block_y if block_y is not None else csc_matrix((1, ny))
        bc = block_c if block_c is not None else csc_matrix((1, ny))
        rows.append(hstack([bx, by, bc]))
        lo.append(low); hi.append(high)

    # squad size, position quotas, budget, club limits
    add(block_x=csc_matrix(np.ones((1, nx))), low=SQUAD_SIZE, high=SQUAD_SIZE)
    for p, q in SQUAD_QUOTA.items():
        add(block_x=csc_matrix((pos == p).astype(float).reshape(1, -1)),
            low=q, high=q)
    add(block_x=csc_matrix(cost.reshape(1, -1)), low=0.0, high=float(budget))
    for club in np.unique(team):
        add(block_x=csc_matrix((team == club).astype(float).reshape(1, -1)),
            low=0.0, high=MAX_PER_CLUB)

    def gw_block(vec_n: np.ndarray, g: int) -> csc_matrix:
        m = np.zeros((1, ny))
        m[0, g * n:(g + 1) * n] = vec_n
        return csc_matrix(m)

    ones = np.ones(n)
    for g in range(G):
        add(block_y=gw_block(ones, g), low=XI_SIZE, high=XI_SIZE)
        add(block_c=gw_block(ones, g), low=1.0, high=1.0)
        for p, (lo_p, hi_p) in XI_BOUNDS.items():
            add(block_y=gw_block((pos == p).astype(float), g), low=lo_p, high=hi_p)

    # y[i,g] <= x[i]  and  c[i,g] <= y[i,g], as sparse blocks
    eye = csc_matrix(np.eye(n))
    link_y = vstack([hstack([-eye,
                             csc_matrix((n, n * g)), eye,
                             csc_matrix((n, n * (G - g - 1))),
                             csc_matrix((n, ny))]) for g in range(G)])
    link_c = vstack([hstack([csc_matrix((n, nx)),
                             csc_matrix((n, n * g)), -eye,
                             csc_matrix((n, n * (G - g - 1))),
                             csc_matrix((n, n * g)), eye,
                             csc_matrix((n, n * (G - g - 1)))]) for g in range(G)])
    constraints = [LinearConstraint(vstack(rows).tocsc(), np.array(lo), np.array(hi)),
                   LinearConstraint(link_y, -np.inf, 0.0),
                   LinearConstraint(link_c, -np.inf, 0.0)]

    res = milp(c=c_vec, constraints=constraints,
               integrality=np.ones(total), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-4})
    if not res.success or res.x is None:
        return [], 0.0
    chosen = np.where(res.x[:nx] > 0.5)[0]
    return [int(ids[i]) for i in chosen], float(-res.fun)


def best_xi_from_squad(rows: pd.DataFrame, col: str):
    """Exact best legal XI for one gameweek from a held squad."""
    by = {p: rows[rows["model_position"].eq(p)].sort_values(col, ascending=False)
          for p in SQUAD_QUOTA}
    if by["GK"].empty:
        return rows.head(0), rows
    gk = by["GK"].iloc[[0]]
    best, best_val = None, -np.inf
    for d in range(XI_BOUNDS["DEF"][0], XI_BOUNDS["DEF"][1] + 1):
        for m in range(XI_BOUNDS["MID"][0], XI_BOUNDS["MID"][1] + 1):
            f = 10 - d - m
            if not (XI_BOUNDS["FWD"][0] <= f <= XI_BOUNDS["FWD"][1]):
                continue
            if len(by["DEF"]) < d or len(by["MID"]) < m or len(by["FWD"]) < f:
                continue
            xi = pd.concat([by["DEF"].head(d), by["MID"].head(m), by["FWD"].head(f)])
            val = float(xi[col].sum()) + float(gk[col].iloc[0])
            if val > best_val:
                best_val, best = val, pd.concat([gk, xi])
    if best is None:
        return rows.head(0), rows
    return best, rows[~rows["element"].isin(best["element"])]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    import validate_chip_timing as C

    season = sys.argv[1] if len(sys.argv) > 1 else "2025-26"
    panel = C.build_panel(season)
    gws = sorted(panel["gameweek"].unique())[:6]
    import time
    t0 = time.time()
    squad, obj = optimise_squad(panel, gws, budget=1000)
    print(f"season {season}, gameweeks {gws}")
    print(f"solved in {time.time()-t0:.1f}s   objective {obj:.1f}   squad size {len(squad)}")
    held = panel[panel["element"].isin(squad) & panel["gameweek"].eq(gws[0])]
    print(held[["element", "model_position", "team", "cost", "xp"]]
          .sort_values("model_position").to_string(index=False))
    print(f"total cost {held['cost'].sum()} / 1000")
