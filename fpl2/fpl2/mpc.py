"""
Receding-horizon transfer planning: the season as the MDP it actually is.

S27 measured the thing that justifies this module. Opening-XI quality and
season total are UNCORRELATED within season (-0.083 / +0.214 / -0.290 across 16
arms). Squad construction does not propagate to the score. What sits between a
good squad and a good season is the TRANSFER POLICY, and the one in
`simulate.simulate_season` is myopic: a single greedy swap per week above a
fixed gain threshold, with no banking, no hits and no planning.

The state is (squad, bank, free transfers, gameweek); the actions are
transfers, line-up and captain; the reward is realised points. That is a finite
MDP, but an exact one is hopeless -- the squad space alone is C(600,15). The
standard approximation is model-predictive control: plan H weeks ahead with a
time-indexed MILP, commit only this week's decision, re-plan at the next
deadline.

THE LEAK THIS INVITES (M17, and the reason that mistake is cited here rather
than merely remembered). At the week-t0 deadline the walk-forward forecast for
t0+3 DOES NOT EXIST: it was fitted on data from t0+1 and t0+2. Handing it to
the planner is precisely M17 -- a policy reading a calendar it cannot have.
`forecast_mode` therefore selects:

    "honest"  future weeks reuse the week-t0 per-player xp, adjusted only by
              the published fixture list (0 for a blank, x2 for a double).
              Fixtures are confirmed roughly four weeks out, which is why
              `chips.KNOWN` is 4 and why HORIZON_MAX is 4 here.
    "oracle"  future weeks use each week's true forecast. LEAKY. An upper
              bound for diagnosis, never a shippable policy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from . import config as C

HORIZON_MAX = 4          # M17: beyond this the fixture list is not yet knowable
FT_MAX = 2               # free transfers that may be banked
HIT_COST = 4.0           # points charged per transfer above the free allowance
POOL_PER_POS = 40        # candidate buys per position, by horizon xp


def _candidate_pool(frames: list[pd.DataFrame], squad: list) -> pd.DataFrame:
    """
    Current squad plus the strongest plausible buys.

    The full pool is ~600 players; at five binaries per player per week an
    unrestricted program is ~12,000 integers and HiGHS stalls. Restricting to
    the squad plus the top `POOL_PER_POS` per position is a heuristic on the
    ACTION space, not on the constraints -- every returned plan is exactly
    feasible and exactly optimal within the pool.
    """
    allrows = pd.concat(frames, ignore_index=True)
    strength = (allrows.groupby(C.PLAYER_KEY, as_index=False)
                .agg(_xp=("xp_plan", "sum"), cost=("cost", "first"),
                     position=("position", "first"), team=("team", "first")))
    keep = set(squad)
    for pos in C.POSITIONS:
        sub = strength[strength["position"].eq(pos)]
        keep |= set(sub.nlargest(POOL_PER_POS, "_xp")[C.PLAYER_KEY])
        # also the cheapest few: a plan that frees money needs somewhere to put
        # the saving, and bench fodder never ranks on xp
        keep |= set(sub.nsmallest(6, "cost")[C.PLAYER_KEY])
    return strength[strength[C.PLAYER_KEY].isin(keep)].reset_index(drop=True)


def plan(frames: list[pd.DataFrame], squad: list, bank: float,
         free_transfers: int = 1, discount: float = 1.0) -> dict | None:
    """
    Solve the H-week plan and return ONLY week 0's committed decision.

    `frames` is one frame per gameweek in the horizon, each carrying `xp_plan`
    (the planning forecast, already made honest or oracle by the caller),
    `cost`, `position`, `team`. `frames[0]` is the week being decided.

    Returns {"buy": [...], "sell": [...], "hits": int} or None if infeasible.
    """
    H = min(len(frames), HORIZON_MAX)
    if H == 0:
        return None
    pool = _candidate_pool(frames[:H], squad)
    n = len(pool)
    if n < C.SQUAD_SIZE:
        return None
    code = pool[C.PLAYER_KEY].to_numpy()
    idx = {c: i for i, c in enumerate(code)}
    if any(c not in idx for c in squad):
        return None                      # a held player fell out of the pool
    cost = pool["cost"].to_numpy(float)
    own0 = np.zeros(n)
    own0[[idx[c] for c in squad]] = 1.0
    spend_cap = bank + float(cost[own0.astype(bool)].sum())

    # per-week xp over the pool, aligned to `code`
    xp = np.zeros((H, n))
    for t, fr in enumerate(frames[:H]):
        m = fr.set_index(C.PLAYER_KEY)["xp_plan"]
        xp[t] = pd.Series(code).map(m).fillna(0.0).to_numpy(float)

    # ---- variable layout --------------------------------------------------
    # s[t,i] x[t,i] c[t,i] buy[t,i] sell[t,i]   then  hits[t]  ft[t]
    B = H * n
    off = {k: j * B for j, k in enumerate(("s", "x", "c", "buy", "sell"))}
    off["hits"] = 5 * B
    off["ft"] = 5 * B + H
    N = 5 * B + 2 * H

    def scalar(block, t):
        a = np.zeros(N); a[off[block] + t] = 1.0; return a

    rows, lo, hi = [], [], []
    def add(a, l, h):
        rows.append(a); lo.append(l); hi.append(h)

    pos_mask = {p: (pool["position"] == p).to_numpy(float) for p in C.POSITIONS}
    for t in range(H):
        sl = slice(off["s"] + t * n, off["s"] + (t + 1) * n)
        xl = slice(off["x"] + t * n, off["x"] + (t + 1) * n)
        a = np.zeros(N); a[sl] = 1.0; add(a, C.SQUAD_SIZE, C.SQUAD_SIZE)
        a = np.zeros(N); a[xl] = 1.0; add(a, C.XI_SIZE, C.XI_SIZE)
        a = np.zeros(N); a[off["c"] + t * n: off["c"] + (t + 1) * n] = 1.0
        add(a, 1, 1)
        for p, k in C.SQUAD_QUOTA.items():                 # squad quota
            a = np.zeros(N); a[sl] = pos_mask[p]; add(a, k, k)
        for p, (blo, bhi) in C.XI_BOUNDS.items():          # legal formation
            a = np.zeros(N); a[xl] = pos_mask[p]; add(a, blo, bhi)
        for tm in pool["team"].dropna().unique():          # club cap
            a = np.zeros(N); a[sl] = (pool["team"] == tm).to_numpy(float)
            add(a, 0, C.MAX_PER_CLUB)
        a = np.zeros(N); a[sl] = cost; add(a, 0, spend_cap)   # budget
        for i in range(n):                                 # x<=s and c<=x
            a = np.zeros(N)
            a[off["s"] + t * n + i] = 1.0; a[off["x"] + t * n + i] = -1.0
            add(a, 0, 1)
            a = np.zeros(N)
            a[off["x"] + t * n + i] = 1.0; a[off["c"] + t * n + i] = -1.0
            add(a, 0, 1)
            # continuity: s[t,i] - s[t-1,i] - buy + sell = 0
            a = np.zeros(N)
            a[off["s"] + t * n + i] = 1.0
            a[off["buy"] + t * n + i] = -1.0
            a[off["sell"] + t * n + i] = 1.0
            if t == 0:
                add(a, own0[i], own0[i])
            else:
                a[off["s"] + (t - 1) * n + i] = -1.0
                add(a, 0, 0)
        # hits[t] >= transfers[t] - ft[t]
        a = np.zeros(N)
        a[off["buy"] + t * n: off["buy"] + (t + 1) * n] = 1.0
        a[off["hits"] + t] = -1.0
        a[off["ft"] + t] = -1.0
        add(a, -np.inf, 0)
        # ft[0] given; ft[t] <= ft[t-1] - transfers[t-1] + 1
        if t == 0:
            add(scalar("ft", 0), free_transfers, free_transfers)
        else:
            a = np.zeros(N)
            a[off["ft"] + t] = 1.0
            a[off["ft"] + (t - 1)] = -1.0
            a[off["buy"] + (t - 1) * n: off["buy"] + t * n] = 1.0
            add(a, -np.inf, 1)

    c = np.zeros(N)
    for t in range(H):
        g = discount ** t
        c[off["x"] + t * n: off["x"] + (t + 1) * n] = -g * xp[t]
        c[off["c"] + t * n: off["c"] + (t + 1) * n] = -g * xp[t]
        c[off["hits"] + t] = g * HIT_COST

    integrality = np.zeros(N)
    integrality[:5 * B] = 1                      # s, x, c, buy, sell binary
    integrality[off["ft"]: off["ft"] + H] = 1    # free transfers integer
    ub = np.ones(N)
    ub[off["hits"]: off["hits"] + H] = C.SQUAD_SIZE
    ub[off["ft"]: off["ft"] + H] = FT_MAX

    res = milp(c=c, constraints=LinearConstraint(np.array(rows), lo, hi),
               integrality=integrality, bounds=Bounds(np.zeros(N), ub))
    if not res.success:
        return None
    x = res.x
    buy0 = [code[i] for i in range(n)
            if round(x[off["buy"] + i]) == 1]
    sell0 = [code[i] for i in range(n)
             if round(x[off["sell"] + i]) == 1]
    # `hits` is read from the TRANSFERS, not from the slack variable. The
    # constraint is only a lower bound (hits >= T - ft), so when HIT_COST is 0
    # nothing pushes it down and it drifts to its upper bound -- 15 hits
    # against 2 transfers, which is not a count of anything.
    return {"buy": buy0, "sell": sell0,
            "hits": max(0, len(buy0) - int(free_transfers)),
            "objective": float(-res.fun)}


def honest_frames(panel: pd.DataFrame, gw: int, horizon: int,
                  mode: str = "honest") -> list[pd.DataFrame]:
    """
    Build the planning frames for the deadline at `gw`.

    "honest" is the shippable one: every future week reuses THIS week's xp per
    player, scaled only by that week's published fixture count (0 for a blank,
    x2 for a double). Nothing from a future week's forecast is read.
    """
    gws = [g for g in sorted(panel["gameweek"].unique()) if gw <= g < gw + horizon]
    now = panel[panel["gameweek"].eq(gw)]
    base = now.set_index(C.PLAYER_KEY)["xp"]
    per_fx = now.assign(
        _f=pd.to_numeric(now["fixtures"], errors="coerce").fillna(0).clip(lower=1))
    base_per_fixture = (per_fx.set_index(C.PLAYER_KEY)["xp"]
                        / per_fx.set_index(C.PLAYER_KEY)["_f"])
    out = []
    for t, g in enumerate(gws):
        fr = panel[panel["gameweek"].eq(g)].copy()
        if t == 0 or mode == "oracle":
            fr["xp_plan"] = fr["xp"].fillna(0.0)
        else:
            f = pd.to_numeric(fr["fixtures"], errors="coerce").fillna(0).clip(0, 2)
            fr["xp_plan"] = (fr[C.PLAYER_KEY].map(base_per_fixture).fillna(0.0)
                             * f.to_numpy())
        fr.loc[pd.to_numeric(fr["fixtures"], errors="coerce").fillna(0) <= 0,
               "xp_plan"] = 0.0
        out.append(fr)
    return out
