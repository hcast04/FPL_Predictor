#!/usr/bin/env python3
"""
O52 phase 2 -- the optimiser decisions that need TRANSFERS to be visible.

Two sub-experiments, separated because they need different pairing and only
one of them can be measured at high power. Conflating them is how
`bench_weight` came to be quoted as a 338-point effect (S27) when that number
was the instrument's fork variance.

  A. TRANSFER rules -- `min_gain`, greedy vs MPC.
     Opening squad is HELD FIXED at the neutral squads, identical across arms,
     so the comparison is paired (common random numbers) and only the transfer
     path differs. High power.

  B. SQUAD-BUILDING rules -- `bench_weight`, `opening_horizon`.
     These DECIDE the opening squad, so they cannot be handed one. Each arm
     builds its own from its own objective, the paths fork from gameweek 1,
     and K restarts only average that down rather than removing it. Power is
     structurally lower and the MDE is printed next to every row: if a row's
     MDE exceeds any plausible effect, the honest reading is "this instrument
     cannot answer it", not "no effect" (R24, R31).

Signal is FIXED across every arm, so nothing here is confounded with O51.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, measure, simulate                      # noqa: E402
from run_bakeoff_score import cluster, load, HOLDOUTS                # noqa: E402

SCRATCH = Path(os.environ.get("BAKE_DIR", "/tmp/bakeoff"))
K = int(os.environ.get("KOPT2", "10"))
SIGNAL = os.environ.get("OPT_SIGNAL", "m_full_fpl")
# Season-suffixed outputs so this can be fanned out one process per holdout.
# Each arm here is a FULL season simulation (~7s), 12 arms x 8 seasons x K
# restarts, so serial is the better part of an hour.
TAG = ("_" + HOLDOUTS[0]) if len(HOLDOUTS) == 1 else ""

# A: transfer rules -- paired on a fixed opening squad
TRANSFER = {
    "base":        dict(),                                  # REFERENCE
    "gain_0.0":    dict(min_gain=0.0),
    "gain_1.0":    dict(min_gain=1.0),
    "gain_2.0":    dict(min_gain=2.0),
    "mpc_h2":      dict(policy="mpc", mpc_horizon=2),
    "mpc_h4":      dict(policy="mpc", mpc_horizon=4),
}
# B: squad-building rules -- each arm builds its own opening squad
BUILD = {
    "build_base":  dict(),                                  # REFERENCE
    "bench_0.0":   dict(bench_weight=0.0),
    "bench_0.25":  dict(bench_weight=0.25),
    "bench_0.5":   dict(bench_weight=0.5),
    "horizon_1":   dict(opening_horizon=1),
    "horizon_10":  dict(opening_horizon=10),
}


def report(M, ref_name, title, key):
    ref = M[M["arm"].eq(ref_name)].set_index(key)["value"]
    print("\n" + "=" * 78); print(title)
    print("season-clustered, bar |t| > 2.36 (G=8)"); print("=" * 78)
    print(f"{'arm':<14}{'mean':>9}{'t':>8}{'p':>8}{'95% CI':>20}{'MDE':>8}  seasons")
    rows = []
    for a in M["arm"].unique():
        if a == ref_name:
            continue
        cur = M[M["arm"].eq(a)].set_index(key)["value"]
        j = pd.concat({"a": cur, "r": ref}, axis=1).dropna()
        d = (j["a"] - j["r"]).to_numpy(float)
        gg = j.index.get_level_values("season").to_numpy()
        m, se, tt, pp, lo, hi = cluster(d, gg)
        wins = sum(1 for _, dd in pd.Series(d).groupby(gg) if dd.mean() > 0)
        rows.append({"arm": a, "mean": m, "t": tt, "p": pp, "mde": 2.8 * se})
        print(f"{a:<14}{m:>+9.1f}{tt:>8.2f}{pp:>8.3f}"
              f"{f'[{lo:+.1f}, {hi:+.1f}]':>20}{2.8*se:>8.1f}  {wins}/8", flush=True)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    _, grids, _ = load()
    print(f"[{time.time()-t0:4.0f}s] signal={SIGNAL}, K={K}", flush=True)
    A, B = [], []
    for s in HOLDOUTS:
        if s not in grids:
            continue
        # `mpc.honest_frames` reads the forecast column by the literal name
        # "xp" (mpc.py:209), because every earlier caller fed it a
        # `model.walk_forward` frame. This bake-off names each arm's column
        # after the arm, so the MPC arms raise KeyError: 'xp'. Alias rather
        # than edit mpc.py: the two names must point at the SAME column, which
        # an alias guarantees and a second argument would not.
        g = grids[s].assign(xp=lambda d: d[SIGNAL])
        squads = measure.neutral_squads(g, k=K, seed=1)
        for a, kw in TRANSFER.items():                 # paired
            r = measure.restart_score(g, squads, SIGNAL,
                                      simulate.simulate_season, **kw)
            r["season"] = s; r["arm"] = a; A.append(r)
        # B is measured PATH-FREE, not through a season simulation. Each arm
        # builds its 15 at gameweek 1 from its own objective, then the squad is
        # HELD and only XI + captain move -- so the transfer fork, which is the
        # dominant variance term (S38: within-arm restart sd 102 points), is
        # removed by construction. That turns n = 8 season totals into
        # 8 x 38 paired (season, gameweek) cells at the same G = 8.
        for a, kw in BUILD.items():
            sq = simulate._opening_squad(
                g, SIGNAL, kw.get("opening_horizon", 5), C.BUDGET,
                kw.get("bench_weight", 1.0))
            if sq is None or len(sq) != C.SQUAD_SIZE:
                print(f"   !! {s} {a}: no legal squad", flush=True)
                continue
            sc = measure.fixed_squad_score(g, [sq[C.PLAYER_KEY].tolist()],
                                           SIGNAL, actual="points")
            sc["season"] = s; sc["arm"] = a
            B.append(sc)
        pd.concat(A, ignore_index=True).to_pickle(SCRATCH / f"opt2_transfer{TAG}.pkl")
        pd.concat(B, ignore_index=True).to_pickle(SCRATCH / f"opt2_build{TAG}.pkl")
        print(f"   [{time.time()-t0:5.0f}s] {s} done", flush=True)

    if TAG:
        # Fanned out one process per holdout, so this process holds G = 1 and
        # the season-clustered SE is undefined -- `G / (G - 1)` divides by
        # zero. The per-season pickles are already written above; combining
        # and reporting is `run_opt_bakeoff2_combine.py`, which sees all eight.
        print(f"\n[{time.time()-t0:5.0f}s] single-holdout run: pickles written, "
              f"reporting deferred to the combine step (G=1 cannot cluster)")
        return
    MA = pd.concat(A, ignore_index=True)
    report(MA, "base", f"A. TRANSFER RULES vs base -- paired on fixed squads, "
                       f"signal {SIGNAL}", ["season", "squad"]) \
        .to_pickle(SCRATCH / f"opt2_transfer_res{TAG}.pkl")
    MB = pd.concat(B, ignore_index=True)
    MB["value"] = MB["value"] * 38.0                 # per-season equivalent
    report(MB, "build_base",
           "B. SQUAD-BUILDING vs build_base -- path-free, XI+captain only",
           ["season", "gameweek"]).to_pickle(SCRATCH / f"opt2_build_res{TAG}.pkl")
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
