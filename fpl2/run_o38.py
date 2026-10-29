#!/usr/bin/env python3
"""
O38: receding-horizon transfer policy, against the gate pre-registered 2026-10-22.

  1. honest arm, mean over 3 holdouts, must beat greedy by >= +100 pts/season
  2. 3/3 -> SHIP; 2/3 -> real but not established; <=1/3 -> REFUTED
  3. H in {1,2,3,4}, whole curve reported, nothing selected on the run.
     H > 4 forbidden (M17). H=1 is a correctness check against greedy.
  4. R26 control: xp shuffled within (gameweek, position) -> MPC must not beat
     greedy by more than the noise floor
  5. oracle arm (true future xp) reported as the M17 leak control
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, simulate  # noqa: E402

GRIDS = C.cache("fpl2_o37_grids.pkl")
HORIZONS = (1, 2, 3, 4)
FLOOR = 150.0                      # R1 season-total noise floor


def main():
    t0 = time.time()
    grids = pickle.load(open(GRIDS, "rb"))
    rows = []
    for s in C.HOLDOUT_SEASONS:
        g = grids[s]
        base = simulate.simulate_season(g, "xp").total
        rows.append(dict(season=s, arm="greedy", H=0, total=base))
        print(f"[{time.time()-t0:5.0f}s] {s} greedy {base:.0f}", flush=True)
        for H in HORIZONS:
            r = simulate.simulate_season(g, "xp", policy="mpc", mpc_horizon=H)
            rows.append(dict(season=s, arm="honest", H=H, total=r.total,
                             transfers=r.transfers))
            print(f"[{time.time()-t0:5.0f}s]   honest H={H} {r.total:.0f} "
                  f"({r.total-base:+.0f})", flush=True)
        r = simulate.simulate_season(g, "xp", policy="mpc", mpc_horizon=4,
                                     mpc_mode="oracle")
        rows.append(dict(season=s, arm="oracle", H=4, total=r.total))
        print(f"[{time.time()-t0:5.0f}s]   oracle H=4 {r.total:.0f} "
              f"({r.total-base:+.0f})", flush=True)
    R = pd.DataFrame(rows)

    print("\n" + "="*76)
    print("HORIZON SWEEP (delta vs greedy)")
    print("="*76)
    piv = R.pivot_table(index="season", columns=["arm", "H"], values="total")
    print(piv.round(0).to_string())
    base = R[R.arm.eq("greedy")].set_index("season")["total"]
    print("\n  delta vs greedy:")
    print(f"  {'season':<10}" + "".join(f"{'H='+str(H):>9}" for H in HORIZONS)
          + f"{'oracle':>10}")
    for s in C.HOLDOUT_SEASONS:
        line = f"  {s:<10}"
        for H in HORIZONS:
            v = R[(R.season == s) & (R.arm == "honest") & (R.H == H)]["total"]
            line += f"{(float(v.iloc[0]) - base[s]):>+9.0f}"
        v = R[(R.season == s) & (R.arm == "oracle")]["total"]
        line += f"{(float(v.iloc[0]) - base[s]):>+10.0f}"
        print(line)

    print("\n" + "="*76)
    print("GATE (pre-registered 2026-10-22)")
    print("="*76)
    for H in HORIZONS:
        d = np.array([float(R[(R.season == s) & (R.arm == "honest")
                              & (R.H == H)]["total"].iloc[0]) - base[s]
                      for s in C.HOLDOUT_SEASONS])
        npos = int((d > 0).sum())
        verdict = ("SHIP" if d.mean() >= 100 and npos == 3 else
                   "REAL BUT NOT ESTABLISHED" if d.mean() >= 100 and npos == 2 else
                   "FAIL")
        print(f"  H={H}  mean {d.mean():+7.1f}  positive {npos}/3  "
              f"[need >=+100 and 3/3]  -> {verdict}")

    do = np.array([float(R[(R.season == s) & (R.arm == "oracle")]["total"].iloc[0])
                   - base[s] for s in C.HOLDOUT_SEASONS])
    dh = np.array([float(R[(R.season == s) & (R.arm == "honest")
                           & (R.H == 4)]["total"].iloc[0]) - base[s]
                   for s in C.HOLDOUT_SEASONS])
    print(f"\n  M17 LEAK CONTROL: oracle mean {do.mean():+.1f} vs honest {dh.mean():+.1f}")
    print(f"    gap = {do.mean()-dh.mean():+.1f} pts/season = the value of knowledge")
    print(f"    we do not have. honest ~= oracle would mean the honest arm leaks.")

    print("\n" + "="*76)
    print("NEGATIVE CONTROL (R26): xp shuffled within (gameweek, position)")
    print("="*76)
    rng = np.random.default_rng(0)
    cd = []
    for s in C.HOLDOUT_SEASONS:
        g = grids[s].copy()
        g["xp"] = (g.groupby(["gameweek", "position"])["xp"]
                     .transform(lambda v: rng.permutation(v.to_numpy())))
        bg = simulate.simulate_season(g, "xp").total
        bm = simulate.simulate_season(g, "xp", policy="mpc", mpc_horizon=4).total
        cd.append(bm - bg)
        print(f"  {s}: greedy {bg:.0f}  mpc {bm:.0f}  delta {bm-bg:+.0f}")
    print(f"  mean {np.mean(cd):+.1f}   [noise floor {FLOOR:.0f}]")
    print(f"  -> control {'PASSES' if abs(np.mean(cd)) < FLOOR else 'FAILS'}")

    pickle.dump(R, open(C.cache("fpl2_o38.pkl"),"wb"))
    print(f"\n[{time.time()-t0:.0f}s] cached -> ~/fpl2_o38.pkl")


if __name__ == "__main__":
    main()
