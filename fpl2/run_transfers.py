#!/usr/bin/env python3
"""
O23 remainder: WHICH transfers get made?

O13 established the shape: from a shared opening 15, freezing the squad leaves
a -53 gap, allowing transfers widens it to ~-146. Both signals make the SAME
NUMBER of moves (36.7 vs 36.3), so legacy is not transferring more or less --
it is transferring better, by roughly 90 points a season.

Every transfer is a bet: this player will outscore that one from here. So the
bet is directly scoreable, with no simulation and no noise floor to argue about:

  realised(IN) - realised(OUT) over the next K gameweeks

Reported for K = 1, 3, 5 and rest-of-season, for our signal and for legacy,
both starting from the SAME opening 15 so squad differences cannot contaminate
it. Also the hit rate (share of transfers that gained) and the claimed gain,
which is what the policy THOUGHT it was buying -- the gap between claimed and
realised is the measure of how much the signal is fooling itself.
"""
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, legacy, optimise, simulate   # noqa: E402

ARMS = C.cache("fpl2_o19_arms.pkl")
HORIZONS = (1, 3, 5, 99)


def main():
    arms = pickle.load(open(ARMS, "rb"))
    panel = data.build_panel()
    leg = legacy.load_fold_forecasts().rename(columns={"xp": "xp_leg"})

    rows = []
    for s in C.HOLDOUT_SEASONS:
        grid = data.with_blanks(panel, s)
        g = grid.merge(arms["glm"][s][["season", "gameweek", C.PLAYER_KEY, "xp"]],
                       on=["season", "gameweek", C.PLAYER_KEY], how="left")
        g = g.merge(leg, on=["season", "gameweek", C.PLAYER_KEY], how="left")
        for c in ("xp", "xp_leg"):
            g[c] = g[c].fillna(0.0)
            g.loc[g["fixtures"] <= 0, c] = 0.0
        # shared opening 15, picked by legacy, so only the transfers differ
        gws = sorted(g["gameweek"].unique())[:5]
        w = g[g["gameweek"].isin(gws)]
        st = (w.groupby(C.PLAYER_KEY)
               .agg(xp_leg=("xp_leg", "mean"), cost=("cost", "first"),
                    position=("position", "first"), team=("team", "first")).reset_index())
        sq = optimise.optimise_squad(st, "xp_leg")
        open15 = sq[C.PLAYER_KEY].tolist() if sq is not None else None

        pts = (g[g["fixtures"] > 0]
               .set_index([C.PLAYER_KEY, "gameweek"])["points"])

        def realised(code, gw0, k):
            try:
                v = pts.loc[code]
            except KeyError:
                return 0.0
            sel = v[(v.index >= gw0) & (v.index < gw0 + k)]
            return float(sel.sum())

        for arm, sig in (("ours(glm)", "xp"), ("legacy", "xp_leg")):
            r = simulate.simulate_season(g, sig, opening_squad=open15)
            for gw, out, inn, claimed in r.transfer_log:
                rows.append({"season": s, "arm": arm, "gw": gw, "claimed": claimed,
                             **{f"d{k}": realised(inn, gw, k) - realised(out, gw, k)
                                for k in HORIZONS}})

    d = pd.DataFrame(rows)
    print(f"TRANSFER QUALITY  (realised IN - OUT, from a SHARED opening 15)\n")
    print(f"{'arm':12s}{'n':>5}{'claimed':>9}" +
          "".join(f"{'d'+str(k) if k < 99 else 'rest':>9}" for k in HORIZONS) +
          f"{'hit@5':>8}", flush=True)
    for arm, sub in d.groupby("arm"):
        print(f"{arm:12s}{len(sub):>5}{sub.claimed.mean():>9.2f}" +
              "".join(f"{sub[f'd{k}'].mean():>9.2f}" for k in HORIZONS) +
              f"{(sub['d5'] > 0).mean():>8.1%}", flush=True)

    # n is ~110 per arm and per-transfer differences are wildly variable, so the
    # means above mean nothing without their spread. Unpaired: the two arms make
    # DIFFERENT transfers, so they cannot be matched up.
    print(f"\nIS THE DIFFERENCE REAL?  (unpaired, Welch)", flush=True)
    o = d[d.arm.eq("ours(glm)")]; l = d[d.arm.eq("legacy")]
    print(f"{'metric':10s}{'ours':>9}{'sd':>8}{'legacy':>9}{'sd':>8}"
          f"{'diff':>8}{'se':>7}{'t':>7}", flush=True)
    for k in list(HORIZONS) + ["claimed"]:
        col = f"d{k}" if k != "claimed" else "claimed"
        a, b = o[col].to_numpy(float), l[col].to_numpy(float)
        diff = a.mean() - b.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        lbl = col if k != 99 else "rest"
        print(f"{lbl:10s}{a.mean():>9.2f}{a.std(ddof=1):>8.2f}{b.mean():>9.2f}"
              f"{b.std(ddof=1):>8.2f}{diff:>+8.2f}{se:>7.2f}"
              f"{diff/se if se else float('nan'):>7.2f}", flush=True)
    print("  |t| < 2 means the arms are indistinguishable on that metric.", flush=True)

    print(f"\nPER SEASON  (d5 = realised gain over the next 5 gameweeks)", flush=True)
    print(f"{'season':10s}{'ours_n':>8}{'ours_d5':>9}{'leg_n':>7}{'leg_d5':>9}"
          f"{'ours_claim':>12}{'leg_claim':>11}", flush=True)
    for s, sub in d.groupby("season"):
        o = sub[sub.arm.eq("ours(glm)")]; l = sub[sub.arm.eq("legacy")]
        print(f"{s:10s}{len(o):>8}{o.d5.mean():>9.2f}{len(l):>7}{l.d5.mean():>9.2f}"
              f"{o.claimed.mean():>12.2f}{l.claimed.mean():>11.2f}", flush=True)

    print(f"\nSELF-DECEPTION  (claimed gain minus realised, per transfer)", flush=True)
    for arm, sub in d.groupby("arm"):
        print(f"  {arm:12s} claimed {sub.claimed.mean():5.2f}/gw  "
              f"realised d1 {sub.d1.mean():5.2f}  "
              f"overstatement {sub.claimed.mean()-sub.d1.mean():+5.2f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
