#!/usr/bin/env python3
"""
O23: is the opening-squad gap SHAPE rather than ranking?

Eliminated so far for legacy's advantage: overall ranking (S10), top-end
calibration (W13), elite identification (S15), transfer count and the min_gain
threshold (O13), and self-normalised scale (O20). Now also eliminated: early-
season signal quality -- within the top 150 the optimiser actually picks from,
OUR signal out-ranks legacy in 2 of 3 seasons (gbdt 0.293/0.550/0.347, glm
0.419/0.460/0.305, legacy 0.389/0.468/0.265).

Both squads come from the SAME MILP under the same budget. If the rankings are
comparable and the opening 15 differ by ~500 season points (S11: 1683 vs 2184),
the difference must be the VALUES, not the order -- the optimiser trades xp
against cost, so a stretched top buys premiums and a flat one spreads.

Arms, all using OUR signal for every decision after gameweek 5:
  A own          our signal picks the opening 15                (baseline)
  B shape        our RANKING, legacy's VALUE DISTRIBUTION, per gameweek
  C legacy       legacy's signal picks the opening 15           (S11 reference)

  B ~ C  -> the gap is SHAPE, and it is fixable without a better forecast
  B ~ A  -> shape is not it, and the opening-squad gap stays unexplained
"""
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, legacy, optimise, simulate   # noqa: E402

ARMS = C.cache("fpl2_o19_arms.pkl")


def match_shape(vals, template):
    v = np.asarray(vals, float)
    t = np.sort(np.asarray(template, float))
    if len(t) != len(v):
        t = np.interp(np.linspace(0, 1, len(v)), np.linspace(0, 1, len(t)), t)
    return t[np.argsort(np.argsort(v))]


def opening(g, sig):
    gws = sorted(g["gameweek"].unique())[:5]
    w = g[g["gameweek"].isin(gws)]
    st = (w.groupby(C.PLAYER_KEY)
           .agg(**{sig: (sig, "mean"), "cost": ("cost", "first"),
                   "position": ("position", "first"), "team": ("team", "first")})
           .reset_index())
    sq = optimise.optimise_squad(st, sig)
    return sq[C.PLAYER_KEY].tolist() if sq is not None and len(sq) == C.SQUAD_SIZE else None


def main():
    arms = pickle.load(open(ARMS, "rb"))
    panel = data.build_panel()
    leg = legacy.load_fold_forecasts().rename(columns={"xp": "xp_leg"})

    print(f"{'season':10s}{'arm':10s}{'open15_pts':>12}{'season_tot':>12}"
          f"{'vs_legacy':>11}{'cost':>7}", flush=True)
    print("-" * 62, flush=True)
    agg = {}
    for s in C.HOLDOUT_SEASONS:
        season_pts = panel[panel.season.eq(s)].groupby(C.PLAYER_KEY).points.sum()
        grid = data.with_blanks(panel, s)
        g = grid.merge(arms["glm"][s][["season", "gameweek", C.PLAYER_KEY, "xp"]],
                       on=["season", "gameweek", C.PLAYER_KEY], how="left")
        g = g.merge(leg, on=["season", "gameweek", C.PLAYER_KEY], how="left")
        for c in ("xp", "xp_leg"):
            g[c] = g[c].fillna(0.0)
            g.loc[g["fixtures"] <= 0, c] = 0.0
        g["xp_shape"] = (g.groupby("gameweek", group_keys=False)
                          .apply(lambda d: pd.Series(
                              match_shape(d["xp"].to_numpy(), d["xp_leg"].to_numpy()),
                              index=d.index)))
        g["xp_shape"] = g["xp_shape"].fillna(0.0)
        g.loc[g["fixtures"] <= 0, "xp_shape"] = 0.0
        leg_tot = float(simulate.simulate_season(g, "xp_leg").total)
        cost = g.groupby(C.PLAYER_KEY).cost.first()

        for arm, sig in (("A own", "xp"), ("B shape", "xp_shape"),
                         ("C legacy", "xp_leg")):
            sq = opening(g, sig)
            if sq is None:
                continue
            r = simulate.simulate_season(g, "xp", opening_squad=sq)
            o15 = float(season_pts.reindex(sq).fillna(0).sum())
            agg.setdefault(arm, []).append((o15, float(r.total) - leg_tot))
            print(f"{s if arm=='A own' else '':10s}{arm:10s}{o15:>12.0f}"
                  f"{r.total:>12.0f}{r.total-leg_tot:>+11.0f}"
                  f"{cost.reindex(sq).sum():>7.1f}", flush=True)
        print(flush=True)

    print("MEAN OVER SEASONS", flush=True)
    print(f"{'arm':10s}{'open15_pts':>12}{'vs_legacy':>11}", flush=True)
    for arm, v in agg.items():
        print(f"{arm:10s}{np.mean([x[0] for x in v]):>12.0f}"
              f"{np.mean([x[1] for x in v]):>+11.0f}", flush=True)
    a, b, c = (np.mean([x[0] for x in agg[k]]) for k in ("A own", "B shape", "C legacy"))
    if abs(c - a) > 1e-9:
        print(f"\n  shape recovers {100*(b-a)/(c-a):.0f}% of the own->legacy "
              f"opening-squad gap", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
