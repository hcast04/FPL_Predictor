#!/usr/bin/env python3
"""
Ceiling features: A/B against the same model without them, then the G1 gate.

WHY: every rolling/expanding feature was a MEAN. Ranking captains by a lagged
75th percentile of points beat argmax(xp) by +1.52 pts/gw on the league-wide
best XI and sat 3.83 sd above a shuffled control -- so the model lacks
information it could use. As a captain rule it was worth nothing (+19 over four
seasons); as features it reaches every decision.

Gates, declared before the run:

  A/B (ceiling ON vs OFF, identical folds and seasons)
    decision metric : win all 3 holdout seasons AND mean margin > 148 pts
                      (R1's noise floor -- the same bar G1 was held to)
    forecast quality: spearman must not fall in any season
    A season-total move under 148 is NOT evidence either way (M9/R24); the
    forecast-quality panel is reported because it is far more sensitive.

  G1 (rebuilt+ceiling vs legacy): win all 3 seasons AND mean margin > 148.
"""
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, legacy, model, team, simulate  # noqa: E402

NOISE_FLOOR = 148.0
CEIL_SUFFIX = ("_q75", "_sd", "_haul_rate", "_q75_r10")


def main():
    t0 = time.time()
    raw = pd.concat([data.load_raw(), pd.read_pickle("/tmp/fpl2_live2627.pkl")],
                    ignore_index=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    ext = pd.read_pickle("/tmp/fpl2_ext_understat.pkl")
    feats, names = features.build(panel, tf)
    cols = [c for c in ext.columns if c.startswith("ext_")
            and c not in ("ext_match_method", "ext_source", "ext_name",
                          "ext_league", "ext_squads")]
    feats = feats.merge(ext[[C.PLAYER_KEY, "season"] + cols],
                        on=[C.PLAYER_KEY, "season"], how="left")
    feats["ext_available"] = feats["ext_available"].fillna(0.0)
    full = names + cols
    ceil = [n for n in full if n.endswith(CEIL_SUFFIX)]
    without = [n for n in full if n not in ceil]
    print(f"[{time.time()-t0:4.0f}s] features: {len(full)} with ceiling, "
          f"{len(without)} without ({len(ceil)} ceiling)\n", flush=True)

    fc = {"on": {}, "off": {}}
    for s in C.HOLDOUT_SEASONS:
        fc["on"][s] = model.walk_forward(feats, tf, full, s)
        fc["off"][s] = model.walk_forward(feats, tf, without, s)
        print(f"[{time.time()-t0:4.0f}s] {s} both arms done", flush=True)

    print(f"\nFORECAST QUALITY (all rows with fixtures)", flush=True)
    print(f"{'season':10s}{'sp_off':>9}{'sp_on':>9}{'d':>8}"
          f"{'ratio_off':>11}{'ratio_on':>10}", flush=True)
    sp_fail = 0
    for s in C.HOLDOUT_SEASONS:
        a = fc["off"][s]; b = fc["on"][s]
        a = a[a.fixtures > 0]; b = b[b.fixtures > 0]
        spa = a.xp.corr(a.points, method="spearman")
        spb = b.xp.corr(b.points, method="spearman")
        if spb < spa:
            sp_fail += 1
        print(f"{s:10s}{spa:>9.4f}{spb:>9.4f}{spb-spa:>+8.4f}"
              f"{a.xp.sum()/a.points.sum():>11.4f}{b.xp.sum()/b.points.sum():>10.4f}",
              flush=True)

    print(f"\nDECISION METRIC (budget-constrained squad)", flush=True)
    print(f"{'season':10s}{'off':>8}{'on':>8}{'margin':>9}", flush=True)
    margins = []
    base = data.build_panel()
    for s in C.HOLDOUT_SEASONS:
        grid0 = data.with_blanks(base, s)
        tot = {}
        for arm in ("off", "on"):
            g = grid0.merge(fc[arm][s][["season", "gameweek", C.PLAYER_KEY, "xp"]],
                            on=["season", "gameweek", C.PLAYER_KEY], how="left")
            g["xp"] = g["xp"].fillna(0.0)
            g.loc[g["fixtures"] <= 0, "xp"] = 0.0
            r = simulate.simulate_season(g, "xp")
            tot[arm] = float(r.total) if r else float("nan")
        margins.append(tot["on"] - tot["off"])
        print(f"{s:10s}{tot['off']:>8.0f}{tot['on']:>8.0f}{margins[-1]:>+9.0f}",
              flush=True)

    wins = sum(1 for m in margins if m > 0)
    mean_m = float(np.mean(margins))
    print(f"\nA/B GATE", flush=True)
    print(f"  decision: {wins}/3 seasons, mean margin {mean_m:+.0f} "
          f"(need 3/3 and > {NOISE_FLOOR:.0f})  "
          f"{'PASS' if wins == 3 and mean_m > NOISE_FLOOR else 'FAIL'}", flush=True)
    print(f"  spearman: fell in {sp_fail}/3 seasons  "
          f"{'PASS' if sp_fail == 0 else 'FAIL'}", flush=True)
    if abs(mean_m) < NOISE_FLOOR:
        print(f"  NOTE: |margin| < noise floor -- the decision metric is "
              f"uninformative here (M9/R24). Judge on forecast quality.", flush=True)

    pickle.dump({k: v[["season","gameweek","player_code","position","team","cost",
                       "fixtures","points","xp","xp_raw"]]
                 for k, v in fc["on"].items()}, open("/tmp/fpl2_xp.pkl", "wb"))
    print(f"\n[{time.time()-t0:4.0f}s] wrote /tmp/fpl2_xp.pkl -- run_gate_g1.py "
          f"can now run against the ceiling model", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
