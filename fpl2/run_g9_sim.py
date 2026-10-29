#!/usr/bin/env python3
"""
G9, instrument 2: budget-constrained squad simulation.

The unconstrained best-XI test (run_g9_xi.py) FAILED its gate: 2/4 seasons,
worst -9.0. That instrument picks from the whole league with no budget, so it
is dominated by elite players and Championship arrivals are never in the pool
-- in 2023-24 it moved by exactly 0.0 points while 91 players gained priors.

This instrument has a 100.0 budget and a 15-man squad, where cheap promoted-club
players actually compete for enabler and bench slots.

THIS IS THE SECOND INSTRUMENT TRIED ON THE SAME HYPOTHESIS. A marginal pass is
therefore weak evidence -- the multiple-testing budget is already spent once.
So a negative control is required to pass as well:

  arm A  understat_only
  arm B  understat + manual Championship priors
  arm C  understat + manual priors SHUFFLED between players (same players hold
         a prior, same value distribution, correspondence destroyed)

Gate, declared before the run:
  * B beats A in >=3 of 4 seasons, AND
  * B beats C in >=3 of 4 seasons, AND
  * aggregate (B-A) > 2x aggregate (C-A)
If B ~ C the gain is perturbation, not information, and the result is rejected
however good B looks against A alone.
"""
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, external, features, model, team, simulate  # noqa: E402

SEASONS = list(C.HOLDOUT_SEASONS) + ["2026-27"]
SEED = 20261021


def shuffle_manual(ext, rng):
    """Permute manual priors between the players who hold them, per season."""
    out = ext.copy()
    cols = [c for c in out.columns if c.startswith("ext_")
            and c not in ("ext_available", "ext_match_method", "ext_source",
                          "ext_name", "ext_league", "ext_squads")]
    for s in out["season"].unique():
        m = out["season"].eq(s) & out["ext_source"].eq("manual")
        n = int(m.sum())
        if n < 2:
            continue
        perm = rng.permutation(n)
        block = out.loc[m, cols].to_numpy()
        out.loc[m, cols] = block[perm]
    return out


def build_ext(base_ext, panel, arm, rng):
    ext = base_ext.copy()
    if arm == "understat_only":
        return ext
    parts = [external.with_manual(ext[ext.season.eq(s)].copy(), panel, s)
             for s in SEASONS if not ext[ext.season.eq(s)].empty]
    ext = pd.concat(parts, ignore_index=True)
    if arm == "shuffled":
        ext = shuffle_manual(ext, rng)
    return ext


def main():
    t0 = time.time()
    raw = pd.concat([data.load_raw(), pd.read_pickle("/tmp/fpl2_live2627.pkl")],
                    ignore_index=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    base_ext = pd.read_pickle("/tmp/fpl2_ext_understat.pkl")
    feats, names = features.build(panel, tf)
    rng = np.random.RandomState(SEED)
    print(f"[{time.time()-t0:4.0f}s] setup done\n", flush=True)

    built = {}
    for arm in ("understat_only", "manual", "shuffled"):
        ext = build_ext(base_ext, panel, arm, rng)
        cols = [c for c in ext.columns if c.startswith("ext_")
                and c not in ("ext_match_method", "ext_source", "ext_name",
                              "ext_league", "ext_squads")]
        f = feats.merge(ext[[C.PLAYER_KEY, "season"] + cols],
                        on=[C.PLAYER_KEY, "season"], how="left")
        f["ext_available"] = f["ext_available"].fillna(0.0)
        built[arm] = (f, names + cols)

    print(f"{'season':10s}{'A base':>10}{'B manual':>10}{'C shuf':>10}"
          f"{'B-A':>8}{'C-A':>8}", flush=True)
    res = {}
    for s in SEASONS:
        grid0 = data.with_blanks(panel, s)
        tot = {}
        for arm, (f, n) in built.items():
            out = model.walk_forward(f, tf, n, s)
            g = grid0.merge(out[["season", "gameweek", "player_code", "xp"]],
                            on=["season", "gameweek", "player_code"], how="left")
            g["xp"] = g["xp"].fillna(0.0)
            g.loc[g["fixtures"] <= 0, "xp"] = 0.0
            r = simulate.simulate_season(g, "xp")
            tot[arm] = float(r.total) if r is not None else float("nan")
        a, b, c = tot["understat_only"], tot["manual"], tot["shuffled"]
        res[s] = (a, b, c)
        print(f"{s:10s}{a:>10.0f}{b:>10.0f}{c:>10.0f}{b-a:>+8.0f}{c-a:>+8.0f}",
              flush=True)

    ba = [b - a for a, b, _ in res.values()]
    ca = [c - a for a, _, c in res.values()]
    bc = [b - c for _, b, c in res.values()]
    print(f"\nB beats A in {sum(x>0 for x in ba)}/4   (need >=3)", flush=True)
    print(f"B beats C in {sum(x>0 for x in bc)}/4   (need >=3)", flush=True)
    print(f"aggregate B-A = {sum(ba):+.0f}, C-A = {sum(ca):+.0f}   "
          f"(need B-A > 2x C-A)", flush=True)
    ok = (sum(x > 0 for x in ba) >= 3 and sum(x > 0 for x in bc) >= 3
          and sum(ba) > 2 * max(sum(ca), 0.0))
    print("  VERDICT:", "PASS" if ok else "FAIL", flush=True)
    pickle.dump(res, open("/tmp/fpl2_g9_sim.pkl", "wb"))
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
