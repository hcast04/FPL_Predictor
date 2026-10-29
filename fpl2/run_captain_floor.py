#!/usr/bin/env python3
"""
O12 step 1: the noise floor for CAPTAIN points. Measure before judging.

F9 rejected two captain rules on per-gameweek deltas of +0.66/-1.45/-0.03 with
no standard error, no test, and no noise floor -- the M6/M12 pattern. R24 says
state the threshold in units of the measured floor, so measure it first.

Method: hold the XI fixed (chosen by unjittered xp, so the squad decision is
identical across arms, exactly as a captain-rule comparison would be), then
perturb xp by information-free multiplicative lognormal noise and re-pick the
armband inside that same XI. The spread of season armband totals across seeds
is the smallest captain effect any test could detect.

Forecasts are cached so later captain experiments need no refitting.
"""
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, model, team, optimise, noise  # noqa: E402

SEASONS = list(C.HOLDOUT_SEASONS) + ["2026-27"]
FC = Path("/tmp/fpl2_captain_forecasts.pkl")
SEEDS = 24


def forecasts():
    if FC.exists():
        print("  (forecasts from cache)", flush=True)
        return pickle.load(open(FC, "rb"))
    raw = pd.concat([data.load_raw(), pd.read_pickle("/tmp/fpl2_live2627.pkl")],
                    ignore_index=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    ext = pd.read_pickle("/tmp/fpl2_ext_understat.pkl")
    feats, names = features.build(panel, tf)
    cols = [c for c in ext.columns if c.startswith("ext_")
            and c not in ("ext_match_method", "ext_source", "ext_name",
                          "ext_league", "ext_squads")]
    f = feats.merge(ext[[C.PLAYER_KEY, "season"] + cols],
                    on=[C.PLAYER_KEY, "season"], how="left")
    f["ext_available"] = f["ext_available"].fillna(0.0)
    out = {}
    for s in SEASONS:
        out[s] = model.walk_forward(f, tf, names + cols, s)
        print(f"  forecast {s} done", flush=True)
    pickle.dump(out, open(FC, "wb"))
    return out


def weekly_xi(out):
    """Best legal XI per gameweek, by unjittered xp. Fixed across all arms."""
    xis = []
    for gw, g in out[out.fixtures > 0].groupby("gameweek"):
        if len(g) < 50:
            continue
        xi = optimise.best_xi(g, "xp")
        if xi is not None:
            xis.append((gw, xi))
    return xis


def main():
    t0 = time.time()
    fc = forecasts()
    print(f"[{time.time()-t0:4.0f}s] forecasts ready\n", flush=True)

    print(f"{'season':10s}{'gws':>5}{'armband_mean':>14}{'sigma':>8}"
          f"{'sd(season mean)':>17}{'sd(season tot)':>16}", flush=True)
    summary = {}
    for s in SEASONS:
        xis = weekly_xi(fc[s])
        base = [float(xi.loc[xi.xp.idxmax()].points) for _, xi in xis]
        for sigma in (0.001, 0.01, 0.05):
            means = []
            for seed in range(SEEDS):
                rng = np.random.default_rng(10_000 * seed + int(sigma * 1e5))
                tot = []
                for _, xi in xis:
                    fac = rng.lognormal(-0.5 * sigma**2, sigma, size=len(xi))
                    tot.append(float(xi.points.to_numpy(float)[np.argmax(
                        xi.xp.to_numpy(float) * fac)]))
                means.append(np.mean(tot))
            sd = float(np.std(means, ddof=1))
            summary[(s, sigma)] = sd
            print(f"{s:10s}{len(xis):>5}{np.mean(base):>14.3f}{sigma:>8.3f}"
                  f"{sd:>17.4f}{sd*len(xis):>16.1f}", flush=True)
        print(flush=True)

    print("MINIMUM DETECTABLE CAPTAIN EFFECT (per gameweek)", flush=True)
    print("  A rule must beat argmax(xp) by more than the spread that "
          "information-free\n  jitter produces, or the test cannot see it.\n", flush=True)
    for sigma in (0.001, 0.01, 0.05):
        sds = [summary[(s, sigma)] for s in SEASONS]
        print(f"  sigma={sigma:.3f}  per-season sd {min(sds):.4f}-{max(sds):.4f} "
              f"pts/gw   ~2sd = {2*max(sds):.3f} pts/gw", flush=True)
    print(f"\n  F9's rejected deltas were +0.66 / -1.45 / -0.03 and "
          f"+0.24 / -0.55 / -0.34 pts/gw.", flush=True)
    pickle.dump(summary, open("/tmp/fpl2_captain_floor.pkl", "wb"))
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
