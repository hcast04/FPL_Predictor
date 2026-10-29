#!/usr/bin/env python3
"""
O12 step 5: the negative control for the captain upside rules.

run_captain_rules.py found xp + 0.5*up_q75 beating argmax(xp) by +1.52 pts/gw,
t=2.94, 4/4 seasons, with a clean dose-response. That is the strongest captain
result on record -- which is exactly when a control is mandatory (R26, and F16
today, where a shuffled control beat the real priors in the live season).

The floor check in that run was INVALID: those rules perturb xp by 8-130%, far
beyond the largest sigma (0.05) ever measured, so there was no matched floor to
compare against.

This replaces it with a direct control. Within EACH gameweek's XI, permute the
upside vector between the 11 players. Same values, same distribution, same
perturbation magnitude -- only the player<->ceiling correspondence is destroyed.

  real     : argmax(xp + k*upside)
  shuffled : argmax(xp + k*permute(upside))     <- 40 seeds

If the real gain sits inside the shuffled distribution, the rule is exploiting
perturbation, not information, and must be rejected however good its t looks.
"""
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, optimise   # noqa: E402

SEASONS = list(C.HOLDOUT_SEASONS) + ["2026-27"]
SEEDS = 40
ARMS = [("up_q75", 0.25), ("up_q75", 0.5), ("up_q75", 1.0), ("up_sd", 1.0)]


def upside_features(panel):
    d = panel.sort_values(["season", "gameweek"]).copy()
    g = d.groupby(C.PLAYER_KEY)["points"]
    d["up_sd"] = g.transform(lambda s: s.shift(1).expanding(min_periods=3).std())
    d["up_q75"] = g.transform(lambda s: s.shift(1).expanding(min_periods=3).quantile(0.75))
    return d[["season", "gameweek", C.PLAYER_KEY, "up_sd", "up_q75"]]


def main():
    t0 = time.time()
    fc = pickle.load(open("/tmp/fpl2_captain_forecasts.pkl", "rb"))
    raw = pd.concat([data.load_raw(), pd.read_pickle("/tmp/fpl2_live2627.pkl")],
                    ignore_index=True)
    up = upside_features(data.build_panel(raw))

    weeks = []
    for s in SEASONS:
        out = fc[s].merge(up, on=["season", "gameweek", C.PLAYER_KEY], how="left")
        for gw, g in out[out.fixtures > 0].groupby("gameweek"):
            if len(g) < 50:
                continue
            xi = optimise.best_xi(g, "xp")
            if xi is None or len(xi) < 11:
                continue
            weeks.append((s, xi.xp.to_numpy(float), xi.points.to_numpy(float),
                          pd.to_numeric(xi.up_sd, errors="coerce").fillna(0.0).to_numpy(float),
                          pd.to_numeric(xi.up_q75, errors="coerce").fillna(0.0).to_numpy(float)))
    print(f"[{time.time()-t0:4.0f}s] {len(weeks)} gameweeks\n", flush=True)

    base = np.array([y[int(np.argmax(xp))] for _, xp, y, _, _ in weeks])
    print(f"{'rule':16s}{'real':>9}{'shuf_mean':>11}{'shuf_sd':>9}"
          f"{'z':>7}{'pctile':>9}{'verdict':>16}", flush=True)
    print("-" * 78, flush=True)
    out = {}
    for col, k in ARMS:
        idx = 3 if col == "up_sd" else 4
        real = np.array([y[int(np.argmax(xp + k * w[idx]))]
                         for w in weeks for _, xp, y, _, _ in [w[:1] + w[1:3] + (None, None)]][:0]) \
               if False else np.array([w[2][int(np.argmax(w[1] + k * w[idx]))] for w in weeks])
        real_d = float(np.mean(real - base))

        shuf = []
        for seed in range(SEEDS):
            rng = np.random.default_rng(7919 * seed + k_hash(col, k))
            tot = [w[2][int(np.argmax(w[1] + k * rng.permutation(w[idx])))] for w in weeks]
            shuf.append(float(np.mean(np.array(tot) - base)))
        shuf = np.array(shuf)
        z = (real_d - shuf.mean()) / shuf.std(ddof=1) if shuf.std(ddof=1) > 0 else np.nan
        pct = float((shuf < real_d).mean())
        verdict = "INFORMATION" if pct >= 0.975 else ("noise" if pct <= 0.90 else "marginal")
        out[(col, k)] = (real_d, shuf.mean(), shuf.std(ddof=1), z, pct)
        print(f"{col}_k{k}".ljust(16) + f"{real_d:>9.3f}{shuf.mean():>11.3f}"
              f"{shuf.std(ddof=1):>9.3f}{z:>7.2f}{pct:>9.1%}{verdict:>16}", flush=True)

    print("\n  pctile = fraction of shuffled draws the real rule beats.", flush=True)
    print("  >=97.5% is the one-sided 0.025 bar. Anything at or below 90% means", flush=True)
    print("  scrambled ceilings do the same job and the rule carries no signal.", flush=True)
    pickle.dump(out, open("/tmp/fpl2_captain_control.pkl", "wb"))
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


def k_hash(col, k):
    return abs(hash((col, k))) % 100000


if __name__ == "__main__":
    main()
