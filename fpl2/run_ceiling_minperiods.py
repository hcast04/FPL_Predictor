#!/usr/bin/env python3
"""
min_periods sweep for the ceiling signal.

up_q75 uses min_periods=3, so a player with three appearances and one haul gets
a wildly optimistic ceiling. The shuffled control rules out the effect being a
pure artefact, but not that the EARLY-SEASON portion is noise a higher
threshold would clean up.

Probe: the captain-XI metric (xp + 0.5*up_q75 vs argmax(xp)), which needs no
refitting and is a sensitive read on whether the ceiling information is good.
Each threshold carries its own shuffled control, because a threshold that
raises the effect by perturbing more is not an improvement.

Also reported: coverage (how many XI rows have a defined ceiling at all). A
higher threshold buys cleanliness with missingness, and the trade has to be
visible rather than assumed.
"""
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, optimise   # noqa: E402

SEASONS = list(C.HOLDOUT_SEASONS) + ["2026-27"]
K = 0.5
SEEDS = 30
GRID = (3, 5, 10, 20)


def main():
    t0 = time.time()
    fc = pickle.load(open("/tmp/fpl2_captain_forecasts.pkl", "rb"))
    raw = pd.concat([data.load_raw(), pd.read_pickle("/tmp/fpl2_live2627.pkl")],
                    ignore_index=True)
    panel = data.build_panel(raw)
    d = panel.sort_values(["season", "gameweek"]).copy()
    g = d.groupby(C.PLAYER_KEY)["points"]
    for mp in GRID:
        d[f"q75_{mp}"] = g.transform(
            lambda s, m=mp: s.shift(1).expanding(min_periods=m).quantile(0.75))
    up = d[["season", "gameweek", C.PLAYER_KEY] + [f"q75_{m}" for m in GRID]]
    print(f"[{time.time()-t0:4.0f}s] setup done\n", flush=True)

    weeks = []
    for s in SEASONS:
        out = fc[s].merge(up, on=["season", "gameweek", C.PLAYER_KEY], how="left")
        for gw, gr in out[out.fixtures > 0].groupby("gameweek"):
            if len(gr) < 50:
                continue
            xi = optimise.best_xi(gr, "xp")
            if xi is None or len(xi) < 11:
                continue
            weeks.append((s, xi.xp.to_numpy(float), xi.points.to_numpy(float),
                          {m: pd.to_numeric(xi[f"q75_{m}"], errors="coerce").to_numpy(float)
                           for m in GRID}))
    print(f"  {len(weeks)} gameweeks\n", flush=True)

    base = np.array([y[int(np.argmax(xp))] for _, xp, y, _ in weeks])
    print(f"{'min_per':>8}{'coverage':>10}{'disagree':>10}{'real':>9}"
          f"{'shuf_mean':>11}{'shuf_sd':>9}{'z':>7}{'seasons':>9}", flush=True)
    print("-" * 73, flush=True)
    for mp in GRID:
        cov, dis, real = [], [], []
        for (_, xp, y, qs) in weeks:
            v = np.nan_to_num(qs[mp], nan=0.0)
            cov.append(np.mean(~np.isnan(qs[mp])))
            i = int(np.argmax(xp + K * v))
            dis.append(i != int(np.argmax(xp)))
            real.append(y[i])
        real = np.array(real)
        real_d = float(np.mean(real - base))
        shuf = []
        for seed in range(SEEDS):
            rng = np.random.default_rng(1301 * seed + mp)
            tot = [w[2][int(np.argmax(w[1] + K * rng.permutation(
                np.nan_to_num(w[3][mp], nan=0.0))))] for w in weeks]
            shuf.append(float(np.mean(np.array(tot) - base)))
        shuf = np.array(shuf)
        z = (real_d - shuf.mean()) / shuf.std(ddof=1) if shuf.std(ddof=1) > 0 else np.nan
        by_season = {}
        for (s, xp, y, qs), r in zip(weeks, real):
            by_season.setdefault(s, []).append(r - y[int(np.argmax(xp))])
        wins = sum(1 for v in by_season.values() if np.mean(v) > 0)
        print(f"{mp:>8}{np.mean(cov):>9.1%}{np.mean(dis):>10.1%}{real_d:>9.3f}"
              f"{shuf.mean():>11.3f}{shuf.std(ddof=1):>9.3f}{z:>7.2f}"
              f"{wins}/{len(by_season)}".rjust(9), flush=True)

    print("\n  coverage = share of XI players with a DEFINED ceiling.", flush=True)
    print("  A higher threshold trades noise for missingness; z is the read on", flush=True)
    print("  whether the remaining signal is information rather than churn.", flush=True)
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
