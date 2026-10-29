#!/usr/bin/env python3
"""
O20: does quantile-normalising the signal help the optimiser?

O19 split the GLM's +210 into 57% ORDERING (3/3 seasons, +241/+39/+77) and 43%
SCALE (+8/+362/-98, sign-flipping).

My stated prediction -- "normalise away the volatile half and keep the reliable
one" -- was TOO STRONG, and reading the numbers again says so: the scale
component has mean +91, so it is volatile but not zero-mean. Removing it trades
~91 points of mean for lower variance. This test decides whether that trade pays,
it does not assume it.

Already known from O19: matching to the GBDT's own per-gameweek distribution
gives the GLM a mean of -148 vs legacy, WORSE than raw's -57. That template is
itself bad, so it is included only as the control.

Templates (all applied per gameweek, preserving each signal's ordering exactly):
  raw            no transform (control)
  self_pooled    the signal's own quantiles pooled over PRIOR SEASONS -- the
                 only leak-free template available, and the only one that could
                 ship
  gbdt_gw        O19's: the gbdt's distribution for that same gameweek (control)
  legacy_gw      legacy's distribution for that gameweek. Uses a signal we have
                 at decision time, so it is legitimate, though it imports
                 legacy's shape wholesale.

GATE, declared now: a template must beat RAW on mean-vs-legacy AND win in at
least as many seasons. Variance reduction alone, with a worse mean, is recorded
as NOT a result -- the decision metric is points, not stability.
"""
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, legacy, simulate   # noqa: E402

ARMS = C.cache("fpl2_o19_arms.pkl")


def apply_template(values, template):
    """Give `values` the multiset of `template`, preserving values' ORDER."""
    v = np.asarray(values, float)
    if len(v) == 0:
        return v
    t = np.sort(np.asarray(template, float))
    if len(t) != len(v):            # resample the template to this gameweek's size
        t = np.interp(np.linspace(0, 1, len(v)), np.linspace(0, 1, len(t)), t)
    return t[np.argsort(np.argsort(v))]


def main():
    t0 = time.time()
    arms = pickle.load(open(ARMS, "rb"))
    panel = data.build_panel()
    leg_fc = legacy.load_fold_forecasts().rename(columns={"xp": "xp_leg"})
    print(f"[{time.time()-t0:4.0f}s] setup\n", flush=True)

    # leak-free template: each arm's own xp quantiles pooled over PRIOR seasons
    pooled = {}
    for arm in ("gbdt", "glm"):
        for s in C.HOLDOUT_SEASONS:
            prev = [arms[arm][p] for p in C.HOLDOUT_SEASONS if p < s]
            if prev:
                v = pd.concat(prev); v = v[v.fixtures > 0].xp.to_numpy(float)
                pooled[(arm, s)] = np.quantile(v, np.linspace(0, 1, 500))

    print(f"{'arm':6s}{'template':14s}{'2023-24':>10}{'2024-25':>10}{'2025-26':>10}"
          f"{'mean':>9}{'wins':>6}", flush=True)
    print("-" * 65, flush=True)
    out = {}
    for arm in ("gbdt", "glm"):
        for tmpl in ("raw", "self_pooled", "gbdt_gw", "legacy_gw"):
            margins = []
            for s in C.HOLDOUT_SEASONS:
                grid = data.with_blanks(panel, s)
                g = grid.merge(arms[arm][s][["season", "gameweek", C.PLAYER_KEY, "xp"]],
                               on=["season", "gameweek", C.PLAYER_KEY], how="left")
                g = g.merge(arms["gbdt"][s][["season", "gameweek", C.PLAYER_KEY, "xp"]]
                            .rename(columns={"xp": "xp_g"}),
                            on=["season", "gameweek", C.PLAYER_KEY], how="left")
                g = g.merge(leg_fc, on=["season", "gameweek", C.PLAYER_KEY], how="left")
                for c in ("xp", "xp_g", "xp_leg"):
                    g[c] = g[c].fillna(0.0)
                    g.loc[g["fixtures"] <= 0, c] = 0.0
                if tmpl == "raw":
                    g["sig"] = g["xp"]
                else:
                    key = {"gbdt_gw": "xp_g", "legacy_gw": "xp_leg"}.get(tmpl)
                    if tmpl == "self_pooled" and (arm, s) not in pooled:
                        margins.append(np.nan); continue
                    g["sig"] = (g.groupby("gameweek", group_keys=False)
                                 .apply(lambda d: pd.Series(apply_template(
                                     d["xp"].to_numpy(),
                                     pooled[(arm, s)] if tmpl == "self_pooled"
                                     else d[key].to_numpy()), index=d.index)))
                g["sig"] = g["sig"].fillna(0.0)
                g.loc[g["fixtures"] <= 0, "sig"] = 0.0
                r = simulate.simulate_season(g, "sig")
                l = simulate.simulate_season(g, "xp_leg")
                margins.append(float(r.total) - float(l.total) if r and l else np.nan)
            ok = [m for m in margins if np.isfinite(m)]
            mean = float(np.mean(ok)) if ok else float("nan")
            out[(arm, tmpl)] = (margins, mean)
            print(f"{arm:6s}{tmpl:14s}" + "".join(
                f"{m:>10.0f}" if np.isfinite(m) else f"{'n/a':>10}" for m in margins)
                + f"{mean:>9.0f}{sum(1 for m in ok if m > 0):>4}/{len(ok)}", flush=True)
        print(flush=True)

    print("GATE: a template must beat RAW on mean AND win >= as many seasons.", flush=True)
    for arm in ("gbdt", "glm"):
        rawm = out[(arm, "raw")][1]
        raww = sum(1 for m in out[(arm, "raw")][0] if np.isfinite(m) and m > 0)
        for tmpl in ("self_pooled", "gbdt_gw", "legacy_gw"):
            m = out[(arm, tmpl)][1]
            w = sum(1 for x in out[(arm, tmpl)][0] if np.isfinite(x) and x > 0)
            if not np.isfinite(m):
                continue
            print(f"  {arm:5s} {tmpl:12s} mean {m:+7.0f} vs raw {rawm:+7.0f}   "
                  f"wins {w} vs {raww}   "
                  f"{'PASS' if m > rawm and w >= raww else 'FAIL'}", flush=True)
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
