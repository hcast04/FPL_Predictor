#!/usr/bin/env python3
"""
O54 -- does BLENDING recent form into the armband help?

User's hypothesis, from watching the live season: the model forecast
B.Fernandes 5.7-7.1 every week while he returned 2,23,2,2,2,1,1, and raised
him to 7.1 AFTER the single haul. It does not respond to six weeks of
nothing. So: weight form more, especially for captaincy and for premium
players.

WHY THIS IS NOT `cap_form5`, WHICH IS ALREADY REFUTED. That arm captained by
form ALONE and measured -27.6, p 0.022, 1/8 seasons. This is a BLEND -- the
model's xp stays the base and form perturbs the ordering. A blend can only be
evaluated on its own because the refuted arm is its lambda -> infinity limit,
and the interesting region is small lambda.

Both signals are z-scored WITHIN (season, gameweek) before blending, because
xp is in points and form is in points-per-game and a raw sum would just be
whichever has the larger spread.

  cap_blend_<L>       argmax( z(xp) + L * z(points_r5) )
  cap_blendprem_<L>   the same, but the form term applies ONLY to players
                      priced at or above `PREMIUM` -- the user's second
                      hypothesis, that form matters more for the expensive
                      players you would actually captain

Reference `cap_argmax` on the REPAIRED signal (R37: an optimiser change
measured on a broken signal inverts).
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, measure, optimise                      # noqa: E402
from run_bakeoff_score import cluster, load, HOLDOUTS                # noqa: E402

SCRATCH = Path(os.environ.get("BAKE_DIR", "/tmp/bakeoff"))
K = int(os.environ.get("KOPT", "10"))
SIGNAL = os.environ.get("OPT_SIGNAL", "m_full_us")
LAMBDAS = [0.1, 0.25, 0.5, 1.0]
PREMIUM = 9.0            # GBPm; roughly the price of a captainable asset


def _z(df, col):
    """z-score within (season, gameweek), over players with a fixture."""
    g = df.groupby(["season", "gameweek"])[col]
    mu, sd = g.transform("mean"), g.transform("std")
    return ((df[col] - mu) / sd.replace(0, np.nan)).fillna(0.0)


def main():
    t0 = time.time()
    _, grids, _ = load()
    print(f"[{time.time()-t0:4.0f}s] signal={SIGNAL}, K={K}, premium>=£{PREMIUM}m",
          flush=True)
    rows = []
    for s in HOLDOUTS:
        if s not in grids:
            continue
        g = grids[s].copy()
        pl = g["fixtures"] > 0
        form_col = "z_points_r5" if "z_points_r5" in g.columns else "b_form5"
        g["_zxp"] = _z(g, SIGNAL)
        g["_zform"] = _z(g, form_col)
        g.loc[~pl, ["_zxp", "_zform"]] = -9.0
        rules = {"cap_argmax": g[SIGNAL].to_numpy(float)}
        prem = (g["cost"].to_numpy(float) >= PREMIUM)
        for L in LAMBDAS:
            rules[f"cap_blend_{L}"] = (g["_zxp"] + L * g["_zform"]).to_numpy(float)
            rules[f"cap_blendprem_{L}"] = (
                g["_zxp"] + L * np.where(prem, g["_zform"], 0.0)).to_numpy(float)
        for n, v in rules.items():
            g["_r_" + n] = np.nan_to_num(v, nan=-9.0)
        squads = measure.neutral_squads(g, k=K, seed=1)
        for si, squad in enumerate(squads):
            held = g[g[C.PLAYER_KEY].isin(squad)]
            for gw, wk in held.groupby("gameweek", sort=True):
                for n in rules:
                    v = optimise.xi_value(wk, SIGNAL, actual="points",
                                          captain_score="_r_" + n)
                    if np.isfinite(v):
                        rows.append({"season": s, "squad": si, "gameweek": int(gw),
                                     "rule": n, "value": float(v)})
        pd.DataFrame(rows).to_pickle(SCRATCH / "opt_blend.pkl")
        print(f"   [{time.time()-t0:5.0f}s] {s}", flush=True)

    M = pd.DataFrame(rows)
    ref = M[M["rule"].eq("cap_argmax")].set_index(["season", "squad", "gameweek"])["value"]
    print("\n" + "=" * 78)
    print(f"FORM-BLENDED ARMBAND vs argmax(xp) -- XI fixed on {SIGNAL}")
    print("per season (x38), season-clustered, bar |t| > 2.36 (G=8)")
    print("=" * 78)
    print(f"{'rule':<20}{'mean':>9}{'t':>8}{'p':>8}{'95% CI':>20}{'MDE':>7}  seasons")
    out = []
    for n in sorted(M["rule"].unique()):
        if n == "cap_argmax":
            continue
        cur = M[M["rule"].eq(n)].set_index(["season", "squad", "gameweek"])["value"]
        j = pd.concat({"a": cur, "r": ref}, axis=1).dropna()
        d = (j["a"] - j["r"]).to_numpy(float) * 38.0
        gg = j.index.get_level_values("season").to_numpy()
        m, se, tt, pp, lo, hi = cluster(d, gg)
        w = sum(1 for _, dd in pd.Series(d).groupby(gg) if dd.mean() > 0)
        out.append({"rule": n, "mean": m, "t": tt, "p": pp, "mde": 2.8 * se, "wins": w})
        print(f"{n:<20}{m:>+9.1f}{tt:>8.2f}{pp:>8.3f}"
              f"{f'[{lo:+.1f}, {hi:+.1f}]':>20}{2.8*se:>7.1f}  {w}/8", flush=True)
    pd.DataFrame(out).to_pickle(SCRATCH / "opt_blend_res.pkl")
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
