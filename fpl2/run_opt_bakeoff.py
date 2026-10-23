#!/usr/bin/env python3
"""
O52 -- race the OPTIMISER the way O51 races the model.

The forecast is only half the pipeline. Given a signal, four decisions follow:
which 15 to own, which 11 to start, who wears the armband, and when to
transfer. This file races the ARMBAND, which is the one the user has named as
failing and the one metric 1 can measure at full power.

THE DESIGN POINT. Every rule is scored on a FIXED signal with a FIXED XI --
`optimise.xi_value(..., captain_score=rule)` picks the eleven on the reference
and lets only the armband move. So nothing here is confounded with the model
bake-off: a rule wins or loses on captaincy alone.

Ledger history, because none of these are new ideas:
  F9  variance-adjusted captain rules -- rejected at G=3 on the instrument
      S37 later showed was unfit, and flagged in the ledger as "highest-value
      reopen on the board (S8, O12)". This is that reopen.
  S8  the model's entire margin over dumb baselines came from captaincy.
  S20 the model captained Bruno for 1,1,2 while the crowd captained Haaland;
      six routes to fixing it were closed.

`cap_premium` -- armband on the most expensive player in the XI -- is in here
because it is the only rule that would have described the crowd's Haaland
behaviour, and it has never been measured on this project.
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
SIGNAL = os.environ.get("OPT_SIGNAL", "m_full_us")    # XI + reference armband
# Outputs are suffixed by the signal. The first run of this file used
# `m_full_fpl`, which M26 showed forecasts zero goals for every player on five
# of eight folds -- so every rule that consults a SECOND model scored a model
# difference, not a rule difference. Both runs are kept; the comparison
# between them is what separates the two.
SUF = "_" + SIGNAL


def points_dispersion(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Strictly-lagged sd of the last 5 gameweeks' points, per player.

    Built here rather than read from the feature frame because no dispersion
    feature was carried through phase A. Same construction as
    `features._lagged_roll_sd`: row t sees rows < t only.
    """
    d = panel.sort_values(["player_code", "season", "gameweek"]).copy()
    g = d.groupby("player_code", sort=False)["points"]
    d["z_points_sd_r5"] = g.transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).std())
    return d[["season", "gameweek", "player_code", "z_points_sd_r5"]]


def build_rules(g: pd.DataFrame) -> dict:
    """arm name -> the column its armband maximises. All strictly lagged."""
    xp = g[SIGNAL].to_numpy(float)
    sd = np.nan_to_num(g["z_points_sd_r5"].to_numpy(float), nan=0.0)
    p60 = np.nan_to_num(g.get("z_p_sixty", pd.Series(1.0, index=g.index))
                        .to_numpy(float), nan=0.0)
    out = {"cap_argmax": xp}                       # REFERENCE: argmax(xp)
    out["cap_upside_03"] = xp + 0.3 * sd
    out["cap_upside_05"] = xp + 0.5 * sd
    out["cap_upside_10"] = xp + 1.0 * sd
    out["cap_safe"] = xp * p60                     # penalise minutes risk
    out["cap_premium"] = g["cost"].to_numpy(float)  # the crowd's Haaland rule
    if "b_form5" in g.columns:
        out["cap_form5"] = g["b_form5"].to_numpy(float)
    # The quantile head is a DIFFERENT MODEL as well as a different functional,
    # so on its own it cannot separate "predicting the tail is the right
    # objective for an armband" from "XGBoost is a better model". cap_xgb_mean
    # is the control: same model family, conditional mean. Then
    #   cap_xgb_mean vs cap_argmax  = the model
    #   cap_q80      vs cap_xgb_mean = the functional
    if 'd_xgb' in g.columns:
        out['cap_xgb_mean'] = g['d_xgb'].to_numpy(float)
    if 'd_xgb_q80' in g.columns:
        out['cap_q80'] = g['d_xgb_q80'].to_numpy(float)
    return out


def main():
    t0 = time.time()
    panel, grids, _ = load()
    disp = points_dispersion(panel)
    print(f"[{time.time()-t0:4.0f}s] loaded; signal={SIGNAL}, K={K}", flush=True)

    rows = []
    for s in HOLDOUTS:
        if s not in grids:
            continue
        g = grids[s].merge(disp, on=["season", "gameweek", "player_code"],
                           how="left")
        rules = build_rules(g)
        for name, vals in rules.items():
            g["_r_" + name] = np.nan_to_num(vals, nan=0.0)
        squads = measure.neutral_squads(g, k=K, seed=1)
        for si, squad in enumerate(squads):
            held = g[g["player_code"].isin(squad)]
            for gw, wk in held.groupby("gameweek", sort=True):
                for name in rules:
                    v = optimise.xi_value(wk, SIGNAL, actual="points",
                                          captain_score="_r_" + name)
                    if np.isfinite(v):
                        rows.append({"season": s, "squad": si, "gameweek": int(gw),
                                     "rule": name, "value": float(v)})
        pd.DataFrame(rows).to_pickle(SCRATCH / f"opt_captain{SUF}.pkl")
        print(f"   [{time.time()-t0:5.0f}s] {s} done ({len(squads)} squads)",
              flush=True)

    M = pd.DataFrame(rows)
    ref = M[M["rule"].eq("cap_argmax")].set_index(["season", "squad", "gameweek"])["value"]
    print("\n" + "=" * 78)
    print(f"CAPTAIN RULES vs cap_argmax -- XI fixed on {SIGNAL}, armband only")
    print("per season (x38), season-clustered, bar |t| > 2.36 (G=8)")
    print("=" * 78)
    print(f"{'rule':<16}{'mean':>9}{'t':>8}{'p':>8}{'95% CI':>20}{'MDE':>7}  seasons")
    out = []
    for name in sorted(M["rule"].unique()):
        if name == "cap_argmax":
            continue
        cur = M[M["rule"].eq(name)].set_index(["season", "squad", "gameweek"])["value"]
        j = pd.concat({"a": cur, "r": ref}, axis=1).dropna()
        d = (j["a"] - j["r"]).to_numpy(float) * 38.0
        gg = j.index.get_level_values("season").to_numpy()
        m, se, tt, pp, lo, hi = cluster(d, gg)
        wins = sum(1 for _, dd in pd.Series(d).groupby(gg) if dd.mean() > 0)
        out.append({"rule": name, "mean": m, "t": tt, "p": pp, "lo": lo,
                    "hi": hi, "mde": 2.8 * se, "wins": wins})
        print(f"{name:<16}{m:>+9.1f}{tt:>8.2f}{pp:>8.3f}"
              f"{f'[{lo:+.1f}, {hi:+.1f}]':>20}{2.8*se:>7.1f}  {wins}/8", flush=True)
    pd.DataFrame(out).to_pickle(SCRATCH / f"opt_captain_res{SUF}.pkl")

    print("\n" + "=" * 78)
    print("absolute season equivalent per rule")
    print("=" * 78)
    tot = (M.groupby(["rule", "season"])["value"].mean() * 38).unstack()
    tot["MEAN"] = tot.mean(axis=1)
    print(tot.round(0).sort_values("MEAN", ascending=False).to_string(), flush=True)
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
