#!/usr/bin/env python3
"""
Blended forecast (component + GBM), and what every variant is worth in
season-end points.

Part 1 — the blend
------------------
The component model under-predicts the top of the distribution; the gradient
boosted model over-predicts it (trees cannot extrapolate). Those errors have
opposite signs, so a blend should cancel part of both:

    blend = w * component_xp + (1 - w) * gbm_xp

w is fitted on an inner chronological split of the TRAINING seasons only, never
on the fold being scored.

ACCEPTANCE GATES (declared before the first run, per the project audit rules).
The blend is accepted only if ALL hold in BOTH chronological folds:

  G1  upper-tail (5+ xP) |bias| is smaller than BOTH parents
  G2  MAE is no worse than the component model
  G3  NDCG@10 is no worse than the component model by more than 0.002
  G4  mean top-10 realised points per gameweek beats the component model
      by at least 0.25
  G5  premium (price >= 8.0) MID/FWD Spearman is no worse than the component
      model by more than 0.003

Part 2 — season totals
----------------------
Per-gameweek metrics like NDCG and "top-10 realised points" are hard to read.
This converts each variant into season-end points by picking, each gameweek,
the best legal starting XI by predicted xP (1 GK, 3-5 DEF, 3-5 MID, 1-3 FWD)
and doubling the captain, then summing realised points over the season.

IMPORTANT: this is deliberately UNCONSTRAINED — no budget, no 3-per-club, no
transfer limit, no price changes. It measures forecast ranking quality in units
of season points. It is NOT a predicted FPL score. The notebook's own
greedy_1gw_no_hits policy backtest (mean 2196.7 realised points) remains the
realistic anchor.

Run:  python blend_and_season.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fpl_xp_fixes as fx  # noqa: E402
import validate_xp_fixes as V  # noqa: E402

FORMATIONS = [(3, 5, 2), (3, 4, 3), (4, 5, 1), (4, 4, 2), (4, 3, 3),
              (5, 4, 1), (5, 3, 2), (5, 2, 3)]


def gbm() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error", max_iter=400, learning_rate=0.06,
        max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=0,
    )


def fit_clips(train: pd.DataFrame) -> None:
    V.FEATURE_CLIPS.clear()
    td = train.reindex(columns=V.FEATURES).astype(float).replace([np.inf, -np.inf], np.nan)
    for c in V.FEATURES:
        col = td[c].dropna()
        if len(col) > 100:
            V.FEATURE_CLIPS[c] = (float(col.quantile(0.001)), float(col.quantile(0.999)))


def component_and_gbm(train: pd.DataFrame, test: pd.DataFrame):
    """Return (forecast frame, baseline xP, v2.6 xP, gbm xP)."""
    fit_clips(train)
    fc = V.build_forecast(train, test).reset_index(drop=True)
    af = fx.fit_event_conversion_factors(
        train, realised_col="assists", expected_col="expected_assists",
        bounds=fx.ASSIST_CONVERSION_BOUNDS,
    )
    base = V.components(fc, V.GOAL_POINTS_BASELINE)
    fixed = fx.apply_event_conversion(base.copy(), fc, assist_factors=af)
    y = pd.to_numeric(train["target_fpl_points"], errors="coerce")
    ok = y.notna()
    model = gbm().fit(V.design(train)[ok.to_numpy()], y[ok])
    return (fc,
            base["expected_points"].to_numpy(float),
            fixed["expected_points"].to_numpy(float),
            model.predict(V.design(test)))


def fit_blend_weight(train_seasons: list[str], raw: pd.DataFrame) -> float:
    """Fit w on an inner chronological split of the training seasons only."""
    if len(train_seasons) < 2:
        return 0.5
    inner_tr = raw[raw["season"].isin(train_seasons[:-1])]
    inner_te = raw[raw["season"].eq(train_seasons[-1])]
    inner_te = inner_te[inner_te["gameweek"] >= 6]
    if inner_tr.empty or inner_te.empty:
        return 0.5
    fc, _, comp, g = component_and_gbm(inner_tr, inner_te)
    actual = fc["actual_points"].to_numpy(float)
    best_w, best = 0.5, np.inf
    for w in np.arange(0.0, 1.01, 0.05):
        mae = np.mean(np.abs(w * comp + (1 - w) * g - actual))
        if mae < best:
            best, best_w = mae, float(w)
    return best_w


def tail_bias(fc: pd.DataFrame, xp: np.ndarray, lo: float = 5.0) -> float:
    m = xp >= lo
    if m.sum() < 5:
        return np.nan
    return float(np.mean(xp[m] - fc["actual_points"].to_numpy(float)[m]))


def premium_spearman(fc: pd.DataFrame, xp: np.ndarray) -> float:
    m = (fc["model_position"].isin(["MID", "FWD"]).to_numpy()
         & (pd.to_numeric(fc["price"], errors="coerce").fillna(0).to_numpy() >= 8.0))
    out = []
    for _, g in fc.assign(_p=xp, _m=m)[lambda f: f["_m"]].groupby("gameweek"):
        if len(g) >= 8:
            out.append(spearmanr(g["_p"], g["actual_points"]).statistic)
    return float(np.nanmean(out)) if out else np.nan


def best_xi_points(group: pd.DataFrame, pred: str) -> float:
    """Best legal XI by predicted xP, captain doubled, scored on actuals."""
    by = {p: group[group["model_position"].eq(p)].nlargest(6, pred)
          for p in ("GK", "DEF", "MID", "FWD")}
    if by["GK"].empty:
        return 0.0
    gk = by["GK"].iloc[0]
    best, best_pred = None, -np.inf
    for d, m, f in FORMATIONS:
        if len(by["DEF"]) < d or len(by["MID"]) < m or len(by["FWD"]) < f:
            continue
        xi = pd.concat([by["DEF"].head(d), by["MID"].head(m), by["FWD"].head(f)])
        total = float(xi[pred].sum()) + float(gk[pred])
        if total > best_pred:
            best_pred, best = total, xi
    if best is None:
        return 0.0
    squad = pd.concat([best, gk.to_frame().T])
    captain = squad.loc[squad[pred].astype(float).idxmax()]
    return float(squad["actual_points"].sum()) + float(captain["actual_points"])


def weekly_xi_points(fc: pd.DataFrame, xp: np.ndarray) -> np.ndarray:
    work = fc.assign(_p=xp)
    return np.array([best_xi_points(g, "_p")
                     for _, g in work.groupby("gameweek") if len(g) >= 50])


def season_total(fc: pd.DataFrame, xp: np.ndarray, label: str) -> dict:
    weeks = weekly_xi_points(fc, xp)
    return {"variant": label, "gameweeks": len(weeks),
            "pts/GW": float(np.mean(weeks)), "sd/GW": float(np.std(weeks)),
            "projected_38GW": float(np.mean(weeks) * 38)}


def paired_table(fc: pd.DataFrame, variants: list, reference: str) -> pd.DataFrame:
    """Paired weekly comparison against a reference variant.

    The variants share a player pool and mostly pick the same players, so the
    per-gameweek SD of a single variant badly overstates the uncertainty on a
    DIFFERENCE. This pairs them week by week instead.
    """
    weekly = {lab: weekly_xi_points(fc, p) for lab, p in variants}
    ref = weekly[reference]
    rows = []
    for lab, w in weekly.items():
        if lab == reference:
            continue
        d = w - ref
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        rows.append({
            "vs " + reference: lab,
            "delta pts/GW": float(d.mean()),
            "sd of delta": float(d.std(ddof=1)),
            "SE": float(se),
            "t": float(d.mean() / se) if se and se > 0 else np.nan,
            "delta over 38GW": float(d.mean() * 38),
            "95% CI over 38GW": f"[{38*(d.mean()-1.96*se):+.0f}, {38*(d.mean()+1.96*se):+.0f}]",
        })
    return pd.DataFrame(rows).set_index("vs " + reference)


def run_fold(test_season: str, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_seasons = [s for s in V.SEASONS if s < test_season]
    train = raw[raw["season"].isin(train_seasons)].copy()
    test = raw[raw["season"].eq(test_season)].copy()
    test = test[test["gameweek"] >= 6].copy()

    w = fit_blend_weight(train_seasons, raw)
    fc, xp_base, xp_v26, xp_gbm = component_and_gbm(train, test)
    xp_blend = w * xp_v26 + (1 - w) * xp_gbm

    variants = [("component v2.5 (baseline)", xp_base),
                ("component v2.6", xp_v26),
                ("direct GBM", xp_gbm),
                (f"blend w={w:.2f}", xp_blend)]

    metrics = pd.DataFrame([
        {**V.evaluate(fc, pd.Series(p, index=fc.index), lab),
         "tail_bias_5+": tail_bias(fc, p),
         "premium_rho": premium_spearman(fc, p)}
        for lab, p in variants
    ]).set_index("variant")

    seasons = pd.DataFrame([season_total(fc, p, lab) for lab, p in variants]).set_index("variant")
    # reference points
    seasons.loc["-- oracle (perfect hindsight)"] = season_total(
        fc, fc["actual_points"].to_numpy(float), "oracle")
    seasons.loc["-- random legal XI"] = season_total(
        fc, np.random.default_rng(0).random(len(fc)), "random")

    comp = metrics.loc["component v2.6"]
    gbmr = metrics.loc["direct GBM"]
    bl = metrics.loc[f"blend w={w:.2f}"]
    gates = {
        "G1 tail |bias| < both parents":
            abs(bl["tail_bias_5+"]) < min(abs(comp["tail_bias_5+"]), abs(gbmr["tail_bias_5+"])),
        "G2 MAE <= component": bl["MAE"] <= comp["MAE"],
        "G3 NDCG@10 >= component - 0.002": bl["ndcg@10"] >= comp["ndcg@10"] - 0.002,
        "G4 top-10 >= component + 0.25": bl["top10_pts"] >= comp["top10_pts"] + 0.25,
        "G5 premium rho >= component - 0.003": bl["premium_rho"] >= comp["premium_rho"] - 0.003,
    }
    return metrics, seasons, gates, fc, variants


def main() -> None:
    raw = V.add_history_features(V.add_team_strength(V.load_seasons(mask_coverage=True)))
    all_gates = {}
    for season in ("2024-25", "2025-26"):
        print("=" * 80)
        print(f"FOLD — held out {season}")
        print("=" * 80)
        metrics, seasons, gates, fc, variants = run_fold(season, raw)
        print("\nForecast metrics")
        print(metrics.round(4).to_string())
        print("\nSeason-end points — best legal XI each gameweek, captain doubled")
        print("(UNCONSTRAINED: no budget, no 3-per-club, no transfer limit)")
        print(seasons.round(1).to_string())
        print("\nPaired weekly comparison vs the current production model")
        print(paired_table(fc, variants, "component v2.5 (baseline)").round(3).to_string())

        print("\nAcceptance gates for the blend")
        for n, ok in gates.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {n}")
        all_gates[season] = gates
        print()

    print("=" * 80)
    every = all(all(g.values()) for g in all_gates.values())
    print(f"BLEND VERDICT across both folds: {'ACCEPT' if every else 'REJECT'}")
    for season, g in all_gates.items():
        failed = [n for n, ok in g.items() if not ok]
        print(f"  {season}: {'all gates pass' if not failed else 'failed -> ' + '; '.join(failed)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
