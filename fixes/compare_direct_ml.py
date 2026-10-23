#!/usr/bin/env python3
"""
Would a direct ML model for xP beat the component model?

The notebook forecasts each scoring event (xG, xA, saves, bonus, cards) plus a
minutes hierarchy, then applies the FPL scoring rules analytically. The obvious
alternative is to skip all that and regress realised FPL points on the same
features with a gradient-boosted tree.

This script runs that comparison on the same chronological folds and the same
decision metrics the rest of the package uses, plus a hybrid that gives the
tree the component xP as an extra feature.

    A  component model          the notebook's architecture (v2.6)
    B  direct GBM               HistGradientBoosting on target_fpl_points
    C  hybrid GBM               same tree, plus component xP as a feature

Reading the result
------------------
If B beats A, the component structure is costing accuracy.
If B matches A, the limit is feature signal, not architecture.
If C beats both, the components carry information the raw features do not.

Run:  python compare_direct_ml.py [2024-25|2025-26]
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


def gbm() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=0,
    )


def main(test_season: str = "2025-26") -> None:
    train_seasons = [s for s in V.SEASONS if s < test_season]
    print("=" * 78)
    print(f"Direct ML vs component model — train {train_seasons}, test {test_season}")
    print("=" * 78)

    raw = V.add_history_features(V.add_team_strength(V.load_seasons(mask_coverage=True)))
    train = raw[raw["season"].isin(train_seasons)].copy()
    test = raw[raw["season"].eq(test_season)].copy()
    test = test[test["gameweek"] >= 6].copy()

    V.FEATURE_CLIPS.clear()
    td = train.reindex(columns=V.FEATURES).astype(float).replace([np.inf, -np.inf], np.nan)
    for c in V.FEATURES:
        col = td[c].dropna()
        if len(col) > 100:
            V.FEATURE_CLIPS[c] = (float(col.quantile(0.001)), float(col.quantile(0.999)))

    # ---- A: component model (v2.6) ----
    fc = V.build_forecast(train, test).reset_index(drop=True)
    af = fx.fit_event_conversion_factors(
        train, realised_col="assists", expected_col="expected_assists",
        bounds=fx.ASSIST_CONVERSION_BOUNDS,
    )
    comp = V.components(fc, V.GOAL_POINTS_BASELINE)
    comp = fx.apply_event_conversion(comp, fc, assist_factors=af)
    xp_component = comp["expected_points"].to_numpy(float)

    # ---- B: direct GBM on realised points ----
    Xtr, Xte = V.design(train), V.design(test)
    ytr = pd.to_numeric(train["target_fpl_points"], errors="coerce")
    ok = ytr.notna()
    model_b = gbm().fit(Xtr[ok.to_numpy()], ytr[ok])
    xp_direct = model_b.predict(Xte)

    # ---- C: hybrid, component xP as an extra feature ----
    # The component xP for the TRAINING rows must be built the same way, so
    # refit the component layer on an inner chronological split.
    inner_train = train[train["season"] < train_seasons[-1]]
    inner_test = train[train["season"].eq(train_seasons[-1])]
    if len(inner_train) and len(inner_test):
        fc_in = V.build_forecast(inner_train, inner_test).reset_index(drop=True)
        af_in = fx.fit_event_conversion_factors(
            inner_train, realised_col="assists", expected_col="expected_assists",
            bounds=fx.ASSIST_CONVERSION_BOUNDS,
        )
        comp_in = fx.apply_event_conversion(
            V.components(fc_in, V.GOAL_POINTS_BASELINE), fc_in, assist_factors=af_in
        )
        Xc_tr = V.design(inner_test).copy()
        Xc_tr["component_xp"] = comp_in["expected_points"].to_numpy(float)
        yc_tr = pd.to_numeric(inner_test["target_fpl_points"], errors="coerce")
        ok_c = yc_tr.notna()
        Xc_te = Xte.copy()
        Xc_te["component_xp"] = xp_component
        model_c = gbm().fit(Xc_tr[ok_c.to_numpy()], yc_tr[ok_c])
        xp_hybrid = model_c.predict(Xc_te)
        n_inner = int(ok_c.sum())
    else:
        xp_hybrid, n_inner = None, 0

    variants = [
        ("A component model (v2.6)", xp_component),
        ("B direct GBM", xp_direct),
    ]
    if xp_hybrid is not None:
        variants.append((f"C hybrid GBM (+component xP, {n_inner:,} train rows)", xp_hybrid))

    rows = [V.evaluate(fc, pd.Series(p, index=fc.index), label) for label, p in variants]
    res = pd.DataFrame(rows).set_index("variant")
    print(f"\ntrain rows {int(ok.sum()):,}   test rows {len(test):,}\n")
    print(res.round(4).to_string())

    base = res.iloc[0]
    print("\ndeltas vs the component model:")
    for label in res.index[1:]:
        d = res.loc[label] - base
        print(f"  {label:46s} MAE {d['MAE']:+.4f}  spearman {d['spearman']:+.4f}  "
              f"ndcg {d['ndcg@10']:+.4f}  top10 {d['top10_pts']:+.3f}")

    print("\nBIAS BY PREDICTED-xP BUCKET")
    for label, p in variants:
        print(f"\n-- {label} --")
        print(V.bias_table(fc, pd.Series(p, index=fc.index)).round(3).to_string())

    print("\n" + "=" * 78)
    print("Interpretation: the component model is correct by construction for the")
    print("scoring rules and survives rule changes (defensive contribution, BPS)")
    print("that a direct model trained on older seasons cannot see at all.")
    print("A direct model is worth keeping as a BENCHMARK, not a replacement,")
    print("unless it wins clearly on the decision metrics above.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2025-26")
