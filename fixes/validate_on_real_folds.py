#!/usr/bin/env python3
"""
Definitive evaluation — on the notebook's OWN cached walk-forward forecasts.

Every earlier experiment in this package was run on an independent
re-implementation. That harness mispredicted twice: it said the assist
conversion was worth +2.2 top-10 points per gameweek (the real pipeline lost
0.3-1.7), and that dropping the wildcard structural gate was worth +40 points a
season (it reversed once the contaminated forecasts were removed). Both failures
came from the same cause: the harness is a weaker forecaster, so corrections
that boost attacking xP help it and over-correct a stronger model.

This script removes that weakness entirely by reading
`data/walk_forward_fold_cache_base_v2_4_0/*.pkl.gz` -- the notebook's real
out-of-sample fold forecasts, with its own `expected_points`.

It evaluates every candidate on three metrics:

  NDCG@10 / top-10 realised   what the project has always used
  BEST LEGAL XI points        what actually decides a gameweek: 1 GK, 3-5 DEF,
                              2-5 MID, 1-3 FWD, captain doubled

The third matters because the two disagree. A model can win top-10 by ranking
ten forwards highest and still be unable to field a team. The v2.4 layer was
rejected on NDCG and top-10, which is why it is re-tested here.

Candidates
----------
  base                 the notebook's expected_points, unchanged
  + assist conversion  F1, the correction that failed on the real pipeline
  + v2.4 upper tail    the rejected layer, re-tested on best-XI

Run:  python validate_on_real_folds.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fpl_xp_fixes as fx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "walk_forward_fold_cache_base_v2_4_0"
V24_ARTIFACT = ROOT / "data" / "targeted_forecast_layer_v2_4_0.json"
XI_BOUNDS = {"DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}


def load_folds() -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(glob.glob(str(CACHE / "*.pkl.gz"))):
        payload = pd.read_pickle(path, compression="gzip")
        frame = payload["bundle"]["holdout_expected_points"].copy()
        out[str(payload["holdout_season"])] = frame
    return out


# ----------------------------------------------------------------- candidates

def candidate_assist_conversion(frame: pd.DataFrame, develop: pd.DataFrame) -> np.ndarray:
    """F1: rescale projected assists onto the realised-assist scale."""
    factors = fx.fit_event_conversion_factors(
        develop, realised_col="assists", expected_col="expected_assists",
        bounds=fx.ASSIST_CONVERSION_BOUNDS,
    )
    pos = frame["model_position"]
    mult = (pos.map({k: v for k, v in factors.items() if k != "__pooled__"})
            .astype(float).fillna(factors.get("__pooled__", 1.0)))
    delta = 3.0 * pd.to_numeric(frame["projected_assists"], errors="coerce").fillna(0.0) * (mult - 1.0)
    return (pd.to_numeric(frame["expected_points"], errors="coerce") + delta).to_numpy(float)


def candidate_v24_upper_tail(frame: pd.DataFrame, season: str) -> np.ndarray | None:
    """The rejected v2.4 upper-tail correction, reproducing _v24_upper_tail_design.

    Uses THIS fold's own fitted coefficients, so the test stays leakage-safe.
    The design is the two ramp weights alone -- an earlier version of this
    function multiplied them by xP, which inflated the adjustment several-fold
    and produced a spuriously large gain.
    """
    if not V24_ARTIFACT.exists():
        return None
    art = json.loads(V24_ARTIFACT.read_text(encoding="utf-8"))
    bundle = (art.get("fold_candidate_bundles") or {}).get(season) or {}
    coeffs = ((bundle.get("upper_tail") or {}).get("coefficients"))
    if not coeffs or len(coeffs) != 2:
        return None

    xp = pd.to_numeric(frame["expected_points"], errors="coerce").fillna(0.0)
    price = pd.to_numeric(frame.get("price_at_collection"), errors="coerce").fillna(0.0)
    mins = pd.to_numeric(frame.get("expected_minutes"), errors="coerce").fillna(0.0)
    xp_w = ((xp - 2.5) / (5.5 - 2.5)).clip(0.0, 1.0)        # RAMP_START/FULL
    price_w = ((price - 6.5) / (10.0 - 6.5)).clip(0.0, 1.0)  # PRICE_RAMP_START/FULL
    design = np.column_stack([xp_w.to_numpy(), price_w.to_numpy()])
    adj = np.clip(design @ np.asarray(coeffs, float), -1.0, 1.0)   # MAX_ABS_ADJUSTMENT
    adj = np.where((mins >= 45.0).to_numpy(), adj, 0.0)            # MIN_EXPECTED_MINUTES
    return (xp.to_numpy(float) + adj)


# ------------------------------------------------------------------- metrics

def best_xi_points(group: pd.DataFrame, pred: str) -> float:
    by = {p: group[group["model_position"].eq(p)].nlargest(6, pred) for p in
          ("GK", "DEF", "MID", "FWD")}
    if by["GK"].empty:
        return np.nan
    gk = by["GK"].iloc[0]
    best, best_pred = None, -np.inf
    for d in range(*(XI_BOUNDS["DEF"][0], XI_BOUNDS["DEF"][1] + 1)):
        for m in range(*(XI_BOUNDS["MID"][0], XI_BOUNDS["MID"][1] + 1)):
            f = 10 - d - m
            if not (XI_BOUNDS["FWD"][0] <= f <= XI_BOUNDS["FWD"][1]):
                continue
            if len(by["DEF"]) < d or len(by["MID"]) < m or len(by["FWD"]) < f:
                continue
            xi = pd.concat([by["DEF"].head(d), by["MID"].head(m), by["FWD"].head(f)])
            tot = float(xi[pred].sum()) + float(gk[pred])
            if tot > best_pred:
                best_pred, best = tot, xi
    if best is None:
        return np.nan
    squad = pd.concat([best, gk.to_frame().T])
    cap = squad.loc[squad[pred].astype(float).idxmax()]
    return float(squad["target_fpl_points"].sum()) + float(cap["target_fpl_points"])


def ndcg(pred, actual, k=10):
    order = np.argsort(-np.asarray(pred))
    gain = np.clip(np.asarray(actual)[order][:k], 0, None)
    disc = 1.0 / np.log2(np.arange(2, len(gain) + 2))
    ideal = np.clip(np.sort(actual)[::-1][:k], 0, None)
    den = (ideal * disc[: len(ideal)]).sum()
    return float((gain * disc).sum() / den) if den > 0 else np.nan


def score(frame: pd.DataFrame, pred: np.ndarray, label: str) -> dict:
    work = frame.assign(_p=pred)
    work = work[pd.to_numeric(work["target_fpl_points"], errors="coerce").notna()]
    sp, nd, top, xi = [], [], [], []
    for _, g in work.groupby("gameweek"):
        if len(g) < 50:
            continue
        a = pd.to_numeric(g["target_fpl_points"], errors="coerce").to_numpy(float)
        p = g["_p"].to_numpy(float)
        sp.append(spearmanr(p, a).statistic)
        nd.append(ndcg(p, a))
        top.append(a[np.argsort(-p)][:10].sum())
        xi.append(best_xi_points(g, "_p"))
    return {"variant": label,
            "MAE": float(np.mean(np.abs(work["_p"] - pd.to_numeric(
                work["target_fpl_points"], errors="coerce")))),
            "spearman": float(np.nanmean(sp)),
            "ndcg@10": float(np.nanmean(nd)),
            "top10_pts": float(np.nanmean(top)),
            "best_XI_pts": float(np.nanmean(xi))}


def main() -> None:
    folds = load_folds()
    if not folds:
        print(f"No cached folds found under {CACHE}")
        return
    print("=" * 84)
    print("DEFINITIVE EVALUATION — on the notebook's own cached fold forecasts")
    print("=" * 84)

    all_rows = []
    for season, frame in folds.items():
        payload = pd.read_pickle(
            CACHE / f"{season}.pkl.gz", compression="gzip")
        develop = payload["bundle"]["development_points_data"]

        base = pd.to_numeric(frame["expected_points"], errors="coerce").to_numpy(float)
        cands = [("base (current production)", base),
                 ("+ assist conversion (F1)", candidate_assist_conversion(frame, develop))]
        v24 = candidate_v24_upper_tail(frame, season)
        if v24 is not None:
            cands.append(("+ v2.4 upper tail (rejected)", v24))
            # Do the two stack? v2.4's ramp is a function of expected_points,
            # which F1 shifts, so the combination is not simply additive.
            f1_frame = frame.copy()
            f1_frame["expected_points"] = candidate_assist_conversion(frame, develop)
            both = candidate_v24_upper_tail(f1_frame, season)
            if both is not None:
                cands.append(("+ F1 AND v2.4 (combined)", both))
        else:
            print(f"    (no v2.4 coefficients fitted for {season} - skipped)")

        rows = [score(frame, p, lab) for lab, p in cands]
        table = pd.DataFrame(rows).set_index("variant")
        print(f"\n--- holdout {season}   ({len(frame):,} rows) ---")
        print(table.round(4).to_string())
        d = table - table.loc["base (current production)"]
        print("\n  deltas vs base:")
        print(d.drop(index="base (current production)").round(4).to_string())
        for r in rows:
            r["season"] = season
        all_rows.extend(rows)

    print("\n" + "=" * 84)
    print("PAIRED DELTAS vs base, averaged over the folds where each applies")
    print("=" * 84)
    cols = ["MAE", "spearman", "ndcg@10", "top10_pts", "best_XI_pts"]
    df = pd.DataFrame(all_rows)
    base = df[df["variant"].eq("base (current production)")].set_index("season")[cols]
    # v2.4 has fitted coefficients for only some folds, so levels must never be
    # averaged across different fold sets -- pair each variant to ITS OWN folds.
    out = []
    for variant, grp in df[~df["variant"].eq("base (current production)")].groupby("variant"):
        g = grp.set_index("season")[cols]
        delta = (g - base.loc[g.index]).mean()
        delta["folds"] = f"{len(g)} ({', '.join(sorted(g.index))})"
        delta.name = variant
        out.append(delta)
    print(pd.DataFrame(out)[["folds"] + cols].round(4).to_string())
    print("\nIf top10_pts and best_XI_pts disagree in sign, best_XI_pts is the one")
    print("that reflects an actual gameweek decision.")


if __name__ == "__main__":
    main()
