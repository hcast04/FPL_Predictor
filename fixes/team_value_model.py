#!/usr/bin/env python3
"""
Team value: is it worth modelling, and how much is it worth in points?

The notebook tracks purchase prices and selling values for budget accounting,
but it does not model FUTURE price changes, so it has no notion of building
team value. This quantifies the whole chain:

  PART A  What is a pound actually worth?
          Solve the exact squad MILP at a range of budgets and read off the
          extra expected points per extra GBP 1.0m. This is the conversion
          factor everything else depends on, and it is measured rather than
          assumed.

  PART B  Are price changes predictable?
          FPL moves a price by GBP 0.1m off net transfer flow. Fit a model for
          P(rise) and P(fall) next gameweek from transfers_balance relative to
          ownership, and score it out of sample.

  PART C  What is a value-aware policy worth?
          Simulate a season of greedy transfers where near-tied candidates are
          broken by predicted price movement, and measure the team value
          accumulated versus the value-blind policy. Convert through Part A.

ACCEPTANCE GATE (declared before the first run): a value-aware tiebreak is
worth recommending only if it accumulates positive extra team value in BOTH
held-out seasons AND the converted points gain exceeds the xP sacrificed by
breaking ties on price instead of on xP.

Run:  python team_value_model.py [2024-25 2025-26]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import squad_optimizer as SO  # noqa: E402
import validate_chip_timing as C  # noqa: E402
import validate_xp_fixes as V  # noqa: E402

TIE_BAND = 0.15          # xP within which two candidates count as tied


# ------------------------------------------------------------------- Part A

def value_of_budget(panel: pd.DataFrame, gameweeks: list[int],
                    budgets=(960, 980, 1000, 1020, 1040)) -> pd.DataFrame:
    """
    Exact optimal squad xP as a function of budget.

    Two things matter for this to measure what it claims:

    * The candidate pool must be wide enough that extra budget still has
      somewhere to go. With the default pruning (top 32 per position) the
      objective saturates around GBP 105m and the marginal value collapses to
      zero -- that is the pool running out, not real diminishing returns.
    * The budget range must be realistic. FPL squads run about GBP 100-104m, so
      sampling out to 115m measures a region no manager occupies.
    """
    rows = []
    for b in budgets:
        _, obj = SO.optimise_squad(panel, gameweeks, budget=int(b),
                                   top_per_position=70, cheap_per_position=20)
        rows.append({"budget_m": b / 10.0, "horizon_xp": obj})
    out = pd.DataFrame(rows)
    out["xp_per_gw"] = out["horizon_xp"] / len(gameweeks)
    out["xp_per_gw_per_1m"] = out["xp_per_gw"].diff() / out["budget_m"].diff()
    return out


# ------------------------------------------------------------------- Part B

PRICE_FEATURES = ["tb_share", "tb_share_l3", "selected_share", "value",
                  "form_proxy", "gameweek"]


def build_price_frame(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.sort_values(["season", "element", "gameweek"]).copy()
    for c in ("value", "transfers_balance", "selected", "total_points"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    g = d.groupby(["season", "element"], sort=False)
    d["next_value"] = g["value"].shift(-1)
    d["delta"] = d["next_value"] - d["value"]
    d["tb_share"] = d["transfers_balance"] / d["selected"].clip(lower=1)
    d["tb_share_l3"] = g["tb_share"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    d["selected_share"] = d["selected"] / d.groupby(
        ["season", "gameweek"])["selected"].transform("sum").clip(lower=1)
    d["form_proxy"] = g["total_points"].transform(
        lambda s: s.shift().rolling(4, min_periods=1).mean())
    return d[d["delta"].notna()].copy()


def fit_price_model(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    X_tr, X_te = train[PRICE_FEATURES], test[PRICE_FEATURES]
    out = test.copy()
    for label, target in (("rise", train["delta"] > 0), ("fall", train["delta"] < 0)):
        model = Pipeline([("i", SimpleImputer(strategy="median")),
                          ("s", StandardScaler()),
                          ("m", LogisticRegression(max_iter=2000, C=1.0))])
        model.fit(X_tr, target.astype(int))
        out[f"p_{label}"] = model.predict_proba(X_te)[:, 1]
    out["expected_delta"] = out["p_rise"] - out["p_fall"]
    return out


# ------------------------------------------------------------------- Part C

def simulate_value(panel: pd.DataFrame, price: pd.DataFrame,
                   value_aware: bool) -> dict:
    """Greedy season; when candidates are xP-tied, optionally prefer risers."""
    gws = sorted(panel["gameweek"].unique())
    squad = C.pick_initial_squad(panel[panel["gameweek"] <= gws[0] + 4], gws[0])
    costs = panel.groupby("element")["cost"].first()
    clubs = panel.groupby("element")["team"].first()
    bank = C.BUDGET - int(costs.reindex(squad).sum())
    start_value = int(costs.reindex(squad).sum()) + bank
    # A double gameweek gives a player two rows, so collapse to one entry per
    # (gameweek, element) before building the lookups -- otherwise .get()
    # returns a Series and the comparisons below are ambiguous.
    flat = (price.groupby(["gameweek", "element"], as_index=False)
            .agg(expected_delta=("expected_delta", "mean"),
                 delta=("delta", "mean")))
    pr = {(int(r.gameweek), int(r.element)): float(r.expected_delta)
          for r in flat.itertuples(index=False)}
    realised = {(int(r.gameweek), int(r.element)): float(r.delta)
                for r in flat.itertuples(index=False)}

    value_gain = 0.0
    xp_sacrificed = 0.0
    for gw in gws:
        # realised price movement on the squad this week
        for e in squad:
            d = realised.get((int(gw), int(e)))
            if d is not None and np.isfinite(d):
                value_gain += d

        nxt = panel[panel["gameweek"].eq(gw + 1)]
        if nxt.empty:
            continue
        owned = nxt[nxt["element"].isin(squad)]
        best, best_key = None, None
        for _, outp in owned.iterrows():
            cand = nxt[nxt["model_position"].eq(outp["model_position"])
                       & ~nxt["element"].isin(squad)]
            cand = cand[cand["cost"].le(bank + int(costs.get(outp["element"], 40)))]
            if cand.empty:
                continue
            top_xp = float(cand["xp"].max())
            pool = cand[cand["xp"] >= top_xp - TIE_BAND]
            if value_aware and len(pool) > 1:
                pool = pool.assign(_pd=[pr.get((int(gw) + 1, int(e)), 0.0)
                                        for e in pool["element"]])
                pick = pool.loc[pool["_pd"].idxmax()]
            else:
                pick = pool.loc[pool["xp"].idxmax()]
            gain = float(pick["xp"]) - float(outp["xp"])
            if gain <= 0.30:
                continue
            counts = clubs.reindex(squad).value_counts()
            if clubs.get(pick["element"]) != clubs.get(outp["element"]) and \
               counts.get(clubs.get(pick["element"]), 0) >= C.MAX_PER_CLUB:
                continue
            key = gain
            if best_key is None or key > best_key:
                best_key = key
                best = (int(outp["element"]), int(pick["element"]),
                        int(costs.get(outp["element"], 40)), int(pick["cost"]),
                        top_xp - float(pick["xp"]))
        if best:
            o, i, co, ci, sac = best
            squad = [i if e == o else e for e in squad]
            bank += co - ci
            xp_sacrificed += sac
    return {"value_gain_tenths": value_gain, "xp_sacrificed": xp_sacrificed,
            "start_value": start_value}


def main() -> None:
    seasons = sys.argv[1:] or ["2024-25", "2025-26"]
    raw_all = V.add_history_features(V.add_team_strength(V.load_seasons(mask_coverage=True)))
    price_all = build_price_frame(raw_all)
    results = {}

    for season in seasons:
        print("=" * 78)
        print(f"TEAM VALUE — held out {season}")
        print("=" * 78)
        panel = C.build_panel(season)
        gws = sorted(panel["gameweek"].unique())

        print("\nPART A — what is a pound worth? (exact MILP squad at each budget)")
        tbl = value_of_budget(panel, gws[:6])
        print(tbl.round(3).to_string(index=False))
        per_m = float(tbl["xp_per_gw_per_1m"].dropna().mean())
        print(f"\n  marginal value of budget: {per_m:.3f} xP per gameweek per GBP 1.0m")
        print(f"  over 38 gameweeks        : {per_m*38:.1f} points per GBP 1.0m")

        print("\nPART B — are price changes predictable?")
        tr = price_all[price_all["season"] < season]
        te = price_all[price_all["season"] == season]
        scored = fit_price_model(tr, te)
        auc_r = roc_auc_score((scored["delta"] > 0).astype(int), scored["p_rise"])
        auc_f = roc_auc_score((scored["delta"] < 0).astype(int), scored["p_fall"])
        print(f"  rows {len(scored):,}   AUC rise {auc_r:.3f}   AUC fall {auc_f:.3f}")
        top = scored.nlargest(int(len(scored) * 0.05), "p_rise")
        print(f"  top 5% by P(rise): actual rise rate {(top['delta']>0).mean():.3f} "
              f"vs base rate {(scored['delta']>0).mean():.3f}")

        print("\nPART C — what does a value-aware tiebreak accumulate?")
        blind = simulate_value(panel, scored, value_aware=False)
        aware = simulate_value(panel, scored, value_aware=True)
        extra = (aware["value_gain_tenths"] - blind["value_gain_tenths"]) / 10.0
        sac = aware["xp_sacrificed"] - blind["xp_sacrificed"]
        print(f"  value-blind squad value change : "
              f"{blind['value_gain_tenths']/10:+.2f}m")
        print(f"  value-aware squad value change : "
              f"{aware['value_gain_tenths']/10:+.2f}m")
        print(f"  extra team value from awareness: {extra:+.2f}m")
        print(f"  xP sacrificed breaking ties    : {sac:+.2f} over the season")
        print(f"  converted at {per_m*38:.1f} pts/m -> {extra*per_m*38:+.1f} points")
        print(f"  NET                            : {extra*per_m*38 - sac:+.1f} points")
        results[season] = {"per_m_38gw": per_m * 38, "extra_value_m": extra,
                           "xp_sacrificed": sac,
                           "net": extra * per_m * 38 - sac}
        print()

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(pd.DataFrame(results).T.round(2).to_string())
    ok = all(r["extra_value_m"] > 0 and r["net"] > 0 for r in results.values())
    print(f"\n  gate (positive extra value AND positive net in BOTH folds): "
          f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
