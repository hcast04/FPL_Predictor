#!/usr/bin/env python3
"""
Rank-based bonus model.

The notebook predicts bonus as an independent per-90 Poisson rate
(`alpha=2.0`, pooled across positions). FPL bonus is not a rate: within each
fixture the three highest BPS scores are awarded 3, 2 and 1 points. A per-90
rate model cannot represent that, and it loses the dispersion -- real bonus/90
spans 0.000 to 0.838 across player-season deciles (3.9x top-to-median), which a
heavily regularised pooled Poisson flattens.

This models the actual mechanism:

  1. Predict BPS per 90 for every player (Poisson GLM on the same leakage-safe
     features the notebook already builds).
  2. Convert predicted BPS into expected bonus by simulating the within-fixture
     ranking. For each fixture, draw each player's BPS from a negative-binomial
     around its predicted mean, rank, award 3/2/1, and average over draws.
     Ties are split the way FPL splits them (joint top both get 3, etc.).

Step 2 is what makes a 20-BPS midfielder in a high-scoring fixture worth less
than a 20-BPS midfielder in a tight one: the competition is modelled, not
assumed away.

ACCEPTANCE GATES (declared before the first run):
  G1  bonus MAE improves against the per-90 rate baseline
  G2  bonus dispersion ratio (sd predicted / sd actual) moves toward 1.0
  G3  per-gameweek Spearman against realised bonus does not degrade
  G4  both must hold on BOTH chronological folds

Run:  python bonus_model.py [2024-25 2025-26]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_xp_fixes as V  # noqa: E402

N_DRAWS = 400


def fit_bps_dispersion(train: pd.DataFrame) -> float:
    """
    Fit the negative-binomial shape k in Var = mu * (1 + mu * k).

    k is strongly mu-dependent (about 41 at mu<1, about 0.45 at mu~19), because
    low-minutes players are dominated by whether they played at all. Bonus is
    only ever decided among high-BPS players, so k is fitted on the top of the
    range -- fitting it pooled would be dominated by players who never compete.
    """
    d = train.copy()
    d["bps"] = pd.to_numeric(d["bps"], errors="coerce")
    d["minutes"] = pd.to_numeric(d["minutes"], errors="coerce")
    d = d[(d["minutes"] > 0) & d["bps"].notna()]
    if len(d) < 500:
        return 0.55
    share = (d["minutes"] / 90.0).clip(0, 1)
    mu = (d.groupby("element")["bps"].transform("mean") * share).clip(0.5, None)
    d = d.assign(mu=mu)
    high = d[d["mu"] >= d["mu"].quantile(0.75)]
    if len(high) < 200:
        return 0.55
    m, v = float(high["mu"].mean()), float(high["bps"].var())
    k = (v / m - 1.0) / m
    return float(np.clip(k, 0.05, 3.0))


def fit_bps_rate(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Predicted BPS per 90, same model family the notebook uses for rates."""
    y = 90.0 * pd.to_numeric(train["bps"], errors="coerce") / pd.to_numeric(
        train["minutes"], errors="coerce"
    )
    mins = pd.to_numeric(train["minutes"], errors="coerce")
    ok = y.notna() & np.isfinite(y) & mins.gt(0)
    weight = (mins[ok] / 90.0).clip(0.05, 1.0)
    model = Pipeline([
        ("i", SimpleImputer(strategy="median")),
        ("s", StandardScaler()),
        ("m", PoissonRegressor(alpha=1.0, max_iter=3000)),
    ])
    model.fit(V.design(train)[ok.to_numpy()], y[ok].clip(0, 60),
              m__sample_weight=weight)
    return np.clip(model.predict(V.design(test)), 0, 60)


def fit_bonus_rate_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """The notebook's current approach: bonus as a per-90 Poisson rate."""
    y = 90.0 * pd.to_numeric(train["bonus"], errors="coerce") / pd.to_numeric(
        train["minutes"], errors="coerce"
    )
    mins = pd.to_numeric(train["minutes"], errors="coerce")
    ok = y.notna() & np.isfinite(y) & mins.gt(0)
    weight = (mins[ok] / 90.0).clip(0.05, 1.0)
    model = Pipeline([
        ("i", SimpleImputer(strategy="median")),
        ("s", StandardScaler()),
        ("m", PoissonRegressor(alpha=2.0, max_iter=3000)),
    ])
    model.fit(V.design(train)[ok.to_numpy()], y[ok].clip(0, 3),
              m__sample_weight=weight)
    return np.clip(model.predict(V.design(test)), 0, 3)


def expected_bonus_by_rank(frame: pd.DataFrame, bps_col: str,
                           n_draws: int = N_DRAWS, seed: int = 0,
                           dispersion: float = 0.55) -> np.ndarray:
    """
    Expected bonus from simulating the within-fixture BPS ranking.

    FPL awards 3/2/1 to the top three BPS in a fixture, with ties sharing the
    higher award. Draws are negative-binomial around the predicted mean so that
    the variance grows with the mean, which matches BPS.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(len(frame), dtype=float)
    positions = {idx: i for i, idx in enumerate(frame.index)}

    for _, group in frame.groupby("fixture", sort=False):
        mu = group[bps_col].to_numpy(dtype=float)
        mu = np.clip(mu * group["minutes_share"].to_numpy(dtype=float), 0.01, None)
        n_players = len(mu)
        if n_players == 0:
            continue
        # negative binomial with mean mu and variance mu*(1+mu*k)
        k = dispersion
        r = 1.0 / k
        p = r / (r + mu)
        draws = rng.negative_binomial(r, p[None, :].repeat(n_draws, axis=0))

        award = np.zeros(n_players, dtype=float)
        for d in range(n_draws):
            row = draws[d]
            # award 3/2/1 to the top three distinct BPS values; players level
            # on a value all receive that award, as FPL does.
            ranks = np.unique(row)[::-1]
            for points, value in zip((3.0, 2.0, 1.0), ranks[:3]):
                tied = np.where(row == value)[0]
                award[tied] += points
        award /= n_draws
        for local, idx in enumerate(group.index):
            out[positions[idx]] = award[local]
    return out


def evaluate(season: str) -> dict:
    train_seasons = [s for s in V.SEASONS if s < season]
    raw = V.add_history_features(V.add_team_strength(V.load_seasons(mask_coverage=True)))
    train = raw[raw["season"].isin(train_seasons)].copy()
    test = raw[raw["season"].eq(season)].copy()
    test = test[(test["gameweek"] >= 6) & (test["minutes"] > 0)].copy()

    V.FEATURE_CLIPS.clear()
    td = train.reindex(columns=V.FEATURES).astype(float).replace([np.inf, -np.inf], np.nan)
    for c in V.FEATURES:
        col = td[c].dropna()
        if len(col) > 100:
            V.FEATURE_CLIPS[c] = (float(col.quantile(0.001)), float(col.quantile(0.999)))

    test = test.reset_index(drop=True)
    test["minutes_share"] = (pd.to_numeric(test["minutes"], errors="coerce")
                             .fillna(0) / 90.0).clip(0, 1)
    actual = pd.to_numeric(test["bonus"], errors="coerce").fillna(0).to_numpy()

    baseline = fit_bonus_rate_baseline(train, test) * test["minutes_share"].to_numpy()
    test["pred_bps90"] = fit_bps_rate(train, test)
    dispersion = fit_bps_dispersion(train)
    ranked = expected_bonus_by_rank(test, "pred_bps90", dispersion=dispersion)
    print(f"fitted BPS dispersion k = {dispersion:.3f}")

    def metrics(pred, label):
        sp = []
        for _, g in test.assign(_p=pred).groupby("gameweek"):
            if len(g) >= 30 and g["bonus"].nunique() > 1:
                sp.append(spearmanr(g["_p"], g["bonus"]).statistic)
        return {"variant": label,
                "MAE": float(np.mean(np.abs(pred - actual))),
                "bias": float(np.mean(pred - actual)),
                "sd_pred": float(np.std(pred)),
                "sd_actual": float(np.std(actual)),
                "dispersion": float(np.std(pred) / np.std(actual)),
                "spearman": float(np.nanmean(sp))}

    return {"season": season,
            "rows": len(test),
            "baseline": metrics(baseline, "per-90 rate (current)"),
            "ranked": metrics(ranked, "rank-based (new)")}


def main() -> None:
    seasons = sys.argv[1:] or ["2024-25", "2025-26"]
    results, gates = [], {}
    for season in seasons:
        print("=" * 76)
        print(f"BONUS MODEL — held out {season}")
        print("=" * 76)
        r = evaluate(season)
        table = pd.DataFrame([r["baseline"], r["ranked"]]).set_index("variant")
        print(f"rows {r['rows']:,}\n")
        print(table.round(4).to_string())
        b, n = r["baseline"], r["ranked"]
        gates[season] = {
            "G1 MAE improves": n["MAE"] < b["MAE"],
            "G2 dispersion closer to 1.0":
                abs(n["dispersion"] - 1) < abs(b["dispersion"] - 1),
            "G3 Spearman not degraded": n["spearman"] >= b["spearman"] - 0.003,
        }
        print()
        for k, v in gates[season].items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        print()
        results.append(r)

    ok = all(all(g.values()) for g in gates.values())
    print("=" * 76)
    print(f"VERDICT across {len(seasons)} folds: {'ACCEPT' if ok else 'REJECT'}")
    for s, g in gates.items():
        bad = [k for k, v in g.items() if not v]
        print(f"  {s}: {'all gates pass' if not bad else 'failed -> ' + '; '.join(bad)}")
    print("=" * 76)


if __name__ == "__main__":
    main()
