#!/usr/bin/env python3
"""
F3 validation: the defensive-contribution minutes double-count.

The headline +2.18 / +2.27 top-10 result from validate_xp_fixes.py does NOT
cover F3 -- that harness excludes defensive contribution entirely, because the
pre-2025/26 training seasons have no CBIT/recovery data. This script closes
that gap on 2025-26, the only season where DC exists.

What is being tested
--------------------
estimate_defensive_return_probability() blends a Poisson threshold model with
an empirical award frequency:

    blended = 0.30 * P_poisson + 0.70 * P_empirical
    blended = blended * (expected_minutes / 90) ** 0.65        # baseline

P_poisson is computed from `expected_defensive_actions`, which is already
rate_per90 * expected_minutes / 90. So the baseline applies minutes twice to
that branch. The fix applies the scale only to the empirical branch:

    blended = 0.30 * P_poisson + 0.70 * P_empirical * minute_scale

Acceptance criteria (declared before inspecting the result)
-----------------------------------------------------------
G1  Brier score on the realised DC award improves (lower).
G2  Mean predicted P(DC) moves closer to the realised award rate.
G3  Defender ranking by DC xP does not degrade (Spearman, NDCG@10).

Run:  python validate_dc_fix.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_xp_fixes import DATA_DIR, ensure_data  # noqa: E402

SEASON = "2025-26"
THRESHOLD = {"GK": np.inf, "DEF": 10.0, "MID": 12.0, "FWD": 12.0}
PRIOR_FIXTURES = 8.0


def load() -> pd.DataFrame:
    ensure_data()
    d = pd.read_csv(DATA_DIR / f"{SEASON}.csv", low_memory=False)
    d["position"] = d["position"].replace({"GKP": "GK"})
    d = d[d["position"].isin(["GK", "DEF", "MID", "FWD"])].copy()
    for c in ["minutes", "defensive_contribution", "round", "total_points"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["kickoff_time"] = pd.to_datetime(d["kickoff_time"], errors="coerce", utc=True)
    d = d.sort_values(["element", "kickoff_time"]).reset_index(drop=True)

    d["threshold"] = d["position"].map(THRESHOLD)
    d["earned"] = (
        d["defensive_contribution"].ge(d["threshold"]) & d["threshold"].notna()
    ).astype(float)

    # Leakage-safe history: shift, then roll.
    g = d.groupby("element", sort=False)
    actions = g["defensive_contribution"].transform(lambda s: s.shift())
    minutes = g["minutes"].transform(lambda s: s.shift())
    earned = g["earned"].transform(lambda s: s.shift())

    num = actions.groupby(d["element"], sort=False).transform(
        lambda s: s.rolling(10, min_periods=3).sum()
    )
    den = minutes.groupby(d["element"], sort=False).transform(
        lambda s: s.rolling(10, min_periods=3).sum()
    )
    d["action_rate_per90"] = 90.0 * num / den.where(den >= 180.0)
    d["empirical_award_rate"] = earned.groupby(d["element"], sort=False).transform(
        lambda s: s.rolling(10, min_periods=3).mean()
    )
    d["prior_n"] = g["minutes"].transform(lambda s: s.shift().gt(0).cumsum())
    return d


def blended_probabilities(d: pd.DataFrame, expected_minutes: pd.Series):
    """Return (baseline, fixed) DC probabilities for the same inputs."""
    position = d["position"]
    priors_action = (
        d.groupby("position")
        .apply(lambda f: 90.0 * f["defensive_contribution"].sum() / f["minutes"].sum(),
               include_groups=False)
    )
    priors_award = d.groupby("position")["earned"].mean()

    # Shrink sparse evidence toward the position prior, as the notebook does.
    n = d["prior_n"].fillna(0.0).clip(lower=0.0)
    reliability = n / (n + PRIOR_FIXTURES)
    rate = (
        reliability * d["action_rate_per90"].fillna(position.map(priors_action))
        + (1 - reliability) * position.map(priors_action)
    )
    empirical = (
        reliability * d["empirical_award_rate"].fillna(position.map(priors_award))
        + (1 - reliability) * position.map(priors_award)
    ).clip(0.0, 1.0)

    frac = (expected_minutes / 90.0).clip(0.0, 1.0)
    expected_actions = rate * frac

    p_poisson = pd.Series(0.0, index=d.index, dtype=float)
    for pos, thr in THRESHOLD.items():
        m = position.eq(pos)
        if not m.any() or not np.isfinite(thr):
            continue
        p_poisson.loc[m] = poisson.sf(int(thr) - 1, expected_actions.loc[m])

    minute_scale = frac.pow(0.65)
    baseline = ((0.30 * p_poisson + 0.70 * empirical) * minute_scale).clip(0, 1)
    fixed = (0.30 * p_poisson + 0.70 * empirical * minute_scale).clip(0, 1)
    return baseline, fixed


def ndcg(pred, actual, k=10):
    order = np.argsort(-np.asarray(pred))
    gain = np.clip(np.asarray(actual)[order][:k], 0, None)
    disc = 1.0 / np.log2(np.arange(2, len(gain) + 2))
    ideal = np.clip(np.sort(actual)[::-1][:k], 0, None)
    denom = (ideal * disc[: len(ideal)]).sum()
    return float((gain * disc).sum() / denom) if denom > 0 else np.nan


def main() -> None:
    d = load()
    # Outfield players with usable history. Goalkeepers cannot earn DC.
    use = d[
        d["position"].isin(["DEF", "MID", "FWD"])
        & d["action_rate_per90"].notna()
        & d["empirical_award_rate"].notna()
        & d["minutes"].gt(0)
    ].copy()
    print("=" * 74)
    print(f"F3 validation - defensive contribution, {SEASON}")
    print("=" * 74)
    print(f"rows: {len(use):,}   realised award rate: {use['earned'].mean():.4f}\n")

    # Realised minutes are used so the test isolates the DC formula from
    # minutes-model error. This is the most favourable case for the baseline.
    base, fixed = blended_probabilities(use, use["minutes"])
    y = use["earned"].to_numpy(float)

    def brier(p):
        return float(np.mean((np.asarray(p) - y) ** 2))

    rows = []
    for label, p in [("baseline (minutes applied twice)", base), ("v2.6 fixed", fixed)]:
        rows.append({
            "variant": label,
            "Brier": brier(p),
            "mean_pred": float(np.mean(p)),
            "bias": float(np.mean(p) - y.mean()),
            "xP_DC_mean": float(2 * np.mean(p)),
        })
    res = pd.DataFrame(rows).set_index("variant")
    print(res.round(4).to_string())

    # Ranking among defenders, by gameweek.
    dfn = use[use["position"].eq("DEF")].copy()
    dfn["base"], dfn["fixed"] = base[use["position"].eq("DEF")], fixed[use["position"].eq("DEF")]
    sp, nd = {}, {}
    for key in ("base", "fixed"):
        s, n = [], []
        for _, g in dfn.groupby("round"):
            if len(g) < 30:
                continue
            s.append(spearmanr(g[key], g["earned"]).statistic)
            n.append(ndcg(g[key].to_numpy(), g["earned"].to_numpy()))
        sp[key], nd[key] = float(np.nanmean(s)), float(np.nanmean(n))
    print(f"\nDefender ranking on the realised DC award (per gameweek):")
    print(f"   Spearman  baseline {sp['base']:.4f}   fixed {sp['fixed']:.4f}   "
          f"delta {sp['fixed'] - sp['base']:+.4f}")
    print(f"   NDCG@10   baseline {nd['base']:.4f}   fixed {nd['fixed']:.4f}   "
          f"delta {nd['fixed'] - nd['base']:+.4f}")

    b = res.loc["baseline (minutes applied twice)"]
    f = res.loc["v2.6 fixed"]
    gates = {
        "G1 Brier improves": f["Brier"] < b["Brier"],
        "G2 |bias| moves toward zero": abs(f["bias"]) < abs(b["bias"]),
        "G3 defender ranking not degraded": (sp["fixed"] >= sp["base"] - 0.003)
                                            and (nd["fixed"] >= nd["base"] - 0.002),
    }
    print("\nACCEPTANCE GATES (declared before inspecting results)")
    for name, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  OVERALL: {'ACCEPT' if all(gates.values()) else 'REJECT'}")
    # --- second scenario: forecast-style minutes ---------------------------
    # Realised minutes cluster at 90, where minute_scale ~ 1 and the double
    # count barely bites. A forecast is shrunk and spread out, which is where
    # the bug should actually matter. Proxy it with the player's leakage-safe
    # rolling mean minutes.
    g = use.groupby("element", sort=False)
    proxy = g["minutes"].transform(
        lambda s: s.shift().rolling(5, min_periods=2).mean()
    )
    sub = use[proxy.notna()].copy()
    pm = proxy[proxy.notna()]
    b2, f2 = blended_probabilities(sub, pm)
    y2 = sub["earned"].to_numpy(float)
    print("\n" + "-" * 74)
    print("Second scenario: forecast-style (rolling-mean) minutes instead of realised")
    print(f"   rows {len(sub):,}   mean proxy minutes {pm.mean():.1f} "
          f"(vs realised {sub['minutes'].mean():.1f})")
    print(f"   {'variant':34s} {'Brier':>8} {'mean pred':>10} {'bias':>8}")
    for lab, p in [("baseline (minutes twice)", b2), ("v2.6 fixed", f2)]:
        print(f"   {lab:34s} {np.mean((p-y2)**2):8.4f} {np.mean(p):10.4f} "
              f"{np.mean(p)-y2.mean():+8.4f}")
    print(f"\n   Brier delta {np.mean((f2-y2)**2)-np.mean((b2-y2)**2):+.5f}  "
          f"(negative = fix better)")


if __name__ == "__main__":
    main()
