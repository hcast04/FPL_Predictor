"""
The noise floor. Build this before anything it will be used to judge.

A season total is a sum of ~38 highly variable gameweeks filtered through a
discrete optimiser, so two signals that are identical in expectation can still
produce very different seasons. Without knowing that spread there is no way to
tell a real improvement from a reshuffle.

Method: perturb the signal by a multiplicative jitter that encodes NO new
information, re-run the full season, and measure how much the answer moves.
Anything smaller than that spread is not a result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .simulate import simulate_season


def jitter(panel: pd.DataFrame, signal: str, sigma: float, seed: int,
           out_col: str = "_jittered") -> pd.DataFrame:
    """Multiplicative lognormal noise: same expectation, no new information."""
    rng = np.random.default_rng(seed)
    p = panel.copy()
    factor = rng.lognormal(mean=-0.5 * sigma ** 2, sigma=sigma, size=len(p))
    p[out_col] = p[signal].to_numpy(float) * factor
    return p


def noise_floor(season_panels: dict[str, pd.DataFrame], signal: str,
                sigmas=(0.001, 0.01, 0.05), seeds: int = 8,
                actual: str = "points") -> pd.DataFrame:
    """
    For each jitter size, the spread of season totals across seeds.

    Returns one row per (sigma, season) plus the sd of the cross-season mean --
    the number to compare any claimed improvement against.
    """
    rows = []
    for sigma in sigmas:
        per_season_totals = {}
        for season, panel in season_panels.items():
            totals = []
            for seed in range(seeds):
                p = jitter(panel, signal, sigma, seed)
                r = simulate_season(p, "_jittered", actual=actual)
                if r is not None:
                    totals.append(r.total)
            if totals:
                per_season_totals[season] = totals
                rows.append({"sigma": sigma, "season": season,
                             "n": len(totals), "mean": float(np.mean(totals)),
                             "sd": float(np.std(totals, ddof=1)),
                             "range": float(np.max(totals) - np.min(totals))})
        if per_season_totals:
            n = min(len(v) for v in per_season_totals.values())
            means = [np.mean([per_season_totals[s][i] for s in per_season_totals])
                     for i in range(n)]
            rows.append({"sigma": sigma, "season": "ALL (mean of seasons)",
                         "n": n, "mean": float(np.mean(means)),
                         "sd": float(np.std(means, ddof=1)),
                         "range": float(np.max(means) - np.min(means))})
    return pd.DataFrame(rows)


def detectable(noise: pd.DataFrame, sigma: float = 0.01, k: float = 3.0) -> float:
    """
    Smallest season-total difference worth believing: k standard deviations of
    the cross-season mean at the given jitter.
    """
    row = noise[(noise["sigma"] == sigma)
                & noise["season"].str.startswith("ALL")]
    return float(row["sd"].iloc[0] * k) if len(row) else float("nan")
