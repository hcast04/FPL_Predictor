"""
Bridge to the OLD pipeline's cached out-of-sample forecasts.

Temporary and clearly fenced off. It exists so the new harness can be validated
against a signal whose behaviour is already known, before any new model exists.
Delete once the rebuilt model produces its own walk-forward forecasts.

The fold cache holds, per holdout season, the old notebook's `expected_points`
for every player-fixture it forecast. That is genuine out-of-sample output, not
a re-implementation.
"""
from __future__ import annotations

import glob

import pandas as pd

from . import config as C


def load_fold_forecasts() -> pd.DataFrame:
    """One row per (season, gameweek, player_code) with the legacy xP."""
    frames = []
    for path in sorted(glob.glob(str(C.FOLD_CACHE / "*.pkl.gz"))):
        payload = pd.read_pickle(path, compression="gzip")
        fr = payload["bundle"]["holdout_expected_points"].copy()
        fr["season"] = str(payload["holdout_season"])
        fr["xp"] = pd.to_numeric(fr["expected_points"], errors="coerce")
        frames.append(fr[["season", "gameweek", C.PLAYER_KEY, "xp"]])
    d = pd.concat(frames, ignore_index=True)
    # a double gameweek has two forecast rows; the player plays both, so sum
    return (d.groupby(["season", "gameweek", C.PLAYER_KEY], as_index=False)
              .agg(xp=("xp", "sum")))


def attach(panel: pd.DataFrame) -> pd.DataFrame:
    """Join legacy xP onto the panel. Unforecast rows get xp = 0."""
    fc = load_fold_forecasts()
    out = panel.merge(fc, on=["season", "gameweek", C.PLAYER_KEY], how="left")
    out["xp"] = out["xp"].fillna(0.0)
    return out
