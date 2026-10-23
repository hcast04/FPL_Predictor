"""
FPL Predictor — mechanism-level xP corrections (v2.6 candidate).

Diagnosed against 4 seasons of archived FPL data (2022-23 .. 2025-26,
vaastav/Fantasy-Premier-League). Each correction here targets a *mechanism*
that was measured, not the upper-tail symptom that the rejected v2.4
calibration tried to patch.

Findings behind each fix
------------------------
F1  FPL `expected_assists` systematically understates realised assists.
    Pooled 2023-26 ratio assists/xA: DEF 1.31, MID 1.47, FWD 2.39, GK ~4.5.
    `calculate_component_expected_points` uses predicted xA directly as
    expected assists, so `xp_assists` is understated by ~25-45%. The absolute
    shortfall grows with xA, which is exactly the upper-tail / high-minutes
    bias recorded in audit sections 7.4 and 7.5.

F2  2022-23 has fabricated ZERO (not null) `expected_goals`/`expected_assists`
    for gameweeks 1-15: 4,036 rows (35.6% of the season) containing 370 real
    goals and 331 real assists. These pass the model's `.notna()` training
    filter and are learned as genuine zero-xG observations, depressing every
    attacking rate model that trains on DEVELOPMENT_SEASONS.

F3  The defensive-contribution layer applies minutes twice to its Poisson
    branch: `expected_defensive_actions` is already minutes-scaled, then the
    blended probability is multiplied by `minute_scale = (mins/90)**0.65`.
    Only the empirical branch needs that multiplier.

F4  WITHDRAWN. An earlier revision changed GOAL_POINTS_BY_POSITION["GK"] from
    10.0 to 6.0 on the belief that FPL pays 6 for a goalkeeper goal. That is
    out of date: current FPL awards 10 for a goalkeeper goal (raised from 6).
    The notebook's original 10.0 was correct and is restored here. No
    goalkeeper scored in 2024-25 or 2025-26, so this cannot be settled from
    the archived data -- the reconstruction check below validates every other
    position (DEF goal = 6: 126/126 and 128/128 exact) but is silent on GK.

F5  `apply_v24_targeted_forecast_layer` is idempotent by *resetting* the value
    column to `pre_v24_expected_points`. That baseline is never invalidated
    when the v2.5 role integration recomputes `expected_points`, so a later
    v2.4 call silently reverts the v2.5 forecast.

NOT changed (tested and found sound)
------------------------------------
*   The component -> FPL points arithmetic. Monte-Carlo scoring of 400k
    simulated matches per archetype agrees with
    `calculate_component_expected_points` to within 0.1% for nailed starters.
*   The clean-sheet exposure rule `exp(-lambda * minutes / 90)`. Fitting
    `exp(-lambda * frac**gamma)` on 2022-25 and testing on 2025-26 gives
    gamma = 0.975 and the current form wins on Brier and log-loss. Leaving
    the pitch early genuinely protects a clean sheet (+7.6pp within the same
    team-match).
*   `expected_goals`: league-wide goals/xG is 0.94-1.00 for 2023-26, so xG is
    already a usable expected-goals estimate. A conversion factor is fitted
    anyway so the assumption is monitored rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "GOAL_POINTS_BY_POSITION",
    "CONVERSION_SHRINKAGE_EVENTS",
    "APPLY_GOAL_CONVERSION_BY_DEFAULT",
    "ASSIST_CONVERSION_BOUNDS",
    "GOAL_CONVERSION_BOUNDS",
    "mask_fabricated_event_coverage",
    "fit_event_conversion_factors",
    "apply_event_conversion",
    "fix_defensive_contribution_minutes",
    "assert_no_stale_v24_baseline",
    "clear_stale_layer_baselines",
]

PLAYER_SCORING_POSITIONS = ["GK", "DEF", "MID", "FWD"]

# Current FPL scoring. A goalkeeper goal is worth 10 (raised from 6); a
# defender goal is worth 6. Do not "correct" the goalkeeper value to 6.
GOAL_POINTS_BY_POSITION = {
    "GK": 10.0,
    "DEF": 6.0,
    "MID": 5.0,
    "FWD": 4.0,
}

# Empirical-Bayes strength for the conversion factors, in event units.
CONVERSION_SHRINKAGE_EVENTS = 150.0

# Conversion factors are bounded so a degraded data feed cannot produce an
# extreme multiplier. The observed 2023-26 assist range is 1.31 - 2.39.
ASSIST_CONVERSION_BOUNDS = (1.0, 2.6)
GOAL_CONVERSION_BOUNDS = (0.85, 1.25)

# Goal conversion is OFF by default. League-wide goals/xG is 0.94-1.00 for
# 2023-26, so there is no demonstrated defect to correct and fitting another
# multiplier only adds variance. Enable it only behind its own ablation.
APPLY_GOAL_CONVERSION_BY_DEFAULT = False


def _position_series(frame: pd.DataFrame) -> pd.Series:
    """Return canonical GK/DEF/MID/FWD labels from whichever column exists."""
    for column in ("model_position", "position"):
        if column in frame.columns:
            return (
                frame[column]
                .astype("string")
                .str.strip()
                .str.upper()
                .replace({
                    "GKP": "GK",
                    "GOALKEEPER": "GK",
                    "DEFENDER": "DEF",
                    "MIDFIELDER": "MID",
                    "FORWARD": "FWD",
                })
            )
    raise KeyError("No 'model_position' or 'position' column is available.")


# ---------------------------------------------------------------------------
# F2 — fabricated zero xG/xA coverage
# ---------------------------------------------------------------------------

def mask_fabricated_event_coverage(
    frame: pd.DataFrame,
    *,
    season_col: str = "season",
    gameweek_col: str = "gameweek",
    minutes_col: str = "minutes",
    event_columns: tuple[str, ...] = ("expected_goals", "expected_assists"),
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Set xG/xA to NaN for (season, gameweek) blocks where the feed reported no
    data at all.

    A gameweek in which *every* player with minutes has exactly 0.0 xG is a
    missing feed, not 20 goalless teams. Converting those to NaN lets the
    existing median imputer and `.notna()` training filters do the right thing
    instead of learning them as real zeros.

    Call this on the raw per-gameweek history immediately after loading it and
    before `add_player_event_targets`.
    """
    result = frame.copy()
    if season_col not in result.columns:
        result[season_col] = "__single__"

    minutes = pd.to_numeric(result.get(minutes_col), errors="coerce").fillna(0.0)
    played = minutes > 0

    masked_blocks: list[tuple] = []
    for column in event_columns:
        if column not in result.columns:
            continue
        values = pd.to_numeric(result[column], errors="coerce")
        totals = (
            values.where(played)
            .groupby([result[season_col], result[gameweek_col]])
            .sum(min_count=1)
        )
        counts = (
            played.groupby([result[season_col], result[gameweek_col]]).sum()
        )
        dead = totals.index[(totals.fillna(0.0) <= 0.0) & (counts > 0)]
        if len(dead) == 0:
            continue
        key = pd.MultiIndex.from_arrays(
            [result[season_col], result[gameweek_col]]
        )
        blanket = key.isin(dead)
        result.loc[blanket, column] = np.nan
        masked_blocks.extend((column, *k) for k in dead)

    if verbose and masked_blocks:
        affected = len({(b[1], b[2]) for b in masked_blocks})
        rows = int(
            pd.MultiIndex.from_arrays([result[season_col], result[gameweek_col]])
            .isin({(b[1], b[2]) for b in masked_blocks})
            .sum()
        )
        print(
            f"[fpl_xp_fixes] masked fabricated event coverage in "
            f"{affected} season-gameweek blocks ({rows:,} rows) -> NaN"
        )
    return result


# ---------------------------------------------------------------------------
# F1 — xA -> assists and xG -> goals conversion
# ---------------------------------------------------------------------------

def fit_event_conversion_factors(
    training_data: pd.DataFrame,
    *,
    realised_col: str,
    expected_col: str,
    bounds: tuple[float, float],
    minutes_col: str = "target_minutes",
    shrinkage_events: float = CONVERSION_SHRINKAGE_EVENTS,
) -> dict[str, float]:
    """
    Estimate realised/expected conversion by position with shrinkage to the
    pooled factor.

    Fit this on development/training rows only. The factor corrects the fact
    that FPL's expected_assists is not on the same scale as a realised assist.
    """
    data = training_data.copy()
    data["_pos"] = _position_series(data)
    realised = pd.to_numeric(data.get(realised_col), errors="coerce")
    expected = pd.to_numeric(data.get(expected_col), errors="coerce")
    minutes = pd.to_numeric(data.get(minutes_col), errors="coerce").fillna(0.0)

    valid = realised.notna() & expected.notna() & (minutes > 0)
    data = data.loc[valid]
    realised = realised.loc[valid]
    expected = expected.loc[valid]
    if data.empty or expected.sum() <= 0:
        return {position: 1.0 for position in PLAYER_SCORING_POSITIONS}

    pooled = float(realised.sum() / expected.sum())
    pooled = float(np.clip(pooled, *bounds))

    factors: dict[str, float] = {}
    grouped = pd.DataFrame({
        "pos": data["_pos"], "r": realised, "e": expected
    }).groupby("pos").sum()

    for position in PLAYER_SCORING_POSITIONS:
        if position not in grouped.index:
            factors[position] = pooled
            continue
        r = float(grouped.loc[position, "r"])
        e = float(grouped.loc[position, "e"])
        if e <= 0:
            factors[position] = pooled
            continue
        raw = r / e
        weight = e / (e + float(shrinkage_events))
        factors[position] = float(
            np.clip(weight * raw + (1.0 - weight) * pooled, *bounds)
        )

    factors["__pooled__"] = pooled
    return factors


def apply_event_conversion(
    components: pd.DataFrame,
    data: pd.DataFrame,
    *,
    assist_factors: dict[str, float] | None = None,
    goal_factors: dict[str, float] | None = None,
    goal_points: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Rescale projected assists/goals onto the realised scale and rebuild the
    affected xP components.

    Apply immediately after `calculate_component_expected_points`, passing the
    same input frame. `expected_points` is recomputed by delta so every other
    component is left untouched.
    """
    result = components.copy()
    position = _position_series(data).reindex(result.index)
    goal_points = goal_points or GOAL_POINTS_BY_POSITION

    result["pre_conversion_expected_points"] = pd.to_numeric(
        result["expected_points"], errors="coerce"
    )

    delta = pd.Series(0.0, index=result.index, dtype=float)

    if assist_factors:
        factor = position.map(
            {k: v for k, v in assist_factors.items() if k != "__pooled__"}
        ).astype(float).fillna(assist_factors.get("__pooled__", 1.0))
        old = pd.to_numeric(result["xp_assists"], errors="coerce").fillna(0.0)
        result["assist_conversion_factor"] = factor
        result["projected_assists"] = (
            pd.to_numeric(result["projected_assists"], errors="coerce").fillna(0.0)
            * factor
        )
        result["xp_assists"] = 3.0 * result["projected_assists"]
        delta = delta + (result["xp_assists"] - old)

    if goal_factors:
        factor = position.map(
            {k: v for k, v in goal_factors.items() if k != "__pooled__"}
        ).astype(float).fillna(goal_factors.get("__pooled__", 1.0))
        old = pd.to_numeric(result["xp_goals"], errors="coerce").fillna(0.0)
        result["goal_conversion_factor"] = factor
        result["projected_goals"] = (
            pd.to_numeric(result["projected_goals"], errors="coerce").fillna(0.0)
            * factor
        )
        result["xp_goals"] = result["projected_goals"] * position.map(
            goal_points
        ).astype(float).fillna(0.0)
        delta = delta + (result["xp_goals"] - old)
    elif goal_points != {"GK": 10.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}:
        # F4 alone: repoint goal values without rescaling the projection.
        old = pd.to_numeric(result["xp_goals"], errors="coerce").fillna(0.0)
        result["xp_goals"] = pd.to_numeric(
            result["projected_goals"], errors="coerce"
        ).fillna(0.0) * position.map(goal_points).astype(float).fillna(0.0)
        delta = delta + (result["xp_goals"] - old)

    result["expected_points"] = (
        result["pre_conversion_expected_points"] + delta
    )
    result["event_conversion_adjustment"] = delta
    return result


# ---------------------------------------------------------------------------
# F3 — defensive contribution minutes double-count
# ---------------------------------------------------------------------------

def fix_defensive_contribution_minutes(
    components: pd.DataFrame,
    data: pd.DataFrame,
    *,
    poisson_weight: float = 0.30,
    exponent: float = 0.65,
) -> pd.DataFrame:
    """
    Rebuild the DC probability applying `minute_scale` to the empirical branch
    only.

    `expected_defensive_actions` already carries `expected_minutes / 90`, so
    the Poisson branch must not be scaled by minutes a second time.
    """
    result = components.copy()
    minutes = (
        pd.to_numeric(data["expected_minutes"], errors="coerce")
        .reindex(result.index)
        .fillna(0.0)
        .div(90.0)
        .clip(0.0, 1.0)
    )
    minute_scale = minutes.pow(exponent)

    poisson_p = pd.to_numeric(
        result["poisson_defensive_return_probability"], errors="coerce"
    ).fillna(0.0)
    empirical_p = pd.to_numeric(
        result["empirical_defensive_return_probability"], errors="coerce"
    ).fillna(0.0)

    old = pd.to_numeric(
        result["defensive_return_probability"], errors="coerce"
    ).fillna(0.0)
    corrected = (
        poisson_weight * poisson_p
        + (1.0 - poisson_weight) * empirical_p * minute_scale
    ).clip(0.0, 1.0)

    result["defensive_return_probability"] = corrected
    result["xp_defensive_contribution"] = 2.0 * corrected
    result["expected_points"] = (
        pd.to_numeric(result["expected_points"], errors="coerce")
        + 2.0 * (corrected - old)
    )
    return result


# ---------------------------------------------------------------------------
# F5 — stale post-layer baselines
# ---------------------------------------------------------------------------

_BASELINE_COLUMNS = (
    "pre_v24_expected_points",
    "pre_premium_expected_points",
    "raw_expected_points",
)


def clear_stale_layer_baselines(
    frame: pd.DataFrame,
    columns: tuple[str, ...] = _BASELINE_COLUMNS,
) -> pd.DataFrame:
    """
    Drop idempotence baselines after a layer has recomputed `expected_points`.

    Call this inside `integrate_v25_probabilistic_minutes_xp` (or immediately
    after it) so a later `apply_v24_targeted_forecast_layer` cannot reset the
    forecast to a pre-v2.5 value.
    """
    result = frame.copy()
    return result.drop(columns=[c for c in columns if c in result.columns])


def assert_no_stale_v24_baseline(
    frame: pd.DataFrame,
    *,
    value_col: str = "expected_points",
    baseline_col: str = "pre_v24_expected_points",
    tolerance: float = 1e-9,
) -> None:
    """
    Fail loudly when a stored v2.4 baseline no longer matches the live
    forecast lineage, which means a later layer would be silently reverted.
    """
    if baseline_col not in frame.columns:
        return
    adjustment = frame.get("v24_targeted_layer_adjustment")
    if adjustment is None:
        return
    expected = pd.to_numeric(frame[baseline_col], errors="coerce") + pd.to_numeric(
        adjustment, errors="coerce"
    )
    actual = pd.to_numeric(frame[value_col], errors="coerce")
    drift = (expected - actual).abs()
    if drift.max() > tolerance:
        raise AssertionError(
            f"Stale '{baseline_col}' detected: a later layer changed "
            f"'{value_col}' without refreshing the v2.4 idempotence baseline. "
            f"Max drift {drift.max():.6f}. Call clear_stale_layer_baselines()."
        )
