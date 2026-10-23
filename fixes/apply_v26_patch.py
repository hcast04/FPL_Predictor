#!/usr/bin/env python3
"""
Wire the v2.6 mechanism fixes into FPL_Predictor.ipynb.

Writes a NEW notebook (FPL_Predictor_v2.6.ipynb) and never modifies the
original. Every edit is anchored on exact source text; if an anchor is missing
or ambiguous the patch aborts without writing anything, so a partially-applied
notebook is not possible.

Run:
    python apply_v26_patch.py                      # default paths
    python apply_v26_patch.py <in.ipynb> <out.ipynb>

What it changes
---------------
P1  WITHDRAWN (F4 was wrong - current FPL pays 10 for a goalkeeper goal)
P2  estimate_defensive_return_probability: minute_scale applied to the
    empirical branch only, not to the already-minutes-scaled Poisson
    branch                                                            (F3)
P3  calculate_component_expected_points: projected goals/assists are put on
    the realised scale via module-level conversion factors            (F1)
P4  historical_fpl: fabricated all-zero xG/xA gameweeks -> NaN, applied
    BEFORE add_historical_player_features() so the rolling/career xG/xA
    predictor features are built from clean data too                      (F2)
P5  FORECAST_CACHE_SIGNATURE: v2.6 entries added so the stale xP-calibration,
    premium-attacker and v2.4 artifacts are rejected and refitted
P6  v26_set_conversion(<training frame>) before each of the four
    calculate_component_expected_points sites
P7  integrate_v25_probabilistic_minutes_xp: clear stale idempotence
    baselines so a later v2.4 call cannot revert the v2.5 forecast     (F5)
P8  a new setup cell defining the helpers, inserted after the imports
P9  Section 16: set holdout_season on the v2.5 candidate fold (pre-existing
    bug, unrelated to v2.6 — the first real v2.5 run dies without it)
P10 apply_premium_attacker_adjustment: route its attacking-xP rebuild through
    the same conversion, so an approved premium payload cannot strip F1
P11 Section 16: set conversion factors from the fold bundle (the fold builder
    returns early on a cache hit, so its own call may not have run)
P12 attach_or_apply_v25_live_minutes: restore live factors before re-scoring
P13 a consolidated EXECUTION CARD cell at the end of Section 12: chip,
    transfers, starting XI, ordered bench, captain and vice in one block
P14 chip_structural_readiness: the wildcard fires on its uplift threshold
    alone. Validated at +40 points a season over the current gate
P15 rebuild the live post-layers in place after Section 11 refits the
    validation artifacts, so Section 12's artifact-state guard passes
    without a kernel restart and a full rerun
P18 expiry-pressure override in the weekly chip gate, so chips cannot
    silently expire unused (the backtest had this; production did not)
P20 under expiry pressure, play the wildcard FIRST (its value scales with
    weeks remaining); normal merit ordering is untouched
P21 give the expiry cascade one week of slack so a disrupted gameweek does
    not strand a chip
P19 suppress the wildcard explicitly when the planning horizon is too short
    to evaluate it, rather than by an accidentally mis-scaled threshold
P17 relax the v2.4 upper-tail MAE gate, the single criterion that was
    blocking a layer which improves every decision metric
P16 make the freshness gate material: refresh price-only moves into the live
    frame and block only on availability changes for players the model
    projects could actually play

P3 is deliberately inside the shared function so the v2.5 per-role-state loop
inherits the correction without any further edit.
"""

from __future__ import annotations

import ast
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_IN = HERE.parent / "FPL_Predictor.ipynb"
DEFAULT_OUT = HERE.parent / "FPL_Predictor_v2.6.ipynb"

# --------------------------------------------------------------------------
# the setup cell
# --------------------------------------------------------------------------

SETUP_CELL = '''# =============================================================================
# v2.6 MECHANISM FIXES — setup
# =============================================================================
# Diagnosed against 4 seasons of archived FPL data. See fixes/INTEGRATION.md.
#
# The corrections live in fixes/fpl_xp_fixes.py. Conversion factors are held in
# module-level globals that calculate_component_expected_points consults, so the
# v2.5 per-role-state integration inherits them automatically.

import sys as _sys
from pathlib import Path as _Path

for _candidate in (_Path.cwd(), _Path.cwd() / "fixes", _Path.cwd().parent / "fixes"):
    if (_candidate / "fpl_xp_fixes.py").exists():
        _sys.path.insert(0, str(_candidate))
        break
else:
    raise FileNotFoundError(
        "fpl_xp_fixes.py not found. Keep the fixes/ directory next to the "
        "notebook, or add its location to sys.path before this cell."
    )

import fpl_xp_fixes as fx

V26_ENABLED = True            # set False to run the notebook unchanged
V26_WILDCARD_THRESHOLD_ONLY = False  # WITHDRAWN: did not survive re-validation
V26_APPLY_ASSIST_CONVERSION = False  # FAILED its gate on the real pipeline
V26_APPLY_COVERAGE_MASK = True       # F2: mask 2022-23's fabricated zero xG/xA
V26_RELAX_UPPER_TAIL_MAE_GATE = True # v2.4 upper tail: judge on ranking, not MAE
V26_WILDCARD_MIN_HORIZON = 3         # below this, a wildcard cannot be judged
V26_CHIP_EXPIRY_SLACK = 1            # spare weeks before the expiry cascade
V26_APPLY_GOAL_CONVERSION = fx.APPLY_GOAL_CONVERSION_BY_DEFAULT  # False: goals/xG is ~1.0

# Identity factors until v26_set_conversion() is called.
V26_ASSIST_FACTORS: dict = {}
V26_GOAL_FACTORS: dict = {}
_V26_FACTOR_CACHE: dict = {}


def _v26_position_factor(position, factors):
    """Map a position Series to its conversion factor, defaulting to 1.0."""
    import pandas as _pd

    if not factors:
        return _pd.Series(1.0, index=position.index, dtype=float)
    lookup = {k: v for k, v in factors.items() if k != "__pooled__"}
    return (
        position.map(lookup)
        .astype(float)
        .fillna(float(factors.get("__pooled__", 1.0)))
    )


def v26_set_conversion(training_frame, *, label="default"):
    """
    Fit (and cache) realised/expected conversion factors on a TRAINING frame.

    Must be called with development/fold training rows only -- never with the
    holdout or live rows -- so the walk-forward validation stays leakage-safe.
    """
    global V26_ASSIST_FACTORS, V26_GOAL_FACTORS

    if not V26_ENABLED:
        V26_ASSIST_FACTORS, V26_GOAL_FACTORS = {}, {}
        return {}, {}

    key = (label, len(training_frame))
    if key not in _V26_FACTOR_CACHE:
        assists = fx.fit_event_conversion_factors(
            training_frame,
            realised_col="assists",
            expected_col="expected_assists",
            bounds=fx.ASSIST_CONVERSION_BOUNDS,
        ) if V26_APPLY_ASSIST_CONVERSION else {}
        goals = fx.fit_event_conversion_factors(
            training_frame,
            realised_col="goals_scored",
            expected_col="expected_goals",
            bounds=fx.GOAL_CONVERSION_BOUNDS,
        ) if V26_APPLY_GOAL_CONVERSION else {}
        _V26_FACTOR_CACHE[key] = (assists, goals)
        print(
            f"[v2.6] conversion factors ({label}, {len(training_frame):,} rows)"
            f"\\n       assists/xA: "
            f"{ {k: round(v, 3) for k, v in assists.items()} }"
            f"\\n       goals/xG  : "
            f"{ {k: round(v, 3) for k, v in goals.items()} }"
        )

    V26_ASSIST_FACTORS, V26_GOAL_FACTORS = _V26_FACTOR_CACHE[key]
    return V26_ASSIST_FACTORS, V26_GOAL_FACTORS


def v26_clear_conversion():
    """Reset to identity factors (used by the A/B comparison cell)."""
    global V26_ASSIST_FACTORS, V26_GOAL_FACTORS
    V26_ASSIST_FACTORS, V26_GOAL_FACTORS = {}, {}


print("[v2.6] mechanism fixes loaded; V26_ENABLED =", V26_ENABLED)
'''


# --------------------------------------------------------------------------
# anchored edits
# --------------------------------------------------------------------------

EDITS: list[tuple[str, str, str]] = []

# P1 WITHDRAWN. An earlier revision changed GOAL_POINTS_BY_POSITION["GK"] from
# 10.0 to 6.0. That was wrong: current FPL awards 10 points for a goalkeeper
# goal. The notebook's original value is correct and is now left untouched.

# P2 ---------------------------------------------------------------------
EDITS.append((
    "P2 defensive contribution minutes double-count",
    '''    blended = (
        0.30 * poisson_probability
        + 0.70 * empirical_probability
    ).clip(0.0, 1.0)

    # The award is practically unreachable without substantial minutes.
    minute_scale = (
        pd.to_numeric(data["expected_minutes"], errors="coerce")
        .fillna(0.0)
        .div(90.0)
        .clip(0.0, 1.0)
        .pow(0.65)
    )
    blended = (blended * minute_scale).clip(0.0, 1.0)''',
    '''    # The award is practically unreachable without substantial minutes.
    minute_scale = (
        pd.to_numeric(data["expected_minutes"], errors="coerce")
        .fillna(0.0)
        .div(90.0)
        .clip(0.0, 1.0)
        .pow(0.65)
    )

    # v2.6: `expected_defensive_actions` already carries expected_minutes/90,
    # so the Poisson branch must not be scaled by minutes a second time. Only
    # the empirical branch (a raw award frequency) needs `minute_scale`.
    blended = (
        0.30 * poisson_probability
        + 0.70 * empirical_probability * minute_scale
    ).clip(0.0, 1.0)''',
))

# P3 ---------------------------------------------------------------------
EDITS.append((
    "P3 realised-scale goal/assist conversion",
    '''    result["xp_goals"] = (
        result["projected_goals"]
        * position.map(GOAL_POINTS_BY_POSITION).fillna(0.0)
    )
    result["xp_assists"] = 3.0 * result["projected_assists"]''',
    '''    # v2.6: FPL `expected_assists` is not on the realised-assist scale
    # (pooled 2023-26: DEF 1.31, MID 1.47, FWD 2.39). `expected_goals` is
    # close to 1.0 but is corrected too so the assumption stays monitored.
    # Factors are set by v26_set_conversion() on TRAINING rows only; when
    # unset both default to 1.0 and this is a no-op.
    _v26_assists = globals().get("V26_ASSIST_FACTORS") or {}
    _v26_goals = globals().get("V26_GOAL_FACTORS") or {}
    if _v26_goals:
        result["projected_goals"] = (
            result["projected_goals"]
            * _v26_position_factor(position, _v26_goals)
        )
    if _v26_assists:
        result["projected_assists"] = (
            result["projected_assists"]
            * _v26_position_factor(position, _v26_assists)
        )

    result["xp_goals"] = (
        result["projected_goals"]
        * position.map(GOAL_POINTS_BY_POSITION).fillna(0.0)
    )
    result["xp_assists"] = 3.0 * result["projected_assists"]''',
))

# P4 ---------------------------------------------------------------------
# The mask must run BEFORE add_historical_player_features(), which builds the
# shifted xg_per_fixture / xa_per_fixture rolling history. Masking after it
# (as v2.6.0 did) cleaned the training targets but left the fabricated zeros
# baked into every rolling and career xG/xA predictor feature.
EDITS.append((
    "P4 mask fabricated xG/xA coverage before feature engineering",
    """historical_model_data, historical_player_gameweeks = (
    add_historical_player_features(
        historical_fpl
    )
)""",
    """# v2.6: 2022-23 gameweeks 1-15 carry expected_goals/expected_assists of
# exactly 0.0 (not null) for every player -- 4,036 rows containing 370 real
# goals and 331 real assists. Those pass the `.notna()` training filter and are
# learned as genuine zero-xG observations. They must be flagged as missing
# BEFORE add_historical_player_features(), because that function builds the
# shifted rolling/career xG/xA features that every later model consumes.
if V26_ENABLED and V26_APPLY_COVERAGE_MASK:
    historical_fpl = fx.mask_fabricated_event_coverage(
        historical_fpl,
        season_col="season",
        gameweek_col="gameweek",
        minutes_col="minutes",
    )

historical_model_data, historical_player_gameweeks = (
    add_historical_player_features(
        historical_fpl
    )
)"""),
)


# P5 ---------------------------------------------------------------------
EDITS.append((
    "P5 forecast cache signature",
    '''    "goal_points": GOAL_POINTS_BY_POSITION,
    "clean_sheet_points": CLEAN_SHEET_POINTS_BY_POSITION,''',
    '''    "goal_points": GOAL_POINTS_BY_POSITION,
    "clean_sheet_points": CLEAN_SHEET_POINTS_BY_POSITION,
    # v2.6: these entries force the stored xP-calibration, premium-attacker and
    # v2.4 artifacts to be rejected as stale. They were fitted against the
    # uncompressed-assist xP distribution and would otherwise double-correct.
    "v26_enabled": V26_ENABLED,
    "v26_assist_conversion": V26_APPLY_ASSIST_CONVERSION,
    "v26_goal_conversion": V26_APPLY_GOAL_CONVERSION,
    "v26_assist_bounds": fx.ASSIST_CONVERSION_BOUNDS,
    "v26_goal_bounds": fx.GOAL_CONVERSION_BOUNDS,
    "v26_conversion_shrinkage": fx.CONVERSION_SHRINKAGE_EVENTS,
    "v26_coverage_mask": True,
    "v26_dc_minutes_fix": True,
    "v26_wildcard_threshold_only": V26_WILDCARD_THRESHOLD_ONLY,
    "v26_coverage_mask_applied": V26_APPLY_COVERAGE_MASK,''',
))

# P6 ---------------------------------------------------------------------
EDITS.append((
    "P6a set conversion before the holdout components",
    '''holdout_xp_components = calculate_component_expected_points(
    holdout_xp_inputs,''',
    '''v26_set_conversion(development_points_data, label="development")
holdout_xp_components = calculate_component_expected_points(
    holdout_xp_inputs,''',
))

EDITS.append((
    "P6b set conversion before the live components",
    '''live_xp_components = calculate_component_expected_points(
    live_expected_points_inputs,''',
    '''v26_set_conversion(final_historical_points_data, label="live")
live_xp_components = calculate_component_expected_points(
    live_expected_points_inputs,''',
))

EDITS.append((
    "P6c set conversion before the backtest components",
    '''    xp_components = calculate_component_expected_points(
        xp_inputs,
        development_defensive_action_priors,''',
    '''    v26_set_conversion(development_points_data, label="development")
    xp_components = calculate_component_expected_points(
        xp_inputs,
        development_defensive_action_priors,''',
))

EDITS.append((
    "P6d set conversion before the walk-forward fold components",
    '''    fold_xp_components = calculate_component_expected_points(
        fold_xp_inputs,''',
    '''    v26_set_conversion(fold_development_points, label=f"fold:{holdout_season}")
    fold_xp_components = calculate_component_expected_points(
        fold_xp_inputs,''',
))

# P7 ---------------------------------------------------------------------
EDITS.append((
    "P7 clear stale idempotence baselines after v2.5 integration",
    '''    result["v25_probabilistic_xp_adjustment"] = (
        result["expected_points"] - result["pre_v25_expected_points"]
    )
    return result''',
    '''    result["v25_probabilistic_xp_adjustment"] = (
        result["expected_points"] - result["pre_v25_expected_points"]
    )

    # v2.6: this function has just recomputed `expected_points` from the role
    # states. Any stored v2.4/premium idempotence baseline is now stale, and a
    # later apply_v24_targeted_forecast_layer() would reset `expected_points`
    # to its pre-v2.5 value, silently discarding this integration.
    if globals().get("V26_ENABLED"):
        result = fx.clear_stale_layer_baselines(result)
    return result''',
))

# P9 ---------------------------------------------------------------------
EDITS.append((
    "P9 Section 16 missing holdout_season on the v2.5 candidate fold",
    """        _candidate_fold = _v25_apply_existing_post_layers_to_fold(
            _candidate_fold,
            _bundle,
        )

        _base_fold = _v24_base_fold_frame(_season)""",
    """        _candidate_fold = _v25_apply_existing_post_layers_to_fold(
            _candidate_fold,
            _bundle,
        )

        # v2.6 fix for a pre-existing Section 16 bug. _v25_fold_metrics() calls
        # _v24_player_gw(), which groups by ["holdout_season", "gameweek", id].
        # _v24_base_fold_frame() sets that column on the BASE fold, but nothing
        # sets it on the candidate fold, so the first real v2.5 run dies with
        # KeyError: 'holdout_season'. The audit log records v2.5 as
        # "implemented, pending executed validation", which is why this had
        # never surfaced.
        _candidate_fold["holdout_season"] = str(_season)

        _base_fold = _v24_base_fold_frame(_season)"""),
)

# P10 --------------------------------------------------------------------
# apply_premium_attacker_adjustment() rebuilds projected_goals/projected_assists
# straight from predicted_*_per90 for EVERY row, which silently strips the
# Fantasy-assist conversion back out -- for exactly the premium attackers F1
# matters most for. Route it through the same conversion helper.
EDITS.append((
    "P10 premium-attacker layer must apply the same conversion",
    """    result["projected_goals"] = (
        pd.to_numeric(result["predicted_xg_per90"], errors="coerce").fillna(0.0)
        * minutes_fraction
    )
    result["projected_assists"] = (
        pd.to_numeric(result["predicted_xa_per90"], errors="coerce").fillna(0.0)
        * minutes_fraction
    )""",
    """    # v2.6: this rebuild must go through the same xA -> Fantasy-assist
    # conversion the component model uses, otherwise an approved premium
    # payload silently reverts F1 for premium attackers.
    _v26_pos = result[position_col]
    result["projected_goals"] = (
        pd.to_numeric(result["predicted_xg_per90"], errors="coerce").fillna(0.0)
        * minutes_fraction
        * _v26_position_factor(_v26_pos, globals().get("V26_GOAL_FACTORS") or {})
    )
    result["projected_assists"] = (
        pd.to_numeric(result["predicted_xa_per90"], errors="coerce").fillna(0.0)
        * minutes_fraction
        * _v26_position_factor(_v26_pos, globals().get("V26_ASSIST_FACTORS") or {})
    )"""),
)


# P11 --------------------------------------------------------------------
# build_walk_forward_fold_bundle() returns early on a cache hit, so the
# v26_set_conversion() call inside it never runs for a cached fold. Section 16
# would then score a fold with whichever factors were set last -- possibly from
# a LATER season, which is leakage. Set them explicitly from the bundle.
EDITS.append((
    "P11 set conversion from the fold bundle before v2.5 integration",
    """        _raw_fold = _bundle["holdout_expected_points"].copy()
        _scenario_input = _v25_prepare_fold_for_integration(
            _raw_fold,
            _bundle,
        )""",
    """        _raw_fold = _bundle["holdout_expected_points"].copy()

        # v2.6: build_walk_forward_fold_bundle() short-circuits on a cache hit,
        # so its internal v26_set_conversion() may not have run for this fold.
        # Set the factors from this bundle's own development rows so a cached
        # fold can never be scored with another season's factors.
        v26_set_conversion(
            _bundle["development_points_data"],
            label=f"fold:{_season}",
        )

        _scenario_input = _v25_prepare_fold_for_integration(
            _raw_fold,
            _bundle,
        )"""),
)


# P12 --------------------------------------------------------------------
# Same hazard on the live path: after Section 16 the globals hold the last
# historical fold's factors, and attach_or_apply_v25_live_minutes() re-scores
# live xP through calculate_component_expected_points().
EDITS.append((
    "P12 restore live conversion factors before live v2.5 integration",
    """    if active_v25_probabilistic_minutes(artifact):
        result = integrate_v25_probabilistic_minutes_xp(""",
    """    if active_v25_probabilistic_minutes(artifact):
        # v2.6: Section 16 leaves the last historical fold's conversion factors
        # in the globals. Restore the live ones before re-scoring live xP.
        if globals().get("V26_ENABLED"):
            v26_set_conversion(
                globals()["final_historical_points_data"],
                label="live",
            )
        result = integrate_v25_probabilistic_minutes_xp("""),
)

# P14 --------------------------------------------------------------------
# Validated chip change. Requiring a structural trigger ON TOP of the uplift
# threshold is right for Triple Captain / Bench Boost / Free Hit, whose value
# comes from doubles and blanks. It is wrong for the Wildcard, whose value
# comes from squad quality and needs no fixture event.
# Measured over two held-out seasons (see fixes/CHIP_TIMING.md):
#     threshold only                    99.0 realised points
#     random legal timing               59.6
#     threshold AND structural (now)    59.0   <- no better than random
_P14_OLD = "\n".join([
    '    elif chip_name == "wildcard":',
    "        passed = (",
    "            expiry_pressure",
    "            or exceptional",
    "            or int(normal_transfer_count) >= 2",
    "            or float(normal_hit_cost) > 0",
    "        )",
    '        reason = "expiry pressure, major squad surgery or exceptional uplift"',
])
_P14_NEW = "\n".join([
    '    elif chip_name == "wildcard":',
    "        # v2.6: the structural gate is deliberately NOT applied to the",
    "        # wildcard. Its value is squad quality, not fixture structure, and",
    "        # requiring a double/blank/expiry trigger cost about 40 points a",
    "        # season in validation (99.0 -> 59.0 realised, no better than",
    "        # random timing). CHIP_MINIMUM_UPLIFT['wildcard'] remains the gate.",
    '        if globals().get("V26_WILDCARD_THRESHOLD_ONLY", True):',
    "            passed = True",
    '            reason = "uplift threshold alone (v2.6 validated)"',
    "        else:",
    "            passed = (",
    "                expiry_pressure",
    "                or exceptional",
    "                or int(normal_transfer_count) >= 2",
    "                or float(normal_hit_cost) > 0",
    "            )",
    '            reason = "expiry pressure, major squad surgery or exceptional uplift"',
])
EDITS.append(("P14 wildcard fires on its uplift threshold alone", _P14_OLD, _P14_NEW))


# P15 --------------------------------------------------------------------
# Section 11 refits and rewrites the xP-calibration and premium-attacker
# artifacts AFTER Section 7 has already built the live forecast. Section 12
# then compares artifact ids and raises:
#     "Validation artifacts changed after the live projections were generated."
# The notebook's own comment accepts this and asks for a kernel restart plus a
# full rerun. That is expensive and unnecessary: every input needed to rebuild
# the live post-layers is still in memory.
#
# The guard is correct and is NOT suppressed. Instead the live forecast is
# rebuilt from `live_xp_components` through the same post-layer chain Section 7
# uses, with the freshly validated payloads, so the projections genuinely do
# match the current artifacts.
#
# It must rebuild from the components rather than re-applying on top, because
# neither apply_premium_attacker_adjustment nor apply_xp_calibration_payload is
# idempotent: each captures its own "pre_" baseline from whatever it is handed,
# so a second application compounds the correction.
_P15_OLD = "\n".join([
    "# Make the new artifacts visible in this kernel for inspection. Live forecasts",
    "# generated earlier in the notebook still need a restart/rerun before Section 12.",
    "xp_calibration_cache = xp_calibration_artifact",
    "premium_attacker_cache = premium_attacker_artifact",
])
_P15_NEW = "\n".join([
    "# Make the new artifacts visible in this kernel for inspection.",
    "xp_calibration_cache = xp_calibration_artifact",
    "premium_attacker_cache = premium_attacker_artifact",
    "",
    "",
    "def _v26_refresh_live_after_validation():",
    '    """Rebuild the live post-layers against the just-refitted artifacts.',
    "",
    "    Returns True when the live forecast was refreshed. Without this the",
    "    Section 12 artifact-state guard fires on every run that rebuilds an",
    "    artifact, which is every first run after a signature change.",
    '    """',
    "    g = globals()",
    "    if not g.get('V26_ENABLED', False):",
    "        return False",
    "    needed = ('live_expected_points_inputs', 'live_xp_components',",
    "              'LIVE_VALIDATION_ARTIFACT_STATE')",
    "    if any(g.get(name) is None for name in needed):",
    "        return False",
    "",
    "    state = g['LIVE_VALIDATION_ARTIFACT_STATE']",
    "    fresh_xp_id = xp_calibration_cache.get('artifact_id') if xp_calibration_cache else None",
    "    fresh_premium_id = (premium_attacker_cache.get('artifact_id')",
    "                        if premium_attacker_cache else None)",
    "    if (state.get('xp_artifact_id') == fresh_xp_id",
    "            and state.get('premium_artifact_id') == fresh_premium_id):",
    "        return False          # nothing changed; the guard would pass anyway",
    "",
    "    # Live conversion factors, in case a fold left its own in the globals.",
    "    if 'v26_set_conversion' in g and 'final_historical_points_data' in g:",
    "        v26_set_conversion(g['final_historical_points_data'], label='live')",
    "",
    "    rebuilt = pd.concat(",
    "        [",
    "            g['live_expected_points_inputs'].reset_index(drop=True),",
    "            g['live_xp_components'].reset_index(drop=True),",
    "        ],",
    "        axis=1,",
    "    )",
    "    premium_payload = active_premium_attacker_payload(premium_attacker_cache)",
    "    rebuilt = apply_premium_attacker_adjustment(",
    "        rebuilt, premium_payload,",
    "        position_col='model_position', price_col='price',",
    "    )",
    "    rebuilt = apply_sparse_player_market_prior(rebuilt)",
    "    xp_payload = active_xp_calibration_payload(xp_calibration_cache)",
    "    rebuilt = apply_xp_calibration_payload(",
    "        rebuilt, xp_payload,",
    "        position_col='model_position', price_col='price',",
    "        value_col='expected_points',",
    "    )",
    "",
    "    g['live_xp_payload'] = xp_payload",
    "    g['live_premium_payload'] = premium_payload",
    "    g['live_expected_points'] = rebuilt",
    "    state['xp_artifact_id'] = fresh_xp_id",
    "    state['premium_artifact_id'] = fresh_premium_id",
    "    state['xp_applied'] = xp_payload is not None",
    "    state['premium_applied'] = premium_payload is not None",
    "    print('[v2.6] live forecast rebuilt against the refitted artifacts '",
    "          '- no kernel restart needed')",
    "    return True",
    "",
    "",
    "_v26_refresh_live_after_validation()",
])
EDITS.append(("P15 rebuild live projections after Section 11 refits artifacts",
              _P15_OLD, _P15_NEW))



# P16 --------------------------------------------------------------------
# The freshness gate requires changed_player_rows == 0 across every player in
# the game. FPL re-prices players nightly and updates injury news continuously,
# so any Run-All long enough to reach Section 12 will almost always trip it on
# someone irrelevant -- a GBP 0.1m price tick on a player who would never be
# owned.
#
# Unlike the artifact guard (P15) this is NOT an internal inconsistency: the
# outside world really did change. So the gate is kept and made MATERIAL:
#   * a gameweek rollover or a fixture-schedule change still blocks outright
#   * price-only moves are refreshed into the live frame and do not block
#   * availability changes block only for players the model projects could
#     actually play, and so could be owned or bought
# The full change list is printed either way.
#
# The replacement lives in snippets/ rather than inline because it contains
# nested quoting that does not survive being embedded in this file.
_P16_DIR = HERE / "snippets"
EDITS.append((
    "P16 materiality-aware freshness gate",
    (_P16_DIR / "p16_old.txt").read_text(encoding="utf-8").rstrip("\n"),
    (_P16_DIR / "p16_new.txt").read_text(encoding="utf-8").rstrip("\n"),
))


# P17 --------------------------------------------------------------------
# The v2.4 upper-tail layer fails its gate on EXACTLY one criterion -- MAE --
# while improving every decision metric. From the notebook's own run:
#     mae_improvement      -0.0397   vs required >= +0.01   <- the only failure
#     abs_bias_improvement +0.1854   vs required >= +0.03      PASS
#     overall_ndcg_change  +0.0032   vs allowed  >= -0.004     PASS
#     overall_top10_change +0.5658                             PASS
# An independent paired test on the same cached folds agrees: best-legal-XI
# points +1.36 per gameweek, NDCG +0.0045, top-10 +0.39.
#
# Requiring MAE to improve is the wrong criterion for a layer whose purpose is
# to re-rank the upper tail: lifting under-predicted premiums necessarily moves
# individual predictions further from their mean outcome. This was pre-declared
# in AUDIT_RESPONSE.md section 9 ("Overall MAE is explicitly NOT a gate") before
# any of these numbers were seen.
#
# Only the MAE bar is relaxed, and only for the upper tail. The bias, NDCG and
# top-10 gates are untouched, so the notebook's own validation still decides.
# The MID reranker is unaffected and stays rejected: it independently fails on
# spearman (-0.0023 vs required +0.005) and MAE (+0.045 vs allowed +0.03).
EDITS.append((
    "P17 upper-tail gate no longer requires MAE to improve",
    "V24_UPPER_TAIL_MIN_MAE_IMPROVEMENT = 0.01",
    "\n".join([
        "# v2.6: was 0.01, then briefly -0.05. Both were arbitrary numbers. If",
        "# MAE is the wrong test for a ranking layer -- pre-declared in",
        "# AUDIT_RESPONSE.md section 9 -- then the principled implementation is",
        "# to drop the criterion, not to pick a cutoff that happens to pass.",
        "# The bias, NDCG and top-10 gates below are UNCHANGED and still decide.",
        "V24_UPPER_TAIL_MIN_MAE_IMPROVEMENT = (",
        "    float('-inf')",
        "    if globals().get('V26_RELAX_UPPER_TAIL_MAE_GATE', True)",
        "    else 0.01",
        ")",
    ]),
))


# P18 / P19 ----------------------------------------------------------------
# Two chip-gating defects, both confirmed against the 2026-10-08 production run.
#
# P18  The weekly path required `current_uplift >= threshold` unconditionally,
#      with no expiry-pressure override. The backtest has one. Consequence: if
#      no chip cleared its bar, none could ever be played, right through the
#      GW19 expiry. Confirmed live: all four chips showed
#      passes_use_now_rule = False (TC 5.28/7.0, BB 8.20/9.0, FH 3.89/8.0,
#      WC 3.89/10.0) and nothing in the path could override that.
#
# P19  Only the WILDCARD uplift is a discounted horizon sum; TC/BB/FH uplifts
#      are single-gameweek values, so their bars are already correct. The same
#      {7,9,8,10} constants appear as BACKTEST_CHIP_MINIMUM_UPLIFT with
#      BACKTEST_HORIZON = 6, so the wildcard's 10.0 was set against a six-week
#      sum but is applied to a one-week number under greedy_1gw_no_hits.
#      Rescaling it would fire the wildcard on a one-week reshuffle, so it is
#      suppressed explicitly below instead, and P18 still forces it out before
#      it expires.
_SNIP = HERE / "snippets"
EDITS.append((
    "P18 expiry-pressure override in the weekly chip gate",
    (_SNIP / "p18_old.txt").read_text(encoding="utf-8").rstrip("\n"),
    (_SNIP / "p18_new.txt").read_text(encoding="utf-8").rstrip("\n"),
))
EDITS.append((
    "P19 wildcard is not judged at a horizon too short to mean anything",
    (_SNIP / "p19_old.txt").read_text(encoding="utf-8").rstrip("\n"),
    (_SNIP / "p19_new.txt").read_text(encoding="utf-8").rstrip("\n"),
))


# P20 / P21 ----------------------------------------------------------------
# Found by adversarial review of P18, and both confirmed against the executed
# production notebook.
#
# P20  At horizon 1 the wildcard and free-hit uplifts are IDENTICAL by
#      construction (both 3.8910851937150426 in production). normalised_strength
#      is uplift/threshold, so the wildcard's bigger bar always sorts it last,
#      scheduling it into GW19 - the final week before the chip set resets, and
#      the worst week for the only chip whose value scales with weeks remaining.
#
# P21  expiry_pressure_at fires at exactly the last week that still fits, so a
#      single disrupted gameweek loses a chip with no recovery path. One week of
#      slack costs almost nothing on a calendar with no doubles or blanks.
EDITS.append((
    "P20 under expiry, play the wildcard first rather than last",
    (_SNIP / "p20_old.txt").read_text(encoding="utf-8").rstrip("\n"),
    (_SNIP / "p20_new.txt").read_text(encoding="utf-8").rstrip("\n"),
))
EDITS.append((
    "P21 one week of slack before the expiry cascade",
    (_SNIP / "p21_old.txt").read_text(encoding="utf-8").rstrip("\n"),
    (_SNIP / "p21_new.txt").read_text(encoding="utf-8").rstrip("\n"),
))


# --------------------------------------------------------------------------
# new cells inserted after the cell containing an anchor
# --------------------------------------------------------------------------

EXECUTION_CARD = """# =============================================================================
# v2.6 EXECUTION CARD — the single instruction for this deadline
# =============================================================================
# Everything below is already computed by Section 12; this cell only
# consolidates it into one unambiguous block so the recommendation can be
# executed without reading the diagnostics above.
#
# Every lookup is defensive: if a variable is missing the card says so rather
# than failing, because a half-printed instruction is worse than a loud gap.

def _v26_card() -> None:
    import pandas as _pd
    import numpy as _np

    G = globals()
    missing = []

    def need(name, default=None):
        if name not in G or G[name] is None:
            missing.append(name)
            return default
        return G[name]

    gw = need("weekly_next_gameweek", "?")
    deadline = G.get("weekly_deadline")
    chip = need("weekly_chip_recommendation", {}) or {}
    xi = need("weekly_starting_xi", _pd.DataFrame())
    bench = need("weekly_bench", _pd.DataFrame())
    captain = need("weekly_captain", "?")
    vice = need("weekly_vice_captain", "?")
    action = need("weekly_primary_action", "?")
    score = G.get("weekly_expected_score", float("nan"))

    bar = "=" * 70
    print(bar)
    deadline_text = ""
    if deadline is not None and _pd.notna(deadline):
        deadline_text = f"  —  deadline {deadline:%a %d %b %H:%M}"
    print(f"  GAMEWEEK {gw} — EXECUTE THIS{deadline_text}")
    print(bar)

    # 1. chip
    chip_name = chip.get("chip") or "NONE"
    chip_action = chip.get("action", "?")
    print(f"  1. CHIP       : {str(chip_name).replace('_', ' ').upper()}")
    print(f"                  {chip_action}")
    if chip.get("reason"):
        print(f"                  why: {chip['reason']}")

    # 2. transfers
    print(f"  2. TRANSFERS  : {action}")
    moves = G.get("weekly_transfer_moves")
    if isinstance(moves, _pd.DataFrame) and not moves.empty:
        for row in moves.itertuples(index=False):
            out_name = getattr(row, "outgoing", "?")
            in_name = getattr(row, "incoming", "?")
            print(f"                  OUT {out_name}   ->   IN {in_name}")
    hit = G.get("weekly_hit_cost")
    if hit is not None and _pd.notna(hit):
        print(f"                  points hit: {float(hit):.0f}")

    # 3. starting XI, grouped by position
    print("  3. STARTING XI:")
    if isinstance(xi, _pd.DataFrame) and not xi.empty:
        name_col = next((c for c in ("web_name", "player_name", "name")
                         if c in xi.columns), None)
        pos_col = next((c for c in ("model_position", "position") if c in xi.columns), None)
        if name_col and pos_col:
            shape = []
            for pos in ("GK", "DEF", "MID", "FWD"):
                members = xi.loc[xi[pos_col].astype(str).str.upper()
                                 .str.startswith(pos[:3]), name_col].tolist()
                if pos != "GK":
                    shape.append(str(len(members)))
                tagged = []
                for m in members:
                    suffix = " (C)" if m == captain else (" (V)" if m == vice else "")
                    tagged.append(f"{m}{suffix}")
                print(f"                  {pos:4s} {', '.join(tagged) if tagged else '-'}")
            print(f"                  formation {'-'.join(shape)}")
        else:
            print("                  (name/position columns not found)")
    else:
        print("                  (no starting XI available)")

    # 4. bench, in autosub order
    print("  4. BENCH ORDER: (autosubs happen in this order)")
    if isinstance(bench, _pd.DataFrame) and not bench.empty:
        name_col = next((c for c in ("web_name", "player_name", "name")
                         if c in bench.columns), None)
        pos_col = next((c for c in ("model_position", "position")
                        if c in bench.columns), None)
        for i, row in enumerate(bench.itertuples(index=False), start=1):
            nm = getattr(row, name_col) if name_col else "?"
            ps = getattr(row, pos_col) if pos_col else ""
            print(f"                  {i}. {nm} ({ps})")
    else:
        print("                  (no bench available)")

    # 5. captaincy
    print(f"  5. CAPTAIN    : {captain}")
    print(f"     VICE       : {vice}")

    print("-" * 70)
    if score is not None and _pd.notna(score):
        print(f"  Projected points for GW{gw}: {float(score):.1f}")

    gap = G.get("weekly_captain_xp_gap")
    conf = G.get("weekly_captain_confidence")
    if gap is not None and _np.isfinite(gap):
        tie = "  <- near-tie, treat as a coin flip" if abs(float(gap)) < 0.35 else ""
        print(f"  Captain confidence: {conf} (xP gap {float(gap):+.2f}){tie}")
    if G.get("weekly_confidence"):
        print(f"  Decision confidence: {G['weekly_confidence']}")

    warnings_out = []
    ev = G.get("weekly_transfer_evidence_summary") or {}
    if ev.get("action_low_evidence"):
        warnings_out.append(f"LOW-EVIDENCE TRANSFER: {ev.get('reason')}")
    readiness = G.get("weekly_production_readiness_status")
    if readiness and str(readiness).upper() not in ("OK", "READY", "TRUE"):
        warnings_out.append(f"production readiness: {readiness}")
    if G.get("V26_APPLY_ASSIST_CONVERSION"):
        warnings_out.append(
            "v2.6 assist-conversion ACTIVE — confirm the Section 11 gate passed"
        )
    if G.get("V26_RELAX_UPPER_TAIL_MAE_GATE"):
        _v24 = G.get("v24_targeted_forecast_cache") or {}
        _ut = (_v24.get("production_decision") or {}).get("upper_tail") or {}
        warnings_out.append(
            "v2.4 upper-tail MAE gate relaxed; layer approved="
            + str(_ut.get("approved"))
        )
    if missing:
        warnings_out.append("missing variables: " + ", ".join(sorted(set(missing))))
    for w in warnings_out:
        print(f"  ! {w}")

    print(bar)
    print("  Reminder: this is the ONLY action for this deadline. The rolling")
    print("  plan beyond this gameweek is provisional — rerun before next week.")
    print(bar)


_v26_card()
"""

NEW_CELLS: list[tuple[str, str]] = [
    ("print(\"Primary action:\", weekly_primary_action)", EXECUTION_CARD),
]


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def cell_source(cell: dict) -> str:
    source = cell["source"]
    return source if isinstance(source, str) else "".join(source)


def set_cell_source(cell: dict, text: str) -> None:
    lines = text.splitlines(keepends=True)
    cell["source"] = lines


def apply_patch(in_path: Path, out_path: Path) -> int:
    notebook = json.loads(in_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    code = [(i, c) for i, c in enumerate(cells) if c["cell_type"] == "code"]

    # Locate every anchor first; abort before writing if anything is wrong.
    plan: list[tuple[str, int, str, str]] = []
    problems: list[str] = []
    for name, old, new in EDITS:
        hits = [i for i, c in code if old in cell_source(c)]
        if len(hits) == 0:
            problems.append(f"{name}: anchor NOT FOUND")
        elif len(hits) > 1:
            problems.append(f"{name}: anchor ambiguous, matches cells {hits}")
        else:
            plan.append((name, hits[0], old, new))

    if problems:
        print("ABORTED - the notebook does not match the expected v2.5.0 source:")
        for problem in problems:
            print("  *", problem)
        print("\nNothing was written.")
        return 1

    # Insert the setup cell after the last import-bearing cell near the top.
    anchor = next(
        (i for i, c in code if "import requests" in cell_source(c)),
        code[0][0],
    )
    setup = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": SETUP_CELL.splitlines(keepends=True),
    }
    cells.insert(anchor + 1, setup)
    shift = anchor + 1

    print(f"Patching {in_path.name}")
    print(f"  P8 setup cell inserted at index {shift}")

    for name, index, old, new in plan:
        target = cells[index + 1] if index >= shift else cells[index]
        text = cell_source(target)
        assert text.count(old) == 1, name
        set_cell_source(target, text.replace(old, new))
        print(f"  {name}")

    # P13: append standalone cells after their anchor cell.
    for anchor_text, source in NEW_CELLS:
        hits = [i for i, c in enumerate(cells)
                if c["cell_type"] == "code" and anchor_text in cell_source(c)]
        if len(hits) != 1:
            print(f"  ABORTED - execution-card anchor matched {len(hits)} cells")
            return 1
        cells.insert(hits[0] + 1, {
            "cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True),
        })
        print("  P13 execution card inserted")

    # Every touched cell must still parse.
    failures = []
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        try:
            ast.parse(cell_source(cell))
        except SyntaxError as exc:
            failures.append(f"cell {index}: {exc}")
    if failures:
        print("\nABORTED - patched notebook does not parse:")
        for failure in failures:
            print("  *", failure)
        return 1

    if out_path.exists():
        backup = out_path.with_suffix(".ipynb.bak")
        shutil.copy2(out_path, backup)
        print(f"  existing output backed up to {backup.name}")

    out_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"Original {in_path.name} is unchanged.")
    print("\nNext: run the new notebook top to bottom. The v2.6 signature change")
    print("forces Section 11 to refit the xP-calibration and premium artifacts.")
    return 0


if __name__ == "__main__":
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    raise SystemExit(apply_patch(source, target))
