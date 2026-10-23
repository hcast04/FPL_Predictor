---
project: FPL Predictor
document: v2.6 mechanism-fix integration notes
as_of: 2026-10-05
applies_to: FPL_Predictor.ipynb (v2.5.0)
---

# v2.6 — mechanism fixes for upper-tail / high-minutes xP error

## Why this is not another v2.4

The audit log rejected the v2.4 upper-tail calibration because it corrected the
*symptom* (bias at high xP) with a fitted residual layer, and FPL decision
quality got worse: NDCG −0.0028, top-10 realised **−0.64 points/GW**.

Everything here changes a **mechanism** instead. On chronological held-out
seasons the direction reverses:

| Held-out season | ΔMAE | Δbias | ΔSpearman | ΔNDCG@10 | Δtop-10 realised |
|---|---:|---:|---:|---:|---:|
| 2024-25 (train 22-24) | +0.0087 | +0.0227 | +0.0012 | **+0.0116** | **+2.18 pts/GW** |
| 2025-26 (train 22-25) | +0.0069 | +0.0178 | +0.0016 | **+0.0113** | **+2.27 pts/GW** |
| *v2.4, for contrast* | — | — | — | *−0.0028* | *−0.64 pts/GW* |

Reproduce with `python validate_xp_fixes.py 2025-26` (downloads its own data).

### Honest caveat on the acceptance gates

Gate **G1 ("mean signed bias moves toward zero") FAILED** in both folds, and the
gate was declared before the result was seen. It should be read with care:
the validation harness is an independent, simpler re-implementation whose
baseline bias is **+0.044 (over-prediction)**, whereas the real notebook's
baseline bias is **negative** (audit §7.5: −0.512 at the 6+ bucket). Adding the
missing assist points moves bias up in both, which helps the notebook and hurts
the harness. The harness also produces almost no mass above 6 xP (8–9 rows of
26k), so it cannot measure upper-tail calibration directly.

**Therefore: treat the ranking/top-K gains as the validated result, and
re-check bias on the real notebook before enabling F1 in production.**

### Update after a full run — the upper tail IS measurable

An earlier version of the harness let unbounded per-90 history features blow up
to the model caps, which polluted the top buckets. With train-fitted
winsorisation in place the upper tail is populated and well behaved, and it
shows the predicted effect directly:

| Fold | Region | Baseline bias | v2.6 bias | Change | n (base/v2.6) |
|---|---|---:|---:|---:|---|
| 2024-25 | 5+ xP | −1.169 | −0.699 | **+0.470** | 55 / 72 |
| 2024-25 | 4+ xP | −0.012 | +0.081 | +0.093 | 243 / 343 |
| 2025-26 | 5+ xP | −0.924 | −0.780 | **+0.144** | 14 / 20 |
| 2025-26 | 4+ xP | +0.190 | +0.213 | +0.023 | 197 / 245 |

Both folds move toward zero in exactly the region audit §7.5 flagged. The
2024-25 fold is the informative one (55→72 rows at 5+ xP); 2025-26 has only
14–20 and should not be leaned on.

**G1 still fails, and the reason is now clear:** overall signed bias is a
weighted average dominated by the `<1` xP bucket (14,399 of 26,163 rows), where
this harness over-predicts. The gate measured the wrong thing for the
hypothesis under test. A correctly specified gate would have been "upper-tail
bias moves toward zero", which passes in both folds. Stating that after the
fact is weaker evidence than declaring it up front, so the real notebook's §7.5
table remains the deciding check.

---

## F1 — assists are scored on the wrong scale (primary fix)

`calculate_component_expected_points` does:

```python
result["projected_assists"] = data["predicted_xa_per90"].fillna(0.0) * minute_fraction
result["xp_assists"] = 3.0 * result["projected_assists"]
```

This treats FPL's `expected_assists` as expected assists. It is not. Pooled
2023-26 archived data, players with minutes > 0:

| Position | Assist pts awarded | 3 × xA (what the model uses) | Shortfall |
|---|---:|---:|---:|
| DEF | 2,016 | 1,619.6 | **+396** |
| MID | 5,733 | 4,254.8 | **+1,478** |
| FWD | 1,170 | 555.5 | **+615** |
| GK | 54 | 11.7 | +42 |

**844 assist points per season** are structurally invisible to the model. Per
match for a premium creator on 0.30 xA/90, 85 minutes:

| Position | Model | Corrected | Gap |
|---|---:|---:|---:|
| MID | 0.85 xP | 1.16 xP | **+0.31** |
| FWD | 0.85 xP | 1.53 xP | **+0.68** |
| DEF | 0.85 xP | 1.03 xP | +0.18 |

Audit §7.5 measured −0.512 bias at the 6+ bucket. This mechanism accounts for
most of it, and because the gap scales with both xA and minutes, it reproduces
exactly the pattern you described: **the players with the most minutes and the
most points carry the worst error.**

`expected_goals` needs no such fix — league-wide goals/xG is 0.94–1.00 for
2023-26. A goal factor is fitted anyway so the assumption stays monitored
(it comes out at 0.88–1.00 by position).

### Patch

In the cell that builds `holdout_expected_points` (cell 96) and the live cell
(cell ~130, `live_xp_components`), after `calculate_component_expected_points`:

```python
import sys; sys.path.insert(0, "fixes")
import fpl_xp_fixes as fx

assist_factors = fx.fit_event_conversion_factors(
    development_points_data,            # live: final_historical_points_data
    realised_col="assists", expected_col="expected_assists",
    bounds=fx.ASSIST_CONVERSION_BOUNDS,
)
goal_factors = fx.fit_event_conversion_factors(
    development_points_data,
    realised_col="goals_scored", expected_col="expected_goals",
    bounds=fx.GOAL_CONVERSION_BOUNDS,
)
holdout_xp_components = fx.apply_event_conversion(
    holdout_xp_components, holdout_xp_inputs,
    assist_factors=assist_factors, goal_factors=goal_factors,
    goal_points=fx.GOAL_POINTS_BY_POSITION,
)
```

Fit the factors on development/training rows **only**, and refit per fold inside
the walk-forward loop (cell ~177) so the validation stays leakage-safe.

---

## F2 — 2022-23 trains the attacking models on fabricated zeros

2022-23 gameweeks 1–15 have `expected_goals` and `expected_assists` of exactly
**0.0, not null**, for every player: 4,036 rows (35.6% of the season) containing
**370 real goals and 331 real assists**. `add_player_event_targets` divides by
minutes, giving `target_xg_per90 = 0.0`, and `fit_player_event_rate_models`
filters on `.notna()` — so these pass straight into training as genuine zero-xG
observations and depress every attacking rate.

2023-24, 2024-25 and 2025-26 are clean.

### Patch

Immediately after loading the archived seasons, before
`add_player_event_targets`:

```python
historical_player_data = fx.mask_fabricated_event_coverage(historical_player_data)
```

It flags any (season, gameweek) where every player with minutes has 0.0 xG as a
missing feed and sets those cells to NaN, so the existing imputer and `.notna()`
filters handle them correctly. Either apply this or drop 2022-23 from
`DEVELOPMENT_SEASONS`.

---

## F3 — defensive contribution applies minutes twice

In `estimate_defensive_return_probability`, `expected_defensive_actions` is
already `rate_per90 × expected_minutes/90`, so the Poisson branch carries
minutes. The blended probability is then multiplied again by
`minute_scale = (expected_minutes/90) ** 0.65`. Only the empirical branch
(a raw award frequency with no minutes in it) needs that multiplier.

DC is worth 0.7–1.85 xP of a defender's ~4.0 total, so the bug is real.

**Validated separately, and the decision impact is negligible.** The headline
+2.18 / +2.27 result does *not* cover F3 — `validate_xp_fixes.py` excludes DC
because the pre-2025/26 training seasons have no CBIT data. `validate_dc_fix.py`
closes that gap on 2025-26 (7,927 rows):

| Variant | Brier | mean P(DC) | bias | DEF Spearman | DEF NDCG@10 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.1072 | 0.1292 | −0.0266 | 0.3916 | 0.5059 |
| v2.6 fixed | 0.1072 | 0.1297 | −0.0261 | 0.3909 | 0.5059 |

Bias improves slightly, Brier is unchanged to four decimals, ranking is flat.
With forecast-style (rolling-mean) minutes instead of realised, the Brier delta
is −0.0002. So: keep it as a correctness fix, but **do not expect it to change
any decision**, and do not count it toward the F1 headline.

```python
holdout_xp_components = fx.fix_defensive_contribution_minutes(
    holdout_xp_components, holdout_xp_inputs
)
```

---

## F4 — WITHDRAWN (this was my error)

v2.6.0 changed `GOAL_POINTS_BY_POSITION["GK"]` from 10.0 to 6.0, claiming FPL
pays 6 for a goalkeeper goal. **That is wrong.** Current FPL awards **10** for
a goalkeeper goal — it was raised from 6. The notebook's original value was
correct, and v2.6.1 leaves it untouched.

This was a stale-knowledge error on my part: I treated a deliberate value as a
bug because it disagreed with an older version of the rules. No goalkeeper
scored in 2024-25 or 2025-26, so it cannot be settled from the archived data —
a points reconstruction validates every other position exactly (DEF goal = 6:
126/126 and 128/128 rows) but is silent on goalkeepers.

---

## F5 — stale idempotence baseline silently reverts v2.5

`apply_v24_targeted_forecast_layer` is idempotent by *resetting*
`expected_points` to `pre_v24_expected_points`. That column is never
invalidated when `integrate_v25_probabilistic_minutes_xp` recomputes
`expected_points`, so any later v2.4 call throws the v2.5 forecast away.
Demonstrated with the notebook's own function:

```
1. after v2.4          : 6.7 | baseline = 6.0
2. after v2.5 integrate: 7.4 | baseline = 6.0   <-- stale
3. any later v2.4 call : 6.7  <-- v2.5 result silently discarded
```

The current Run-All order does not trigger it, but re-running the Section 11D
cell does. Related: when v2.5 is active the v2.4 *correction* is dropped while
`apply_v24_uncertainty_payload` is still applied, so the uncertainty bands
describe a forecast that is no longer in use.

### Patch

At the end of `integrate_v25_probabilistic_minutes_xp`:

```python
result = fx.clear_stale_layer_baselines(result)
```

and optionally `fx.assert_no_stale_v24_baseline(live_expected_points)` as a
Section 12 guard.

---

## Tested and found sound — do not change

**The component → FPL points arithmetic.** Monte-Carlo scoring of 400k
simulated matches per archetype against real FPL rules agrees with
`calculate_component_expected_points` to within 0.1% for nailed starters
(DEF 3.342 vs 3.342; MID 6.748 vs 6.738; GK 3.670 vs 3.668). Rotation-risk
players are over-valued by ~3%, which is the point-estimate-minutes effect that
v2.5 targets.

**The clean-sheet exposure rule.** `exp(-λ · minutes/90)` looks suspicious
because it makes xP *rise* as minutes fall, but it is empirically correct:
fitting `exp(-λ · frac**γ)` on 2022-25 and testing on 2025-26 gives **γ = 0.975**
and the current form wins on Brier (0.1969 vs 0.1979 for a team-only model) and
log-loss. Within the same team-match, a player subbed between 60' and 89' really
does gain **+7.6pp** clean-sheet probability. Leave it alone.

---

## Open item — bonus is structurally mis-specified

Not fixed here; flagged with evidence. Bonus is modelled as a per-90 Poisson
rate (`alpha=2.0`, pooled across positions), but FPL bonus is **rank-based**
(top-3 BPS within a fixture). Real spread, player-seasons ≥900 minutes:

| Bonus decile | D1 | D5 | D10 |
|---|---:|---:|---:|
| bonus/90 | 0.000 | 0.214 | **0.838** |
| BPS/90 | 12.45 | 15.54 | 22.32 |

A 3.9× top-to-median spread; a pooled Poisson at `alpha=2.0` loses ~28% of
dispersion in simulation. The principled fix is to predict **BPS per 90**, then
convert BPS to expected bonus through a within-fixture rank model. That is a
larger change and deserves its own OOS gate.

---

## REQUIRED: invalidate the signed artifacts, or you will double-correct

`STRICT_CACHE_SIGNATURES = True`, and `load_signed_json_artifact` drops any
artifact whose `forecast_signature` differs from `FORECAST_CACHE_SIGNATURE`.
That protection only fires if the signature actually changes.

The stored xP-calibration payload, the premium-attacker payload and the v2.4
artifact were all fitted against the **uncompressed-assist** xP distribution.
Their whole job is to correct compression that F1 removes. If you apply these
fixes through a post-hoc module without touching the signature dict, the
signature is unchanged, the stale payloads load, and the correction is applied
**twice**.

Add the conversion configuration to the signature dict (cell ~116):

```python
FORECAST_CACHE_SIGNATURE = stable_configuration_signature({
    ...
    "v26_event_conversion": True,
    "v26_assist_bounds": fx.ASSIST_CONVERSION_BOUNDS,
    "v26_goal_bounds": fx.GOAL_CONVERSION_BOUNDS,
    "v26_conversion_shrinkage": fx.CONVERSION_SHRINKAGE_EVENTS,
    "v26_coverage_mask": True,
    "v26_dc_minutes_fix": True,
    "goal_points": fx.GOAL_POINTS_BY_POSITION,   # replaces the GK=10 dict
})
```

Then run Section 11 so the calibration and premium artifacts are refitted
against corrected xP. The policy scorecard also becomes stale by the project's
own cache rule — `greedy_1gw_no_hits` remains the sensible fallback, but it is
no longer a signed result for this forecast semantics.

---

## Suggested order

1. **F2** (data contamination) and **F5** (stale baseline) — pure correctness,
   apply now.
2. **F4** — trivial.
3. **F3** — small, well-understood.
4. **F1** — the one that matters. Apply, then re-run Section 11 and confirm
   the §7.5 bias table improves on the *real* notebook before promoting it.

## One more thing about v2.5

Minutes is the audit's stated next target, but the Monte-Carlo result above
shows the point-estimate-minutes error is ~0% for nailed starters and ~3% for
rotation players. The bias you are chasing is concentrated in **high-minutes,
high-points** players — the population v2.5 barely touches. v2.5 is still worth
running on its own merits, but it cannot fix this.

---

## Captaincy and chips — what exists, and what is actually validated

### Captaincy (Section 8) — implemented, never validated against alternatives

`PRODUCTION_CAPTAIN_HIGHEST_XP = False`, so the joint objective is used:

```
uplift = captain_xP + P(captain does not appear) * vice_xP
```

That is the correct FPL rule (the vice takes over only if the captain does not
play) and it is risk-neutral: it maximises expected points, with no ceiling or
differential weighting. No alternative captain rule has ever been scored
against it.

**Captain impact is not measurable at this sample size.** v2.6.0 reported
−0.08 pts/GW, but that was computed with the *wrong rule* — a naive highest-xP
argmax, which is the `PRODUCTION_CAPTAIN_HIGHEST_XP = True` fallback branch,
not the joint objective production actually uses. `captain_impact.py` now uses
the joint rule, and the sign flips:

| Season | Captain pick changed | Captain pts/GW baseline → v2.6 |
|---|---:|---|
| 2024-25 | 0 of 33 GWs (0%) | 7.21 → 7.21 (+0.00) |
| 2025-26 | 5 of 33 GWs (15%) | 6.36 → 6.94 (+0.58) |

Mean +0.29 pts/GW (+0.58 doubled) — but it rests on **five changed picks in one
season**. Neither the old −0.08 nor the new +0.29 is distinguishable from zero.
The correct statement is that this harness cannot measure captain impact, and
the starting XI is proxied (top 11 by xP) rather than optimised, so it is not a
production captain validation either way. The v2.6 evidence is for **squad and
transfer ranking**.

### Chips (Section 10) — implemented, NOT validated

All four chips are scored by incremental expected value against the best
no-chip path, with two chip sets (first expires at GW19), Free Hit blocked in
consecutive gameweeks, and a lazy/coarse lookahead. Production is deliberately
cautious: `WEEKLY_CHIP_DECISION_MODE = "conservative"` and
`WEEKLY_CHIP_REQUIRE_STRUCTURAL_TRIGGER = True`, so unless the horizon shows a
blank/double gameweek or expiry pressure, the recommendation is **"Save all
chips"** regardless of uplift.

Two things to be clear about:

1. **The minimum-uplift thresholds are judgement, not evidence.**
   `CHIP_MINIMUM_UPLIFT = {triple_captain: 7.0, bench_boost: 9.0, free_hit: 8.0,
   wildcard: 10.0}` — the notebook calls them "transparent conservative
   thresholds". They were not fitted or back-tested.

2. **No chip-policy scorecard exists.** Searching the notebook for
   `chip_policy` / `chip_scorecard` / `chip_validation` returns nothing. Chips
   appear in the single-season Section 10A backtest
   (`BACKTEST_INCLUDE_CHIPS = True`), but that is one season and compares
   nothing.

### The transfer-policy result has a caveat worth knowing

`WALK_FORWARD_INCLUDE_CHIPS = False` — *"Cross-season tests disable chips
because historical chip rules differ."* So the scorecard that selected
`greedy_1gw_no_hits` describes a **chip-free season**. It does not tell you how
the policy interacts with wildcards or free hits.

On strength of evidence, from the audit's own numbers (n = 3 seasons):

| Comparison | Gap | Season ranks | Read |
|---|---:|---|---|
| greedy vs rolling_6gw_no_hits | 80.0 pts | 1,3,1 vs 5,4,5 | greedy wins all three — reasonably convincing |
| greedy vs rolling_3gw_no_hits | 70.4 pts | 1,3,1 vs 2,5,3 | greedy ahead |
| greedy vs rolling_6gw_hits | 36.7 pts | 1,3,1 vs 3,2,4 | greedy ahead; supports not taking hits |
| greedy vs bank_aware | 8.4 pts | 1,3,1 vs 4,1,2 | a coin flip — the audit is right to call it diagnostic only |

"Don't take hits, plan one gameweek at a time" is the defensible conclusion.
"Greedy beats bank-aware" is not.

### Also: v2.6 makes all of the above stale

Policy, chip and captain decisions are all functions of `expected_points`. The
v2.6 signature change correctly invalidates the xP-calibration and
premium-attacker artifacts, but the **policy scorecard is not signature-gated**
— `WEEKLY_POLICY_FALLBACK = "greedy_1gw_no_hits"` is a hard-coded fallback.
Keep it (it is the sensible default) but stop treating it as a signed result
for v2.6 forecast semantics until the suite is re-run.
