---
document: chip-timing validation
version: v2.6.1
as_of: 2026-10-06
---

# Chip timing — the experiment the project never ran

> ## ⚠ CORRECTION (2026-10-06, after the first real pipeline run)
>
> Everything below was measured on forecasts that had the **v2.6 assist
> conversion (F1)** applied. F1 subsequently **failed its predeclared gate on
> the real notebook** (top-10 realised down 0.3–1.7 points per gameweek across
> all three seasons), so these numbers rest on distorted xP.
>
> Re-running the wildcard study with F1 **off** reverses the headline result:
>
> | Policy | F1 on (as published below) | F1 off (corrected) |
> |---|---:|---:|
> | oracle | 166.5 | 137.0 |
> | best predicted | 80.0 | **111.0** |
> | threshold only (>= 10 xP) | **99.0** | 70.5 |
> | notebook rule (threshold AND structural) | 59.0 | **74.0** |
> | random legal | 59.6 | 55.0 |
>
> With F1 off the **notebook's existing rule (74.0) beats threshold-only
> (70.5)**. The claimed +40 points a season from dropping the structural gate
> does not survive; the 3.5-point difference that remains is noise over two
> seasons.
>
> **Patch P14 is therefore withdrawn.** Set
> `V26_WILDCARD_THRESHOLD_ONLY = False`.
>
> What *does* survive is forecaster-independent: every double and blank
> gameweek in both test seasons falls in the second half, and the live
> 2026-27 calendar currently has **no doubles or blanks scheduled at all**
> (they appear later as cup postponements force rearrangements). So the
> structural trigger genuinely cannot fire on fixture grounds before GW19.
> That is a true fact about the calendar — it just does not translate into a
> measurable gain from acting on it.
>
> The TC/BB/FH results below carry the same contamination. "Chips are worth
> ~92 points a season versus never playing them" is robust (not playing scores
> zero by definition). The finer comparisons against random timing are not.


Reproduce with `python validate_chip_timing.py` (~15 min, both seasons).

## Method

A full season is simulated per fold: 15-player squad, £100.0m, max 3 per club,
one greedy no-hit transfer per gameweek on next-gameweek xP — mirroring the
validated `greedy_1gw_no_hits` policy. For every gameweek both the **predicted**
and the **realised** value of each chip is recorded given that week's squad.
Policies decide on predicted value and are scored on realised value.

Triple Captain, Bench Boost and Free Hit are covered. **Wildcard is excluded** —
its value is multi-week and inseparable from the transfer optimiser.

## Result

Mean realised chip points per season, both chip sets, three chips:

| Policy | 2024-25 | 2025-26 | mean | vs never | vs random |
|---|---:|---:|---:|---:|---:|
| oracle (hindsight) | 166.0 | 145.0 | 155.5 | +155.5 | +100.0 |
| **notebook rule** | **132.0** | **53.0** | **92.5** | **+92.5** | **+37.0** |
| greedy threshold | 101.0 | 58.0 | 79.5 | +79.5 | +24.0 |
| best predicted | 68.0 | 58.0 | 63.0 | +63.0 | +7.5 |
| random legal | 55.0 | 56.1 | 55.6 | +55.6 | 0.0 |
| forced at expiry | 61.0 | 40.0 | 50.5 | +50.5 | −5.0 |
| never | 0.0 | 0.0 | 0.0 | 0.0 | −55.6 |

## Four conclusions

### 1. Chips are the biggest lever in the system — ~92 points a season

That is 5–30× the entire forecast-layer effort (v2.6 is worth +3 to +17 season
points). The earlier estimate of 80–160 was the right order of magnitude.

**The single most important thing is that chips get played at all.** "Never"
scores zero. The notebook defaults to `WEEKLY_CHIP_DECISION_MODE = "conservative"`
with `WEEKLY_CHIP_REQUIRE_STRUCTURAL_TRIGGER = True`, so its standard output is
"Save all chips". That is safe only if it eventually fires.

### 2. The rule works — in the second half, and not reliably

| Policy | first half | second half |
|---|---:|---:|
| oracle | 70.0 | 85.5 |
| **notebook rule** | **27.5** | **65.0** |
| greedy threshold | 18.0 | 61.5 |
| best predicted | 18.5 | 44.5 |
| forced at expiry | 28.0 | 22.5 |
| random legal | 24.8 | 30.7 |

Second half: 65.0 vs 30.7 random — the rule captures 76% of the hindsight
maximum. That is genuinely good and should be kept.

First half: 27.5 vs 24.8 random, and *worse* than simply being forced at expiry
(28.0). The rule adds nothing before GW19.

### 3. Why the first half fails — there is nothing to trigger on

| Season | Double gameweeks | Blank gameweeks |
|---|---|---|
| 2024-25 | 24, 25, 32, 33 | 15, 29, 34 |
| 2025-26 | 26, 33, 36 | 31, 34 |

Every double gameweek in both seasons falls in the **second** half. The
structural gate is `doubles > 0 or blanks > 0 or expiry_pressure or exceptional`,
so before GW19 it can essentially only fire on expiry pressure — which, with
four chips and a GW19 expiry, first triggers at **GW16**:

```
expiry_pressure = remaining_available_chips >= remaining_weeks
GW16: 4 chips left, 4 weeks left -> fires
```

and `clears_uplift = uplift >= threshold or expiry_pressure` bypasses the value
threshold entirely. So the first chip set is saved until GW16 and then spent one
per week regardless of value.

### 4. The first-half gap is mostly — but not entirely — unrecoverable

My first pass said first-half chip value was simply "unforecastable". That was
too strong: it rested on three timing policies, not on measuring the signal.
`validate_wildcard.py` Part A measures it directly — Spearman between each
decision-time signal and the realised chip value:

| Chip | Signal | 24-25 first | 25-26 first | 24-25 second | 25-26 second |
|---|---|---:|---:|---:|---:|
| Triple Captain | predicted value | **+0.324** | **+0.167** | +0.502 | +0.180 |
| Triple Captain | squad doubles | — | — | +0.574 | +0.089 |
| Free Hit | predicted value | +0.264 | +0.025 | +0.466 | +0.275 |
| Free Hit | squad doubles | — | — | **+0.764** | **+0.518** |
| Bench Boost | predicted value | +0.139 | **−0.588** | +0.142 | −0.262 |
| Bench Boost | squad doubles | — | — | −0.220 | −0.206 |

Three things fall out:

- **The strong signal is "does my squad have a double this week"** — up to
  ρ = 0.76 for Free Hit. That is exactly why the rule works after GW19, and it
  is literally undefined before it (the "—" cells: no club plays twice).
- **Triple Captain has a weak but sign-consistent signal in the first half**
  (+0.324, +0.167), and Free Hit a weaker one (+0.264, +0.025). So there *is*
  something there.
- **Bench Boost has none** (+0.139 then −0.588). Its payoff is whether four
  bench players happen to haul, which is variance.

But a correlation of ~0.25 is not enough for a single argmax pick over ~14
candidate weeks to beat random — which is exactly what the policy table shows
("best predicted" 18.5 vs random 24.8). The expected gain from exploiting a
signal that weak is a few points, concentrated in Triple Captain.

**Revised advice:** in the first half, do not wait for a structural trigger that
cannot fire. Use predicted value for Triple Captain and Free Hit, ignore it for
Bench Boost, and expect single digits — not the 42-point oracle gap.

## What I would actually do

1. **Keep the rule.** It is the best implementable policy tested, by +37 a
   season over random.
2. **Make sure the chips get played.** The failure mode that costs real points
   is not bad timing (±5 vs random) but deferral to zero (−92). If the weekly
   output says "Save all chips" every week into GW15, that is the rule working
   as designed — but check it fires.
3. **Treat the 2025-26 result as a variance warning**, not a defect. The rule
   was worse than random in that season (53 vs 56.1). One bad season out of two
   means the +37 edge is real on average and unreliable week to week.
4. **In the first half, stop waiting for a trigger that cannot fire.** Use
   predicted value for Triple Captain (ρ ≈ +0.32/+0.17, sign-consistent) and
   Free Hit, ignore it for Bench Boost (ρ ≈ +0.14/−0.59, no signal), and expect
   single-digit points — the signal is too weak for more.
5. **Wildcard is the best chip opportunity found** — worth ~**+99 points a
   season** with the threshold rule, +39 over random timing. Your +10 uplift
   bar is reachable (it is 1.84 points/GW over a 6-week discounted horizon).
6. **Solver quality is itself a lever.** Swapping a greedy rebuild for the exact
   MILP raised the wildcard oracle by 45% and turned several
   "value-destroying" policies positive. If any part of the production
   pipeline still approximates the squad problem, that is worth auditing.

## Wildcard

This section was rewritten twice, and both rewrites mattered.

1. The first version compared only *position* heuristics (earliest / mid-half /
   forced / random) with no forecast in it — an evaluator, not a predictor.
2. The second added a predicted wildcard value but still rebuilt the squad with
   a two-pass greedy knapsack. It reported that most wildcard policies **lose**
   points, and flagged that this was probably an artefact of the weak rebuild.

It was. Replacing the greedy with an exact MILP (`squad_optimizer.py`, the same
`scipy.optimize.milp` formulation the notebook itself uses) changes every number
and reverses the conclusion.

Mean realised points over both seasons:

| Policy | first half | second half | total | (greedy rebuild) |
|---|---:|---:|---:|---:|
| oracle (hindsight) | 65.5 | 101.0 | **166.5** | *114.5* |
| **threshold rule (>= 10 xP)** | **40.5** | **58.5** | **99.0** | *41.0* |
| best predicted | 5.0 | 75.0 | 80.0 | *70.0* |
| mid-half | 28.0 | 49.0 | 77.0 | *−23.0* |
| random legal | 19.3 | 40.4 | 59.6 | *17.5* |
| forced at expiry | 17.5 | 17.5 | 35.0 | *11.0* |
| earliest legal | −8.0 | 8.5 | 0.5 | *−1.5* |
| never | 0.0 | 0.0 | 0.0 | *0.0* |

Three things changed:

- **No policy loses points any more.** "mid-half" went from −23.0 to +77.0. The
  earlier "wildcarding destroys value" result was entirely an artefact of the
  rebuild quality, not a property of wildcards.
- **The oracle rose 45%** (114.5 to 166.5). A better optimiser does not just
  improve the policies — it raises the ceiling, because the wildcard's value
  *is* the squad it builds.
- **The implementable threshold rule now scores 99.0**, ahead of the
  semi-implementable "best predicted" (80.0). A wildcard is worth roughly
  **+99 points a season** against never playing it, and **+39 against random
  timing**.

### Is the notebook's +10 threshold reachable? Yes.

Earlier I warned it might be unreachable. **That was wrong, twice over.**

First, a scale error. `wildcard_uplift = wildcard_horizon_score -
wildcard_baseline_score` is a *discounted sum over the planning horizon*, with
`PLANNING_HORIZON = 6` and `GAMEWEEK_DISCOUNT = 0.96`. The discount weights sum
to 5.431, so a 10.0 bar means **1.84 points per gameweek** of squad
improvement — a modest ask, not a high one. I had compared it against a
differently-scaled per-decision number.

Second, the pessimism I observed was my broken optimiser. With the exact
rebuild, mean predicted uplift over an 8-gameweek horizon is **+1.8 to +29.4**
(it was −1.5 to −12.4 with the greedy). The bar is comfortably clearable.

One genuine finding survives: `required_player_ids=baseline_squad_ids` passed
to `optimise_chip_squad` only keeps the incumbent squad in the **candidate
pool** via `kept_ids`; it does not force those players into the solution. The
rebuild is correctly unconstrained. That is right, and worth knowing it is
right.

### Forecastability

| Season | half | n | Spearman(pred, real) | mean predicted | mean realised |
|---|---|---:|---:|---:|---:|
| 2024-25 | first | 13 | +0.505 | +14.1 | +46.8 |
| 2024-25 | second | 18 | **+0.791** | +29.4 | +55.5 |
| 2025-26 | first | 13 | +0.347 | +1.8 | −8.3 |
| 2025-26 | second | 18 | +0.128 | +11.8 | +25.2 |

Now positive in all four cells (+0.13 to +0.79), where the greedy version had a
negative cell. Predicted wildcard value is a usable signal — still systematically
*under*-stated relative to realised, so if anything the threshold could be
lowered rather than raised.

## Caveats

- Two seasons. The project's own standard is a minimum of two chronological
  folds; this meets it and no more, and the two folds disagree sharply.
- The squad simulation is a simplified `greedy_1gw_no_hits`: no price changes,
  no injuries beyond what the forecast captures, no autosubs, a crude initial
  squad. Absolute levels are not your pipeline's; the policy *ranking* is the
  result.
- Free Hit value is computed as the best pool XI minus your own XI for that
  week, which is the correct single-week valuation but ignores that a Free Hit
  can also rescue a blank gameweek you would otherwise field 7 players in.

---

## Production fix, 2026-10-08 (P18–P21)

The chip-timing study above compared *policies*. Running the real pipeline exposed
that the production code could not execute any of them.

### P18 — the weekly gate had no expiry override

The sequential backtest gates on `uplift >= threshold or expiry_pressure`. The
weekly production path required the threshold unconditionally; `expiry_pressure_at`
appeared only in the coarse-lookahead veto, where it can *remove* a candidate and
never admit one. Confirmed live: all four chips showed `passes_use_now_rule=False`
(TC 5.28/7.0, BB 8.20/9.0, FH 3.89/8.0, WC 3.89/10.0) with nothing able to override.
**No chip could ever have been played, through the GW19 expiry** — the ~92
points/season failure mode, live in production.

P18 mirrors the backtest, plus a **positive-uplift floor**: under expiry a chip is
played only if its uplift is above zero. Free-hit and wildcard uplifts really do go
negative (an executed run showed −0.276), because the chip squad is rebuilt at
current market prices while the held squad was bought at purchase prices. Playing a
negative chip is strictly worse than letting it expire.

### P19 — only the wildcard bar was mis-scaled

`triple_captain_uplift`, `bench_boost_uplift` and the free-hit uplift are all
**single-gameweek** quantities, so 7.0 / 9.0 / 8.0 are correct per gameweek at any
horizon. Only `wildcard_horizon_uplift` is a discounted sum over the horizon. The
identical constants appear as `BACKTEST_CHIP_MINIMUM_UPLIFT` with
`BACKTEST_HORIZON = 6`, but the live policy runs at horizon 1 — so the wildcard's
10.0 is roughly 5.4× too strict.

Rescaling it was rejected: at horizon 1 the wildcard "uplift" measures a one-week
squad reshuffle and ignores that you keep the squad, so a scaled bar (1.84) would
have fired the wildcard this week on a 3.89 one-week gain. It is suppressed
explicitly instead, so the limitation is visible rather than accidental.

### P20 — the wildcard was being scheduled last

At horizon 1, `evaluate_wildcard_at` and `evaluate_free_hit_at` solve the same
one-gameweek problem, so their uplifts are **identical by construction** —
confirmed in production, both `3.8910851937150426`. Selection sorts on
`normalised_strength = uplift / threshold`, so the wildcard's larger bar always
sorted it below the free hit, scheduling it into GW19 — the last week before the
chip set resets, and the worst possible week for the only chip whose value scales
with weeks remaining.

Under expiry pressure the wildcard now goes **first**. It buys a better squad for
every remaining week, and it raises the value of the single-gameweek chips that
follow. Normal merit-based ordering is untouched.

### P21 — the cascade had zero slack

`remaining_chips >= remaining_weeks` is the must-start-now boundary, not a margin:
with four chips it fired at exactly GW16, leaving four weeks for four chips and no
recovery room. One disrupted week lost a chip permanently — and a free hit blocks
the following gameweek, so a late free hit could strand another chip by itself.
One week of slack costs almost nothing on a calendar with no doubles or blanks.

### Resulting schedule

```
GW6-14  save
GW15    PLAY wildcard        <- first, buys 5 weeks of a better squad
GW16    PLAY bench boost
GW17    PLAY triple captain
GW18    PLAY free hit
GW19    spare week (slack)
```

Verified: the GW6 recommendation is byte-identical before and after all four
patches — same transfers, same captain, same expected score 51.15172734480787.
P20 and P21 were found by adversarial review of P18, not by me.
