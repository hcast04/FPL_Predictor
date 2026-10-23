---
document: response to the v2.6 audit
version: v2.6.1
as_of: 2026-10-06
---

# Response to the v2.6 audit

Every finding in the audit was checked against the code. **All of them hold.**
Five changed the package; two produced new measurements that changed a stated
conclusion. Summary first, evidence below.

| Audit finding | Verdict | Action in v2.6.1 |
|---|---|---|
| F4 GK goal = 6 is wrong | **Correct, my error** | P1 withdrawn; GK stays 10 |
| F2 masked after feature engineering | **Correct** | P4 moved before `add_historical_player_features()` |
| Global conversion state leaks across folds | **Correct** | P11 + P12 added |
| Premium layer silently reverts F1 | **Correct** | P10 added |
| F3 not covered by the headline validation | **Correct** | `validate_dc_fix.py` added — impact is negligible |
| Captain script tests the wrong rule | **Correct** | Rewritten to the joint objective; conclusion withdrawn |
| Goal conversion unjustified | **Correct** | Default OFF |
| Assist factor may not be stationary | Reasonable, **not supported by the data** | Pooling kept, evidence below |

---

## 1. F4 — I was wrong, and it is reverted

Current FPL awards **10** for a goalkeeper goal. I changed it to 6 on stale
knowledge and treated a deliberate value as a bug. P1 is withdrawn entirely.

I could not settle this from the archived data: **no goalkeeper scored in
either 2024-25 or 2025-26**. A full points reconstruction validates every other
position exactly (DEF goal = 6 reproduces 126/126 and 128/128 rows) but is
silent on goalkeepers. The FPL API does not publish the stat→points mapping,
and the official rules pages are JS-rendered behind a consent wall. So this
rests on your citation plus the notebook author's own value — both of which say
10, against one stale recollection.

## 2. F2 — mask moved before feature engineering

Confirmed exactly as described. `add_historical_player_features()` at the
`historical_fpl` stage builds `xg_per_fixture` / `xa_per_fixture` rolling and
career history; v2.6.0 masked `historical_model_data` *after* that, so the
fabricated zeros stayed baked into every shifted predictor feature.

P4 now targets `historical_fpl` directly, immediately before that call. Both
the training targets and the rolling/career features are built from clean data,
which also makes the notebook and the independent harness test the same thing —
the harness already did `load_seasons(mask_coverage=True)` → `add_history_features()`.

## 3. Global conversion state — two new edits

Both hazards confirmed:

```python
def build_walk_forward_fold_bundle(holdout_season):
    if holdout_season in walk_forward_fold_cache:
        return walk_forward_fold_cache[holdout_season]      # P6d never runs
```

There is a second early return for the disk cache. So on a cache hit the
globals keep whatever fold ran last — possibly a *later* season, which is
leakage into the v2.5 validation.

- **P11** sets the factors from `_bundle["development_points_data"]` immediately
  before each Section 16 fold integration, so a cached fold can never be scored
  with another season's factors. It is placed before
  `_v25_prepare_fold_for_integration()`, which calls the premium layer.
- **P12** restores the live factors inside `attach_or_apply_v25_live_minutes()`
  before it re-scores live xP.

I agree the globals should eventually move into the fold bundle. P11/P12 close
the correctness hole without a larger refactor; the architectural change is
worth doing separately.

## 4. Premium layer reverting F1 — P10

Confirmed. `apply_premium_attacker_adjustment()` rebuilds

```python
result["projected_assists"] = predicted_xa_per90 * minutes_fraction
result["xp_assists"] = 3.0 * result["projected_assists"]
```

for **every row**, not just the premium mask — the recompute sits outside the
position loop. With an approved payload that strips the conversion from exactly
the players F1 matters most for. P10 routes both rebuilds through the same
`_v26_position_factor()` helper the component model uses.

## 5. F3 — now validated, and the honest answer is "negligible"

You were right that the headline numbers did not cover it.
`validate_dc_fix.py` tests it on 2025-26 (7,927 outfield rows, leakage-safe
rolling history), with gates declared in the file before the run:

| Variant | Brier | mean P(DC) | bias | DEF Spearman | DEF NDCG@10 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.1072 | 0.1292 | −0.0266 | 0.3916 | 0.5059 |
| v2.6 fixed | 0.1072 | 0.1297 | −0.0261 | 0.3909 | 0.5059 |

G1 (Brier improves) **fails** — it is unchanged to four decimals. G2 and G3
pass. With forecast-style minutes instead of realised, the Brier delta is
−0.0002.

The reason is simple: most DC-relevant rows are near 90 minutes, where
`minute_scale ≈ 1` and the double-count barely bites. **Keep F3 as a
correctness fix; do not expect it to move any decision.**

## 6. Captain conclusion withdrawn

You are right that `captain_impact.py` tested the fallback branch. Rewritten to
the production joint objective, and the sign flips:

| Season | Picks changed | baseline → v2.6 |
|---|---:|---|
| 2024-25 | 0 / 33 | 7.21 → 7.21 (+0.00) |
| 2025-26 | 5 / 33 | 6.36 → 6.94 (+0.58) |

Mean +0.29 pts/GW. But that rests on **five changed picks in one season**.
Neither the old −0.08 nor the new +0.29 is distinguishable from zero. The
correct statement is that this harness cannot measure captain impact — and it
proxies the XI by top-11 xP rather than optimising it, so it is not a
production captain test either way.

## 7. Goal conversion — now OFF by default

Agreed, and implemented: `APPLY_GOAL_CONVERSION_BY_DEFAULT = False`. goals/xG
is 0.94–1.00, there is no demonstrated defect, and fitting a second multiplier
only adds variance. The factor is still *fitted and printed* so the assumption
stays monitored, but it is not applied unless explicitly enabled.

This also means the reported +2.18 / +2.27 is now attributable to the assist
conversion alone, which is cleaner than before.

## 8. Assist-factor stationarity — checked, and pooling survives

A reasonable concern given the 2025/26 Fantasy-assist broadening, but the data
does not show a regime break for outfield positions (2022-23 excluded, broken
feed):

| Position | 2023-24 | 2024-25 | 2025-26 | 2025-26 vs mean(23-25) |
|---|---:|---:|---:|---:|
| MID | 1.345 | 1.371 | 1.325 | −2.4% |
| FWD | 2.128 | 2.077 | 2.105 | +0.2% |
| DEF | 1.329 | 1.107 | 1.292 | +6.1% |

MID and FWD are flat across the change. DEF's *year-to-year* swing
(1.329 → 1.107 → 1.292) is larger than any era effect, so a recency-weighted
factor would add variance without removing bias. Goalkeepers have ~1.0 xA per
season, so the GK factor is pure noise — the existing shrinkage already pulls it
to ~99% of the pooled value, which is the right behaviour.

I have kept simple pooling and left the hierarchical variant unimplemented
rather than adding a parameter the evidence does not justify. Worth revisiting
after a full 2026/27 season under the new rules.

## 9. The predeclared F1 gate

Agreed that redefining the gate after seeing results is post-hoc. Declaring the
real-notebook gate now, before it is run:

> **F1 acceptance, real pipeline, chronological OOS.** Accept only if all hold:
> 1. §7.5 bias in the 5+ xP buckets moves toward zero, weighted by bucket count;
> 2. §7.5 bias in the `<2` xP buckets does not worsen by more than 0.05;
> 3. NDCG@10 does not degrade by more than 0.002;
> 4. mean top-10 realised points per gameweek does not fall by more than 0.25;
> 5. premium (price ≥ 8.0) MID/FWD Spearman does not degrade by more than 0.003.
>
> Overall MAE is explicitly *not* a gate: F1 trades a small MAE increase for
> upper-tail calibration, and that trade is the hypothesis.

Run once, report whatever comes out.

## What I did not change

- The global-state architecture (P11/P12 patch the hole; the refactor is yours
  to schedule).
- Bonus. Still the strongest remaining lead, still deserves its own experiment
  after assists and minutes settle — and as you note, 2026/27 BPS changes make
  it more regime-sensitive, not less.
- Policy and chip revalidation. Deferred until the forecast layer is accepted.

## Status

Research quality and production readiness were fairly rated. v2.6.1 addresses
the four integration bugs and the rules error; what remains before merge is a
single run of the real pipeline against the gate above — which is the one thing
I cannot do from here.
