---
document: direct-ML experiment, blended forecast, and season-points translation
version: v2.6.1
as_of: 2026-10-06
---

# Direct ML, the blend, and what any of it is worth in season points

Two questions, one answer each.

1. Should the component model be replaced (or blended) with a direct ML model?
   **No. Both rejected by predeclared gates.**
2. What do the validation metrics mean in season-end points?
   **Much less than the per-gameweek headlines imply — and for one variant they
   point the opposite way.**

Reproduce with `python blend_and_season.py`.

---

## 1. The headline finding: top-10 points and NDCG are the wrong metric

This is the most important result in this document, and it affects how the
whole project validates changes.

On per-gameweek decision metrics the direct GBM looked clearly better than the
component model:

| Fold | metric | component v2.6 | direct GBM |
|---|---|---:|---:|
| 2024-25 | top-10 realised pts/GW | 45.24 | **48.61** |
| 2024-25 | NDCG@10 | 0.3717 | **0.3891** |
| 2025-26 | top-10 realised pts/GW | 40.48 | **41.97** |
| 2025-26 | Spearman | 0.7068 | **0.7201** |

But when the same forecasts are used to pick a **legal starting XI** each week
(1 GK, 3-5 DEF, 3-5 MID, 1-3 FWD, captain doubled) the ordering reverses:

| Fold | component v2.5 | component v2.6 | direct GBM |
|---|---:|---:|---:|
| 2024-25 | 2079.6 | **2096.9** | 2061.2 |
| 2025-26 | 2003.6 | **2007.1** | 1879.3 |

The GBM wins top-10 by +1.5 to +3.4 points per gameweek and *loses* legal-XI
selection by −0.5 to −3.3 points per gameweek.

**Why:** "top-10 realised points" ignores position constraints. A model that
ranks ten forwards highest scores brilliantly on top-10 and cannot field a
team. The GBM also over-predicts the upper tail (trees cannot extrapolate),
which concentrates its top picks in the wrong places.

The project's audit log used NDCG@10 and top-10 realised points to reject the
v2.4 layer. Those metrics have this blind spot. **Best-legal-XI points should
be the primary decision metric from here on**, with NDCG as a secondary.

## 2. Direct ML — rejected

Beyond the XI result, three structural reasons not to replace the component
model even if the numbers had held:

- **Rule changes.** This comparison *removed* defensive-contribution points
  from the 2025-26 target to make it fair. Production cannot do that: a GBM
  trained on 2022-25 has never seen DC, and 2026/27 changed BPS again. The
  component model absorbs new rules by construction.
- **Tail sign flip.** 2025-26 `6+` bucket bias: component −1.23, GBM **+1.24**.
  Wrong at exactly the end that drives captaincy.
- **Downstream dependencies.** Bench boost, autosubs, clean-sheet logic and
  captain uncertainty all consume component outputs, not just the total.

Keep it as a **benchmark** — if a GBM ever beats the component model on
best-XI points across both folds, that is a real signal worth acting on.

## 3. The blend — rejected, and unstable

```
blend = w * component_xp + (1 - w) * gbm_xp
```

with `w` fitted on an inner chronological split of the training seasons only.

| Fold | fitted w | gates failed |
|---|---:|---|
| 2024-25 | 0.55 | G1 (tail bias) |
| 2025-26 | **0.00** | G1, G3 (NDCG), G5 (premium ρ) |

The fitted weight collapsed to **zero** on the second fold — the optimiser
discarded the component model entirely, so the "blend" became the plain GBM.
A blend weight that swings from 0.55 to 0.00 between adjacent folds is not
identifiable from this much data, independent of the gate results.

The 2024-25 fold alone looked genuinely good (+3.79 pts/GW, t = 2.25, 38-GW
CI [+18, +269]). That is exactly the kind of single-fold result the project's
two-fold rule exists to catch.

## 4. Season-points translation

What you asked for: the per-gameweek metrics converted into season-end points.

**Method.** Each gameweek, pick the best legal XI by predicted xP, double the
captain, score on realised points, sum, scale to 38 gameweeks.

**This is NOT a predicted FPL score.** No budget, no 3-per-club, no transfer
limit, no price changes, and the forecaster here is a simplified
re-implementation, not your notebook. Absolute levels are not comparable to
your pipeline — the notebook's own `greedy_1gw_no_hits` backtest (mean 2196.7
realised, with full constraints) remains the realistic anchor. **Only the
deltas between variants mean anything.**

### Levels

| Variant | 2024-25 | 2025-26 |
|---|---:|---:|
| oracle (perfect hindsight) | 5648 | 5636 |
| component v2.5 (current production) | 2079.6 | 2003.6 |
| component v2.6 | 2096.9 | 2007.1 |
| direct GBM | 2061.2 | 1879.3 |
| random legal XI | 517 | 449 |

### Paired weekly deltas vs current production

The variants share a player pool and mostly pick the same players, so the
per-gameweek SD of a single variant badly overstates the uncertainty on a
*difference*. Pairing week by week:

| Fold | Variant | Δ pts/GW | SD of Δ | t | Δ over 38 GW | 95% CI |
|---|---|---:|---:|---:|---:|---|
| 2024-25 | component v2.6 | +0.455 | 5.11 | 0.51 | **+17.3** | [−49, +84] |
| 2024-25 | direct GBM | −0.485 | 17.59 | −0.16 | −18.4 | [−246, +210] |
| 2025-26 | component v2.6 | +0.091 | 3.57 | 0.15 | **+3.5** | [−43, +50] |
| 2025-26 | direct GBM | −3.273 | 12.91 | −1.46 | −124.4 | [−292, +43] |

### How to read this

- **v2.6 is worth roughly +3 to +17 season points**, positive in both folds.
  Not the **+84** that "+2.2 top-10 points per gameweek × 38" implied. That
  headline overstated the real benefit by roughly five-fold, because most of
  the ranking improvement lands on players who were never going to make the XI.
- **Neither fold reaches significance** (t = 0.51 and 0.15). The honest
  statement is "small, consistently positive, not separable from zero in two
  seasons."
- **The paired SD is small** (3.6–5.1 pts/GW) precisely because v2.6 changes
  few picks. That is reassuring about *risk*: v2.6 is very unlikely to make
  things meaningfully worse. The 95% CI floor is −49 and −43 season points.
- **The GBM's SD of Δ is 3-5× larger** (12.9–17.6). It changes many picks, and
  the changes do not pay.

### Scale check

A full season's spread from your own noise analysis is ~±127 points at 95%
over 15 gameweeks, and the gap between the oracle and current production is
roughly **3,550 points**. A +3 to +17 point improvement is real but tiny
against both. It is worth taking because it is free and low-risk, not because
you will notice it.

## 5. What this changes

- **Merge v2.6.1 on correctness grounds**, not on expected points. The assist
  mechanism is genuinely wrong in v2.5 and the fix is close to risk-free, but
  do not expect a visible season effect.
- **Drop the direct-ML and blend lines of work.** Both rejected on
  predeclared gates; the blend weight is not identifiable at this sample size.
- **Change the primary decision metric to best-legal-XI points.** Top-10 and
  NDCG are measuring something that does not survive position constraints.
  This is worth re-checking against the rejected v2.4 layer: it was turned
  down on metrics that have now been shown to disagree with XI selection.
