---
document: bonus model and team value experiments
version: v2.6.1
as_of: 2026-10-06
---

# Bonus and team value — both tested, both rejected

Two of the three things requested. The third (wildcard) **passed** and is now in
production as patch P14; see `CHIP_TIMING.md`.

Both experiments here declared their gates before the first run and both failed
them. They are documented rather than tuned until they pass, because tuning
until a gate clears is exactly the failure mode the project audit log warns
about: *"the project should not optimise whichever metric is easiest to
improve."*

Reproduce with `python bonus_model.py` and `python team_value_model.py`.

---

## 1. Bonus — rank-based model. REJECTED.

### The hypothesis

The notebook predicts bonus as an independent per-90 Poisson rate
(`alpha=2.0`, pooled). FPL bonus is not a rate: within each fixture the three
highest BPS totals take 3, 2 and 1 points. A rate model cannot represent that,
and it loses dispersion — real bonus/90 spans 0.000 to 0.838 across player-season
deciles while the model's predictions barely spread at all.

So: predict **BPS** per 90, then convert to expected bonus by simulating the
within-fixture ranking (negative-binomial draws around the predicted mean, rank,
award 3/2/1 with FPL tie handling, average over 400 draws).

### The result

| Fold | Variant | MAE | dispersion | Spearman |
|---|---|---:|---:|---:|
| 2024-25 | per-90 rate (current) | **0.3514** | 0.173 | **0.2648** |
| 2024-25 | rank-based (new) | 0.3543 | **0.280** | 0.2313 |
| 2025-26 | per-90 rate (current) | **0.3538** | 0.166 | **0.2465** |
| 2025-26 | rank-based (new) | 0.3556 | **0.262** | 0.2421 |

- **G1 MAE improves** — FAIL in both folds.
- **G2 dispersion closer to 1.0** — PASS in both (0.17 → 0.28).
- **G3 Spearman not degraded** — FAIL in both.

### Why it fails, and why that is informative

The rank model does exactly what it was built to do: it recovers dispersion,
nearly doubling it. It still ranks worse.

The reason is that the baseline trains **directly on the target** (bonus), while
the rank model trains on BPS and then transforms. Predicting the outcome
directly beats modelling the mechanism and propagating through it, because the
transformation adds variance without adding information the features did not
already carry.

One honest note on process: `BPS_DISPERSION` was originally a hard-coded 0.55
with a comment claiming it was "fitted below if possible". It was not fitted.
That is now a real fit (`fit_bps_dispersion`), done on the high-BPS region,
because k is strongly mean-dependent — about 41 at mu < 1 and about 0.45 at
mu ~ 19, since low-minutes players are dominated by whether they played at all.
Fitting it pooled would be dominated by players who never compete for bonus.
The fitted value came out at 0.48, close to the original guess, and the verdict
did not change.

### Recommendation

Keep the per-90 rate model. If bonus is revisited, the promising direction is
**not** a better mechanism model but better *features* for the existing one —
BPS components (passes, tackles, saves, big chances) rather than aggregate
history.

---

## 2. Team value — price-change model. MIXED, NOT IMPLEMENTED.

Three parts, and they do not all point the same way.

### Part A — what is a pound worth? Unstable.

Solve the exact squad MILP across a realistic budget range and read off the
marginal expected points per extra £1.0m:

| Fold | £96m | £98m | £100m | £102m | £104m | points per £1m over 38 GW |
|---|---:|---:|---:|---:|---:|---:|
| 2024-25 | 48.5 | 49.1 | 49.6 | 50.0 | 50.4 | **8.9** |
| 2025-26 | 53.1 | 53.2 | 53.4 | 53.4 | 53.4 | **1.3** |

A 7× disagreement between adjacent seasons. I chased two explanations and ruled
both out:

- **Candidate pruning** truncating the top of the pool — widened to 70 per
  position, still saturates.
- **The forecast not ranking expensive players** — Spearman(cost, xP) is 0.513
  and 0.519, essentially identical across folds.

The actual cause is that 2025-26 has a smaller premium-to-cheap gap in my
harness (premium mean xP 3.15 vs 2024-25's 4.03, cheap roughly equal), so the
optimal XI is already affordable at £96m and extra budget has nowhere to go.

**Treat "a pound is worth 1–9 points a season" as the honest range.** It is not
a number to build on.

### Part B — are price changes predictable? Emphatically yes.

This is the one solid result in this document.

| Fold | rows | AUC rise | AUC fall | top 5% by P(rise): actual rise rate vs base |
|---|---:|---:|---:|---|
| 2024-25 | 26,499 | **0.948** | 0.831 | 26.1% vs 2.1% — **12× lift** |
| 2025-26 | 28,916 | **0.959** | 0.829 | 26.8% vs 2.0% — **13× lift** |

A simple logistic model on net transfers relative to ownership predicts next-week
price movement almost perfectly for rises. This is reusable regardless of what
is done with it.

### Part C — is a value-aware tiebreak worth it? Fails the gate.

Simulate a season of greedy transfers where xP-tied candidates (within 0.15 xP)
are broken by predicted price movement, and convert the accumulated value through
Part A:

| Fold | points per £1m | extra team value | xP sacrificed | net |
|---|---:|---:|---:|---:|
| 2024-25 | 6.22 | **+1.25m** | 0.45 | **+7.33** |
| 2025-26 | 0.74 | **−0.60m** | 0.28 | **−0.73** |

Gate (positive extra value **and** positive net in **both** folds): **FAIL.**

It gains value in one season and loses it in the other, and the conversion
factor it depends on is itself unstable. The effect size in the good fold
(+7 points) is within the noise band of a single season anyway.

### Recommendation

Do not implement the value-aware tiebreak on this evidence. **Keep the price
model** — it is accurate and cheap, and it is genuinely useful as *information*
on the execution card ("this player is 85% likely to rise tonight") even when it
is not driving an automated decision. That is a presentation feature, not a
policy change, and it needs no gate.

---

## Summary across all three requests

| Request | Verdict | Status |
|---|---|---|
| Wildcard into production | **PASS** — +40 pts/season | shipped as P14 |
| Bonus model | REJECT — ranks worse | documented, not shipped |
| Team value | REJECT — unstable, fails gate | price model kept as information only |

One passing from three is roughly what the project's own history predicts: of
the layers tested in the audit log, v2.4 upper-tail was rejected, v2.4 MID
reranker was rejected, and the bank-aware threshold was rejected. The gates are
doing their job.
