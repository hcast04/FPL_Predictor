---
project: FPL Predictor
document: Model, Validation, and Decision Audit Log
as_of: 2026-10-05
current_notebook_branch: v2.5.0
audit_scope: v2.2.0 through v2.5.0
---

# FPL Predictor — Model, Validation, and Decision Audit Log

## 1. Purpose

This document records the major modelling, validation, optimisation, operational, and human-decision choices made during the development of the FPL Predictor from the first structured audit through v2.5.0.

The objective is to make the system auditable by clearly separating:

- what was found;
- what was changed;
- what was validated;
- what was rejected;
- what remains provisional;
- where current-season evidence is incomplete;
- which results may be reused without recomputation.

This is a decision log, not a replacement for the notebook code or its generated validation artifacts.

## 2. Status legend

| Status | Meaning |
|---|---|
| **Validated** | Supported by chronological out-of-sample or direct observational evidence and accepted for production use. |
| **Rejected** | Tested and explicitly not accepted because validation criteria were not met. |
| **Operational fix** | Correctness, execution-order, schema, runtime, caching, or resilience fix rather than a modelling hypothesis. |
| **Implemented — pending validation** | Code exists but the acceptance gate has not yet been evaluated on an executed notebook. |
| **Diagnostic only** | Retained to explain or quantify behaviour but not used to alter production xP or policy. |
| **Known limitation** | Evidence is incomplete or a data source is unavailable/degraded. |

---

# 3. Executive summary

The project evolved from a working FPL forecasting/optimisation notebook into an auditable decision system with explicit model, policy, provenance, and caching controls.

The strongest conclusions to date are:

1. **The core historical xP model has real signal.** Across three untouched historical seasons it consistently outperformed a simple history baseline on MAE and decision-ranking metrics.
2. **The validated production transfer policy is `greedy_1gw_no_hits`.** Longer planning horizons and the fixed bank-aware threshold did not improve realised historical performance in the fair-baseline experiment.
3. **Historical top-player ranking is useful, but current-season evidence is weak and still small.** In the two true archived live forecast GWs available at the time of diagnosis, model top-10 selection was effectively tied with the FPL baseline.
4. **Upper-tail underprediction exists, but a direct upper-tail calibration was rejected.** It corrected bias but worsened OOS MAE and decision quality.
5. **A dedicated MID residual reranker was also rejected.** It did not improve the relevant chronological OOS ranking metrics.
6. **Minutes/role uncertainty is the next justified modelling target.** Historical minutes MAE remains around 13–15 minutes and is materially worse for rotation-risk players. v2.5 introduces scenario-based minutes modelling, but this is **not yet validated**.
7. **Human execution provenance matters.** GW4, GW5, and GW6 deviations from the raw notebook recommendation were deliberate conservative overrides advised in review, generally to bank an extra FT. GW2 is the genuine manager deviation. These weeks must not be labelled as simple “failure to follow the model.”
8. **Caching is now a first-class requirement.** Expensive walk-forward forecast folds and new diagnostic artifacts are persisted and separately signed so later iterations do not recompute unchanged work.

Current production stance before v2.5 validation:

- keep the existing base xP architecture;
- keep `greedy_1gw_no_hits` as the policy fallback;
- use uncertainty as confidence context, not as a hidden xP penalty;
- do not activate v2.4 upper-tail or MID corrections;
- validate the v2.5 probabilistic-minutes layer before allowing it to alter production xP.

---

# 4. Version and decision chronology

## v2.2.0 — Initial audit fixes

**Status:** Operational/model-correctness programme.

The first systematic audit identified the following issues:

- clean-sheet xP used full-match clean-sheet probability rather than sufficiently accounting for a player's projected time on pitch;
- autosub legality could be evaluated incorrectly in edge cases;
- live production layers were not always identical to the system validated historically;
- minutes modelling treated appearance/start/60+ probabilities too independently and required hierarchical coherence;
- historical backtests lacked true deadline-time injury/availability snapshots;
- hit-policy evidence was not sufficiently same-state/causal;
- FBref enrichment had broken/unsupported integration paths and zero npxG/xAG coverage;
- the all-competition workload collector could produce hundreds of failed requests;
- cache signatures were not sufficiently strict;
- transfer optimisation runtime was excessive.

Key design principle established here: **production behaviour must be the same behaviour that is validated, and optional external data must fail safely.**

---

## v2.2.1 — Configuration execution-order fix

**Status:** Operational fix.

Issue:
- `PLAYER_SCORING_POSITIONS` was referenced before definition in minutes-model code.

Decision:
- move canonical scoring-position configuration to the central configuration section.

Takeaway:
- shared model constants must be defined before any model family can execute.

---

## v2.2.2 — Position schema compatibility

**Status:** Operational fix.

Issue:
- minutes training expected `model_position`, while historical data could provide `position`, `element_type`, or another equivalent field.

Decision:
- introduce a canonical position resolver supporting:
  - `model_position`;
  - `position`;
  - `element_type`;
  - `position_id`;
  - `GKP -> GK`;
  - numeric position IDs 1–4.

Takeaway:
- model code must not assume one upstream schema representation when equivalent historical/live schemas exist.

---

## v2.2.3 — Model schema contracts

**Status:** Operational fix; important reproducibility control.

Issue:
- team-model feature construction created duplicate/suffixed columns such as `_x` / `_y`, causing missing expected unsuffixed features.
- xG-related team feature naming was inconsistent.

Decision:
- create explicit schema contracts for minutes, team, and player-event model families.
- contracts:
  - enforce required feature order;
  - recover valid suffixed aliases;
  - map known aliases such as `event -> gameweek` and `is_home -> was_home`;
  - resolve position encodings;
  - create optional missing numeric columns as `NaN` for the imputer;
  - record repairs.
- align team career xG naming.

Result:
- full executed notebook completed with no saved exceptions and schema checks passed.

Takeaway:
- schema compatibility is treated as an explicit contract rather than relying on brittle `frame[FEATURE_COLUMNS]` indexing.

---

## v2.2.4 — Production resilience

**Status:** Operational fix.

Problems addressed:

- all-competition circuit breaker existed conceptually but was not actually preventing repeated failed requests;
- fallback policy provenance could be mislabeled as validated;
- robustness subset checks consumed substantial runtime even when the robustness gate was disabled;
- optional FBref features could be degraded without a clear readiness state.

Decisions:

- implement all-competition failure circuit breaker with negative cache;
- make policy provenance explicit (`validated` vs `fallback`);
- prevent smoke tests from overwriting production policy state;
- skip exact subset solves when the relevant robustness gate is off;
- treat optional ESPN/FBref degradation as non-blocking where core inputs remain available;
- add explicit production-readiness reporting.

Observed runtime improvement:
- unnecessary subset checks dropped to effectively zero when disabled.
- base transfer optimisation remained the dominant runtime bottleneck.

Historical forecast metrics remained strong:

| Holdout season | Model MAE | Baseline MAE | Model Spearman | Baseline Spearman |
|---|---:|---:|---:|---:|
| 2023/24 | 0.926 | 1.008 | 0.678 | 0.673 |
| 2024/25 | 1.002 | 1.057 | 0.704 | 0.683 |
| 2025/26 | 0.974 | 1.056 | 0.718 | 0.706 |

Takeaway:
- optional context sources should degrade gracefully; core forecast production should not fail because an enrichment source is unavailable.

---

## v2.2.5 — Adaptive transfer search and policy-validation infrastructure

**Status:** Operational/performance design.

Decision:
- replace always-wide live search with adaptive search:
  - start with a fast profile;
  - widen only when first-action separation is insufficient;
  - keep the old broad search as a reference benchmark, not an automatic production path.
- make policy validation resumable by season/profile with signed checkpoints.
- require a complete scorecard before a policy can be labelled validated.

Takeaway:
- search effort should scale with ambiguity, not be paid every week regardless of need.

---

## v2.2.6 — Autosub invariants

**Status:** Correctness fix.

Issue:
- historical realised-score autosub handling could infer an impossible nominal formation from only appearing players.
- example failure: nominal starter formation reported as `DEF=3, MID=6, FWD=1`.

Decision:
- pass the true nominal XI into realised autosub scoring;
- remove unsafe fallback inference;
- add hard decision invariants:
  - 15 unique squad players;
  - 11 starters;
  - exactly 1 starting GK;
  - legal outfield formation;
  - 1 bench GK;
  - 3 ordered outfield substitutes;
  - captain and vice-captain distinct starters;
- add regression checks over legal formations and random appearance/no-show patterns.
- invalidate policy/sequential artifacts affected by scoring semantics.

Takeaway:
- realised-score backtests require the same substitution legality rules as real FPL, otherwise policy comparisons are not trustworthy.

---

## v2.2.7 — Fair policy comparison, bank-aware diagnostic, sparse-player evidence

**Status:** Mixed: policy result validated; bank-aware threshold rejected as production; sparse-evidence fixes retained.

### Fair policy baseline

A major methodological flaw was identified in the earlier policy comparison: different policies had been allowed to optimise different opening squads. This made whole-season scores incomparable.

Decision:
- all policies must start each holdout season from one shared neutral opening state.
- fail fast if opening-squad/static-benchmark signatures differ.

### Fair policy scorecard

| Policy | Mean realised points | Season ranks | Mean rank | Conclusion |
|---|---:|---|---:|---|
| `greedy_1gw_no_hits` | **2196.7** | 1, 3, 1 | **1.67** | **Selected** |
| `greedy_1gw_bank_aware` | 2188.3 | 4, 1, 2 | 2.33 | Diagnostic only |
| `rolling_6gw_hits` | 2160.0 | 3, 2, 4 | — | Rejected |
| `rolling_3gw_no_hits` | 2126.3 | 2, 5, 3 | — | Rejected |
| `rolling_6gw_no_hits` | 2116.7 | 5, 4, 5 | — | Rejected |

Production decision:
- **`greedy_1gw_no_hits` is the validated production policy.**
- normal production should not take hits.
- the bank-aware threshold of 1.25 objective points per extra transfer is **not validated as a production policy** and remains a diagnostic.

### Sparse-player / defensive-contribution case study

The Khalaili recommendation exposed overconfidence in sparse defensive-action evidence.

Problems found:
- recent defensive-action rates could dominate too quickly;
- sparse-player market prior effectively disappeared after very little PL evidence;
- external prior matching could miss a player's previous-league context.

Decisions:
- reliability-shrink defensive-action rates toward career/position anchors;
- fade sparse-player market prior smoothly rather than removing it abruptly;
- improve external-player matching;
- add evidence-risk flags:
  - sparse PL evidence;
  - defensive-contribution share of xP;
  - missing external event prior;
  - prior weight;
  - cheap near-equivalent alternative;
  - weak marginal value of an extra transfer.

Observed effect:
- Khalaili's projection fell from approximately 3.9 xP to roughly 3.5 xP after the sparse/DC shrinkage.
- the recommendation became materially less dominant.

Takeaway:
- low-sample players should not be blacklisted, but their xP must reflect epistemic uncertainty and shrinkage.

---

# 5. Production-policy decision

## Accepted policy

**Validated:** `greedy_1gw_no_hits`

Rationale:
- best average within-season rank in the fair shared-opening-state experiment;
- best mean realised score among the tested profiles;
- longer horizons did not justify their additional modelling and optimisation complexity;
- hit-enabled policy did not win the cross-season comparison.

## Rejected policy conclusions

- `rolling_6gw_no_hits` lost to greedy in all three fair-baseline seasons.
- `rolling_3gw_no_hits` was also net inferior.
- `rolling_6gw_hits` did not provide sufficient realised-value evidence to justify routine hit-taking.
- fixed bank-aware threshold = 1.25 was not sufficiently validated and should not be treated as production truth.

## Current operational handling

Because later forecast-signature experiments changed post-processing without re-running the expensive policy suite, the notebook uses:

```text
WEEKLY_POLICY_FALLBACK = "greedy_1gw_no_hits"
```

This preserves the historically selected policy without pretending that every later forecast-signature variant has a newly signed policy scorecard.

---

# 6. v2.3.0 — Observational live xP audit

**Status:** Diagnostic only.

Motivation:
- through the early 2026/27 season, the managed team was not consistently beating the overall FPL average.
- aggregate historical MAE/Spearman was not sufficient evidence that the model was good at the actual decision task.

Decision:
- preserve pre-deadline forecasts and evaluate realised outcomes directly.
- do not run new policy simulations.

The online ledger records:

- squad/starting-XI xP vs actual;
- player xP vs actual;
- expected vs actual minutes;
- model top-K ranking;
- captain diagnostics;
- simple FPL baseline comparisons;
- manager score vs overall FPL average;
- transfer-execution provenance.

### Early live evidence

At the time of the first audit:

- true preserved model forecast GWs: **GW4 and GW5**;
- manager vs overall average through GW5: **1 of 5 GWs above average**;
- cumulative manager delta: **-29 points**.

On the same archived player rows for GW4/GW5:

| Metric | Model | FPL `ep_next` | Better |
|---|---:|---:|---|
| MAE | **2.939** | 3.443 | Model |
| Spearman | **0.269** | 0.104 | Model |
| Top-10 realised sum | 48 | **49** | FPL baseline |

Takeaway:
- the model was somewhat better at absolute prediction/ranking on these two snapshots, but **did not convert that into better top-10 selection**.
- two GWs are far too small for tuning; this sample is treated as a shift detector only.

### Captain diagnostic

In the two archived live snapshots:
- GW4 captain Bruno: 2 actual; best displayed candidate: 8; hindsight raw regret 6.
- GW5 captain Bruno: 2 actual; best displayed candidate: 14; hindsight raw regret 12.

Decision:
- do **not** redesign captaincy from two outcomes.
- treat very small xP gaps as low-confidence rather than as precise ordering.

---

# 7. v2.3.1 — Historical decision-quality audit

**Status:** Validated diagnostic evidence.

Purpose:
- determine whether historical global MAE was hiding weak decision quality.
- reuse existing OOS folds only.
- no new fitting and no transfer/season simulation.

## 7.1 Historical ranking results

| Season | Mean model Spearman | Baseline Spearman | Model NDCG@10 | Baseline NDCG@10 | Model top-10 actual | Baseline top-10 actual | Lift | Model win rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2023/24 | 0.331 | 0.239 | 0.417 | 0.358 | 53.368 | 47.868 | **+5.500** | 63.2% |
| 2024/25 | 0.329 | 0.245 | 0.481 | 0.408 | 58.474 | 51.395 | **+7.079** | 65.8% |
| 2025/26 | 0.252 | 0.151 | 0.388 | 0.315 | 50.868 | 40.842 | **+10.026** | 76.3% |

Conclusion:
- **historical decision-ranking signal is genuine.**
- the earlier aggregate metrics were not merely hiding a useless ranking model.
- therefore 2026/27 underperformance should not automatically trigger a wholesale xP rewrite.

## 7.2 Premium attackers

Mean realised points from the model's premium top 3 versus the simple baseline:

| Season | Model | Baseline |
|---|---:|---:|
| 2023/24 | 18.632 | 16.079 |
| 2024/25 | 21.974 | 20.711 |
| 2025/26 | 15.474 | 14.684 |

Conclusion:
- premium-attacker ranking adds historical value overall.
- direct premium inflation should not be assumed to improve decisions.

## 7.3 Position-level results

| Position | Model MAE | Baseline MAE | Model Spearman | Baseline Spearman | Main takeaway |
|---|---:|---:|---:|---:|---|
| DEF | 1.068 | 1.137 | 0.636 | 0.595 | Model adds value |
| FWD | 1.077 | 1.161 | 0.744 | 0.739 | Small ranking gain |
| GK | 0.631 | 0.691 | 0.664 | **0.769** | Model MAE better, ranking worse |
| MID | 0.977 | 1.055 | 0.732 | 0.718 | Aggregate value, but candidate ordering still a concern |

Important nuance:
- MID aggregate ranking can be good across all midfielders while **decision-candidate MID ranking** remains weak.
- the important comparison is not “bench fodder vs elite MID,” but “Saka vs Bruno vs Palmer vs Semenyo, etc.”

## 7.4 Price-band results and premium compression

| Price band | Model MAE | Baseline MAE | Model Spearman | Baseline Spearman | Model bias |
|---|---:|---:|---:|---:|---:|
| <5.0 | 0.677 | 0.694 | 0.626 | **0.644** | +0.003 |
| 5.0–6.4 | 1.412 | 1.578 | 0.659 | 0.593 | -0.171 |
| 6.5–7.9 | 2.029 | 2.264 | 0.600 | 0.536 | -0.414 |
| 8.0–9.9 | 2.785 | 3.224 | 0.433 | 0.371 | -0.650 |
| 10.0+ | 3.238 | 3.464 | 0.538 | 0.547 | -0.682 |

Conclusion:
- the model has a systematic tendency to **underpredict expensive/high-xP players**.
- however, this is a calibration observation, not by itself evidence that increasing premium xP will improve selection.

## 7.5 xP calibration by predicted-xP bucket

| Predicted xP bucket | Mean xP | Mean actual | Bias (predicted - actual) |
|---|---:|---:|---:|
| <1 | 0.226 | 0.208 | +0.018 |
| 1–2 | 1.505 | 1.725 | -0.220 |
| 2–3 | 2.508 | 2.681 | -0.173 |
| 3–4 | 3.340 | 3.642 | -0.302 |
| 4–5 | 4.373 | 4.920 | -0.547 |
| 5–6 | 5.447 | 5.785 | -0.338 |
| 6+ | 7.197 | 7.709 | -0.512 |

Conclusion:
- high-xP compression is real and systematic.
- it required a targeted experiment rather than a manual multiplier.

## 7.6 Minutes metrics

| Holdout season | Appearance Brier | Start Brier | 60+ Brier | Minutes MAE |
|---|---:|---:|---:|---:|
| 2023/24 | 0.0983 | 0.1077 | 0.1087 | 15.246 |
| 2024/25 | 0.0998 | 0.0909 | 0.0941 | 14.360 |
| 2025/26 | 0.0891 | 0.0818 | 0.0839 | 13.249 |

Conclusion:
- minutes modelling improved across folds but remains an important source of uncertainty.
- intermediate/rotation-risk minute ranges produce much larger errors than nailed-starter ranges.
- minutes error propagates into appearance points, clean sheets, goals/assists, saves, goals conceded, bonus, and defensive contributions.

## 7.7 Attacking-event model diagnosis

Historical xG/xA ranking by position remained reasonably informative and approximately unbiased. Example Spearman values:

- MID xG: ~0.565
- MID xA: ~0.684
- FWD xG: ~0.630
- FWD xA: ~0.608

Decision:
- **do not rewrite the attacking-event model wholesale** based on the current evidence.
- minutes/role uncertainty is a more justified next target.

---

# 8. Early-season public comparator

**Status:** External diagnostic only; not our model.

A public archived FPL xP source was used where genuinely timestamped data existed.

Verified public GW1 comparator:

- rows: 610;
- FPL xP MAE: 1.586;
- bias: -0.036;
- Spearman: 0.493;
- FPL top-10 realised points: 51;
- oracle top-10 within scope: 138;
- top-10 oracle efficiency: 0.370.

Important limitation:
- genuine archived pre-deadline **our-model** forecasts for GW1–GW3 were not available.
- no attempt was made to relabel reconstructed/post-deadline data as historical model forecasts.
- where public archived xP was unavailable, the gap remains explicit.

---

# 9. Human execution provenance

This distinction is required for any audit of realised manager performance.

## Genuine deviation

- **GW2:** user decision differed from the recommendation without being the later assistant-advised “bank the extra FT” conservative override pattern.

## Assistant-advised conservative overrides

The notebook's execution-provenance table records:

| GW | Raw model action | Executed action | Provenance |
|---|---|---|---|
| 4 | Maguire -> Richards; Palmer -> Saka | Maguire -> Dunk | Assistant-advised conservative override |
| 5 | Milenković -> N. Williams; Thiaw -> Tarkowski | Milenković -> N. Williams | Assistant-advised conservative override |
| 6 | Dunk -> Hall; Palmer -> Saka | Roll | Assistant-advised conservative override |

Interpretation:
- these are **not** manager noncompliance.
- the extra transfer was intentionally not spent because the incremental model edge was judged too small/fragile relative to uncertainty and future optionality.
- audit attribution must therefore distinguish:
  1. raw model recommendation;
  2. reviewed/assistant-advised decision;
  3. manager execution.

This is essential when analysing whether poor realised performance arose from:
- forecast error;
- optimiser choice;
- conservative human override;
- or manager deviation.

---

# 10. v2.4 — Targeted forecast-correction experiments

**Status:** Both candidate corrections **Rejected**.

Motivation:
- test the two main v2.3.1 diagnostic hypotheses:
  1. upper-tail/premium compression;
  2. weak MID decision-candidate ordering.

The base component model remained unchanged. Both corrections were trained only on historical OOS forecast rows and required chronological acceptance gates.

## 10.1 Upper-tail calibration

Observed aggregate decision:

- evaluated folds: 2;
- MAE improvement: **-0.0336** (worse);
- absolute-bias improvement: **+0.1428**;
- overall NDCG change: **-0.0028**;
- overall top-10 realised change: **-0.6447 points/GW**.

Decision:
- **Rejected**.

Interpretation:
- the correction successfully reduced underprediction bias;
- however, correcting calibration in this way **made FPL decision quality worse**.
- therefore “premium compression exists” does not imply “inflate premium xP.”

## 10.2 MID reranker

Observed aggregate decision:

- evaluated folds: 2;
- MID Spearman change: **-0.0023**;
- MID NDCG@5 change: **-0.0039**;
- MID MAE change: **+0.0464** (worse);
- overall NDCG change: **-0.0052**.

Decision:
- **Rejected**.

Interpretation:
- a small residual Ridge reranker did not solve the decision-candidate MID problem.
- direct post-hoc reranking was therefore not allowed into production.

## 10.3 Combined layer

Observed combined result:

- approved: **False**;
- overall NDCG change: approximately **-0.0077**;
- top-10 realised change: approximately **-1.3158 points/GW**.

Decision:
- **Rejected**.

## 10.4 Uncertainty layer

**Status:** Diagnostic/confidence layer retained.

Decision:
- attach historical minutes/xP uncertainty to forecasts;
- do **not** subtract uncertainty from xP;
- use uncertainty to downgrade confidence when forecast differences are small.

Takeaway:
- a 0.1–0.2 xP gap should not be presented as a strong preference when historical forecast uncertainty is several points.

---

# 11. Why minutes became the next target

After rejecting the v2.4 correction hypotheses, minutes/role modelling remained the strongest evidence-backed target.

Key observations:

- historical fixture-level minutes MAE remained roughly 13–15 minutes;
- error was much larger in uncertain expected-minutes bands;
- players in rotation-risk bands could have P80 minute errors on the order of several tens of minutes;
- a minutes error propagates to almost every FPL scoring component;
- single expected-minutes values poorly represent nonlinear scoring around:
  - appearance;
  - 60-minute appearance threshold;
  - clean-sheet eligibility;
  - saves;
  - goals conceded;
  - defensive-contribution thresholds.

Decision:
- do not alter goal/assist/team-strength architecture yet;
- test a probabilistic role/minutes formulation first.

---

# 12. v2.5 — Probabilistic minutes / role model

**Status:** **Implemented — pending validation**.

The v2.5 notebook has been created but, at the time of this audit log, its chronological acceptance results have not yet been reviewed from an executed run.

## 12.1 Design

The existing hierarchy is retained:

- `P(appearance)`;
- `P(start | appearance)`;
- `P(60+ | start)`.

Only one additional classifier is fitted:

```text
P(75+ | starter and 60+)
```

This produces five mutually exclusive role states:

1. `no_appearance`
2. `substitute`
3. `start_lt60`
4. `start_60_74`
5. `start_75_plus`

FPL scoring is then evaluated conditionally within each state and probability-weighted.

Key principle:
- this is a **post-forecast minutes layer**, not a rewrite of the underlying team/event models.

## 12.2 Acceptance gate

The v2.5 layer must:

- evaluate on at least 2 chronological folds;
- improve `P(75+)` Brier by at least 0.0005;
- improve minutes MAE by at least 0.05;
- not degrade xP MAE by more than 0.02;
- not degrade NDCG by more than 0.002;
- not reduce top-10 realised points by more than 0.25;
- not degrade attacker Spearman by more than 0.003.

No 2026/27 outcomes are used to tune or approve the layer.

If the gate fails:
- production xP remains on the existing base model;
- role-state outputs may still be retained diagnostically.

## 12.3 Current status

**Pending:** execute v2.5 and inspect Section 16.

No conclusion should be drawn yet about whether probabilistic minutes improves production forecasts.

---

# 13. Caching and computation policy

The project now follows a **cache-first** rule for expensive work.

## 13.1 Base walk-forward folds

v2.5 persists expensive base fold bundles under:

```text
data/walk_forward_fold_cache_base_v2_4_0/
```

These bundles contain fitted minutes/team/event models and holdout OOS forecasts.

Important design choice:
- v2.5 intentionally retains the **v2.4 base forecast compatibility signature** because the underlying base models did not change.
- the new probabilistic-minutes layer has a separate signature.

This avoids invalidating expensive work when only a post-processing/diagnostic layer changes.

## 13.2 v2.5 specific caches

```text
data/probabilistic_minutes_v2_5_0.json
data/probabilistic_minutes_oos_v2_5_0.csv.gz
data/probabilistic_minutes_live_model_v2_5_0.pkl.gz
```

They separately persist:

- approval/validation artifact;
- complete OOS role/minutes audit rows;
- full-history live `P(75+ | starter, 60+)` model.

## 13.3 Other reusable artifacts

Examples of reusable state already introduced during the project include:

- online pre-deadline forecast archive;
- v2.4 targeted-correction artifact;
- all-competition negative/failure cache;
- signed policy-validation artifacts/checkpoints;
- historical data caches.

## 13.4 Cache rule for future iterations

A future change should invalidate only the artifacts whose inputs or semantics changed.

Do **not**:
- bump a broad signature merely because the notebook version number changed;
- refit all historical folds for a diagnostic that only consumes existing OOS predictions;
- rerun policy simulations unless forecast/policy semantics materially changed and policy revalidation is actually required.

Do:
- sign caches using relevant model code/configuration/data semantics;
- persist fold-level predictions and fitted objects where safe;
- record whether each result was `loaded` or `computed`;
- provide an explicit force-rebuild switch for audit/reproducibility.

---

# 14. Known data and evidence limitations

## 14.1 Historical deadline availability

**Known limitation:** historical backtests do not have complete true deadline-time injury/availability snapshots for every player/GW.

Implication:
- some historical expected-minutes errors are not purely model errors; the live system may know information that the historical reconstruction cannot reproduce.

## 14.2 Current-season archived forecasts

At the first observational audit:
- true preserved model snapshots were available for GW4 and GW5.
- later notebooks preserve more complete pre-deadline snapshots going forward.
- GW1–GW3 our-model pre-deadline forecasts should not be reconstructed using post-deadline information and then presented as genuine forecasts.

## 14.3 Public early-season baseline

- a verified public FPL xP snapshot was available for GW1.
- missing later public snapshots remain missing rather than being fabricated.

## 14.4 Optional external enrichment

FBref:
- optional enrichment had incomplete coverage for npxG/xAG.
- FPL xG/xA remains the authoritative core attacking source in the affected setup.

All-competition workload:
- source can fail/degrade;
- circuit breaker and negative cache prevent repeated request storms;
- missing optional workload context should be surfaced as degraded context, not silently treated as complete.

## 14.5 Small live sample

The 2026/27 online sample is a **shift detector**, not a tuning set.

Rule:
- do not tune thresholds/model parameters to a handful of current-season GWs.

---

# 15. Rejected or intentionally deferred ideas

The following should not be reintroduced without new evidence:

| Idea | Status | Reason |
|---|---|---|
| Routine rolling 3GW/6GW transfer policies | Rejected | Inferior to greedy 1GW in fair historical policy test |
| Routine hits | Rejected for normal production | Hit-enabled policy did not win fair historical test |
| Fixed 1.25 bank-aware threshold as production policy | Rejected / diagnostic only | Did not outperform greedy overall |
| Direct upper-tail/premium xP inflation | Rejected | Bias improved but OOS MAE/ranking/top-10 worsened |
| Residual MID reranker | Rejected | OOS MID/overall ranking did not improve |
| Treat uncertainty as an xP penalty | Rejected by design | Uncertainty is confidence context, not negative expected value |
| Tune on GW4/GW5 live outcomes | Rejected by methodology | Sample too small and would create leakage/overfitting |
| Re-run full policy simulation for every model diagnostic | Rejected by workflow | Excessive cost; diagnostics should reuse cached OOS folds first |

---

# 16. Current production assumptions and controls

As of v2.5.0 code:

```text
AUTONOMOUS_MODE = True
WEEKLY_POLICY_PROFILE = "auto"
WEEKLY_POLICY_FALLBACK = "greedy_1gw_no_hits"
ONLINE_DIAGNOSTIC_COUNTERFACTUALS = False
V25_PROBABILISTIC_MINUTES_MODE = "auto"
V25_FORCE_REBUILD = False
WALK_FORWARD_FOLD_DISK_CACHE_ENABLED = True
```

Still deliberately disabled unless separately validated:

```text
WEEKLY_TRANSFER_ROBUSTNESS_GATE = False
WEEKLY_STRUCTURAL_REPAIR_GATE = False
```

Interpretation:
- production should prefer validated/simple behaviour over unvalidated wrappers.
- code existing in the notebook is not sufficient justification for enabling it.

---

# 17. Audit questions to apply to every future change

Before accepting any new model or policy layer, answer:

1. **What specific failure mode is this change intended to address?**
2. **Was that failure demonstrated in OOS or live observational evidence?**
3. **Can the test reuse frozen/cached predictions instead of refitting?**
4. **Is the evaluation chronological and leakage-safe?**
5. **Does the candidate improve the actual FPL decision objective, not merely calibration?**
6. **Does it preserve or improve top-K/NDCG/ranking?**
7. **Does it improve the target subgroup without harming the full decision set?**
8. **Was current-season data kept out of parameter tuning?**
9. **Is the acceptance/rejection rule declared before inspecting the result?**
10. **If rejected, is the candidate guaranteed not to affect production?**
11. **Is human execution provenance separated from raw-model provenance?**
12. **Will the resulting artifacts be cached and signed for reuse?**

---

# 18. Recommended next audit step

The immediate next step is **not another broad model change**.

Run v2.5 once and inspect:

- `P(75+)` Brier change;
- minutes MAE change;
- xP MAE safety;
- NDCG/top-10 safety;
- MID/FWD ranking safety;
- the final production approval flag.

Decision rule:

- **If v2.5 passes:** allow the probabilistic role/minutes integration into production and then observe new live pre-deadline forecasts prospectively.
- **If v2.5 fails:** retain the current base xP model and use the role-state output only as confidence/diagnostic information. Do not immediately replace it with another complex minutes model without a new diagnosis.

Only after the forecast layer is stable should policy revalidation be considered, and only if the forecast semantics changed enough to plausibly alter the previously selected `greedy_1gw_no_hits` result.

---

# 19. Primary notebook/artifact lineage

Major notebooks in the audit trail:

1. `FPL_Predictor_v2.2.0_Audit_Fixes.ipynb`
2. `FPL_Predictor_v2.2.1_Audit_Fixes.ipynb`
3. `FPL_Predictor_v2.2.2_Audit_Fixes.ipynb`
4. `FPL_Predictor_v2.2.3_Schema_Contracts.ipynb`
5. `FPL_Predictor_v2.2.4_Production_Resilience.ipynb`
6. `FPL_Predictor_v2.2.5_Adaptive_Search_Policy_Validation.ipynb`
7. `FPL_Predictor_v2.2.6_Backtest_Autosub_Invariants.ipynb`
8. `FPL_Predictor_v2.2.7_Fair_Policy_Bank_Aware_Sparse_Evidence.ipynb`
9. `FPL_Predictor_v2.3.0_Observational_xP_Diagnostics.ipynb`
10. `FPL_Predictor_v2.3.1_Decision_Quality_Diagnostics.ipynb`
11. executed v2.3.1 lineage: `fpl_predictor(20261004-181207).ipynb`
12. `FPL_Predictor_v2.4.0_Targeted_Forecast_Corrections.ipynb`
13. executed v2.4 lineage: `fpl_predictor(20261004-184439).ipynb`
14. `FPL_Predictor_v2.5.0_Probabilistic_Minutes_Cache_First.ipynb` — **implemented, pending executed validation at the time of this log**

---

# 20. Final current-state summary

## Validated / retained

- core historical xP architecture has useful decision signal;
- schema contracts and execution invariants;
- autosub legality fixes;
- sparse-player/defensive-action shrinkage;
- all-competition circuit breaker / degraded-optional-source handling;
- fair shared-opening policy methodology;
- `greedy_1gw_no_hits` as the selected policy;
- observational pre-deadline forecast archive;
- uncertainty reporting as confidence context;
- cache-first walk-forward artifact reuse.

## Rejected

- fixed bank-aware threshold as production policy;
- rolling 3GW/6GW policies as preferred default;
- routine hit policy;
- v2.4 upper-tail calibration;
- v2.4 MID residual reranker;
- combined v2.4 correction layer.

## Pending

- v2.5 probabilistic minutes/role integration.

## Most important modelling lesson so far

The project should not optimise whichever metric is easiest to improve.

A change is only useful if it improves the **actual FPL decision task**—ranking/selecting/captaining/transferring players—under chronological out-of-sample evaluation, and it should not be promoted merely because it makes calibration or a subgroup statistic look cleaner.
