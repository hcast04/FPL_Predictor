---
document: experiment ledger — everything tested, and what it taught
status: LIVING DOCUMENT — append, never rewrite history
last_updated: 2026-10-20 (rebuild: scaffold, scoring verified, minutes model)
---

# Experiment ledger

A running record of every hypothesis this project has tested, what happened, and
— more valuable than any individual result — **the methodological mistakes that
produced wrong answers along the way**.

Most entries are failures. That is the point. The failures are what stop the
same idea being re-tested every few weeks, and the methodology section is what
stops the next experiment being wrong in the same way as the last one.

## How to update this

1. **Append, never delete.** A result that gets overturned stays, with its
   reversal recorded underneath. The reversal history is the most useful part.
2. **Record the verdict, the evidence, and the gate** — what would have counted
   as success, declared *before* the run.
3. **Record who/what refuted it.** Several entries here were overturned by
   adversarial review, not by the original author.
4. **If a number changes, say what the old number was and why it was wrong.**

---

# Part 0 — The standing rules

These were learned the hard way. Each one is here because breaking it produced
a confidently stated wrong answer.

| # | Rule | Why |
|---|---|---|
| R1 | **Measure the noise floor before believing any delta.** | A 0.1% multiplicative jitter moves a single season by up to **112–125 points**; 1% jitter gives a 3-season-mean sd of **39–49**. **Any season-total delta under ~150 points is noise.** Measured twice, by two independent implementations (2026-10-19 and 2026-10-20, `fpl2/run_noise_floor.py`). |
| R2 | **A result must hold in every holdout season.** | With 3 seasons, P(a null signal winning 3/3) = 0.125. |
| R3 | **…and even then, count your comparisons.** | ~45–60 distinct comparisons have been run on this project. At 0.125 each, **~6 spurious 3/3 winners are expected by chance alone**. |
| R4 | **Walk-forward caches leak into the future.** | The cached forecast for gameweek *g+5* was produced using data up to *g+4*. Reading it at time *g* is look-ahead. This invalidated an entire horizon study (below). |
| R5 | **Check the unit of analysis.** | 86,755 rows are player-**fixtures**, not player-gameweeks (1,756 double-gameweek pairs). Aggregating correctly halved a headline bias. |
| R6 | **Baselines must be honest.** | "The model's #1 pick scores 6.07 vs a league median of 0.00" is rigged — the pool includes every non-playing squad member. Against the most-owned player: +0.23, p=0.657. |
| R7 | **A handicap must not be adversarial.** | Removing the top X% *by the model's own xP* bans exactly the players the transfer policy wants later. It cost 5× more than a handicap that removed *more* opening xP. |
| R8 | **Never relax a gate to make a layer pass.** | `V24_UPPER_TAIL_MIN_MAE_IMPROVEMENT` went 0.01 → −0.05 → `float('-inf')`. A gate at −inf is not a gate. |
| R9 | **Unconditional calibration ≠ calibration on selected players.** | The model under-predicts high-xP players overall, but what matters is the bias on the XI it actually picks — a different quantity. |
| R10 | **Exclude unfinished gameweeks from evaluation.** | An unplayed fixture contributes structural zeros. 41% of one "failure rate" was a match that hadn't kicked off. |
| R11 | **Verify provenance before using archived forecasts.** | GW4/GW5 snapshots hold 30 rows (`displayed_top30`) with `captured_at_utc: None`. Unusable as evidence. |
| R12 | **`web_name` is not a key.** | "Palmer" is both Cole Palmer (CHE, MID, £9.7m) and an Ipswich goalkeeper. Join on `player_code` / `id`. |
| R13 | **Never judge an expectation by MAE when the outcome is bimodal.** | Minutes are ~0 or ~90. A naive trailing mean predicts 88.5 against an actual of 79.1 — badly biased — yet beats a well-calibrated model (78.3) on MAE, because MAE rewards predicting the mode, not the expectation. Judge calibration within buckets, and the downstream decision metric. Measured 2026-10-20, `fpl2/minutes.py`. |
| R14 | **A suspiciously perfect result is a bug, not a triumph.** | Held-out rate calibration came back at a ratio of exactly 1.000 for five quantities across three seasons. Cause: the evaluation frame recreated columns with the same names as the predictions, so actuals were being compared against actuals. Predictions are now `pred_`-prefixed so the collision cannot recur. Measured 2026-10-20, `fpl2/rates.py`. |
| R15 | **A rule that changed mid-history cannot be learned from before it existed.** | The defensive-contribution bonus began in 2024-25/2025-26. A classifier trained on earlier seasons learns "never happens" and predicts zero. Restrict training to the period the rule was live, and check for this whenever FPL changes scoring. |
| R16 | **Overall rank correlation is a near-useless model-selection metric here.** | Two signals tied on spearman (0.704 vs 0.701) differed by 347 points a season, because 29k rows are dominated by players scoring ~0 while selection only touches the top. The discriminating metric is **mean realised points of the top-k**: at k=20 the same two signals differ by 0.583 per player. Measured 2026-10-20, gate G1. |---
| R17 | **Per-player OVER-prediction does not persist across seasons; do not build corrections on it.** | 528 players, honest out-of-sample ratios: cross-season correlation 0.132 (1.7% of variance), and flat to three decimals across the three most over-predicted quartiles (0.9421 / 0.9504 / 0.9446). Only the under-predicted tail carries signal. A player the model over-rated last season is no more likely to be over-rated again. Measured 2026-10-20, gate G2. |
| R18 | **A feature whose MEANING depends on context needs that context as a feature too.** | `season_appearances_so_far` is 0 for everyone at gameweek 1, but mid-season 0 means "never selected". With no `gameweek` feature the model could not distinguish them, averaged the two populations, and under-predicted cold-start appearances by 18% (p_appear 0.353 vs actual 0.433; expected minutes 22.2 vs 28.6). Calibration from GW5 onward was fine, which is why season-level metrics never showed it. Measured 2026-10-20. |
| R19 | **A rejected experiment can leave damage behind.** | `season_appearances_so_far` was added only to support shrinkage (F15, rejected). The shrinkage was reverted; the feature was not, and it silently degraded the minutes model. When an experiment is rejected, revert everything it introduced, not just the headline change. |# Part 1 — What survived
| R20 | **A threshold is only as good as the population it is computed on.** | Three separate failures today, same shape: a cap from SINGLE-MATCH rates (q99 goals/90 = 2.000, one player scoring twice) capped nothing; a cap applied to TARGETS left Poisson-boosted predictions unbounded; a cap from player-seasons with no MINUTES FLOOR was inflated by cameos (MID assists q99 1.121 vs 0.634 at >=900 min). Each time the idea was right and the population was wrong. State explicitly which rows a threshold is estimated from, and check it actually binds. |
| R21 | **Never `str.replace` with a computed slice without asserting it is non-empty.** | `s[s.index(a):s.index(b)]` returned "" because b preceded a; `str.replace("", x)` inserts between EVERY character and turned an 8KB source file into 20MB. Edit source with anchored exact-match replacements that fail loudly, not index arithmetic. |
| R22 | **Assume every population can be empty.** Four instances in one session: an empty string slice destroyed a source file; caps computed on the wrong rows never bound; a feature column that is all-NaN in one fold crashed sklearn's binner with an error naming nothing useful; a waiter matched its own command line. Guard the degenerate case explicitly -- `features.usable()` is the pattern: filter per fold, state what was dropped. |
| R23 | **Poll log CONTENTS, never process names.** `pgrep -f`/`pkill -f` match the polling command itself. This silently prevented a gate from running at all, and once discarded a script before it was written. Four occurrences in one session. |
| R24 | **State every gate threshold in units of the measured noise floor.** | The G9 best-XI gate required "lose no more than 5 pts in any season" while R1 says a season total moves **112-125 points** under information-free jitter. Both the pass AND fail criteria sat ~30x below the noise floor, so the verdict carried no information in either direction. R1 is rule number one and was not consulted before the gate was written. |
| R25 | **Never touch a running experiment's inputs.** | Creating placeholder CSVs in `data/external_manual/` mid-run made arm 2 read a dataset changing underneath it -- 2 of 4 league-seasons present, 2 absent, a configuration that will never exist again. The files look like static fixtures; they are read lazily, per season, during the run. Checksum inputs before launch and `md5sum -c` after. Same class as R23: an action that looks inert but is not, because something else is concurrently reading what changed. |
| R31 | **Improving a component metric is not evidence for the decision metric, and has four times now predicted the opposite.** | R9, W13, F19, F28. Each improved exactly the quantity it targeted -- calibration, clean-sheet probability, minutes correlation -- and each lost points. The mechanism is dilution: xP RANKING drives squad selection, and extra columns that sharpen a component can blunt the ranking. Always gate on the season total; report the component as mechanism only, never as partial success. |
| R33 | **A mechanism confirmed is not a benefit delivered, and may deliver the opposite.** | S51 moved squad ownership 13.9% -> 29.2% exactly as designed, and beat%% got WORSE (-2.03, p 0.042) while sd ROSE. Sixth occurrence of R31's pattern and the first where the targeted metric moved AGAINST the hypothesis. Always measure the benefit, never infer it from a working mechanism. |\n| R32 | **Cluster by season. Always.** | Every paired test of 2026-10-22 treated 60-1368 observations as independent when they were clustered in 3 seasons. Clustered, F26 went t=3.52 -> 1.11 (p=0.38) and F28 t=2.77 -> 1.38 (p=0.30); both were reported as rescued findings and are withdrawn. With G=3 the bar is |t| > 4.30 on 2 df. The naive SE answers 'would this win again on these same seasons'; the clustered SE answers 'would it win on a NEW season', which is the only question worth asking. |\n| R30 | **A live-season table is a hypothesis generator, never evidence.** | F24 (price calibration, n=14 in the top band) and F25 (XI-aware objective, +74 over 7 gameweeks) were both motivated by 2026-27 tables that looked clean and both were refuted by the holdout seasons the same day. The live season has 7 of 38 gameweeks and no replication available. Generate the hypothesis there; pre-register the gate; decide on the holdouts. |
| R29 | **A grouping whose membership CHANGES with the treatment cannot be compared across arms.** | Three instances on 2026-10-21, in three disguises. M10: club tiers ranked by realised points, so a club entered the top tier BY scoring. O20: `self_pooled` had no 2023-24 value, so a 2-season mean was compared against a 3-season one and printed a spurious PASS. F19: shrinking lambda moves teams OUT of the top clean-sheet bucket (n 2278 -> 628), so the bucket ratio appears to worsen (1.461 -> 1.502) while overall error improves -- the ratio is computed over a different population at every shrink value. Fix the bucket EDGES to the control arm's assignment, or compare on matched membership. |
| R27 | **A gate on a noisy statistic names a THRESHOLD, never a direction.** | "Must not fall" fails ~half the time under pure noise regardless of the noise floor, and did so twice on 2026-10-21 (M13) -- on moves of 0.0004 and 0.0006. Phrase every gate as "must not fall by more than X", with X measured for that statistic, not borrowed from another one. R1's 148 is for full-squad SEASON TOTALS and does not govern spearman, armband points, or opening-15 quality; each needs its own floor (the captain floor is 0.32-1.62 pts/gw, `fpl2/run_captain_floor.py`). |
| R28 | **Nothing under /tmp survives, INCLUDING the session scratchpad.** | systemd-tmpfiles swept `/tmp` twice on 2026-10-21, destroying every cached forecast (live panel, walk-forward outputs, external priors) and, later, a source-file backup taken for a revert. The scratchpad is itself a subdirectory of `/tmp` and was swept by age -- only files touched in the preceding ~20 minutes survived. The first version of this rule said "cache to the scratchpad instead" and was WRONG. Anything that must outlive an hour -- a cache worth minutes of compute, a backup taken before editing a shared file -- goes inside the project or `$HOME`. A revert plan that depends on a `/tmp` backup is not a plan. |
| R26 | **A negative control is what separates information from perturbation.** | G9's budget simulation beat its baseline in 3/4 seasons, +193 aggregate -- a clean PASS against the baseline alone. Against priors SHUFFLED between players it won only 2/4, and in the live season the shuffled control outscored the real priors (+40 vs +31). Adding any feature block perturbs the fit; only the shuffle says whether the gain came from the information. |

# Part 1 — What survived

| # | Claim | Evidence | Status |
|---|---|---|---|
| S1 | **The xP model beats every dumb baseline at actually managing a team.** Same optimiser, £100m, 2/5/5/3, max 3/club, 1 FT no hits — only the ranking signal swapped. | model_xp **2145** vs template 1923, form5 1894, form3 1868, most-expensive 1602, points-per-£m 1583, random 644. Wins 3/3 seasons; gaps are 6–14× the noise floor. | **SOLID** |
| S2 | **The XI/captain optimiser is correct.** | Exhaustive over all C(15,11) legal subsets: model XI = xP-maximal, diff +0.000. Captain = argmax. Bench order descending xP. | **SOLID** |
| S3 | **The transfer search is near-optimal.** | Brute-forced all 574 legal single transfers for GW8: the recommendation ranks **2nd**, 0.004 xP from best. | **SOLID** |
| S4 | **Chip deferral is the expensive failure, not chip mistiming.** | Deferring to zero costs ~**−92/season**; bad timing costs ±5 vs random. | **BOTH HALVES NOW CHALLENGED, 2026-10-21.** Deferring ALL chips costs +47 to +194 (S23), more than the recorded 92 -- so that half is understated. But the timing half is contradicted twice: a secretary policy and a calendar policy differ by ~147 from the ceiling (F21), and FORCING every chip to play scores 15 points/season WORSE than letting one or two lapse (F22). Mistiming is expensive; marginal deferral is cheap. The ordering in S4 is backwards for this simulator. S23 measures deferral at **+47 to +194/season** depending on timing -- more than the recorded 92. But the spread between a reasonable fixture-list policy (+47) and good timing (+194) is **~147 points**, not ±5. Either S4's timing test compared two already-good policies, or its chip model differed from `fpl2.simulate`'s. Needs re-running before either number is quoted. |
| S5 | **FPL price changes are highly predictable.** | Logistic on net transfers / ownership: AUC **0.95** for rises, 12–13× lift in the top 5%. | **SOLID** (kept as information only) |
| S6 | **The model under-predicts high-xP players.** | At xP ≥ 3.5: n=2,536, 246 players, bias **−0.690**, CI [−0.890, −0.471], same sign all 3 seasons. | **SOLID at ≥3.5** (see W3 for the version that was wrong) |
| S7 | **Role-change miscalibration.** The model over-predicts players whose role shrank and under-predicts those whose role grew. | Downgrades: **+0.124 pts/GW**, 3/3 seasons, p<0.0001, permutation p=0.021. Does **not** decay, so it is not staleness. | **SOLID — firmest unfixed defect** |
| S8 | **Caution: roughly half to two-thirds of S1's margin is captaincy, not squad selection.** | Holding XI+captain fixed at model_xp, the squad-signal gap drops from 222–277 to **79–152**/season. | **SOLID — limits S1's reach** |
| S25 | **Chips are worth +87/season with an HONEST policy, and the model then beats the FPL average by ~67.** | 2026-10-21, `fpl2/run_chip_honest.py`. Policy E sees the fixture schedule only to **gw+4** -- the realistic confirmation horizon -- and combines near-term fixture shape, the current squad's state and this week's xP, with a deadline forcing a play. **+208 / -31 / +85, mean +87, 2/3 seasons**, against the leaky season-long calendar's +69. Season equivalents: model alone 2237, **model + honest chips 2324**, real FPL average **2257** -> **+67 above average**. It also rescues 2024-25 (-31 vs the leaky policy's -105), the season that broke S13, F20, F21 and F22. | **SOLID, and the only chip result not contaminated by M17** |
| S24 | **Chip timing signal is in the FIXTURE LIST, not the forecast. An optimal assignment is the only policy positive in every season.** | **CONTAMINATED -- see M17.** Policies A, C and D all scanned SEASON-LONG double/blank counts, which do not exist in a June fixture list. Their +69 / +62 / +52 are unachievable. The conclusion also does not survive: the best honest policy (E, +87) uses fixture shape AND the forecast AND squad state, so "the calendar not the forecast" was an artefact of comparing three leaky fixture-readers against one honest forecast-reader. | **WITHDRAWN** | | 2026-10-21, `fpl2/run_chip3.py`. Four policies against a +194 ceiling: **A calendar** (greedy on fixtures) +180/-105/+132, mean **+69**, 2/3; **B multiweek** (benefit projected over 5 gameweeks from this week's xP) +105/-122/+90, mean +24, 2/3; **C learned** (Ridge on realised chip gains, trained on PRIOR seasons only) -/+108/+15, mean +62, 2/2; **D assignment** (8 chips to gameweeks solved optimally on fixture features, no forecast at all) +49/**+105**/+3, mean +52, **3/3**. A has the best mean; **D is the only one that never loses**, and by R2 that is the shippable one. **2024-25 -- which broke S13, F20, F21 and F22 -- is finally positive under C (+108) and D (+105) where A loses 105.** Four attempts to beat a fixture-only heuristic using the model's own output have now failed, and the one that leaned on it hardest (B) came last. | **SOLID: the calendar, not the forecast, carries chip timing** |
| S23 | **With chips modelled, the model BEATS the real FPL average.** | 2026-10-21, `fpl2/run_chips.py`. Chips added to `fpl2.simulate` (8 per season: 2 each of wildcard / free hit / bench boost / triple captain, split GW1-19 and GW20-38, per bootstrap-static). `chips=None` reproduces 2143 / 2420 / 2149 exactly, so nothing else moved. A **calendar policy** -- timing taken only from the published fixture list, wildcard at a fixed gameweek per half, bench boost and triple captain on the most-DOUBLE gameweeks, free hit on the most-BLANK -- gains **+156 / -97 / +81, mean +47**. Season equivalents: model alone **2237**, model+chips **2284**, real FPL average **2257**. **The headline the project has been chasing since the rebuild: +27 above the average.** Caveats that bound it: +27 is deep inside the 148 floor, so "at or slightly above average" is the honest reading; the policy wins only 2/3 seasons; and it captures just **24%** of the hindsight ceiling (+47 of +194), so most of the chip value is still unclaimed. | **SOLID on direction; the MARGIN is inside the noise floor** |
| S22 | **The model is LEVEL with the real FPL average, not behind it. The simulator is handicapped by ~287 points/season and that handicap is bigger than the apparent deficit.** | 2026-10-21, `fpl2/run_s1_check.py` + `fpl2/run_template.py`. S1 REPRODUCES: ownership through our simulator scores **1970** against S1's recorded 1923 -- within noise, so a model signal genuinely does beat an ownership signal under identical constraints. But the UNCONSTRAINED most-owned XI scores **54.6-61.5/gw**, matching the real FPL average of **59.4/gw** -- so it is the CONSTRAINTS, not the signal, that cost the template 287 points. Put against the right benchmark: constrained template **1970**, legacy **2215**, our model (GLM + minutes fix) **2237**, real FPL average equivalent **2257**. The simulator has no chips, no wildcard and no hits, and does not model within-season price changes; the average manager has all of those. S4 already prices chip deferral alone at ~92/season. **The model is ~20 points off the average while playing with one hand tied.** **CORRECTION 2026-10-21:** an earlier version of this entry listed "accumulated team value" as part of the gap. That is WRONG -- FPL resets every manager to 100.0m at the start of each season. Team value compounds only WITHIN a season via price rises, which the simulator also ignores (it uses each player's first price all year), but that is worth a few million of upgrade power, not hundreds of points. **Second caveat on the attribution:** the 287 is the gap between the real average and an ownership-SIGNAL squad built by our optimiser, which is not the same thing as the template people actually own -- maximising total ownership under a budget can buy 15 cheap popular players rather than the real template. So part of the 287 is a poor proxy, not a missing feature. The chip component is being measured separately (`fpl2/run_chips.py`). | **SOLID on the headline; the 287 ATTRIBUTION is partly unverified** |
| S21 | **The minutes fix works by DEMOTING over-rated rotated players, not by promoting elites -- and it pays off only where the model was already getting the captain wrong.** | 2026-10-21, `fpl2/run_o27.py`, K=0 vs K=4. **2025-26 (+219):** Haaland's xP moves +0.015 -- nothing -- but Isak -0.41, Martinelli -0.40, Marmoush -0.37, Isidor -0.35 are marked down, and he goes from captained **0/38 to 20/38**. The captain changes in **30 of 38** gameweeks, worth **+118** doubled, with another ~+101 from squad changes (XI overlap 7.0/11). **2024-25 (-33):** Salah gains xP as predicted (+0.155) but was ALREADY captained **32/38** -- nothing to win. The captain changes in **2** gameweeks and both land badly (-38). So the season asymmetry is explained by how wrong the model started, NOT by how dominant the top scorer was. **O27's premise was backwards.** Connects to S14: the GBDT captained Haaland 3/38 in 2025-26 and the GLM 0/38 -- both were missing the best player in the league every week, and the minutes correction is what surfaces him. | **SOLID** |
| S19 | **The minutes model over-shrinks: it under-predicts ever-present starters and over-predicts rotated players, 4/4 seasons.** | 2026-10-21, `fpl2/diag_minutes_coldstart.py`. Grouped by CURRENT-SEASON start rate computed strictly before each gameweek, gameweeks 3-10. Predicted/actual minutes -- ever-present: **0.951 / 0.963 / 0.971 / 0.941**; rotated (<75%): **1.142 / 1.076 / 1.065 / 1.124**. `p_sixty` shows the same (0.79-0.82 predicted vs 0.83-0.85 actual). Classic regression-to-the-mean: the model pulls everyone toward the middle regardless of how unambiguous the current-season evidence is, and it already HAS that evidence (`minutes_r3`, `minutes_r5` are in the feature set). **Magnitude is modest, 3-6%.** Note on the grouping: membership depends on realised starts, so between-group levels are not comparable -- only the within-group prediction/actual ratios are. | **SOLID** |
| S20 | **The live captain inversion is a minutes error, and correcting it flips the ranking.** | 2026-10-21, `fpl2/diag_bruno.py`. The model ranks Bruno (11.8m, 33 pts) above Haaland (15.7m, 58 pts) and captained Bruno three weeks running for 1, 1, 2 (O24). Decomposed: the model KNOWS Haaland scores more (predicted 0.794 goals/90 vs 0.392, exactly double). It loses on assists (-0.737, probably correct), the MID/FWD clean-sheet rule (-0.294, the rules), and a **minutes discount**: predicted 71.96 minutes and p_sixty 0.785, against an ACTUAL 7 starts in 7, mean 88.4, zero blanks. That discount multiplies every rate he has. Substituting actual minutes for both: Bruno 6.006 -> 6.515, Haaland 5.552 -> **6.793** -- **the ranking flips** (+0.454 Bruno ahead becomes -0.278 Haaland ahead). Haaland's 19% error is far above S19's systematic 3-6% because his career history (7 and 6 zero-minute gameweeks in 2023-25) actively contradicts his current form. | **SOLID for this case; it is one pair, hand-corrected with hindsight minutes** |
| S18 | **The clean-sheet model OVER-predicts the best defences and is well calibrated elsewhere.** | 2026-10-21, `fpl2/run_cs_calib.py`, per player among those who played 60+ (which is what `clean_sheet_prob` means -- it enters `expected_points` weighted by `p_sixty`). Pooled by predicted-probability bucket, n=2,278-8,544 per cell: 0.111->0.144 (0.77), 0.201->0.255 (0.79), 0.294->0.302 (**0.97**), 0.393->0.328 (**1.20**), 0.542->**0.371** (**1.46**). Where we say 54% the reality is 37%. Season ratios are mixed (1.281 / 0.896 / 0.853 / 0.943), so this is NOT a uniform bias -- it is specific to the strong-defence tail. Consistent with the live GW6-7 collapse, where the XI was van Dijk / Tarkowski / Branthwaite / Guehi / Richards and the clean sheets did not arrive (MCI predicted 0.552, realised 0.419). **Per-team 2026-27 figures are n=7 fixtures and should not be read** -- Brighton's +0.447 is 5 clean sheets instead of 2. | **SOLID in the pooled buckets; per-team noise** |
| S16 | **The GLM is statistically INDISTINGUISHABLE from legacy. The "deficit" we spent the day explaining is, for our best model, within noise.** | 2026-10-21. GLM mean margin **-57** over three seasons. R1 measured the 3-season-mean sd at **39-49** under information-free jitter, so 3sd is 117-147: **-57 is ~1.2-1.5 sd from zero.** The GBDT at -267 IS genuinely behind (>5 sd); the GLM is not. G1 still FAILS, but G1 requires BEATING legacy by +148 -- a deliberately demanding bar, not a finding that we lose. **Ten mechanisms were tested to explain a gap that may not exist** (see O23). | **SOLID — and it reframes most of 2026-10-21** |
| S17 | **Transfer quality is identical. We are only more CONFIDENT.** | 2026-10-21, `fpl2/run_transfers.py`, ~110 transfers per arm from a shared opening 15, scored directly as realised(IN)-realised(OUT) with no simulator in the path. Welch, unpaired: d1 **t=0.06**, d3 **t=-0.04**, d5 **t=-0.38**, rest-of-season **t=-0.13** -- indistinguishable on every horizon. The ONE significant difference is `claimed` gain: ours 2.84 vs legacy 2.36, **t=+2.45**. We overstate what a transfer will deliver (+0.28 vs realised) while legacy understates (-0.15), and it costs nothing because the moves are equally good. | **SOLID** |
| S14 | **The GBDT has a graded defect at the TOP of the distribution. It is not a tie-break and not general ranking weakness.** | 2026-10-21, `fpl2/run_o22.py`. Mean within-gameweek rank of players who finished the season in each tier: **top-10 gbdt 50.0 / glm 37.4 / legacy 46.2**; 11-50 97.1 / 90.0 / 92.4; 51-150 147.8 / 147.4 / 141.4. The gap is **+12.7 ranks at the top, +7.1 next, +0.4 at the bottom** -- graded, and zero where it does not matter, which is why overall spearman looked identical (0.67-0.71). Share of gameweeks with an elite player in the model's top 3: gbdt **29.8/57.0/28.1%**, glm 60.5/78.9/57.9%, legacy 57.9/78.9/44.7%. 12.7 mean ranks is systematic, not a near-tie lost at the argmax. | **SOLID** |
| S15 | **Elite identification explains GBDT->GLM. It does NOT explain legacy's advantage.** | 2026-10-21. The GLM identifies elite players BETTER than legacy (mean top-10 rank 37.4 vs 46.2; top-3 hit rate 57.9% vs 44.7% in 2025-26) and still LOSES to legacy on season totals (-57 vs 0). So the clean narrative -- "find the best players and you win" -- is false. Whatever legacy does better than the GLM is a different mechanism, still unidentified. | **SOLID — a constraint on explanations, not a mechanism** |
| S13 | **2024-25 is unstable because it contains a 344-point outlier the GBDT barely finds.** | 2026-10-21. Salah scored **344**, beating the next player by **108**; the equivalent gaps in 2023-24 and 2025-26 are 14 and 4. (Gini is actually LOWEST in 2024-25 at 0.619 -- this is one extreme player, not general concentration.) Captaincy of that player: GBDT **15/38**, GLM 28/38, legacy 25/38. Because the GBDT identifies him only 40% of the time, seed perturbations flip the choice and each flip is worth ~2.5 pts/gw doubled -- producing the 162-point seed spread (-307/-230/-145), S12's +401 there being ~90% scale effect (O19), and the 539-point alpha swing. Contrast 2023-24, which is stable and uniformly bad: NOBODY captains Palmer (0/0/0), so there is no lever to be unstable about and every configuration lands at -323 to -428. | **SOLID — explains why one season drives several conclusions** |
| S12 | **The GBDT is actively hurting. A Poisson GLM on the same features beats it by ~194 points a season.** | 2026-10-21, `fpl2/run_glm_replication.py`. Swapping ONE component -- `rates._reg` -- from HistGradientBoosting(poisson) to `PoissonRegressor(alpha=1.0)` on median-imputed, standardised inputs. Mean vs legacy: baseline **-251/-253/-274/-252** across 4 seeds (tight), GLM **-57**. top-20 94.54 vs 90.08, captain-added 7.54 vs 5.41, both 3/3 seasons. **Beats legacy outright in 2024-25 (+94)** -- the only configuration all day to beat legacy in any season. Decomposed: ~55 pts of the gain is median imputation (a GBDT on identical preprocessing reaches -199/-192), ~139 is model class. Alpha dose-response is a clean inverted-U: -103 / **-57** / -256 at alpha 0.1/1.0/10. All four pre-declared replication criteria passed. | **SOLID — but see caveats below** |
| S10 | **Forecast quality is NOT the bottleneck.** Three signals of wildly different complexity rank the same and lose/win the same. | 2026-10-21. spearman over 3 holdout seasons: legacy 0.684/0.705/0.720, the 120-feature GBM 0.668/0.696/0.711, and a NO-TRAINING four-tier lookup (`fpl2/fpl2/simple.py`, builds in 9s) 0.692/0.696/0.721 -- the simple model ranks BETTER than the GBM in 2 of 3. Squad-simulation margin vs legacy: complex **-290**, simple **-304**. Weeks of rate models, ceilings, shrinkage, external priors, per-position models and Understat context were all optimising a quantity on which all three are already equivalent. | **SOLID — reframes the project** |
| S11 | **Half the ~300-point deficit is the OPENING SQUAD, one decision made before any within-season data exists.** | 2026-10-21, `fpl2/run_decompose.py`. Giving our signal legacy's opening 15 moves the margin **-304 -> -156**; everything after that point is identical. An opening squad picked with perfect HINDSIGHT scores -151, and legacy's real one scores -156 -- so legacy is already at the attainable bound and ~150 pts is genuinely recoverable. The other ~150 lives in transfers/XI/captain and is **undiagnosed**. Direct inspection (2025-26): our opening 15 was worth 1683 season points, legacy's 2184, same budget. | **SOLID** |
| S9 | **The new-to-PL defect is a SOURCE-LEAGUE effect, not a club-strength effect, and it is far worse than the pooled number showed.** | 2026-27 GW1-6. The pooled new-to-PL ratio of 0.847 averages two different populations: big-5 arrivals **0.903**, Championship arrivals **0.723**. Controlling for destination club — *within the three promoted squads only*, same fixtures and coaching — Championship-sourced **0.713** (n=33) vs other-sourced **0.884** (n=44). The gap holds at every club tier (0.12–0.23). Club tier was the competing hypothesis and is rejected. | **SOLID — a measurement, independent of whether any fix ships** |

---

# Part 2 — What failed

Declared gates, run, failed. Do not re-test without a new idea.

| # | Hypothesis | Gate | Result |
|---|---|---|---|
| F1 | Assist-conversion rescaling (v2.6 F1) | improve top-10 on the real pipeline | Harness said +2.2/GW; real pipeline **−0.3 to −1.7**. Shipped OFF. |
| F2 | Rank-based bonus model (predict BPS, simulate the 3/2/1 ranking) | MAE + dispersion + Spearman | Recovered dispersion (0.17→0.28) but **ranked worse** in both folds. Rejected. |
| F3 | Value-aware transfer tiebreak (break xP ties on predicted price moves) | positive value *and* positive net, both folds | +7.3 one fold, **−0.7** the other. Rejected. **STATUS REVISED 2026-10-21 — see M12: both deltas are ~3-5% of the season-total noise floor. Uninformative, not refuted.** |
| F4 | Recency: **train on the last X gameweeks only** | beat "train on all", all 3 seasons | More data wins monotonically. Best X beats "all" in **1/3** seasons. Rejected. |
| F5 | Recency: **blend xP with trailing 3-GW form** | improve the decision metric | Top-10 realised points worse at **every** weight in **every** season. Best weight = 0.0. Rejected. |
| F6 | **Understat team context** (PPDA, non-penalty xG, deep completions, own + opponent, 5-GW trailing; 99.3% join) | win all 3 seasons | spearman **0/3** (negative in all three). best_xi 1/3. Rejected. |
| F7 | **Per-position models** (separate model per GK/DEF/MID/FWD — Adham's structure) | win all 3 seasons | spearman **0/3**. best_xi 1/3. Rejected. |
| F8 | **Fixture-concentration cap** (limit XI players drawn from one match) | improve mean *or* cut downside | Loses on mean **and** 10th percentile in all 3 seasons. Rejected — see M5. |
| F9 | **Variance-adjusted captain rules** (top-3 by upside; xP + 0.5·upside) | beat argmax(xP), all 3 seasons | +0.66 / −1.45 / −0.03 and +0.24 / −0.55 / −0.34 **per gameweek** (`selection_experiments.py:133`) = +25/−55/−1 and +9/−21/−13 per season. Rejected. **STATUS REVISED 2026-10-21 — see M12. Sign flips with no consistent direction and no noise floor was ever measured for armband points. Correct as a SHIPPING decision; unsupported as a conclusion that the avenue is closed. Highest-value reopen on the board (S8, O12).** |
| F10 | **Longer decision horizon** (H = 2…12 gameweeks) | beat H=1, all 3 seasons | Leaky version said +196/season. **Leakage-free: H=2–10 all worse.** See M1. |
| F11 | **Stale cross-season team priors hurt club-changers** (points) | changers predicted worse | No effect. Club change costs **<0.15 pts/GW**, now bounded with 5× the original precision. Rejected. |
| F12 | Direct-GBM xP replacing the component model | beat the component pipeline | Lost on the decision metric. Rejected. |
| F13 | v2.4 MID rank reranker | declared ranking gates | Failed. Rejected. |
| F14 | Bank-aware transfer threshold | declared gates | Failed. Rejected. |
| F23 | **Crowd-ownership features** (`owned_lag`, `owned_r3`, `owned_share`, `owned_delta`; lagged, leakage-clean at 0.0 on two cut points) | fix the >25%-owned calibration ratio, hold spearman, improve the decision metric in 2/3 seasons | **FAIL, and the defect it targeted does not exist (M19).** Decision metric **+27 / -144 / -132, mean -83**, 1/3 seasons; spearman flat (+0.0011). The motivating measurement used CURRENT ownership applied retroactively; with contemporaneous per-gameweek `selected` the model's ownership calibration is already 0.92-1.05. **Reverted.** |
| F23 | **Crowd-ownership features** (`owned_lag`, `owned_r3`, `owned_share`, `owned_delta`; lagged, leakage-clean at 0.0 on two cut points) | fix the >25%-owned calibration ratio, hold spearman, improve the decision metric in 2/3 seasons | **FAIL, and the defect it targeted does not exist (M19).** Decision metric **+27 / -144 / -132, mean -83**, 1/3 seasons; spearman flat (+0.0011). The motivating measurement used CURRENT ownership applied retroactively; with contemporaneous per-gameweek `selected` the calibration is already 0.92-1.05. **Code REVERTED 2026-10-21** -- `features.py` is back to 100 features and `data.py` no longer carries `selected`. (The revert ran as a side effect of a partially-executed command rather than by decision; the user had asked for the code to be left in place. The exact patch is recoverable from the session transcript if it is ever wanted: four lagged features off `selected`, plus `"selected"` appended to `data._FIRST_COLS`.) |
| F22 | **Force every chip to be played** (deadline-aware scheduling: when gameweeks-remaining <= chips-remaining, fire the best available chip immediately) | beat the stranding version's +48 and the calendar's +47 | **FAIL, and informative. Playing all 8 chips scores LESS than playing 6-7.** Mean drops **+48 -> +32** while chip usage goes from 6-7 to 8/8 in every season. Visible in 2023-24: 7 chips scored +206, forcing the eighth rescheduled the others to meet deadlines and gave **+124**. **Forcing a chip to fire before its best opportunity costs more than the chip is worth** -- the opposite of S4's "deferral is the expensive failure, timing costs +/-5". Letting a marginal chip lapse is cheap; mistiming the rest is not. |
| F21 | **Optimal-stopping (secretary rule) chip timing** -- observe 37% of each chip window recording the best benefit seen, then take the first gameweek that beats it, forced play at the deadline | beat the calendar policy's +47/season | **FAIL -- identical mean, higher variance.** +206 / -156 / +95, mean **+48** against calendar's +47, capturing ~25% of the +194 ceiling either way. Benefit estimated forward-only (this week's signal, squad and fixture list; never a future forecast, per R4). **The comparison is also contaminated by a policy bug:** it returns ONE chip per gameweek and checks them in a fixed order, so at the deadline the first chip in that order claims the week and starves the rest -- 2024-25 played only **6 of 8** chips, and S23 prices deferral at +47 to +194. **Chip timing remains unsolved**: two independent forward-looking policies both capture ~25% of what is available. |
| F20 | **Blend minutes toward the player's own recent evidence** (`minutes.RECENCY_K`, w = n/(n+K), swept 0-8) to fix the over-shrinkage in S19/S20 | improve the season total in 3/3 holdout seasons -- the DECISION metric, declared as the gate from the start because calibration fixes went 0-for-3 earlier the same day | **FAIL on the gate (2/3 at every K), but the first positive margin vs legacy all session.** At K=4: 2143 / 2420 / 2149 against legacy's 2226 / 2359 / 2061 -- **-83 / +61 / +88, mean +22**, winning 2 of 3. The GLM alone was -57, so a **+79 swing**. One season carries it: 2025-26 gains **+219** (above the 112-125 single-season floor and therefore real); the other two move +52 and -33, which is noise. **The live season barely responds: 355 -> 358 against an FPL average of 416.** Legacy given the same correction post-hoc gets WORSE (mean -59), weak evidence the problem is specific to our minutes model. **Shipped OFF (default 0.0)** -- it failed its declared gate and is one-season-driven. |
| F19 | **Shrink predicted goals-conceded** (`team.CONCEDED_SHRINK`, sweep 0.0-0.5) to fix the clean-sheet over-prediction at strong defences (S18) | top bucket toward 1.0 AND middle bucket held AND mean |ratio-1| improves AND decision metric within the floor | **FAIL at every value, and it did what it was designed to do.** Aggregate calibration improves monotonically (mean |ratio-1| **0.226 -> 0.173**) while the decision metric DEGRADES (**-57 -> -110** vs legacy). Spearman is flat throughout (0.6828-0.6837). Third independent instance of R9/W13: better calibration, worse decisions. **Shipped OFF (default 0.0).** |
| F17 | **Ceiling / dispersion features** (lagged q75, sd, haul-rate of points, bps, goals, xG; 10 features, 100 -> 110) | A/B vs the same model without them: win 3/3 seasons AND mean margin > 148; spearman must not fall | **NEUTRAL, and the gate was mis-specified.** Decision margin +118/-18/+32, mean **+44** -- inside the noise floor, uninformative (M9). Spearman -0.0006 / +0.0053 / -0.0004: the "fell in 2/3 seasons" FAIL is a strict inequality tripped by the FOURTH DECIMAL, while the single gain is an order of magnitude larger. Net positive. Calibration improved where worst (2025-26 ratio 1.031 -> 1.019). No evidence they help, none that they hurt. **Second time in one day I wrote a strict-inequality gate on a noisy statistic -- see M13.** |
| F18 | **Captain rules, properly powered** (reopen of F9 per O12): xp + k*ceiling, 13 arms, dose-response, shuffled control | beat argmax(xp) on the decision metric | **The signal is REAL and the armband cannot use it.** On the league-wide best XI, xp+0.5*q75 beats argmax by **+1.52 pts/gw**, t=2.94, 4/4 seasons, **z=3.83** vs a control with the same ceilings SHUFFLED between players, robust across min_periods 3/5/10/20 (z=4.2-5.7), leakage-clean (10/10 features at 0.0 disagreement, two cut points). In a budget-constrained squad it is worth **+19 points over four seasons** -- inside the floor. Cause: the league-wide XI picks 11 from ~500 with many xp near-ties (52% disagreement); a 15-man squad almost never offers the choice. **F9's rejection was right in outcome, wrong in reasoning.** |
| F16 | **Hand-exported Championship priors** (4 seasons of FBref standard stats, pasted manually; 2162 player-seasons, 308 players filled) | two instruments, see G9 | Calibration gain is REAL (Championship arrivals 0.723 → **0.776**) but does **not** convert to points. Unconstrained best-XI: 2/4 seasons, FAIL — and that whole test ran below the noise floor (R24). Budget-constrained: beats baseline 3/4, **+193 aggregate**, but beats its own SHUFFLED control only 2/4, and in the live season the shuffle **won** (+40 vs +31). Rejected as a points-winning change. |

**Partially true:**

| # | Hypothesis | Result |
|---|---|---|
| P1 | Stale priors hurt club-changers' **minutes** (not points) | Real and decays exactly as a stale prior should: GW1–2 **+5.52 min** → GW7–12 +0.67 → GW13–38 −0.10. Permutation p<0.0005, placebo passes. **But fragile**: dies excluding GKs (p=0.26) and for established regulars (p=0.72); no single season reaches p<0.05. Concentrated in fringe/rotation movers, and decayed to nothing by the time it was found. |

---

# Part 3 — Claimed, then withdrawn

### W18 — "Clean sheets are under-predicted by 25-35% in every season" (2026-10-21)
Produced by `run_cs_calib.py` before its aggregation was corrected, and caught
BEFORE publication rather than after -- the only one of the session's retractions
where the explanation test ran first.

**The flaw:** realised clean sheets were taken as `max(clean_sheets)` over each
TEAM-gameweek. FPL awards a clean sheet to any 60+ minute player who was off the
pitch before a late goal, so a team can concede and several players still record
one. That inflated the realised rate from ~0.24 to ~0.39 against a prediction
that means something narrower.

**What exposed it:** the mechanism reached for to EXPLAIN the finding instead
contradicted its magnitude. If `exp(-lambda)` under-stated P(0), goals-conceded
would have to be heavily over-dispersed. It is not: var/mean **1.078**, actual
P(0) 0.242 vs Poisson 0.227 -- an excess of 1.5 points, nowhere near the 10-15
claimed. The Poisson assumption is sound.

**Corrected finding (S18):** the direction REVERSES. Per player among 60+
minutes, the model is well calibrated in the middle and **over**-predicts the
strong-defence tail (0.542 -> 0.371). The original claim had the direction, the
magnitude and the affected end all wrong.

**Lesson:** a model quantity must be scored against the population it is
DEFINED over. `clean_sheet_prob` is per-player-given-60-minutes, weighted by
`p_sixty` in `scoring.expected_points`; comparing it to a team-level max is a
different quantity that happens to have the same name.

### W17 — "The defensive-contribution gap explains the cheap-defender miss" (2026-10-21)
Proposed on seeing that the model under-rates De Cuyper (5.3m, 59 pts, xp 2.72),
Vuskovic, Bogle and Calafiori, and citing G1's recorded diagnosis that the DC
bonus cannot be learned from pre-2025-26 history.

**Refuted immediately.** Only **2 of the top 10** defenders average 10+
defensive contributions -- De Cuyper 3.3, Bogle 2.9, Calafiori 3.4, all far
below the DEF threshold of 10. DC is not what is scoring them.

**But the run sharpened the underlying data problem, and the ledger had it
slightly wrong.** `defensive_contribution` is not NaN before 2025-26 -- it is
recorded as **0.00** (100% non-null, 0.0% hitting the threshold, across
2022-23/2023-24/2024-25, versus mean 7.5 and ~27% clearing it in 2025-26 and
2026-27). Three of five training seasons teach the model that DC is ALWAYS
ZERO, which is worse than absent: the model learns a false fact rather than
holding no opinion.

**Seventh retracted explanation of the session**, against one confirmed (O22).

### W16 — "Half the deficit is the opening squad" (2026-10-21), and "the gap is signal SHAPE"
Two claims from the same run, both now bounded or refuted.

**S11 was measured on the SIMPLE model and I over-generalised it.** Handing the
simple model legacy's opening 15 moved it -304 -> -156, and I reported that as a
property of the project. With the GLM it does not replicate: legacy's opening 15
gives **-62 against our own -57** -- no help. The reason is that the GLM already
fixed most of it. Opening-15 season points: simple **1683**, GLM **1993**,
legacy 2136. The gap closed from 453 to 143 without anyone targeting it.
S11 stands AS A STATEMENT ABOUT THE SIMPLE MODEL and is wrong as a statement
about the project.

**"The gap is signal shape" is refuted.** Both squads come from the same MILP,
rankings are comparable, so the values looked like the only remaining lever --
a stretched top buys premiums, a flat one spreads. Giving our ranking legacy's
value distribution made the opening squad WORSE (1948 vs 1993), recovering
**-32%** of the gap.

**Also eliminated along the way:** early-season signal quality. Within the top
150 the optimiser picks from, OUR GW1-5 signal out-ranks legacy in 2 of 3
seasons (gbdt 0.293/0.550/0.347, glm 0.419/0.460/0.305, legacy
0.389/0.468/0.265).

**One small finding that did survive:** legacy's opening 15 is worth +143 more
raw season points and STILL does not produce a better season under our signal.
The opening squad and the transfer policy interact -- a squad your own signal
does not rate is not worth inheriting.

**Caveat on all of it:** the season totals in this run are -57 / -44 / -62, an
18-point spread deep inside the 148 floor. Only the open15 column is large
enough to read. That is why the conclusions here are about SQUAD QUALITY, not
about season totals.

### W15 — "All-GLM passes its gate (+165, 3/3)" (2026-10-21, same day)
Reported from `run_o17.py` and **withdrawn within the hour** by the completed
gate it had declared but not implemented.

Two defects, both mine:
1. **The declared gate was not implemented.** It said "wins 3/3 on both top-20
   and captain-added"; the code compared those only as season MEANS, so the
   per-season clause was never evaluated. Same class as O20's mismatched folds.
2. **The comparator was one seed.** `+165` was measured against a single
   baseline. The seed spread is **80 points of mean** (-198 / -251 / -278).
   Against the BEST seed the gain is **+96 -- below the 148 floor.**

Completed gate (`run_o17_gate.py`), vs the BEST of 3 seeds per cell:

| metric | beats best seed |
|---|---|
| top-20 | 2/3 (loses 2024-25) |
| **captain-added** | **3/3** (6.2/9.2/6.7 vs 4.7/7.1/4.7) |
| season margin | 2/3 (loses 2024-25) |

**Verdict: PARTIAL.** All-GLM beats every seed on mean (-102 vs -198) and wins
captain 3/3, but the mean gain is unmeasurable and top-20 does not hold.
Caveat on the captain result: the captain floor measured earlier (0.32-1.62
pts/gw) was for the armband WITHIN a fixed XI; this metric is the top-xp player
in the whole pool with the XI varying, so it is a different quantity and its
floor is UNMEASURED (R27).

**Lesson:** a gate compared against ONE seed of a stochastic baseline is not a
gate. Report against the best comparator available, not a convenient one -- the
difference here was +149 vs +96, i.e. pass vs fail.

### W14 — "Normalising away the volatile scale component will help" (2026-10-21)
Predicted from O19's split (ordering 3/3 reliable, scale sign-flipping) and
written into O20 as the next step. **Refuted by O20 the same day.**

The error was reading "volatile" as "noise". The scale component was
+8 / +362 / -98 -- mean **+91**, high variance but clearly positive. Removing it
is a variance-for-mean trade, not a free win, and the trade loses: the only
shippable template costs 44-66 points per season on matched folds.

**Lesson:** a high-variance component is not automatically a worthless one.
"Inconsistent" and "uninformative" are different claims and need different
evidence. This is the THIRD explanation retracted in one session (W13, W14, and
the mis-designed rescale arm in G10) against zero retracted measurements --
every surviving result today came from measuring a quantity, and every failure
came from reasoning about a mechanism.

### W13 — "Over-prediction at the top is what costs us points" (2026-10-21)
Asserted repeatedly through the 2026-10-21 session and used to interpret the
opening squad (S11), transfer quality (O13) and the captain. The evidence
offered was that legacy under-predicts its top picks (ratio 0.81-0.97) while
the rebuild over-predicts (0.94-1.14), and that over-prediction is expensive
under a budget because you pay premium prices for forecasts that do not land.

**Refuted by O18** (`fpl2/run_o18.py`), which measured top-end calibration
across three signals:

| signal | top-end ratio | season total vs legacy |
|---|---|---|
| legacy | 0.81-0.97 | -- |
| GBDT baseline | 1.03-1.14 | -267 |
| Poisson GLM | **1.03-1.19 (worst)** | **-57 (best of ours)** |

Top-end calibration does not order the outcomes: the worst-calibrated signal is
the best-performing one of ours. The correlation that motivated the claim was
real but ran through only two points (legacy vs rebuild), and a third point
breaks it.

**Lesson:** two signals differing on two axes cannot tell you which axis
matters. The claim was repeated for most of a session before a third point was
measured, and it was written into the ledger as a mechanism attached to a SOLID
finding. **An explanation needs its own test, not the result it explains.**



The reversals. Each of these was stated with confidence and later overturned.

| # | Original claim | Reversal | Caught by |
|---|---|---|---|
| W1 | "FPL pays 6 points for a goalkeeper goal" | **Wrong — it pays 10.** Patch withdrawn. | external audit |
| W2 | Clean-sheet exposure has a bug | Empirically validated as **correct** (γ=0.975). Withdrawn. | re-validation |
| W3 | "xP under-predicts by −1.83 (4.5–5.0) and −1.87 (5.0+)" | **Unit error** (player-fixtures vs player-gameweeks) and the buckets were **2 players** (Haaland, Salah), p=0.072, one-season driven. Correct figure: **−0.69 at xP ≥ 3.5**. | adversarial agent |
| W4 | "The model's #1 pick averages 6.07 vs a median of 0.00" | **Rigged baseline.** vs most-owned: +0.23, **p=0.657**. The #1 pick is Haaland or Salah 75% of the time. | adversarial agent |
| W5 | "Dropping the wildcard structural gate is worth +40/season" | Reversed once F1 contamination was removed (74.0 vs 70.5). Withdrawn. | re-test |
| W6 | "The minutes model loses to a naive last-3 average" | **Refuted.** The gap was an unplayed fixture plus 60 pre-selected rows. On the only clean full-pool finished gameweek the model **wins** (11.88 vs 12.31). | adversarial agent |
| W7 | "Captain ceiling gap = −32 points over GW4–7" | **Double-counted.** The armband's marginal value is the *second* copy: gap is **16**, not 32. Two of four weeks contribute zero. Haaland was unfundable (short £6.5–7.5m on one FT). | adversarial agent |
| W8 | "A weak opening squad costs 181–299 points/season" | **Refuted — adversarial handicap** (R7). A squad built at the user's exact −9.7% deficit costs **0.0 points/season**. Plausible handicaps average −60 (range +37 to −179). | adversarial agent |
| W9 | "Decision horizon H=12 is worth +67/season" | **~85% was further leakage** in the supposedly leakage-free script (see M1). With all three defects fixed: **no horizon beats H=1**. | adversarial agent |
| W10 | "The live XI's calibration has flipped sign (+1.12/player)" | **Refuted on the full sample.** Using all finished gameweeks: bias **−0.11**, 95% CI [−1.36, +1.13], **p=0.86**. The flip was GW6 plus an unfinished GW7. | the user, by challenging n=2 |
| W11 | "Fixture concentration is costing you points" (advised 3× across two post-mortems) | **Refuted** — capping it loses on mean *and* downside in all 3 seasons (F8). | own test, belatedly run |
| W12 | "GW6 average was 55" | Final average was **58**. GW6 deficit was −28, not −25. | re-check |

---

# Part 4 — Methodological failures, in detail

The expensive ones. Each produced a wrong published number.

### M1 — Look-ahead leakage, twice, in the same study
`horizon_sweep.py` scored future gameweeks with the fold cache's own forecasts
for those weeks — forecasts built with data the manager would not have. Result:
a spurious **+196/season**.

The "corrected" `horizon_sweep_clean.py` **still leaked**, three ways:
- the opening squad was chosen on mean horizon score over gameweeks 1..H, so the
  leak *grew with H* — H=12 got the most;
- it multiplied by the current fixture's ease although `model_xp` already prices
  it (the sibling script `price_wildcard_horizon.py` fixes exactly this);
- `sched` took `opponent_short_name` "first", silently **dropping the second
  fixture of every double gameweek** (23/10/10 team-gameweeks).

**Lesson:** writing "leakage-free" in a docstring does not make it so. Diff the
information set available at decision time, explicitly, per feature.

### M2 — The noise floor was never measured until late
Only at the end did anyone perturb the signal and re-run. A **0.1%** jitter
moves a season by up to 112 points. Several "findings" across this project are
smaller than that. **Measure the noise floor first; it tells you what size of
effect is even detectable.**

### M3 — Gate-shopping
`V24_UPPER_TAIL_MIN_MAE_IMPROVEMENT`: 0.01 → −0.05 → `float('-inf')`. The layer
degrades MAE by 0.053 and ships anyway. A flag named
`V26_RELAX_UPPER_TAIL_MAE_GATE` turns out only to suppress the *warning*, not to
restore the gate. **If a layer only passes after the threshold moves, it failed.**

### M4 — Optimising on the metric being measured
"Your squad is 9.7% below optimal" compares a human-assembled squad to one
chosen by maximising the very quantity being compared, and on the 15-man xP
**sum** — which counts four bench players who score nothing. On XI+captain the
gap is 6.3%; against an *achievable* one-transfer squad, 4.9%.

### M5 — Advising on an untested intuition
Fixture concentration was flagged as a problem **three times across two
post-mortems** before anyone tested it. When tested, capping it was harmful on
every measure in every season. **Test before advising, not after.**

### M6 — Reporting an underpowered test as a negative result
The first stale-priors test had a minimum detectable effect of **0.76 pts/GW** —
36% of the baseline MAE — and 19.8% power against a 0.30 effect. It was reported
as "no effect". It was *uninformative*, which is a different thing.

### M8 — Four failed attempts to A/B one layer
Trying to answer "does the upper-tail layer change the recommendation?" took
four notebook runs (~100 min) and produced no answer. Each failure was a
different guess at a control surface nobody had read:

| # | What was changed | Why it failed |
|---|---|---|
| 1 | `V26_RELAX_UPPER_TAIL_MAE_GATE = False` (correct!) | ran the copy from `/tmp`, breaking its relative `fixes/` import. 40 errors. |
| 2 | the MAE threshold `-inf` → `0.0` | the approval is **cached** in `data/targeted_forecast_layer_v2_4_0.json`, restored from backup before the run, so nothing was re-evaluated. |
| 3 | `V24_TARGETED_FORECAST_MODE = "off"` | probably valid — but Section 12 was blocked by the (correct) freshness gate, and the executed notebook was deleted in cleanup before the forecast values were extracted. |
| 4 | — | stopped. |

**The control surface, finally read:**
- `V24_TARGETED_FORECAST_MODE` ∈ {auto, off, force} gates **application** via
  `_validation_mode_allows(mode, approved)` in cell 119.
- `V26_RELAX_UPPER_TAIL_MAE_GATE` selects the MAE threshold between
  `float('-inf')` and `0.01` — it does **not** disable the layer.
- The approval decision is **cached to disk**; changing a threshold without
  invalidating the artifact changes nothing.
- "v2.4 active components" in the output refers to **fitting**, not application.

**To do this properly:** set `V24_TARGETED_FORECAST_MODE = "off"`, run the copy
**inside the project directory**, delete or move the cached artifact first, and
**extract the live xP values before cleaning up** — do not rely on the execution
card, which can be blocked by the freshness gate for unrelated reasons.

**Lesson:** read the code that consumes a flag before changing it. Two minutes
of reading would have saved ~100 minutes of runs.

### M7 — Operational slips
- Self-matching `pgrep` monitor loops (twice) — a 46-minute phantom wait.
- Running a notebook copy from `/tmp`, breaking its relative `fixes/` import.
- `md5sum -c` verification misread under a Spanish locale (`La suma coincide`).
- Name collisions on `web_name` (R12) corrupted a captain-options table.
- Created placeholder input files mid-run, racing the experiment reading them (M11).

### M19 — The ownership "defect" was my own diagnostic leaking
*2026-10-21.* Measured on 2026-27 GW1-7, the model appeared to under-predict
widely-owned players catastrophically and monotonically:

    <5% owned    ratio 1.09 (over-rated)
    25-50%       ratio 0.59
    >50%         ratio 0.56

That motivated four new ownership features and a full A/B. **The gradient was
an artefact.** The ownership figure came from `selected_by_percent` in
bootstrap-static -- **current ownership, captured AFTER GW7** -- and was applied
to every gameweek. Managers buy players who have scored, so "highly owned" was
partly a restatement of "scored well", and the apparent under-prediction was
circular.

Re-measured on the holdout seasons using per-gameweek `selected` --
CONTEMPORANEOUS ownership, what people actually held at the time -- the defect
disappears: >50% band ratios **0.916 / 1.006 / 1.046**, and 25-50% at
0.991 / 0.982 / 0.902. The model's calibration by ownership is fine.

The features consequently failed: **-83 pts/season (1/3 seasons), spearman flat
at +0.0011.** They were fixing nothing.

**Same family as M17 and M18**, which the user caught earlier the same day:
confusing the state of the world NOW with its state at DECISION TIME. M17 was
the fixture list, M18 the panel's extent, this one an ownership snapshot.
**Lesson: a diagnostic needs the same leakage discipline as a model. The
features here passed `truncation_equivalence` cleanly -- it was the MEASUREMENT
that leaked, and nothing tests those.**

A caveat that cannot currently be resolved: Haaland at 76.8% was genuinely
well-owned before GW6, not only after, so some of the live-season effect may be
real. `live.py` does not capture contemporaneous ownership, so the live season
cannot be measured properly at all until it does.

### M19 — The ownership "defect" was my own diagnostic leaking
*2026-10-21.* Measured on 2026-27 GW1-7, the model appeared to under-predict
widely-owned players catastrophically and monotonically:

    <5% owned    ratio 1.09 (over-rated)
    25-50%       ratio 0.59
    >50%         ratio 0.56

That motivated four new ownership features and a full A/B. **The gradient was
an artefact.** The ownership figure came from `selected_by_percent` in
bootstrap-static -- **current ownership, captured AFTER GW7** -- applied to
every gameweek. Managers buy players who have scored, so "highly owned" was
partly a restatement of "scored well" and the apparent under-prediction was
circular.

Re-measured on the holdout seasons with per-gameweek `selected` --
CONTEMPORANEOUS ownership, what people actually held at the time -- the defect
disappears: >50% band ratios **0.916 / 1.006 / 1.046**, 25-50% at
0.991 / 0.982 / 0.902. The model's calibration by ownership is fine.

**Same family as M17 and M18**, both caught the same day: confusing the state
of the world NOW with its state at DECISION TIME. M17 was the fixture list,
M18 the panel's extent, this an ownership snapshot.

**Lesson: a diagnostic needs the same leakage discipline as a model.** The
features themselves passed `truncation_equivalence` cleanly at 0.0 on two cut
points -- it was the MEASUREMENT that leaked, and nothing in the test suite
looks at those.

**Unresolved:** Haaland at 76.8% was genuinely well-owned before GW6, not only
after, so part of the live-season effect may be real. `live.py` does not
capture contemporaneous ownership, so the live season cannot be measured
properly until it does (O36).

### M18 — A policy that mistook the DATA's extent for the SEASON's extent
*2026-10-21, in shipped library code.*

`chips.rolling_policy` built its gameweek universe from the panel:

    gws = sorted(grid["gameweek"].unique())
    left = len([g for g in gws if gw <= g <= hi])   # hi = 19 or 38

In a holdout season the panel holds all 38 gameweeks and this is correct. In a
LIVE run the panel stops at the current gameweek, so at GW7 the policy saw
`left = 4` against 4 unplayed first-half chips, concluded the window was
closing, and fired **all four in GW4-7**: `{4: bboost, 5: wildcard, 6: 3xc,
7: freehit}`. The live total jumped 358 -> **449** and briefly looked like the
model beating the FPL average by +33.

Fixed by separating `season = range(1, 39)` (a calendar fact) from `present`
(what the data happens to contain). After the fix the live run plays **zero**
chips at GW7 -- correct, since the GW1-19 observation window has not closed --
and the holdout result is **bit-identical** (+208 / -31 / +85, mean +87).

This is the same family as M17, which the user had caught twenty minutes
earlier: **confusing what the data contains with what the world contains.**
M17 was "the June fixture list has no doubles yet"; this was "the panel has no
gameweeks 8-38 yet". Both times the data's edge was read as the season's edge.

**It was also the first such bug in SHIPPED library code** rather than a run
script or a gate, and it would have misfired every gameweek of the real season.

### M17 — Chip policies read a fixture schedule that does not exist yet
*2026-10-21, caught by the user, not by me.*

Policies A (calendar), C (learned) and D (assignment) chose chip gameweeks by
scanning season-long counts of DOUBLE and BLANK gameweeks. **A June fixture
list contains neither.** Every double and blank arises later from cup
progression, European conflicts and postponements, and FPL confirms them only
~2-4 gameweeks ahead. Measured: doubles occur in just **3-10 gameweeks a
season**, first appearing at GW19 / GW7 / GW24 / GW26.

I wrote "the published fixture list, known in advance" in those docstrings and
then computed from the REALISED schedule. That is M1's lesson verbatim --
writing leakage-free in a docstring does not make it so -- and it is the second
leak of the session after M14.

**What makes it worse: the leak produced the conclusion.** S24 claimed chip
timing lives in the fixture list rather than the forecast, and the evidence was
three fixture-reading policies beating one forecast-reading policy. The
comparison was rigged toward exactly what it concluded.

**The twist:** removing the leak IMPROVED the result. An honest rolling policy
(E, 4-gameweek horizon) scores **+87/season against the leaky planner's +69**.
Committing in June to "bench boost in GW34" cannot know the bench will be
injured; adapting weekly to the squad's actual state is worth more than knowing
which gameweek is busiest league-wide.

**Lesson: when a policy "plans ahead", check what it is allowed to know at the
moment it plans.** A fixture list, a price, an ownership figure and a forecast
all have different knowledge horizons, and a season-long planner silently
assumes all of them are known at gameweek 1.

### M16 — The wildcard was optimised for a single gameweek
*2026-10-21.* The first chip implementation rebuilt the wildcard squad to
maximise the CURRENT gameweek only:

    pool = week[week["fixtures"] > 0]          # one gameweek
    optimise.optimise_squad(pool.groupby(...).agg(signal="mean"), ...)

A wildcard sets up the next stretch of a season, so it must be optimised over a
forward horizon -- which `_opening_squad` already does, five gameweeks ahead. I
wrote the chip without reusing that construction.

Cost: the chip tore up a working squad for one week's fixtures and the damage
persisted. 2024-25 lost **203 points**; across three seasons the calendar
policy averaged **-34/season**, i.e. playing the chips was worse than not
having them. With the horizon fixed the same policy averages **+47**.

Free hit is the opposite case and was already right: it reverts after one week,
so optimising it on the current gameweek alone is correct. **Lesson: when a new
decision resembles one the codebase already makes, reuse that construction
rather than re-deriving it -- the difference between the two here was 81
points/season and it looked like a finding about chips.**

### M15 — Four gate-implementation bugs in one session
*2026-10-21.* Every one printed a verdict that did not follow from the data:

1. **O20** -- `self_pooled` has no 2023-24 value, so a 2-season mean was
   compared against a 3-season one. Printed a spurious **PASS**.
2. **O17** -- the declared clause "wins 3/3 on both top-20 and captain-added"
   was implemented as a comparison of season MEANS, so the per-season part was
   never evaluated. Printed a **PASS** that a completed gate later downgraded.
3. **F19** -- bucket membership moves with the treatment (top bucket n 2278 ->
   628), so the ratio was computed over a different population per arm. See R29.
4. **O28** -- the control was PERFECT (all 15 untreated gameweeks identical,
   mean 0.00, se 0.00), giving t = 0/0 = nan. `abs(nan) < 2` is False, so a
   zero-variance control was reported as a **FAIL**.

Three of the four flattered a result; the fourth buried one. The common thread
is that each gate was written to handle the expected case and met an edge: a
missing fold, an unimplemented clause, a shifting population, a degenerate
variance. **R22 already said assume every population can be empty.** The gate
code needs the same scepticism as the model code -- it is not scaffolding, it
is the instrument that decides what gets believed.

### M14 — A real leak, shipping in production feature code
*Found 2026-10-21 by running the test suite after an unrelated change.*

`history_at_current_club_r5` / `_r10` compared each row's history against
`s.iloc[-1]` -- the player's LAST team in the whole series, i.e. the club he
ENDS UP at. At gameweek 10 the feature read a transfer that might not happen
until gameweek 25. `test_no_feature_reads_the_future` flagged it at max abs
diff **1.0**.

Two things make it worth recording beyond the fix:

**The correct expression was already there.** One line above, `same` computed
the non-leaky version and was never used. The bug and its fix sat adjacent for
however many sessions.

**A rolling window cannot express the intended quantity.** "Share of my last w
appearances at the club I am at NOW" has a per-row reference, so it needs an
explicit loop (`_history_at_club`). The leaky version is what you get when you
reach for `.rolling()` and need a scalar to compare against.

**Scope.** These are 2 of 100 features in `features.py`, so every `model.py`
result carries them -- G1's -290, F17's ceiling A/B, the G8/G9 external-prior
arms. All of those were FAILs, and removing a leak that inflates performance
can only make a failing model look worse, so **no verdict flips**. But the
exact numbers are contaminated and should be re-run before being quoted again.
S10 and S11 are NOT affected: `simple.py` never touches `features.py`, and
legacy has its own pipeline. If anything the leak flattered the GBM, so S10's
gap is wider than measured.

**The test suite earned its keep.** `test_the_leakage_test_can_actually_fail`
injects a `shift(-1)` feature and asserts it is caught -- that negative control
had only ever been theoretical. It has now caught a real one. 57/57 pass after
the fix.

### M13 — The same bad gate, twice in one day
*2026-10-21.* At 09:00 the G9 spearman gate read "must not fall" and failed on
a move of **0.0004**. The lesson was written up as M9 and R24 the same morning.

At 15:00 the ceiling-feature gate read "spearman must not fall in any season"
and failed on **-0.0006 and -0.0004**, while the one season that moved
meaningfully went **+0.0053** -- net positive, recorded as a FAIL.

Writing the rule did not prevent the repeat, because the rule was phrased about
the NOISE FLOOR and this was a different failure: a **strict inequality on a
stochastic quantity**. Any gate of the form "must not fall" fails roughly half
the time under pure noise, whatever the floor is.

**Rule: a gate on a noisy statistic must name a THRESHOLD, never a direction.**
"must not fall by more than X" where X is that statistic's own measured spread.
See R27.

### M12 — Audit: which rejections were uninformative rather than refuted
*2026-10-21, prompted by "is it possible we discarded things we should not
have?" -- after M9 showed a gate written 25x below the noise floor.*

The question is **what metric decided each rejection**, because R1's ~150-point
floor governs season totals and does NOT govern spearman, MAE, or dose-response
patterns, which are far tighter.

**Sound rejections — robust by design, not by luck:**
- F5 (form blend): worse at *every* weight in *every* season -- dose-response.
- F10 (horizon): H=2-10 *all* worse -- dose-response over 9 values.
- F4 (training window): *monotonic* in window length.
- F8 (fixture cap): loses on mean AND p10, 3/3 seasons, consistent sign.
- F11 (stale priors): a *measured* null, bounded <0.15 pts/GW at 5x precision.
- F2 / F6 / F7: judged on spearman over thousands of rows, 0/3 and negative in
  all three. Spearman is far more precise than a season total.

**Uninformative — reported as refuted, but the test could not have detected a
real effect:**
- **F3** (value-aware tiebreak): +7.3 / -0.7 season points, i.e. 3-5% of the
  noise floor. Nothing was measured.
- **F9** (variance-adjusted captain): +0.66/-1.45/-0.03 and +0.24/-0.55/-0.34
  **per gameweek**. The sign flips across seasons with no consistent direction,
  and no noise floor was ever measured for armband points specifically, so R1's
  150 cannot be applied directly -- but a sign-flipping result with no measured
  floor is exactly the M6 pattern.

**Magnitude never recorded, cannot be audited:** F12, F13, F14.

**The distinction that matters.** These rejections were correct as *shipping
decisions*: a rule that wins one season and loses another is not reliably
better and should not ship. They are NOT evidence that the avenue is dead.
"Not reliably better by an underpowered test" is not "no effect."

**The structural problem.** If a season total moves 112-125 points under
information-free jitter, then almost no realistic model improvement is
detectable by comparing season totals over 3-4 seasons. The project has been
running a gate most true effects would fail. Future decision-metric work needs
variance reduction -- paired designs holding the opening squad and draws fixed,
per-player paired tests, or metrics with better noise properties -- not more
seasons of the same comparison.

### M9 — A gate whose thresholds sat below the noise floor
*2026-10-21.* The G9 best-XI gate read: improve in >=3 of 4 seasons, **lose no
more than 5 points in any season**. R1 -- the first standing rule, measured
twice by independent implementations -- says a 0.1% information-free jitter
moves a single season by **112-125 points**.

So the fail threshold was ~25x smaller than the noise, and the deltas judged
against it (0, +43, -9, +39) were all noise. The verdict was uninformative in
BOTH directions: the FAIL meant nothing, and a PASS would have meant nothing.

The error was not the arithmetic, it was not reading our own Part 0 before
writing the gate. **Lesson: look up the noise floor and state the threshold as
a multiple of it. See R24.**

### M10 — A circular tier definition
*2026-10-21.* To ask "does the model mis-rate players at weak clubs?", clubs
were ranked by **realised GW1-6 points** and forecast-vs-realised compared
within each tier. This produced a clean monotonic story -- top6 0.814,
mid 1.066, bottom 1.154 -- which was reported before the flaw was spotted.

It is mechanical. A club enters the top tier *by having scored*, so its realised
points exceed forecast by construction; the bottom tier is the mirror. Re-cut
with tiers set from the PRIOR season the pattern collapses to non-monotonic
(0.983 / 0.884 / 1.177 / 1.023), and only one cell survives as a real finding
(O10).

What did survive untouched was the *within-tier* comparison, because both groups
face the identical artifact and it cancels -- which is why S9 stands. **Lesson:
a grouping variable must not be derived from the outcome being measured. If it
must be, only within-group contrasts are interpretable.**

### M11 — Mutating a running experiment's inputs
*2026-10-21.* Empty placeholder CSVs were created in `data/external_manual/`
while a two-arm run was executing. `load_manual` reads that directory lazily,
once per season, so arm 2 saw a dataset changing underneath it: the 2022-23
table had landed, 2023-24 and 2024-25 had not. The run reported a configuration
that will never exist again, and its numbers were discarded.

It happened twice in one session -- first by creating empty placeholders, then
again when the real tables were pasted during the following run. Both times the
files looked like inert fixtures.

**Fix applied:** `md5sum` the inputs before launch, `md5sum -c` after, and
report the verification alongside the result. Any run without that check on
file-backed inputs is not reproducible. See R25.

---

# Part 2b — Gates declared BEFORE the run (rebuild)

Declared in advance, timestamped, and judged as written. A gate edited after
seeing the result is how `V24_UPPER_TAIL_MIN_MAE_IMPROVEMENT` reached -inf.

### G3 — winsorise the per-90 rate targets
*Declared 2026-10-20, before the run.*

**Diagnosis:** a player with 20 minutes and one assist trains the model on a
target of 4.5 assists/90. These are small-sample artifacts, not rates, and they
stretch the upper tail: predicted max 1.003 assists/90 (legacy 0.289) and a
single-gameweek xP of **11.02** (legacy 5.04). The captain is `argmax(xP)`, so
it samples that corrupted tail every week -- and the captain is ~85% of the
within-squad gap.

**Change:** winsorise each per-90 training target at the 99th percentile of the
same quantity computed among players with >=60 minutes, where the rate is
actually measurable. Nothing else changes.

**PASS requires, in order:**
1. **Component gate:** captain-added points improve in **all three** holdout
   seasons on a frozen shared squad (the decomposition above: 242/234/169 for
   the new signal). If it fails here it does not reach G1.
2. **Then G1 unchanged:** beat legacy in all three seasons, mean margin > +148.

**Declared in advance:** the winsorising quantile is 0.99 and the minutes
threshold for computing it is 60. If the idea only works after tuning those,
that is a FAIL.

**RESULT — FAIL on its own mechanism, before any score, 2026-10-20.**

| | assists/90 max | xP max |
|---|---:|---:|
| before | 1.003 | 11.02 |
| after winsorising targets | 0.745 - **1.179** | 9.42 - **11.52** |

The tail did not come down; 2024-25 got worse. Two reasons, both my error:

1. **The cap was computed from SINGLE-MATCH rates**, where the 99th percentile
   for goals is **2.000** (one player, two goals, 90 minutes). That is a haul,
   not a rate, so the cap was effectively no cap at all.
2. **Capping targets does not cap predictions.** Poisson-loss boosting outputs
   `exp(raw)`, which can exceed every observed target. I constrained the input
   and left the output unbounded.

**The honest ceiling is the SUSTAINED rate** -- aggregated over a whole
player-season, where a rate is actually a rate:

| | single-match q99 | sustained q99 | legacy's max prediction |
|---|---:|---:|---:|
| goals/90 | 2.000 | **0.793** | **0.781** |
| assists/90 | 1.429 | **0.653** | 0.289 |

Legacy's maximum prediction sits essentially on the sustained ceiling. That is
the behaviour to reproduce.

### G6 — smooth target + robust conversion + sustained-rate ceiling
*Declared 2026-10-20, before the run. Combines G4 (right target) with G3b
(bounded output), which were complementary, not alternatives.*

**What G4 alone got wrong.** Assists tail fell (1.0 -> 0.58-0.61) but **xP max
ROSE to 12.4-12.5** (target well below 11; legacy 5.04). Two causes:

1. **Conversion factors on near-zero denominators.** Measured per position:
   GK assists/xA **4.533**, FWD **2.423**. Predicting a smooth xA and then
   multiplying forwards by 2.4 re-injects the dispersion the smooth target
   removed. Those ratios are unreliable because xA is tiny for those positions.
2. **Changing the target does not bound the output.** Poisson-loss boosting
   extrapolates beyond anything it trained on -- goals/90 max 1.11 against a
   sustained ceiling of 0.793. This is the same mechanism that defeated G3.

**Three changes:**
1. targets stay xG/90 and xA/90 (G4, kept);
2. conversion factors **shrunk toward the pooled ratio** by expected-volume:
   `ratio_pos = (events_pos + K*pooled) / (expected_pos + K)` with **K = 50**
   expected-goal units, so a position with little xA inherits the pooled value;
3. predictions **clipped at the sustained ceiling** -- the 99th percentile of
   per-player-season rates, applied after conversion, per position.

**PASS requires, in order:**
1. **Mechanism:** goals/90 max <= ~0.85, assists/90 max <= ~0.50, **xP max
   below 8** (legacy 5.04, current 12.5).
2. **Component gate:** captain-added points improve in all three seasons
   (baseline 242 / 234 / 169).
3. **Then G1 unchanged.**

**Fixed in advance:** K = 50, ceiling quantile 0.99, aggregation per
player-season. No tuning to rescue it.

**RESULT — FAIL on mechanism, 2026-10-20.**

| gate | required | got |
|---|---|---|
| goals/90 max | <= 0.85 | **0.98 - 1.11** |
| assists/90 max | <= 0.50 | 0.48 - **0.53** |
| xP max | **< 8** | **9.58 - 12.52** |

The conversion shrinkage worked (GK xA factor 4.533 -> 1.781; assists tail
0.58/0.61/0.52 -> 0.48/0.53/0.52). **The ceiling did nothing** -- goals/90 max
was unchanged to three decimals (1.114->1.110, 1.027->1.029, 0.983->0.982).

Cause: the cap never fired. "Sustained" rates were aggregated per player-season
**without requiring meaningful minutes**, so cameo seasons inflated the 99th
percentile. MID assists ceiling 1.121 against predictions of 0.53; FWD goals
ceiling 1.134 against 1.03.

| ceiling | no minutes filter | >=900 min |
|---|---:|---:|
| MID assists/90 | 1.121 | **0.634** |
| FWD assists/90 | 0.917 | **0.479** |
| MID goals/90 | 0.796 | **0.642** |
| DEF goals/90 | 0.343 | **0.168** |

### G10 — the deliberately simple model (S10, S11)
*Declared 2026-10-21, before the run. Judged on the UNMODIFIED G1 gate.*

**Why:** the rebuilt model loses G1 by ~290/season and the diagnosis was top-end
OVER-prediction (at xp>=5 the rebuild runs 1.07-1.14, legacy 0.83-0.93). The
rebuild's tail-inflating machinery -- sustained-rate ceilings, shrunk
conversion, hierarchical minutes, 120 features -- was the suspect, and G3, G3b
and G6 were all prior attempts to tame that same tail. This is the control that
line of work never had.

**Build** (`fpl2/fpl2/simple.py`, no training step, 9 seconds): lagged rolling
mean for minutes; lagged xG90/xA90 with no ceiling and no conversion; opponent
strength as FOUR TIERS from prior-season goal difference with multiplicative
factors; clean sheet `exp(-lambda)`; `scoring.expected_points` to compose.

**Tier factors, fitted on prior seasons only** -- leakiness `att[k]` 1.222 ->
0.744 and danger `atk[k]` 0.736 -> 1.343 across tiers 1-4, both monotone. Note
`att` and `atk` are DIFFERENT quantities: how many goals a tier leaks says
nothing about how many it scores, and an early version conflated them by using
`1/att` for the concession term.

**RESULT — FAIL, and far more interesting than the verdict.**

| season | simple | legacy | margin |
|---|---:|---:|---:|
| 2023-24 | 2,109 | 2,256 | -147 |
| 2024-25 | 1,973 | 2,459 | -486 |
| 2025-26 | 1,794 | 2,072 | -278 |

0/3, mean **-304** against a required +148. But spearman is 0.692 / 0.696 /
0.721 against legacy's 0.684 / 0.705 / 0.720 and the 120-feature GBM's 0.668 /
0.696 / 0.711 -- **the no-training model ranks better than the GBM in two of
three seasons and ties legacy.** See S10.

**Decomposition (`run_decompose.py`):** sharing legacy's opening 15 moves the
margin -304 -> -156. A perfect-HINDSIGHT opening squad scores -151 and legacy's
real one -156, so legacy sits at the attainable bound. See S11.

An arm that rescaled our xp to legacy's spread returned IDENTICAL totals, and
that arm was **mis-designed, not a null**: every selection in the simulation is
a fixed-size optimisation (15 squad, 11 XI, 1 captain), and an affine transform
is invariant for those. Only `min_gain` sees scale, and the greedy rule's best
weekly gain is usually well clear of 0.30. Testing that hypothesis needs a
min_gain sweep, not a rescale.

**Shrinkage (`run_shrink_sweep.py`), k in 0..80:** opening-15 season points rise
1858 -> ~1950 against legacy's 2136 -- the mechanism predicted by the 2025-26
squad inspection (we put 14.5 into Salah for 123 points on an unregressed
2024-25 rate; legacy never buys the 14.5 asset and its opening 15 was worth
2184 to our 1683). Still 0/3 at every k, and the entire margin column spans 111
points against a 148 floor, so **k cannot be selected from it**. k=40/80 give
BETTER squads but WORSE seasons, since heavy shrinkage flattens in-season
discrimination for the remaining 33 gameweeks. k=10 kept on that reasoning,
recorded as mechanism-confirmed, magnitude-unmeasurable.

---

### G9 — hand-exported Championship priors (F16)
*Declared 2026-10-21, before each run.*

**Diagnosis:** G8's external priors reach the big-5 only, so a promoted club's
entire squad arrives with no evidence at all. In 2026-27 GW1-7 the new-to-PL
group is 107 players and 1218 points (19.0% of the season); Coventry, Hull and
Ipswich alone hold 466 of those points.

**Blocker and how it was resolved:** FBref sits behind Cloudflare bot
management. The chromedriver/Chrome version-mismatch diagnosis recorded on
2026-10-20 was WRONG -- driver 153.0.8010.52 matches Chrome exactly and renders
every other site. The page returns the challenge interstitial. No evasion
tooling was used; the four season tables were exported by hand from a normal
browser session into `data/external_manual/` and parsed by
`external.load_manual`.

**Measured in advance:** FBref publishes **no xG/npxG/xAG for the Championship
in any of the four seasons**. The prior is minutes, matches, goals, assists --
strictly weaker than the big-5 priors the model trains on (O9).

**Instrument 1 — unconstrained best-legal-XI points.**
Gate: improve in >=3 of 4 seasons; lose no more than 5 pts in any.
Result: +0.0 / +43 / -9 / +39 -> **2/4, FAIL.**
This gate was invalid as written -- every threshold sat below the noise floor
(M9) -- and the instrument has near-zero power besides: it picks from the whole
league with no budget, so Championship arrivals are never in the pool. 2023-24
moved by *exactly* 0.0 while 91 players gained priors.

**Instrument 2 — budget-constrained squad (100.0, 15 players, 1 FT no hits),
with a negative control.** Three arms: A baseline, B manual priors, C manual
priors SHUFFLED between the players who hold them (same players, same value
distribution, correspondence destroyed).
Gate: B beats A in >=3/4, AND B beats C in >=3/4, AND aggregate B-A > 2x C-A.

| season | A | B | C | B-A | C-A |
|---|---|---|---|---|---|
| 2023-24 | 1717 | 1717 | 1717 | +0 | +0 |
| 2024-25 | 2059 | 2086 | 2042 | +27 | -17 |
| 2025-26 | 1869 | 2004 | 1818 | **+135** | -51 |
| 2026-27 | 361 | 392 | **401** | +31 | **+40** |

B beats A **3/4, +193 aggregate** -- a clean pass against the baseline alone,
and what would have been reported without a control. B beats C only **2/4**,
and in the live season -- three promoted squads, where the effect should be
strongest -- the **shuffled control outscored the real priors**. Strip 2025-26
and B-A falls from +193 to +58 across three seasons, one an exact tie.

**VERDICT: FAIL.** Not from low power: low power explains a null, it does not
explain a shuffled control winning. See R26.

**What the run established anyway:** S9, and that the calibration gain is real
and lands where predicted (Championship arrivals 0.723 -> 0.776, +19% of their
gap closed) while the pooled figure moved only 0.847 -> 0.856 because 78 of 107
newcomers are unmatchable from a Championship table.

**Matching note:** surname+initial was replaced by **surname+club** for this
source. It had matched Hull's Nobel Mendy to Watford's *Nampalys* Mendy while
rejecting Hull's Mohamed Belloumi, who is FBref's *Bachir* Belloumi at the same
club. A surname appearing twice at the same club is dropped, not guessed.

---

### G8 — external priors for players arriving from other leagues
*Declared 2026-10-20, before the run.*

**Target:** new arrivals are **19.2% of points scored** and the model rates
them at **0.834** of their level, consistently under-rated in GW1-6 across
every season (0.977 / 0.972 / 0.929 / 0.834) and recovering only as
within-season data accrues. Their spearman is FINE (0.645, above established
players' 0.608) -- the ordering is right and the level is too low, so the
optimiser never buys them.

**Change:** per-90 priors from the player's previous league (Understat big-5):
`ext_xg90`, `ext_xa90`, `ext_shots90`, `ext_kp90`, `ext_min_per_match`,
`ext_minutes`, `ext_matches`, plus an `ext_available` flag. Two sources -- the
prior season for summer arrivals, and the same season in the other league for
**January arrivals**, whose foreign season is truncated at the transfer and so
is genuinely in the past.

**Known coverage limit, measured in advance:** soccerdata reaches the big-5
only. 47 of 188 new arrivals match (25% by headcount) but those 47 carry
**441 of 1,062 new-arrival points (41.5%)**. Championship, Eredivisie and
Primeira Liga arrivals -- Belloumi, Tzolakis, Ajayi, Rudoni -- stay invisible.
So the realistic ceiling for this layer is roughly **8% of the total points
pool**, not all of the 19.2%.

**PASS requires, in order:**
1. **Mechanism:** the NEW-to-PL GW1-6 ratio moves materially toward 1.0 from
   0.834, without degrading the established-player ratio (0.989).
2. **Component gate:** captain-added points improve in all three holdout
   seasons.
3. **Then G1 unchanged.**

**Stated in advance:** with ~42% points coverage, a full correction is not
achievable. A partial move is the expected outcome, and a partial move is still
a FAIL against the gate as written.

**RESULT — external priors WORK; mover flags REJECTED. Gate FAILS. 2026-10-20.**

New-to-PL ratio, GW1-6:

| season | before | after |
|---|---:|---:|
| 2023-24 | 0.977 | 0.980 (no priors in that fold) |
| 2024-25 | 0.972 | **1.000** |
| 2025-26 | 0.929 | **0.952** |
| 2026-27 | 0.834 | 0.847 |

Mover flags: spearman unchanged (0.454/0.465/0.622/0.525 ->
0.446/0.473/0.610/0.527); calibration worse in 2024-25 (1.075 -> 1.228).

The gate fails on 2026-27 (0.834 -> 0.847, not material). The cause is
**coverage, not the idea** -- 58% of that season's new-arrival points belong to
Championship arrivals soccerdata cannot reach.

**Kept:** external priors. **Dropped:** mover flags -- a flag that IDENTIFIES a
group is not a CORRECTION for its bias, and four features on ~60 players a
season is too thin for the model to learn the right sign from.

### G7 — coverage mask on uncollected expected-stats
*Declared 2026-10-20, before the run.*

**Defect, measured:** in 2022-23, **gameweeks 1-32 have 100% zero xG** among
players with >=60 minutes. The data was not collected. From GW33 the season
looks normal (44-51% zero, matching 2023-26). xA and xGC show the same break
(47% vs 21% zero; 36% vs 0%).

Those are missing values being read as "this player generated no threat". Any
model trained on them learns that a striker playing 90 minutes has zero xG.
This is the same defect the previous project found and fixed with
`V26_APPLY_COVERAGE_MASK`, which **passed its gate there** (ledger F2).

**Change:** for each (season, gameweek), if more than 95% of rows with >=60
minutes have xG == 0, set the expected-stat columns to NaN for that gameweek.
Detected from the data, not hardcoded to 2022-23, so it also catches any future
collection gap.

**Why this is not gate-shopping:** it corrects a known-bad INPUT with an
independently-validated remedy. G6b stays recorded as a FAIL. The gate
thresholds are unchanged.

**PASS requires:**
1. 2023-24 captain-added recovers toward its pre-xG baseline (242) rather than
   the degraded 169;
2. then G1 unchanged -- all three seasons, mean margin > +148.

**Stated in advance:** this cannot fix 2023-24 entirely. That fold trains only
on 2022-23, and masking removes 32 of its 38 gameweeks of expected-stats, so it
will train on very little. A partial recovery is the realistic outcome.

### G6b — same, with a 900-minute floor on the ceiling population
*Declared 2026-10-20, before the run.* Identical to G6 except a player-season
must contain **>=900 minutes** (ten full matches) to count toward the ceiling.
Same mechanism thresholds, same component gate, same G1. No other change.


**Known, stated in advance:** the 2023-24 fold trains only on corrupted
2022-23 xG (goals/xG 1.446, xGC mean 0.301, xGC scale 1.495 -- three
independent symptoms). It may fail for that reason. That is NOT grounds for
excluding the fold; the remedy is a coverage mask on the input, which is a
separate change with its own gate.

### G4 — predict xG/xA instead of goals/assists
*Declared 2026-10-20, before the run. Supersedes G3/G3b, which treated the
symptom (a long tail) rather than the cause (a spiky target).*

**Why.** The model predicts goals and assists directly. Those are low-count
spiky events; xG and xA are the smooth underlying quantities. Measured on
players with >=60 minutes:

| | sd/mean | sustained q99 | max |
|---|---:|---:|---:|
| goals/90 | 3.11 | 0.793 | 2.857 |
| **xG/90** | **2.09** | **0.639** | **1.070** |
| assists/90 | 3.18 | 0.653 | 2.250 |
| **xA/90** | **1.80** | **0.315** | **0.507** |

xG/xA are both **less dispersed** AND **more predictive** of future returns:

| trailing 5gw -> next gameweek | corr |
|---|---:|
| goals -> goals | 0.2405 |
| **xG -> goals** | **0.2625** |
| assists -> assists | 0.1224 |
| **xA -> assists** | **0.1628** |

And it explains the legacy model's well-behaved tail exactly: legacy predicts
`predicted_xg_per90` / `predicted_xa_per90`, and its maximum xA prediction
(0.289) sits on the sustained xA ceiling (0.315). Predicting assists directly
gives a ceiling of 0.653 -- the 2x gap in our maxima.

**Change:** targets become xG/90 and xA/90; predictions are converted back with
per-position factors fitted on the training seasons.

Conversion factors measured: goals/xG **1.00**, assists/xA **1.39**, both
stable across 2023-24/24-25/25-26. (2022-23 reads 1.446 / 2.115 -- an
independent confirmation of the known fabricated-xG problem in that season,
ledger F2.)

**PASS requires, in order:**
1. **Mechanism:** predicted assists/90 max falls to roughly the sustained xA
   ceiling scaled by the conversion factor (~0.44), and xP max drops materially
   from ~11.
2. **Component gate:** captain-added points improve in **all three** seasons
   (baseline 242 / 234 / 169).
3. **Then G1 unchanged.**

**Stated in advance:** the 2023-24 fold trains only on 2022-23, whose
conversion factor is corrupted. That fold may suffer for a data reason. It is
NOT grounds for excluding it after the fact.

### G3b — cap PREDICTIONS at the sustained-rate ceiling
*Declared 2026-10-20, before the run.*

**Change:** compute per-position ceilings as the 99th percentile of per-
player-season aggregated per-90 rates (training seasons only), and clip model
PREDICTIONS at them. Targets are left alone.

**PASS requires, in order:**
1. **Mechanism check:** predicted `assists_per90` max must fall below the
   sustained ceiling, and xP max must drop materially from ~11.
2. **Component gate:** captain-added points improve in **all three** seasons
   (baseline 242 / 234 / 169).
3. **Then G1 unchanged.**

**Declared in advance:** quantile 0.99, aggregation per player-season. No
tuning to rescue it.


### G2 — per-player reliability correction
*Declared 2026-10-20, before the run.*

**Claim:** scaling each player's xP by a shrunk ratio of his realised to
predicted points -- estimated from honest out-of-sample history -- improves the
forecast where it is specifically wrong, without flattening everyone (which
`shrink.py` did, and was rejected for).

**PASS requires, in order:**
1. **Component gate first.** Mean realised points of the top-20 must improve
   in **all three** holdout seasons versus the unadjusted signal. This is the
   metric the optimiser consumes (R16). If it fails here it does not reach G1.
2. **Then G1 unchanged:** beat legacy in all three seasons with a mean margin
   above +148.

**Declared in advance:** `alpha` (the shrink-toward-1 constant) and the clip
range are fixed at 20.0 and [0.60, 1.40] before seeing any result. If the idea
only works after tuning these, that is a FAIL, not a win.

**RESULT — FAIL on the component gate, 2026-10-20. Did not reach G1.**

| season | top-20 base | top-20 adjusted | delta |
|---|---:|---:|---:|
| 2023-24 | 4.229 | 4.100 | **-0.129** |
| 2024-25 | 4.630 | 4.600 | **-0.030** |
| 2025-26 | 4.312 | 4.280 | **-0.032** |

0/3. `alpha` was NOT tuned to rescue it.

A structural worry was stated in advance: 2023-24 has no honest prior season,
so it could have failed for that reason alone. **It did not.** Both seasons
that DID have honest out-of-sample prior residuals (2024-25, 2025-26) also got
worse. The idea is rejected on its merits, not on a technicality.

**WHY it failed — per-player over-prediction does not persist.** 528 players
with >=10 appearances in both 2024-25 and 2025-26, honest out-of-sample ratios
(actual/predicted) in each:

correlation across seasons: pearson **0.132**, spearman 0.135 (p=0.002)

| previous-season quartile | next-season ratio |
|---|---:|
| Q1 most OVER-predicted | **0.9421** |
| Q2 | 0.9504 |
| Q3 | 0.9446 |
| Q4 most UNDER-predicted | 1.0152 |

Significant but worth **1.7% of variance**, and **entirely asymmetric**: Q1-Q3
are flat to three decimals. All the signal is in the under-predicted tail.

**Over-prediction carries no information about next season.** A player the
model over-rated is no more likely to be over-rated again than an average
player -- which is precisely the correction the Salah case required. This
closes the whole family of approaches built on past per-player residuals, not
just this implementation.

### G1 — the rebuilt component model vs the legacy signal
*Declared 2026-10-20, before any comparison was run.*

**Claim to be tested:** the rebuilt component model (`fpl2`) produces a better
decision signal than the old pipeline's cached `expected_points`.

**Decision metric:** season total from `fpl2.simulate.simulate_season` under
real constraints (GBP 100m, 2/5/5/3, max 3 per club, one free transfer, no
hits), decisions on the signal, points on realised.

**PASS requires BOTH:**
1. the new signal beats legacy in **all three** holdout seasons, and
2. the **mean** margin exceeds **+148 points**, the measured noise floor
   (`fpl2/run_noise_floor.py`, 3 sd of the cross-season mean at 1% jitter).

**Anything else is a FAIL**, including a win in two seasons, and including a
three-season win whose mean margin is under 148. A fail is recorded here and
the legacy signal stays.

**Known complication, stated in advance:** the 2023-24 holdout trains on a
single prior season and its rate calibration is poor (goals ratio 0.433 vs
1.009 for 2025-26). This is a property of the data, not the model, and it
applies to the legacy signal equally — so it is NOT grounds for excluding that
fold after the fact.

**Secondary metrics, reported but NOT gated:** spearman, calibration ratio,
per-position bias.

**RESULT — FAIL, 2026-10-20.** Run with the gate exactly as declared.

| season | new | legacy | margin |
|---|---:|---:|---:|
| 2023-24 | 1,944 | 2,226 | **-282** |
| 2024-25 | 1,775 | 2,359 | **-584** |
| 2025-26 | 1,849 | 2,061 | **-212** |

0/3 seasons won, mean margin **-359** against a required **+148**. The loss is
roughly 2.4x the noise floor, so it is real rather than a reshuffle. **The
legacy signal stays in place. The gate was not adjusted.**

Diagnosis so far:
- the composed xP **under-predicts by ~10%** overall (ratios 0.880 / 0.899 / 0.905)
- almost all of it is **defenders**: 2025-26 per-position ratio DEF **0.757** vs
  FWD 0.972, GK 1.078, MID 0.973
- cause is a **data gap, not a model choice**: `tackles`, `recoveries` and
  `clearances_blocks_interceptions` are NaN in every season before 2025-26, so
  the defensive-contribution bonus (worth 1,443 x 2 = 2,886 points in 2025-26)
  cannot be learned from history. The formula itself is confirmed exactly --
  DEF = CBIT, MID/FWD = CBIRT, 100% match where the data exists.
- ranking is comparable (spearman 0.674 / 0.661 / 0.709 vs legacy 0.701), so the
  gap is driven by **calibration distorting squad selection**, not by worse
  ordering

The data gap applies to the legacy signal equally, so it does not excuse the
result -- but it does mean the DEF correction is the first thing to try.

**RE-RUN after fixing the defender gap — still FAIL, 2026-10-20.**

The defensive-contribution component was moved to within-season fitting
(expanding window, week k predicted from weeks < k). Calibration improved a
lot: overall ratio 0.905 -> **1.010**, DEF 0.757 -> **0.969**, all four
positions now within 8% of true. The gate was NOT changed.

| season | new | legacy | margin |
|---|---:|---:|---:|
| 2023-24 | 1,944 | 2,226 | -282 |
| 2024-25 | 1,775 | 2,359 | -584 |
| 2025-26 | 1,887 | 2,061 | **-174** (was -212) |

Mean -347. **Calibration was not the binding problem.**

**The real cause — top-end ranking.** Overall spearman is a dead heat
(0.704 vs 0.701) because 29,000 rows are dominated by players who score near
zero. Selection only cares about the top, and there the gap is large:

| top-k | new | legacy | gap |
|---:|---:|---:|---:|
| 5 | 4.663 | 5.095 | -0.432 |
| 20 | 4.113 | 4.696 | **-0.583** |
| 50 | 3.951 | 4.259 | -0.308 |
| 200 | 3.139 | 3.227 | -0.089 |

20 of the top 50 differ each gameweek. Players picked **only by the new model**
were predicted 3.99 and scored **3.37**; players picked **only by legacy** were
predicted 3.65 and scored **4.14**.

Composition of each top-50: new **DEF 8.2 / MID 29.6 / GK 5.0**, legacy
**DEF 15.1 / MID 24.6 / GK 2.8**.

Two named causes, both in hastily-built components:
1. **Saves are over-predicted** (ratio 1.134-1.187), inflating goalkeepers --
   five of them reach the new model's top 50 against legacy's 2.8.
2. **The clean-sheet model barely discriminates** (AUC 0.593-0.608), so good
   defenders never separate from bad ones and the top 50 under-weights them.

**Lesson (R16 candidate):** overall rank correlation is nearly useless as a
model-selection metric here. Judge the top-k realised points instead -- it is
what the optimiser consumes.

**THIRD RUN — fixture awareness added. Still FAIL, 2026-10-20.**

Found that the player feature set had **zero opponent features and `is_home`
set to NaN** -- the model was fixture-blind, while the docstring claimed
otherwise (the same sin as `horizon_sweep_clean.py`). Added 11 fixture-aware
features; leakage test extended to cover the opponent join and passes.

Components improved (saves ratio 1.187 -> 1.131, top-20 realised up) but the
gate barely moved: mean **-344** vs -347. **Forecast quality was not the
binding constraint either.**

**LOCALISED — half the gap is the OPENING SQUAD.** Running both signals from an
identical opening 15 so only weekly decisions differ:

| season | own opening squad | shared opening squad | recovered |
|---|---:|---:|---:|
| 2023-24 | -251 | -131 | +120 |
| 2024-25 | -556 | -400 | +156 |
| 2025-26 | -226 | **-42** | **+184** |

With a shared squad, 2025-26 is **inside the noise floor** -- a tie on the
season with the most training data.

**ROOT CAUSE — cold-start over-confidence.** The opening squad is chosen on
gameweeks 1-5, when within-season evidence is absent and every feature is
carried from the previous season. The new signal is far more confident than
legacy (mean 4.056 vs 3.394) and wrong:

| bought only by NEW | cost | xP | season pts |
|---|---:|---:|---:|
| Salah | 14.5m | **6.33** | **123** |
| Alisson | 5.5m | 4.17 | 91 |
| Kilman | 4.5m | 2.90 | 36 |

| bought only by LEGACY | cost | xP | season pts |
|---|---:|---:|---:|
| B.Fernandes | 9.0m | 3.71 | **235** |
| Semenyo | 7.0m | 3.39 | **202** |
| Raya | 5.5m | 3.26 | **162** |

Legacy rates nobody above 3.71. Its flatness at the top is correct when there
is no current-season evidence; the new model extrapolates career rates for a
declining premium and spends 14.5m on it. Squad quality: 1,900 season points
vs legacy's 2,184.

**Shrinkage attempted — REJECTED, 2026-10-20 (ledger F15).**

Implemented `xp = w*model + (1-w)*prior(position, price)` with `w = n/(n+k)`,
n = appearances this season. At gameweek 1 n=0, so the forecast is entirely the
prior -- the intended fix for cold-start over-confidence.

First run used k=3 because the selection routine silently fell through to its
default (it splits by season but receives a single-season frame). Swept k
properly afterwards on cached out-of-sample forecasts:

| k | 2023-24 | 2024-25 | 2025-26 | mean | wins |
|---:|---:|---:|---:|---:|---:|
| 0 (none) | 4.229 | 4.630 | 4.312 | **4.390** | - |
| 3 | 4.193 | 4.713 | 4.120 | 4.342 | 1/3 |
| 5 | 4.139 | 4.736 | 4.143 | 4.339 | 1/3 |
| 15 | 4.097 | 4.684 | 3.991 | 4.257 | 1/3 |

**No k wins all three seasons; the mean is worse at every k than at k=0.**
Spearman fell ~0.68 -> ~0.59 and calibration flipped to over-prediction. The
gate also moved only -344 -> -327, inside the noise floor. Not kept --
flattening everyone costs more than the over-confidence it removes.

The underlying diagnosis stands and is still unfixed: a per-player reliability
estimate or an explicit decline/role model would target it without flattening.

**FOURTH RUN — cold-start appearance bug fixed. Still FAIL, 2026-10-20.**

Found that `season_appearances_so_far` (added for the rejected shrinkage
experiment, never reverted) was poisoning the minutes model: it is 0 for
everyone at gameweek 1 but means "never selected" mid-season, and `gameweek`
was not a feature so the model could not tell them apart. See R18, R19.

Adding a season-stage feature fixed it cleanly:

| gw | p_appear before | after | actual |
|---:|---:|---:|---:|
| 1 | 0.353 | **0.439** | **0.433** |
| 5 | 0.421 | 0.427 | 0.412 |
| 20 | 0.378 | 0.380 | 0.377 |

Season ratio 0.991, spearman 0.7074 (best of any version), GW1 p95 now 3.31
against legacy's 3.36. **The gate moved -344 -> -336.**

**WHY component fixes keep not moving the gate — the losses are in three
different decision layers, and a different one dominates each season:**

| season | XI+captain only (frozen squad) | + transfers | opening squad |
|---|---:|---:|---|
| 2023-24 | **+8** (new WINS) | -1 | -387 |
| 2024-25 | -150 | **-381** | -6 |
| 2025-26 | -115 | -114 | -118 |

- 2023-24: XI/captain selection **beats legacy**; the whole gap is the opening squad.
- 2024-25: transfers cost a further **-231** beyond XI selection; opening squad is fine.
- 2025-26: split roughly evenly between opening squad and XI selection.

Three component fixes (defenders +38, fixture awareness +3, cold start +8)
total ~50 points against a 495-point requirement, because **forecast quality
was never the binding constraint in any season.** Improving xP cannot fix a
loss that lives in how the signal is consumed.

## Transfer-rule investigation, 2026-10-20

**The greedy one-gameweek transfer rule is not reliably value-adding.** Same
opening squad, frozen (no transfers) vs the standard policy:

| season | signal | frozen | with transfers | delta |
|---|---|---:|---:|---:|
| 2023-24 | new | 2125 | 2225 | +100 |
| 2023-24 | legacy | 2117 | 2226 | +109 |
| 2024-25 | new | 2026 | 1978 | **-48** |
| 2024-25 | legacy | 2176 | 2359 | **+183** |
| 2025-26 | new | 1997 | 1947 | **-50** |
| 2025-26 | legacy | 2112 | 2061 | **-51** |

**In 2025-26 the rule destroys value for BOTH signals (-50, -51).** This is a
property of the policy, not of either forecast. The live pipeline runs the same
policy (`greedy_1gw_no_hits`), so this is worth testing against production, not
only against the rebuild. Flagged as O8.

**Transfer quality, realised over the next 6 gameweeks per swap:**

| season | new realised | legacy realised | new predicted gain | legacy predicted |
|---|---:|---:|---:|---:|
| 2023-24 | **+269** | +135 | 3.52 | 2.57 |
| 2024-25 | +109 | **+238** | 3.40 | 2.10 |
| 2025-26 | +5 | **+124** | 3.20 | 2.40 |

The new signal predicts **~40% larger gains every season** while realising less
in two of three: over-confident about the DIFFERENCE between players even
though the levels are calibrated (season ratio 0.991).

**The base layer is within-squad ranking.** Frozen, same 15, XI and captain
only -- and the XI optimiser is provably optimal given the xP -- legacy still
wins by **115-150 points** in two seasons. Global top-k ranking is a near-tie
(R16 measured this), but discriminating among fifteen players who are all good
enough to own is a finer problem, and that is where the signal is weaker.

**AND WITHIN THAT, ~85% IS THE CAPTAIN.** Frozen squad, XI and captain
decomposed separately:

| season | signal | XI pts (ex-captain) | captain adds | capture of ceiling |
|---|---|---:|---:|---:|
| 2023-24 | new | 1883 | **242** | 84.3% |
| 2023-24 | legacy | 1886 | **231** | 83.9% |
| 2024-25 | new | 1792 | **234** | 81.3% |
| 2024-25 | legacy | 1815 | **361** | 87.3% |
| 2025-26 | new | 1828 | **169** | 79.9% |
| 2025-26 | legacy | 1858 | **254** | 84.5% |

XI points excluding the captain are near-identical (1883/1886, 1792/1815,
1828/1858 -- differences of 3, 23, 30). The captain accounts for **-127 of the
-150** gap in 2024-25 and **-85 of -115** in 2025-26.

Both signals use the same rule, `argmax(xP)` within the XI. So the defect is
that the new signal's **argmax lands on worse players** -- the captain pick
samples the extreme upper tail, which is exactly where the new distribution is
too long (GW1 max 6.60 vs legacy 4.59; predicted transfer gains ~40% too
large). One underlying defect -- poor discrimination among similar players at
the top -- surfacing in both captaincy and transfers.

This also makes the old project's captaincy findings legible: there, ~half the
model's entire margin over dumb baselines came from captaincy (S8). Captaincy
is where a signal's top-end quality is cashed in, at double weight.

## Model-family sweep — RUN 2026-10-21 (design below was followed as written)

**Outcome: the design's own prior was WRONG, and its mandated control is why we
know.** The prior said "expect model family to be second-order... if a family
appears to win big, suspect the harness first." Family turned out to be
first-order. What saved the result from being dismissed on that prior was the
instruction to run the Ridge/GLM control FIRST -- which is exactly where the
effect was.

| family | top20 3/3 | captain 3/3 | mean vs legacy |
|---|---|---|---|
| baseline HistGB(poisson) | -- | -- | -267 |
| ridge (squared loss, linear) | 2/3 | 3/3 | -217 |
| xgboost | 2/3 | 3/3 | -194 |
| quantile q60 | 2/3 | 3/3 | -231 |
| **glm_poisson (alpha=1.0)** | **3/3** | **3/3** | **-57** |

Every alternative beat the baseline on captain-added 3/3 -- consistent with the
top-end over-prediction diagnosed in S11/G10, since the captain is `argmax(xp)`
and lives entirely at the top of the forecast.

**The first run FAILED its gate on a technicality of my own making**: the
third criterion was implemented as "beat LEGACY by 148" when the design's
wording ("the margin must exceed the noise floor on G1") could equally mean
"beat the BASELINE by 148", which +210 clears. **That ambiguity was NOT
resolved in the direction that passes** -- M3 is what happens when gates are
reinterpreted after seeing results. A separate replication with its own
pre-declared gate was run instead, and passed all four criteria.

**CAVEATS, which bound the claim:**
- **G1 still FAILS.** -57 is a deficit, not a win. The GLM loses to legacy on
  the season total in 2 of 3 seasons.
- **2024-25 is wildly unstable**: +94 at alpha=1.0, -18 at alpha=0.1, -445 at
  alpha=10 -- a 539-point swing in one season across one parameter.
- **One component swapped.** Minutes, team strength and the scoring composition
  are untouched; the sweep's own design calls for testing those separately.
- **Three seasons.** Four families were tried on one component; the design
  budgeted ~1.9 spurious 3/3 winners across the full 15-comparison sweep.

**The explanation above was WRONG and is retracted — see W13.** It read: "a
GBDT overfits the top of the distribution... a Poisson GLM is smooth and cannot
blow up the tail." O18 measured it directly and the GLM over-predicts the top
MORE than the GBDT (mean |1-ratio| 0.095 vs 0.070, closer to 1.0 in only 2 of 9
top-end cells). The win is real and replicated; the stated cause is refuted.

What the data supports instead is **ordering, not calibration**: top-20 realised
points 94.54 vs 90.08 and captain-added 7.54 vs 5.41. The GLM picks better
players at the top while being MORE wrong about their absolute scores. Those are
independent properties and conflating them is what produced the bad explanation.

### Original design (2026-10-20), preserved as written

## Planned: model-family sweep (designed 2026-10-20, NOT yet run)

Worth doing, but it is the single easiest place in this project to manufacture
a false positive, so the design is fixed in advance.

**Prior, stated honestly before running:** today's evidence says the TARGET
matters far more than the estimator. Switching goals -> xG and assists -> xA
changed the upper tail and the captain pick; three separate post-hoc
corrections to the same signal changed nothing. Expect model family to be
second-order. If a family appears to win big, suspect the harness first.

**Design**
- Same 96 features, same strict walk-forward, same three holdout seasons.
- Swap ONE component at a time, not the whole stack -- otherwise a win cannot
  be attributed. The components are: minutes hierarchy (4 models), rates
  (5 per position), team xGC.
- Judge on **top-20 realised points** and the **captain-added** decomposition
  (R16: overall rank correlation is near-useless here), then G1.

**Candidates**
| family | why it might help |
|---|---|
| XGBoost | already installed; different regularisation and split policy |
| LightGBM | leaf-wise growth, often better on wide tabular data |
| CatBoost | ordered boosting reduces target leakage on small data |
| Ridge / GLM on the same features | **the control that matters** -- if a linear model matches, the GBDT is not earning its complexity |
| Quantile regression | directly targets the upper tail, which is where the captain is decided |

**Multiple-testing budget, fixed in advance.** 5 families x 3 components = 15
comparisons. At the measured 0.125 rate of a null signal winning 3/3, that is
**~1.9 spurious winners expected by chance** (R3). So:
- a family must win on BOTH top-20 and captain-added, in all three seasons;
- and the margin must exceed the noise floor (~148 season points) on G1;
- a win on one metric only is recorded as NOT a result.

**Run the Ridge/GLM control FIRST.** If a linear model on these features is
within noise of the GBDT, that is the most useful finding available and it
changes what the rest of the sweep means.

## CORRECTION — the tail gap was mostly an artifact, 2026-10-20

Four gates (G3, G3b, G6, G6b) were aimed at closing an xP tail gap of
"11.02 vs legacy 5.04". **That comparison was not like-for-like.**

The fold cache stores **one row per player-FIXTURE**. The rebuild's panel
stores one row per player-GAMEWEEK and **sums double gameweeks**. Every one of
the top-6 xP predictions was a double (fx=2, ~171 expected minutes) -- Salah at
11.84 is two matches, which is arithmetically correct.

On a like-for-like per-fixture basis:

| | q99 | max |
|---|---:|---:|
| legacy | 3.94 | 6.23 |
| new, single-fixture rows | **5.04** | **7.44** |
| new, double-fixture rows | 10.01 | 11.84 |

The real gap is **+28% at q99**, not +120%. A tail difference exists and is
worth something, but it is far smaller than the four gates assumed, and the
mechanism thresholds those gates were judged against ("xP max below 8") were
meaningless for a panel that sums doubles.

**This is R20 applied to my own success criterion.** Three times the rule caught
a threshold estimated on the wrong population in the MODEL; this is the fourth,
in the GATE. State the unit of the quantity being compared before comparing it.

**What survives from those four gates** (each measured, each independently
defensible regardless of the tail):
- xG/xA targets: smoother and more predictive than goals/assists (G4)
- xGC team target: clean-sheet AUC +0.017 / +0.027 on clean folds (G5)
- conversion factors shrunk toward pooled: GK xA factor 4.533 -> 1.781
- ceilings from sustained rates with a 900-minute floor (R20)

**What this means for the captain diagnosis.** The captain gap is real
(242/234/169 vs legacy 231/361/254) but it was attributed to a tail that is
mostly an artifact. The attribution is now unsupported and must be re-tested
directly.

## NEW-PLAYER COLD START — confirmed, 2026-10-20

The rebuilt model under-rates players arriving from other leagues, and they are
**19.2% of all points scored** in 2026-27 to date.

| | players | model xP | actual | ratio |
|---|---:|---:|---:|---:|
| established | 362 | 1.880 | 1.919 | 0.980 |
| **NEW to PL** | **188** | **0.866** | **1.038** | **0.834** |

Individually: Vuskovic 40 actual vs 14.0 predicted; Belloumi 37 vs 16.6;
Tzolakis 36 vs 16.3 -- roughly 40% of what they scored.

Their **spearman is 0.645, HIGHER than established players (0.608)**. The model
ranks new arrivals correctly against each other and puts the whole group at too
low a level, so the optimiser never buys them.

**A first analysis rejected this as uncorrectable** because the whole-season
ratios flip sign (1.076 / 0.763 / 1.081 / 0.834). **That comparison was
confounded** -- the holdouts cover 38 gameweeks, by which time a "new" player
has 19 gameweeks of history and is not new. Restricted to the comparable
window the bias is consistent:

| season | GW1-6 | GW7-19 | GW20-38 |
|---|---:|---:|---:|
| 2023-24 | **0.977** | 0.954 | 1.170 |
| 2024-25 | **0.972** | 1.031 | 0.649 |
| 2025-26 | **0.929** | 1.050 | 1.140 |
| 2026-27 | **0.834** | - | - |

All four under-predict early and recover as within-season data accrues.
Established-player control over the same windows: 0.762 / 1.053 / 0.989 /
0.989 -- so it is specific to new arrivals, not generic cold start.

**This is R20 a fourth time** (a conclusion drawn from a badly-matched
population), and it was caught by the user, not by me.

**Implication:** the fix is information, not calibration. A fixed boost cannot
know whether a given transfer window delivered good signings. External priors
from the player's previous league can. This is what the legacy pipeline's
`external_xg_per90`, `external_minutes`, `external_role_score` and
`external_prior_reliability` columns exist for, and what its FBref/Understat
calls feed.

# Part 5 — Open questions

| # | Question | Why it matters | Status |
|---|---|---|---|
| O29 | **Is the minutes fix's +76/season real, or magnitude-driven noise?** | O28 conditioned on decisions instead of seasons: captain-changed gameweeks gain **+5.09** (n=44, t=2.15), XI-only changes **+0.27** (t=0.29, nothing), control **exactly 0.00** across 15 gameweeks. Overall +1.99/gw ~ **+76/season**, t=1.99. But both t-statistics sit at the 2.0 boundary and the sign test is weak -- **25W/18L, p~0.19**. The effect is carried by a few large wins rather than consistent small ones, which is the profile that fails to replicate. Needs 2026-27 to finish, or a bootstrap over gameweeks. | **Open. Probably real, not established.** |
| O28 | **Re-test F20 with more seasons.** | **DONE 2026-10-21 via a decision-conditioned gate (`run_o28.py`) -- see O29 and M15.** F20 failed its declared 3/3 gate at 2/3, and that stands. But S21 shows the 2/3 is not a coin-flip 2/3: the winning season improved SYSTEMATICALLY (30 captain changes, +219) while the losing season was INERT (2 changes, -38 of pure variance). A gate counting seasons cannot distinguish "the fix acted and worked" from "the fix did nothing and the noise went the wrong way". With only 3 holdout seasons and one of them inert, the test is underpowered by construction. Needs more seasons, or a gate conditioned on how much the fix actually changes decisions. | **Open. The fix is more promising than its 2/3 suggests, and that is not grounds to relax the gate.** |
| O27 | **Why does 2025-26 gain +219 from the minutes fix and 2024-25 lose 33?** | **RESOLVED 2026-10-21 -- see S21. The premise was backwards.** F20's whole mean rests on one season. 2025-26 is also the season where S13 found no dominant player (Haaland 239, 4 clear of second) while 2024-25 had Salah 344, 108 clear. A fix that corrects minutes for ever-present starters should matter MORE where one such player dominates, not less -- so the season-split runs against the mechanism. Either the mechanism is wrong or something else drives 2025-26. | **Open, and it decides whether F20 is real.** |
| O26 | **Fix the minutes over-shrinkage?** | **ATTEMPTED 2026-10-21 -- see F20. Failed its gate 2/3, shipped OFF.** S19/S20 give a mechanism and a verified case. The obvious move is to weight current-season start evidence above career history when the two disagree. **The prior is poor and should be stated:** calibration fixes failed three times on 2026-10-21 -- W13 (top-end calibration does not order performance), F19 (clean-sheet shrinkage improved calibration 0.226->0.173 and degraded the decision metric -57->-110), and R9 predates both. A minutes fix is another calibration fix. Gate it on the DECISION metric from the start, not on predicted/actual ratios. | **Open. Mechanism is solid; expect the fix to fail anyway.** |
| O24 | **LIVE 2026-27 through GW7: model 355, user 337, average 416.** | The model beats the user (+18) and loses to the average (-61). **The entire deficit is GW6-7**: level through GW5 (293 vs 299), then 62 vs 117. Diagnosis (`fpl2/diag_gw67.py`): the minutes model is FINE (1 blank per week); players play and do not return (1-2 of 11 reach 5 pts); and **Bruno Fernandes was captained three weeks running for 1, 1, 2 points**. The model's own ranking puts Bruno (11.8m, xp 5.89, **33 pts**) above Haaland (15.7m, xp 5.48, **58 pts**). **The captain RULE is fine (F18); the xP feeding it is not.** Those were conflated when O12 was closed. | **The live gap, now localised.** |
| O25 | **Do we under-predict CLEAN SHEETS?** | **ANSWERED 2026-10-21: no -- we OVER-predict the best defences (S18), and FIXING IT MAKES THINGS WORSE (F19).** Original framing below, kept because the 34% figure still motivates the work. Clean sheets are **34%** of the top defenders' points in 2026-27 (120 of 358) -- their single largest source, and more predictable than the rest. The cheap defenders the model does NOT own (De Cuyper 5.3m/59pts, Bogle 4.7m/45, Calafiori 6.0m/44) have 4-5 clean sheets in 7 games. **Caveat that bounds this:** those same players over-performed xA by ~3x (4.13 xA -> 12 assists), so a large part of their scoring is variance the model is RIGHT not to predict. Test the clean-sheet component on its own, not their point totals. | **Open. The one part of the live gap that is not variance.** |
| O31 | **Why does 2024-25 lose under EVERY chip policy?** | Calendar -97, secretary -156, and it has the smallest hindsight ceiling (+125 vs +348 and +109). It is also the model's best no-chip season (2420) and the one with Salah on 344 (S13). Hypothesis, UNTESTED: when a squad is already working, rebuilding chips destroy more than they add, and the wildcard benefit estimate -- best-available-15 minus my-15 over the horizon -- is not capturing that. If true, the rule is "do not wildcard a squad that is outperforming", which no policy here encodes. | **Open. The same season has now broken four separate tests (S13, F20, F21, this).** |
| O34 | **Ship D (assignment) or A (calendar)?** | A has the better mean (+69 vs +52) and D the better consistency (3/3 vs 2/3), and the difference between them is inside the noise floor. R2 says hold in every season, which favours D; the mean favours A. Both are fixture-only and cheap. A tie-break needs either more seasons or a variance-aware criterion, neither of which exists yet. | **Open. A real choice, not a measurement problem.** |
| O35 | **Why do forecast-driven chip policies keep losing to fixture-only ones?** | Four attempts (calendar-beating): secretary-stranding, secretary-forcing, multiweek projection, learned-gain. The one leaning hardest on xP (B, +24) placed last; the one ignoring it entirely (D, 3/3) is the most robust. Hypothesis, UNTESTED: a chip's value depends on the SHAPE of a gameweek (how many of my players have two fixtures, how many have none), which the fixture list states exactly and xP only blurs. If true, chip timing is a combinatorial problem over the calendar and the forecast is a distraction. | **Open, and it would redirect all future chip work.** |
| O33 | **Chip timing is unsolved: 3 policies, all stuck at ~24% of the ceiling.** | calendar **+47**, secretary-stranding +48, secretary-forcing +32, against a +194 hindsight ceiling. The crudest policy (fixture list only) is the best of the three. The per-chip benefit estimates -- captain xP, bench xP, best-available-XI gap, horizon squad gap -- evidently do not rank gameweeks well enough to beat a heuristic that ignores the forecast entirely. Next ideas, none tried: estimate benefit over a multi-week window rather than the current week; learn the benefit function from realised chip outcomes; or treat it as a constrained assignment (8 chips to 38 gameweeks) solved once per season on the fixture list. | **~145 pts/season sitting unclaimed. The largest known gap in the project.** |
| O32 | **Fix the chip-starvation bug and re-measure.** | **DONE 2026-10-21 -- see F22. The fix works and makes scoring WORSE.** | F21's policy fires at most one chip per gameweek in a fixed priority order, so chips later in that order can reach their deadline blocked and go unplayed -- 2 of 8 in 2024-25. Deferral is the expensive failure (S4, S23), so this is pure self-inflicted loss and the F21 vs calendar comparison is not clean until it is fixed. | **Cheap. Do before any further chip work.** |
| O36 | **Capture contemporaneous ownership in `live.py`.** | The historical CSV carries per-gameweek `selected` at 100% coverage; the live panel carries none, so the live season's ownership calibration cannot be measured without using a post-hoc snapshot, which is what produced M19. `selected_by_percent` is on every element of bootstrap-static and costs nothing to store each gameweek. Until then any live ownership claim is unmeasurable. | **CLOSED 2026-10-22.** `live.ownership()` added. The entry above assumed we would have to START storing `selected_by_percent` each gameweek and wait; that was wrong. `element-summary/{id}/history` carries a per-gameweek snapshot of `selected`, `value`, `transfers_in` and `transfers_out` going back to GW1, so the whole season was recoverable retroactively -- 668 players, 4,551 rows, 11 seconds. `value` also supplies the historical price the module docstring says FPL "does not expose in bulk", which retires the standing price caveat in `live.py`. |
| O30 | **Model the parts of the real game the simulator omits.** | S22 prices the gap at **~287 points/season**, larger than any modelling gain in this project. Genuinely missing: **chips** (8 per season -- 2 each of wildcard, free hit, bench boost, triple captain, split GW1-19 / GW20-38, confirmed from bootstrap-static; S4 prices deferral at ~92/season), **hits** (-4 for an extra transfer, currently forbidden outright), and **within-season price changes** (the simulator uses each player's first price all year, so no value is ever gained or lost). **NOT missing: across-season team value** -- FPL resets everyone to 100.0m each season, and an earlier version of this entry wrongly claimed otherwise. Also note part of the 287 may be proxy error rather than missing features: an ownership-signal squad built by the optimiser is not the template people actually own. | **Largest single opportunity; chips now being measured.** |
| O1 | **Why is the live season underperforming by ~10.7 pts/GW?** | -- | **LARGELY ANSWERED 2026-10-21 by S22.** The premise was wrong: the model is level with the average, not 10.7/gw behind it. Live 2026-27 through GW5 ran **59.2/gw against the average's 59.4** -- parity. GW6-7 (30 and 32 against 58 and 59) is variance on top of parity. The apparent season-long deficit was the simulator handicap (~287/season) being read as a model defect. |
| O1_OLD | **Why is the live season underperforming by ~10.7 pts/GW?** | 337 vs 412 after 7 GW; rank 3.2M → 8.5M. | **UNEXPLAINED.** xP, optimiser, squad quality, features and horizon all ruled out. Remaining: variance, or something specific to 2026-27. |
| O2 | Is the high-xP bias **−0.6 (historical)** or **+0.46 (live GW6–7)**? | Opposite signs. Determines whether the top of the forecast can be trusted. | Needs 4–5 more archived gameweeks. |
| O3 | Does the v2.4 upper-tail layer change decisions at all? | It degrades MAE and passes only via M3. | **UNRESOLVED after 4 attempts — see M8.** |
| O4 | Policy scorecard was selected with `WALK_FORWARD_INCLUDE_CHIPS = False` | `greedy_1gw_no_hits` may not be optimal under v2.6 semantics. | Deferred, expensive. |
| O5 | Set-piece / penalty roles never modelled | ~20–30 points per wrong call; a new penalty taker is invisible until history accrues. | Cheap, never done. |
| O6 | `optional_fbref_context: PARTIAL` on every recent run | The external feed we *have* is degraded, before adding new ones. | Unfixed. |
| O7 | Fix the role-change miscalibration (S7) | Firmest unfixed defect: +0.124 pts/GW, 3/3 seasons. | Not attempted. |
| O8 | **Does the greedy 1-gameweek transfer rule destroy value in production too?** | It cost both signals ~50 points in 2025-26 in simulation, and the live pipeline uses the same policy. | Untested against production. |
| O9 | **Would a Championship prior WITH xG close the remaining gap?** | FBref publishes no xG/npxG/xAG for the Championship in any of 2022-23..2025-26 — confirmed on all four tables. The big-5 priors the model trains on carry npxG and xAG, so Championship arrivals reach it with those columns NaN. That may be most of why F16's calibration gain stops at 0.776 rather than 1.000. | Needs a source with Championship xG (Opta/StatsBomb). Not available free. |
| O10 | **Established players at last season's BOTTOM clubs are over-predicted by 18%.** | 2026-27 GW1-6, tiers set from the prior season so the split cannot know the outcome: top6 0.983, mid 0.884, bottom **1.177**, promoted 1.023 (n=105, 826 pts in the bottom cell). Largest unexplained calibration number currently on the board. | Never tested. Note the first version of this split was circular — see M10. |
| O17 | **Swap the OTHER components to a GLM.** | -- | **RESOLVED 2026-10-21. `rates` is the whole story.** Minutes alone +17, team alone **-24** -- neither moves top-20 or captain. All three combined: mean **-102** vs legacy, beating every baseline seed, but see the gate correction below. Components are NOT additive (210+17-24=203 expected, 165 observed). |
| O21 | **Why is 2024-25 unstable in every test?** | -- | **RESOLVED 2026-10-21. It contains a 344-point outlier (Salah) that the GBDT barely identifies.** See S13. |
| O23 | **What does legacy do better than the GLM?** | **ANSWERED 2026-10-21: probably nothing measurable — see S16.** The GLM's -57 is within 1.5 sd of zero. Ten mechanisms were eliminated chasing a gap that is not distinguishable from noise. **Ruled out along the way:** overall ranking (S10), top-end calibration (W13), elite identification (S15 -- the GLM is BETTER), transfer count and the min_gain threshold (O13), self-normalised scale (O20), early-season signal quality (W16 -- ours out-ranks legacy in the candidate pool 2/3), signal shape (W16), and the opening squad as a general claim (W16 -- the GLM already closed it from 453 to 143). **What is left:** WHICH transfers get made (~100 pts, O13's frozen-squad arm), and the residual ~143 of opening-squad quality that does not convert. | **Still unexplained. Eight mechanisms eliminated.** |
| O22 | **Does the GBDT fail to identify ELITE players generally?** | -- | **CONFIRMED 2026-10-21 — see S14.** The first explanation of the day to survive its own test. Graded by tier (+12.7 / +7.1 / +0.4), so it is specific to the top rather than general ranking weakness, and the magnitude rules out a tie-break artefact. But see **S15**: it explains GBDT->GLM only, not legacy's advantage. |
| O18 | **Does the GLM fix the top-end over-prediction directly?** | -- | **RESOLVED 2026-10-21: NO.** The GLM over-predicts the top MORE (|1-ratio| 0.095 vs 0.070). S12 stands, its explanation is retracted (W13). |
| O19 | **Why does the GLM actually win?** | -- | **PARTLY RESOLVED 2026-10-21, `fpl2/run_o19.py`.** Quantile-matching both signals to an identical per-gameweek value distribution (same numbers, assigned by each model's own ranking) keeps **+119 of the +210**: the win is **57% ORDERING, 43% SCALE**. The ordering component is positive in **3/3 seasons** (+241/+39/+77); the scale component is volatile and sign-flipping (+8/+362/-98), supplying most of 2024-25's headline +401 and COSTING 77 in 2025-26. Quantile-matching made the result MORE consistent (3/3 vs 2/3 raw). Caveats: +119 is below the 148 floor, so the direction is established and the magnitude is not; and top-decile spearman moved only 2/3 (+0.073/-0.014/+0.006), so the two ordering measures disagree for reasons currently unknown. |
| O20 | **Feed the optimiser a QUANTILE-NORMALISED signal.** | -- | **REFUTED 2026-10-21, `fpl2/run_o20.py`.** `self_pooled` (the signal's own quantiles from PRIOR seasons -- the only leak-free, shippable template) LOSES for both arms on matched seasons: gbdt -252 vs raw -208, glm -85 vs raw -18. The volatile scale component is earning its keep, not adding noise. `legacy_gw` improves the GLM (-57 -> -21, best figure of the day) but HURTS the gbdt (-267 -> -304) and is inside the noise floor, so it reads as noise; it is also a hybrid in which legacy supplies the scale. **Gate bug worth recording: `self_pooled` has no 2023-24 value, so its 2-season mean was compared against raw's 3-season mean and printed a spurious PASS. Always compare on matched folds.** |
| O16 | **Re-run the contaminated model.py results.** | M14's leak was in 2 of 100 features for an unknown number of sessions. Every `model.py`-based number -- G1 -290, F17, G8, G9 -- carries it. No verdict flips (all were FAILs, and the leak inflates), but the magnitudes are wrong. | Cheap to redo; nobody should quote those figures until it is. |
| O13 | **The other ~150 points: transfers, XI and captain.** | -- | **CLOSED 2026-10-21, and its own decomposition was over-read.** O13 reported frozen-squad -53 vs transferring -146 and attributed ~90 pts to transfers. That 93-point difference is **inside the 148 floor** and was never measurable. S17 then scored ~110 transfers directly, outside the simulator: realised quality is identical (|t|<0.4 on every horizon). The whole item dissolves into S16 -- it was decomposing noise. |
| O14 | **Capture FPL's own strength/FDR fields.** | The API exposes `strength_attack_home/away`, `strength_defence_home/away` and per-fixture `difficulty` live, but the 56-column historical dataset has NONE of them, so fixture difficulty cannot be validated against history and our tiers are computed from prior-season goal difference instead. Starting to log them costs nothing and compounds. | Collector change, never done. |
| O15 | **Does shrinkage belong in the complex model too?** | In the simple model, empirical-Bayes shrinkage toward the position mean lifts opening-15 quality from 1858 to ~1950 (legacy 2136) -- mechanism confirmed, magnitude unmeasurable on season totals (the whole k-sweep spans 111 points against a 148 floor). Heavy k improves the squad but degrades in-season XI/captain, so modest k is principled. | k=10 kept in `simple.py`. Untested in `model.py`. |
| O12 | **Re-run the captain rules (F9) with a powered design.** | The captain is half to two-thirds of the model's whole margin (S8) and ~85% of the within-squad gap, and F9 closed it on a sign-flipping, sub-noise test (M12). | **RESOLVED 2026-10-21 — see F18.** Floor measured first (0.32-1.62 pts/gw). A real ceiling signal exists (z=3.83 vs shuffle) but the armband cannot exploit it (+19 pts over 4 seasons). F9's verdict stands; its reasoning does not. **The captain is not where the points are.** |
| O11 | **Is a season-total instrument capable of detecting a 40-player effect at all?** | Two instruments returned *exactly* +0.0 for 2023-24 while 91 players' features changed. With a ~150-point noise floor and effects of ~40, season totals may be structurally incapable of resolving this class of change. | Would need a metric scoped to the affected players, or a paired/variance-reduced design. |
---

## Price-band calibration (F24) — PRE-REGISTERED then REFUTED, 2026-10-22

**The observation (live season, 2026-27 GW1-7, UNDERPOWERED — n=28 and n=14 in
the top two price bands).** `xp/actual` by contemporaneous price:

| band | n | mean_xp | mean_pts | xp/actual | xp_per_£m | pts_per_£m |
|---|---|---|---|---|---|---|
| <5.0 | 2907 | 0.95 | 0.98 | **0.97** | 0.21 | 0.21 |
| 5.0-6.5 | 1420 | 1.84 | 1.91 | 0.96 | 0.32 | 0.34 |
| 6.5-8.0 | 182 | 2.91 | 3.28 | 0.89 | 0.39 | 0.44 |
| 8.0-10.0 | 28 | 3.98 | 5.50 | **0.72** | 0.43 | 0.60 |
| >10.0 | 14 | 5.40 | 6.50 | 0.83 | 0.39 | 0.47 |

**Why it would matter mechanically.** The optimiser maximises xp subject to a
£100m budget, so what governs selection is xp PER POUND, not xp. A bias that
grows with price flattens `xp_per_£m` (0.21 → 0.43 → 0.39) while the truth,
`pts_per_£m`, keeps rising (0.21 → 0.60 → 0.47). The optimiser therefore sees
no value advantage in a premium and buys fifteen mid-price players instead.
This predicts, and is consistent with, the following already-measured facts:

* Haaland was NEVER in the simulated XI in any of GW1-7, with xp rank 1, 2, 2,
  5 and 10 in five of those weeks, while the crowd held him at 69-77%.
* The GW7 captain was Bruno (xp 6.84) over Haaland (xp 6.18, actual 13).
* The simulated XI is populated by sub-5%-owned cheap defenders (Richards 0.6%,
  Muharemović 4.7%, Van Hecke 7.0%).
* `price` IS already a model feature, so this is not missing information. It is
  over-shrinkage — the same family as the minutes defect (S19/S20/S21).

**GATE, fixed before the holdout numbers are known (R24, R27, R2, R26).**

1. **Replication.** `xp/actual` must decline with price in ALL THREE holdout
   seasons measured independently (2023-24, 2024-25, 2025-26). Monotone within
   a 0.02 tolerance. If it holds in fewer than 3, the effect is not established
   and NOTHING is shipped.
2. **Magnitude.** The corrected model, run end-to-end through the
   budget-constrained simulator, must gain **≥ +100 points/season** on the
   3-season mean. That is ~2 sd of the 3-season mean (R1: sd 39-49), stated in
   noise-floor units per R24.
3. **Consistency.** Positive in at least 2 of 3 seasons.
4. **Negative control (R26).** A SHUFFLED correction — the same per-band
   multipliers, randomly reassigned to bands — must NOT pass gates 2 and 3. If
   it does, the gain is perturbation, not information.

**Stated directionally so it cannot be reinterpreted afterwards:** if the bias
replicates but the end-to-end gain is under +100, the finding is recorded as
REAL BUT NOT WORTH SHIPPING, not as a partial success. If the bias fails to
replicate 3/3, the live-season table above is recorded as an artefact of n=14.

---

## GW6-7 postmortem and the optimiser's objective — 2026-10-22

### F24 — the price-calibration hypothesis is REFUTED (gate above, written first)

| xp/actual | 2023-24 | 2024-25 | 2025-26 | monotone? |
|---|---|---|---|---|
| <5.0 | 1.018 | 0.879 | 0.990 | |
| 5.0-6.5 | 0.838 | 0.961 | 0.963 | |
| 6.5-8.0 | 0.747 | 0.968 | 0.988 | |
| 8.0-10.0 | 0.742 | 0.973 | 1.003 | |
| >10.0 | **0.591** | 0.870 | **0.957** | |
| | **YES** | no | no | **1/3** |

Gate 1 required 3/3 and got 1/3, so gates 2-4 were never run and nothing was
shipped. 2025-26 — the most recent and most relevant season — is well
calibrated in every band. The live-season table that motivated this had n=14 in
its top band; the pre-registration said in advance that a 3/3 failure would
make that an artefact, and that is how it is recorded. **The effect is real in
2023-24 only and is not a property of the model.**

Pre-registering the gate is what stopped this shipping. It is the first time on
this project that a hypothesis was killed by a threshold written before the
numbers existed rather than after. Keep doing it.

### S26 — GW6-7 were not anomalous; the squad is structurally high-variance

The question "why were GW6-7 so hard" assumed a defect that appeared in GW6.
Measured, it did not.

| | per-gameweek mean | sd |
|---|---|---|
| model | 52.4 | **25.4** |
| crowd | 59.4 | **11.9** |

GW6 is z = -1.00 and GW7 z = -0.81 against the model's own distribution. The
largest outlier in the season is GW2 at **z = +1.95** — our best week. The mean
gap to the crowd is **-7.0 ± 6.7 pts/gw, t = -1.05** over seven gameweeks:
not distinguishable from zero, and that is BEFORE chips, which S25 prices at
+87/season (~+2.3/gw).

The decomposition that looked damning is a base rate. "17 of the top-20 scorers
were not rated by us (xp rank >40)" is true in EVERY gameweek, including GW2
where we beat the crowd by +21:

| gw | ours | crowd | top-20 not rated | rated+owned |
|---|---|---|---|---|
| 1 | 44 | 50 | 17 / 201 | 1 / 10 |
| 2 | **102** | 81 | 13 / 142 | **4 / 60** |
| 3 | 46 | 51 | 16 / 169 | 1 / 8 |
| 4 | 67 | 69 | 16 / 182 | 1 / 17 |
| 5 | 49 | 48 | 15 / 177 | 2 / 22 |
| 6 | **27** | 58 | 17 / 184 | **0 / 0** |
| 7 | **32** | 59 | 17 / 179 | **0 / 0** |

The discriminating column is the last one. GW6 and GW7 are the only weeks in
which NOT ONE of the top-20 scorers was both rated and owned. That is a
property of a 12.7%-mean-ownership squad meeting two weeks in which the big
hauls went to players nobody owned (Hato 0.6%, Boscagli 0.3%, Ouattara 0.8%) —
a low-ownership squad has no systematic edge on those, and has given up the
template's floor.

**Conclusion: GW6-7 is the downside tail of a high-variance strategy, not a new
defect.** The same structure produced GW2's +21. The open question is whether
that variance is CHOSEN or ACCIDENTAL, which leads to O37.

### O37 — `optimise_squad` maximises squad xp, but only eleven players score

`optimise.optimise_squad` passes `c = -p[score]` to the MILP: it maximises the
sum of xp over all FIFTEEN selected players. FPL scores eleven. Consequences,
measured on the GW7 squad:

* the opening squad spends the full **£100.0m with £0 in the bank**; at GW7
  the XI costs **£78.1m** and the four-man bench **£21.9m**. A real manager's
  bench at £4.0-4.5m costs ~£17.5m, so **~£4.4m is stranded** where it cannot
  score. Two of the four benched players cost £6.8m and £6.0m — they are not
  bench fodder, they are real players bought to raise a sum that excludes them.
  (An earlier draft of this entry said £77.1m/£22.9m: that used GW7 MARKET
  prices rather than purchase prices. Corrected.)
* upgrading a £4.0m bench goalkeeper (xp 0.30) to a £5.0m one (xp 1.68) scores
  as a **+1.38 gain** to this objective and is worth **exactly zero** in play.
* mean XI ownership **12.7%**; Haaland was never selected in ANY of GW1-7
  despite xp rank 1, 2, 2, 5, 10 in five of them, at 69-77% crowd ownership.

This is a defect in the MEASURING INSTRUMENT, in the same family as S22 (the
simulator lacked chips) — it would be present with a perfect forecast, and it
is not a forecast bias: F24 establishes the forecast is well calibrated by
price in 2025-26. Note the ordering: the calibration hypothesis was the
attractive one and it was WRONG; the boring objective-function one was found
only after the gate killed it.

Fix is a standard two-variable MILP: binaries for squad membership and for XI
membership, `x_i <= s_i`, formation constraints on `x`, budget and club cap on
`s`, maximising `sum(xp_i * ((1-w)*x_i + w*s_i))`. `w = 1.0` reproduces present
behaviour exactly, so the change is parameterised rather than replaced.
**NOT YET BUILT. Gate must be pre-registered first, per F24.**

---

## XI-aware squad optimisation (F25) — PRE-REGISTERED then REFUTED, 2026-10-22

**The defect.** `optimise.optimise_squad` maximises `sum(xp)` over fifteen
players; FPL scores eleven. Measured consequences are in S26/O37 above.

**The change.** Two binary vectors instead of one: `s_i` (in the 15) and `x_i`
(in the starting XI), with `x_i <= s_i`, `sum(x) = 11`, formation bounds on
`x`, budget and club cap on `s`. Objective `sum(xp_i * ((1-w)*x_i + w*s_i))`.
Parameter `bench_weight = w`. **Default stays `w = 1.0`**, which is
algebraically the present objective, so every prior ledger result keeps its
meaning. `w` is NOT zero in the candidate: autosubs mean a benched player does
occasionally score.

**CORRECTNESS CHECK (not a result).** At `w = 1.0` the new code must reproduce
the current season totals EXACTLY, to the point. If it does not, the refactor
is wrong and no gate applies.

**GATE, fixed before any number is seen (R24, R27, R2, R26).**

1. **Primary, decision metric.** Mean season total across the three holdout
   seasons must improve by **>= +100 points/season** versus the `w = 1.0`
   control. That is ~2 sd of the 3-season mean (R1: sd 39-49), stated in
   noise-floor units per R24.
2. **Consistency, two tiers declared in advance so neither can be chosen
   afterwards.** 3/3 seasons positive -> SHIP. Exactly 2/3 -> recorded as
   **REAL BUT NOT ESTABLISHED**, shipped OFF, exactly as F20/O29 were handled.
   0/3 or 1/3 -> REFUTED.
3. **Dose-response.** `w` is swept over {1.0, 0.5, 0.25, 0.1, 0.0} and the
   WHOLE curve is reported. No value is selected on the run. A real mechanism
   should vary smoothly with `w`; a result that appears at one `w` and nowhere
   else is noise.
4. **Negative control (R26).** With `xp` SHUFFLED within (season, gameweek,
   position), the `w = 0.1` minus `w = 1.0` delta must sit inside the noise
   floor. If scrambled forecasts also gain, the gain is structural
   perturbation, not information.

**Also measured, not gated** (the user's request): five arms per holdout
season — legacy, legacy + fix, model, model + fix, FPL average. Legacy is a
cached forecast and exists for 2023-24/24-25/25-26 only, so 2026-27 carries
model arms and the real average only. The "FPL average" season equivalent of
**2257** is 59.4 pts/gw x 38, derived from 2026-27 and applied across seasons:
a PROXY, carried from S22/S23/S25, not a per-season measurement.

**Known un-fixed neighbour.** `simulate_season`'s weekly transfer rule compares
single-player xp deltas and is equally XI-blind — a transfer that upgrades the
bench scores as a gain. If squad construction is fixed but transfers are not,
the squad may drift back toward a flat structure over 38 gameweeks. XI spend at
GW1 vs GW38 will be reported to show whether this happens. NOT in scope here.

---

## F25 — the XI-aware objective is REFUTED, and O37's motivation was wrong

`fpl2/run_o37.py`, gate above written before the code existed.

**Correctness first (this is the part that passed).** At `w = 1.0` the new
two-variable formulation reproduces the recorded season totals EXACTLY:
model **2143 / 2420 / 2149** and legacy **2226 / 2359 / 2061**, identical to
F20's figures of 2026-10-21 to the point. Nothing in the ledger changed
meaning. The R26 negative control is perfect: with xp shuffled within
(season, gameweek, position) the w=0.1 minus w=1.0 delta is **exactly 0.0** in
all three seasons, so the harness invents no movement.

**The gate, failed at every dose.**

| w | 2023-24 | 2024-25 | 2025-26 | mean | positive |
|---|---|---|---|---|---|
| 0.5 | 0 | **-290** | -144 | **-144.7** | 0/3 |
| 0.25 | 0 | +44 | -148 | -34.7 | 1/3 |
| 0.1 | 0 | +46 | -194 | -49.3 | 1/3 |
| 0.0 | -14 | -88 | -64 | -55.3 | 0/3 |

Required >= +100 and 3/3. Got negative means and 0-1/3. **REFUTED.** At w=0.5
the mean is -144.7, roughly 3 sd of the 3-season mean (R1 sd 39-49), so that
dose is measurably HARMFUL rather than merely useless. The remaining deltas sit
inside the 112-125 single-season floor and should be read as noise in both
directions, not as a small loss.

**The motivation was wrong, which matters more than the gate.** O37 claimed the
bench weighting was why Haaland was never bought. Tested directly: at
**w = 0.0** — bench valued at nothing — the optimiser STILL does not buy him,
at any dose. His exclusion is a value-per-pound judgement on the forecast:

| player | xp | price | xp/£m |
|---|---|---|---|
| Haaland | 4.543 | £15.7m | **0.2894** |
| Thiago | 3.993 | £7.8m | 0.5120 |
| Calvert-Lewin | 3.482 | £6.0m | **0.5803** |

Forcing him in costs **-0.58 xp/gw** by the model's own arithmetic. Given its
forecast, the optimiser is behaving correctly. The forecast is what is wrong:
Haaland was predicted **4.54/gw against an actual 8.3/gw** over GW1-7, a 45%
under-prediction that persists WITH `RECENCY_K = 4.0` already shipped. That is
S20, still open, and it is the real target.

**Why a more correct objective loses points — hypothesis, NOT established.**
Concentrating budget in eleven players buys a £4.0m bench. Over 38 gameweeks
with injuries, rotation and blanks, autosubs promote that bench, and the
simulator allows one free transfer a week and no hits, so a concentrated squad
cannot be repaired quickly. The flat squad may be absorbing variance that the
single-gameweek objective cannot see. UNTESTED. It would be testable by
counting autosub events per arm, which the simulator does not currently model.

**Kept, shipped OFF.** `optimise_squad(bench_weight=1.0)` is the default and is
the old behaviour exactly; `simulate_season(bench_weight=...)` threads it
through. 11 new tests in `tests/test_optimise.py`, negative-controlled: run
against the unfixed objective the key assertion fails with max XI gain
**0.0000** on all 10 seeds, and passes at 0.05-1.09 with the fix.

**Five arms, as requested** (FPL average = 59.4/gw x 38, a 2026-27-derived
proxy carried from S22/S23/S25, not a per-season measurement):

| season | legacy | legacy+fix | model | model+fix | FPL avg |
|---|---|---|---|---|---|
| 2023-24 | 2226 | 2288 | 2143 | 2129 | 2257 |
| 2024-25 | 2359 | 2026 | 2420 | 2332 | 2257 |
| 2025-26 | 2061 | 1987 | 2149 | 2085 | 2257 |
| **mean** | **2215** | **2100** | **2237** | **2182** | **2257** |
| 2026-27 (7 gw) | n/a | n/a | 358 | **432** | 416 |

The live season is the only arm where the fix wins (432 vs an average of 416,
+74). It is 7 gameweeks against 0/3 on the holdouts and is NOT counted —
the same shape as F24 earlier today, where an n=14 live-season table motivated
a hypothesis the holdouts refuted.

**Two hypotheses pre-registered and killed in one day (F24, F25).** Both were
mechanically plausible, both were motivated by live-season tables with small n,
and both died on thresholds written before the numbers existed. The pattern is
now explicit enough to be a rule — see R30.

---

## S27 — a better opening squad does NOT produce a better season, 2026-10-22

`fpl2/run_o37_dense.py`. 16 values of `bench_weight` x 3 holdout seasons.

**Season total is CHAOTIC in w.** Roughness (sum of absolute step changes
divided by range; a smooth curve scores ~1.0): **1.56 / 2.27 / 2.30**. The
shape is flat plateaus with cliff edges — the MILP returns one integer squad
across a range of w, then flips. 2024-25 sits at ~2465 for w in [0.05, 0.25],
drops to 2130 for w in [0.30, 0.70], and returns to 2420 at w >= 0.80. Range
within a single season reaches **338 points**, three times the 112-125
single-season noise floor, driven purely by which squad the solver happened to
return. **`bench_weight` is therefore NOT optimisable**: there is no curve to
find an optimum on, and any 'best w' is the arm that forked well.

**The path-free metric behaves perfectly.** Mean opening-XI xp per gameweek
rises smoothly as w falls — 51.299 at w=1.0 to a peak of **51.985** at
w ≈ 0.15-0.25 (roughness 1.00 / 1.05 / 1.94). So the XI-aware objective does
exactly what F25 claimed mechanically: it builds a measurably better starting
eleven, worth ~**+0.69 xp/gw**.

**And it does not matter.** Correlation between opening-XI xp and season total,
WITHIN season, across the 16 arms:

| season | corr |
|---|---|
| 2023-24 | **-0.083** |
| 2024-25 | +0.214 |
| 2025-26 | **-0.290** |

(The pooled +0.479 is Simpson's paradox — it is driven by between-season
differences, not by w. The within-season numbers are the ones that answer the
question.)

**This is the most important result of the day and it reframes the project.**
Squad construction quality is swamped by everything that happens afterwards.
It explains F25 cleanly: the fix built a better squad AND lost points, which
is not a contradiction once construction and outcome are known to be
uncorrelated. It also retires a whole class of work — any further effort on
the opening squad, the horizon, or the selection objective is optimising a
quantity that does not propagate to the season total under the current policy.

**What it points at.** The thing between a good squad and a good season is the
TRANSFER POLICY, and `simulate_season`'s is myopic: one greedy single swap per
week, `min_gain = 0.30`, no planning, no banking, no hits. See O38.

### O38 — treat the season as the MDP it is

State (squad, bank, free transfers, gameweek, chips remaining); actions
(transfers, chip, XI, captain); reward (realised points); transition (prices,
injuries, fixtures). Exact DP is impossible — the state space is C(600,15) —
so the tractable method is **receding-horizon MILP / model-predictive
control**: time-indexed `s[i,t]`, `x[i,t]`, `buy[i,t]`, `sell[i,t]`, continuity
`s[i,t] = s[i,t-1] + buy - sell`, free-transfer accounting, hits at -4, budget
on selling prices, formation and club cap; maximise the discounted sum of XI xp
plus captain over the horizon; commit week t only and re-solve.

**HARD CONSTRAINT, from M17.** Horizon <= 4 gameweeks. M17 was precisely this
leak — a policy reading a fixture calendar that does not exist at decision
time — and `chips.py` already uses `KNOWN = 4` for the same reason. A longer
horizon re-introduces it. Any O38 gate must include the M17 negative control.

**Open, and motivated by a measurement (S27) rather than a story — unlike F24
and F25, both of which died today on exactly that distinction.**

---

## Receding-horizon transfer policy (F26) — REFUTED as built; the CONTROL is the finding, 2026-10-22

**Motivated by a measurement, not a story.** S27: opening-XI quality and season
total are UNCORRELATED within season (-0.083 / +0.214 / -0.290). The transfer
policy, not squad construction, is what converts a squad into a score. The
present policy is one greedy single swap per week at `min_gain = 0.30`, with no
banking, no hits and no planning.

**Formulation.** Time-indexed MILP over H gameweeks, re-solved every week,
committing week 0 only. Binaries `s[i,t]` own, `x[i,t]` start, `c[i,t]`
captain, `buy[i,t]`, `sell[i,t]`; continuous `hits[t] >= 0`; integer `ft[t]`.

```
max  sum_t ( sum_i xp[i,t]*x[i,t] + sum_i xp[i,t]*c[i,t] - 4*hits[t] )
s.t. sum_i s[i,t] = 15 ;  position quotas on s ;  club cap <= 3 on s
     sum_i x[i,t] = 11 ;  x[i,t] <= s[i,t] ;  formation bounds on x
     sum_i c[i,t] = 1  ;  c[i,t] <= x[i,t]
     s[i,t] - s[i,t-1] = buy[i,t] - sell[i,t] ;  s[i,-1] = current squad
     sum_i cost[i]*s[i,t] <= bank0 + value(current squad)
     ft[t+1] <= ft[t] - T[t] + 1 ;  ft[t] <= FT_MAX ;  hits[t] >= T[t] - ft[t]
```

**THE LEAK THIS INVITES, and how it is controlled.** At the week-t0 deadline we
do NOT possess the walk-forward forecast for t0+3: that forecast was built from
data in t0+1 and t0+2. Feeding it to the planner is M17 exactly — a policy
reading a calendar that does not exist yet. Two arms are therefore run:

* **honest** — future weeks reuse the week-t0 per-player xp, adjusted only by
  the published fixture list (zero for a blank, scaled for a double). Fixtures
  are confirmed ~4 weeks out, which is why `chips.py` uses `KNOWN = 4`.
* **oracle** — future weeks use each week's true walk-forward xp. LEAKY, an
  upper bound only, never shippable.

Reporting both is the control: if honest ~= oracle the honest arm is leaking
and the implementation is wrong; if honest << oracle the gap is the value of
knowledge we do not have, and the honest number is the only one that counts.

**GATE, fixed before the code exists (R24, R27, R2, R26, R30).**

1. **Primary.** Mean season total over the 3 holdouts, HONEST arm, must beat
   the current greedy policy by **>= +100 pts/season** (~2 sd of the 3-season
   mean; R1 sd 39-49).
2. **Consistency.** 3/3 positive -> SHIP. 2/3 -> REAL BUT NOT ESTABLISHED,
   shipped OFF. <=1/3 -> REFUTED.
3. **Horizon sweep.** H in {1,2,3,4}, whole curve reported, no H selected on
   the run. **H > 4 is forbidden** (M17). H=1 is a correctness check: it should
   land near the greedy policy, since both then optimise a single week.
4. **Negative control (R26).** With xp shuffled within (gameweek, position),
   MPC minus greedy must sit inside the noise floor. A planner given scrambled
   forecasts must not plan its way to points.
5. **S27 re-test.** If O38 passes, re-run the `bench_weight` sweep underneath
   it. S27 says construction does not propagate under the GREEDY policy; it may
   under a planning one. F25 would then deserve re-opening — but only then.

**Out of scope for v1:** chips (handled separately by `chips.py`), within-season
price changes (prices are frozen in this simulator, so buy price = sell price
and the budget constraint is a single linear row per week), and injuries.

---

## F26 / S28 — the planner works, the multi-week forecast does not exist

`fpl2/fpl2/mpc.py` + `fpl2/run_o38.py`, against the gate pre-registered above.

**Correctness, first.** `policy="greedy"` reproduces **2143 / 2420 / 2149**
exactly — the ledger's own figures. The MPC path is additive and off by
default. 7 tests in `tests/test_mpc.py`, including the M17 leak test: scramble
every gameweek after the one being decided and the honest plan must not move,
while the oracle plan MUST move (without that second clause a planner that
ignored its inputs would pass). Candidate-pool size was checked as a possible
handicap and is not one: 2360 / 2393 / 2324 / 2391 across a 10x range, all
inside the single-season floor.

**The gate, failed at every horizon.**

| | 2023-24 | 2024-25 | 2025-26 | mean | positive |
|---|---|---|---|---|---|
| honest H=1 | -18 | -57 | -88 | -54.3 | 0/3 |
| honest H=2 | +48 | -60 | -142 | -51.3 | 1/3 |
| honest H=3 | -90 | -57 | -202 | -116.3 | 0/3 |
| honest H=4 | -110 | -73 | -10 | -64.3 | 0/3 |
| **oracle H=4** | **+107** | **+66** | **+48** | **+73.7** | **3/3** |

**REFUTED as built.** But the M17 leak control, included to catch a mistake,
instead produced the best-replicating result of the day.

**S28 — MPC performance is monotone in forecast quality, and that is the
whole story.** Three arms differing ONLY in the forecast handed to the planner:

| forecast given to the planner | mean vs greedy |
|---|---|
| shuffled within (gameweek, position) — no information | **-143.3** |
| honest — this week's xp, flat, scaled by fixture count | **-64.3** |
| oracle — each week's true walk-forward forecast | **+73.7** |

Monotone, and the spread is 217 points. The planning machinery is not the
bottleneck; it is sound enough to convert better information into points at a
steady rate. What is missing is the information. **A genuine 4-gameweek-ahead
forecast is worth ~+138 pts/season over what we currently project**, and the
oracle arm is not a fantasy ceiling: it uses FORECASTS, not actuals — the
quality we already achieve at one week, available at four.

This also explains why planning further ahead made things worse (+48 -> -90 ->
-110 in 2023-24). Planning on a bad projection is worse than not planning: the
longer the horizon, the more weight the plan puts on numbers that are wrong.

### O39 — build the multi-week forecast

The honest projection reuses week t's xp scaled by week t+k's fixture COUNT. It
does not use week t+k's OPPONENT, although the fixture list is published four
weeks out and the model already carries `opp_conceded_r5`, `opp_scored_r5`,
`att_vs_opp_def`, `def_vs_opp_att` and `is_home`. The buildable version: freeze
every player-form feature at week t (they are strictly lagged, so they are
legitimately known), substitute week t+k's fixture and opponent columns, and
run `ComponentModel.predict` forward. Nothing about it leaks — it is the same
model applied to a known future fixture.

**Priced at up to +138/season by S28.** This is the first open item on the
project with a measured price tag rather than an argued one.

**Second-order, do not skip:** if O39 lands, F25 and the `bench_weight` sweep
deserve re-opening. S27 showed squad construction does not propagate to the
season total UNDER THE GREEDY POLICY. That may be a property of the policy
rather than of construction.

---

## Multi-week forward forecast (F27) — PRE-REGISTERED then REFUTED, 2026-10-22

**Priced by S28 at up to +138 pts/season.** MPC performance is monotone in
forecast quality (shuffled -143.3, honest -64.3, oracle +73.7). The planner is
sound; the projection it is fed is not. The honest arm reuses week t's xp
scaled by week t+k's fixture COUNT and discards the opponent entirely.

**The construction.** Every player-form feature is already a strictly lagged
aggregate, so its week-t value is legitimately known at week t. The fixture
list is published ~4 weeks out. Therefore, to forecast week t+k at week t:

* FREEZE every player-form and own-team-strength feature at its week-t value
* SUBSTITUTE `gw`, `n_fixtures`, `fixtures`, `is_home` from week t+k's fixtures
* SUBSTITUTE `opp_*` with week t+k's OPPONENT's strength **as known at week t**
* RECOMPUTE `att_vs_opp_def`, `def_vs_opp_att` from the two above
* for the team frame, replace every lagged column (`att_r5` ... `opp_xgc_career`)
  with that team's week-t values
* run `ComponentModel.predict` unchanged

Nothing is refitted and nothing reads a result after week t.

**GATE, fixed before the code exists (R24, R27, R2, R26, R30).**

1. **Leak test is a PRECONDITION, not a gate item.** Mutating every gameweek
   after t must leave the week-t projection for t+k bit-identical. A negative
   control must accompany it: the oracle path on the same mutation MUST move.
   If the leak test fails, nothing else is measured.
2. **Forecast quality, the direct test.** Projected xp for week t+k must beat
   the current flat projection on Spearman against realised points, for
   k = 1,2,3, in all three holdout seasons. This is a 9-cell table and it is
   reported whole.
3. **Decision metric, the gate proper.** MPC at H=4 using the new projection
   must beat the GREEDY policy by **>= +100 pts/season** on the 3-season mean.
   3/3 -> SHIP. 2/3 -> REAL BUT NOT ESTABLISHED. <=1/3 -> REFUTED.
4. **Second, weaker comparison, reported but NOT the gate:** new projection vs
   the flat projection, both under MPC H=4. Beating the flat projection while
   still losing to greedy would mean the projection improved and the policy is
   still not worth running — those are different conclusions and must not be
   merged.
5. **Negative control (R26).** Shuffle the opponent assignment in the fixture
   list — each team still plays someone each week, but the wrong someone. The
   new projection must then NOT beat the flat one on gate 2. If a scrambled
   opponent map forecasts as well, the gain is not opponent information.

**Stated in advance so it cannot be reinterpreted:** S28's +138 is an UPPER
BOUND derived from an oracle that uses true one-week-quality forecasts at four
weeks. A forward projection must be worse than that. If gate 3 fails but gate 2
passes, the recorded conclusion is **the forecast improved and it was not
enough**, not a partial success.

---

## F27 / S29 — the opponent is worth nothing, and S28's +138 is not about fixtures

`fpl2/fpl2/project.py` + `fpl2/run_o39.py`, against the gate pre-registered above.

| gate | result |
|---|---|
| 1 — leak (PRECONDITION) | **PASS**. Scrambling 22,027 player rows and 560 team rows after week t moves the week-t projection of t+3 by **0.000000000000**. Control: the same mutation moves the oracle path by **13.04**, so the test is not vacuous. |
| 2 — forecast quality | PASS **9/9** cells, but the deltas are **+0.0014 to +0.0055** spearman. Technically a sweep, practically nothing. |
| 5 — scrambled opponent map | **FAIL**. Real opponents **0.6446**, scrambled opponents **0.6456**. |
| 3 — decision metric | **REFUTED**. +75 / -109 / -66 vs greedy, mean **-33.3**, **1/3**. |
| 4 — vs flat (reported, not gated) | +185 / -36 / -56, mean +31.0, 1/3. Noise. |

**S29 — opponent identity carries no ranking value one to three weeks out.**
The control is the result. Scrambling WHICH opponent each team faces leaves the
forecast very slightly BETTER. This is not a plumbing failure: a dedicated test
(`test_the_opponent_actually_reaches_the_forecast`) proves the opponent does
change the projected numbers. It changes them without improving the ordering.
So gate 2's 9/9 was never opponent information -- it was model nonlinearity
beating a linear rescale, worth 0.003 spearman.

**This re-reads S28, and that is the real output of the day.** S28 measured an
oracle arm worth **+73.7 pts/season (3/3)** against greedy, a 138-point spread
over the honest projection, and the obvious reading was "build a better
multi-week forecast". F27 built exactly that, honestly, and it bought nothing.
Since the oracle and the projection differ in only two things -- future
OPPONENTS and future PLAYER FORM/MINUTES -- and S29 prices opponents at zero,
**the whole +138 must be the value of knowing players' near-future form and
minutes**. Who will be fit, who will be rotated, whose role is about to change.

That is not a modelling gap we can close by rearranging features we already
hold. It is an information gap: team news, press conferences, rotation
patterns, injury reports -- things a human manager reads and this pipeline has
never ingested. S20 is the same finding seen from the other end (Haaland
predicted 4.54/gw against an actual 8.3/gw).

**Kept, shipped OFF.** `project.py` is additive; nothing calls it unless a
caller passes `mpc_frames`. `simulate_season` defaults are unchanged and
`policy="greedy"` still reproduces 2143 / 2420 / 2149 exactly. 6 tests in
`tests/test_project.py`, two of which are negative controls.

### O40 — the only lever S29 leaves

If the remaining headroom is near-future minutes and availability, the next
move is DATA, not modelling: the FPL API's `chance_of_playing_next_round`,
`news`, `status` and `element-summary` transfer flows are contemporaneous
signals of exactly that, and `live.ownership()` (O36, closed today) already
proves the per-gameweek history is retrievable. **Unmeasured. Do not assume it
works -- four hypotheses were refuted today and three of them looked better
than this one going in.**

---

## Market signals as a proxy for team news (F28) — mechanism CONFIRMED, decision REFUTED, 2026-10-22

**O40 as first written is NOT MEASURABLE and that is recorded before anything
else.** `chance_of_playing_next_round`, `status` and `news` are absent from the
historical file -- they exist only as a current snapshot of bootstrap-static.
There is no holdout data, so those flags cannot be gated on this project at
all. Anything built on them would be live-season-only, which R30 says is a
hypothesis generator and not evidence.

**What IS historical, at 100% coverage across all four seasons:** `selected`,
`transfers_in`, `transfers_out`, `transfers_balance`. These are the crowd's
REACTION to team news. When a player is injured, suspended, or has lost his
place, his owners sell him before the deadline. That is a contemporaneous,
recorded proxy for precisely the quantity S29 isolated as the whole of the
remaining headroom: near-future minutes and availability.

**The hazard, stated because it is the one that would invalidate everything.**
It could not be established whether `transfers_in[N]` is accumulated BEFORE
gameweek N's deadline or DURING gameweek N. Haaland 2026-27 GW2 carries
balance -34,247 while `selected` rose 521,891 from GW1, so the two do not
reconcile and the convention is genuinely ambiguous. **Every feature is
therefore built strictly lagged (`.shift(1)`), which is safe under BOTH
readings.** A truncation-equivalence run is a PRECONDITION, not a gate item.

**Prior that must be stated up front:** F23/M19 already tried ownership
features and cost -83 pts/season. That attempt used a POST-HOC snapshot applied
retroactively and was circular. This one uses strictly lagged recorded history.
Different construction, same family -- so the burden here is higher, not lower.

**Features, all `.shift(1)` or longer:**
`sold_share` = transfers_out/selected, `bought_share` = transfers_in/selected,
`net_share` = transfers_balance/selected, each at r1 and r3; `owned_share` =
selected/(season-gameweek total); `owned_change`.

**GATE (R24, R27, R2, R26, R30).**

1. **PRECONDITION.** `truncation_equivalence` must return max_abs_diff < 1e-9
   for every new feature. If any leaks, nothing is measured.
2. **MECHANISM.** The claim is specifically about MINUTES, so it is tested
   there: the minutes model with these features must improve correlation
   between predicted and actual minutes in **3/3** holdout seasons. A features
   set that helps points but not minutes would NOT confirm the S29 story and
   must not be reported as if it had.
3. **DECISION, the gate proper.** Season total must improve by **>= +100
   pts/season** on the 3-season mean. 3/3 -> SHIP. 2/3 -> REAL BUT NOT
   ESTABLISHED. <=1/3 -> REFUTED.
4. **NEGATIVE CONTROL (R26).** Shuffle the new features within (gameweek,
   position). Gates 2 and 3 must then fail. If scrambled market signals also
   help, the gain is extra columns, not information.

**Declared in advance:** if gate 2 passes and gate 3 fails, the conclusion is
**S29 confirmed, the proxy is too weak to pay** -- not a partial success.

---

## F28 / S30 — the information is real and it still costs 113 points

`fpl2/run_o40.py`. The pre-registration named this exact outcome in advance:
"if gate 2 passes and gate 3 fails, the conclusion is S29 confirmed, the proxy
is too weak to pay -- not a partial success."

| gate | result |
|---|---|
| 1 PRECONDITION leak | **PASS**. All 8 new features at max_abs_diff **0.000000000000**; whole 108-feature set clean. |
| 2 MECHANISM (minutes corr) | **PASS 3/3**: +0.0035 / +0.0021 / +0.0017. Control **0/3** at -0.0009 / -0.0004 / -0.0008. |
| 3 DECISION (season total) | **REFUTED**: -58 / -163 / -119, mean **-113.3**, **0/3**. |
| 4 CONTROL | **PASSES**: shuffled -126.7, 0/3 -- scrambled signals do not pay. |

**S30 — a signal can be real, correctly measured, properly controlled, and
still cost points.** The transfer market does carry information about who will
play: three seasons, same direction, with a scrambled control that lands BELOW
baseline rather than merely failing to beat it. That is as clean a mechanism
confirmation as this project has produced. It improves minutes correlation by
0.002 and loses 113 points a season.

This is the fourth time the pattern has appeared -- R9, W13, F19 and now F28 --
each improving the quantity it targeted and degrading the decision. The cost is
not mysterious: eight extra columns dilute a ranking that was already close to
its ceiling, and xP ranking is what the squad is built on. **Elevated to R31.**

**O40 as originally conceived is dead and the reason is recorded:**
`chance_of_playing_next_round`, `status` and `news` exist only as a current
snapshot. There is no history, so they can never be gated on this project.
Any model using them would be live-season-only, which R30 forbids as evidence.

**Shipped OFF.** `features.MARKET = False` is the default and the feature count
stays at 100. `data.py` now carries `selected` and the transfer columns through
the panel regardless -- that is inert for the model and is what made F28
measurable at all.

---

## Variance-aware minutes recency (F29) — REFUTED on BOTH gates, 2026-10-22

**S20 re-measured on the SHIPPED model (K=4.0), and it still holds.**

| | xp as-is | xp with ACTUAL minutes | actual | minutes |
|---|---|---|---|---|
| Haaland | 5.475 | **6.691** | 8.29 | 71.9 -> 88.4 |
| B.Fernandes | 5.938 | 6.394 | 4.71 | 83.6 -> 90.0 |

As shipped, Bruno leads by +0.463 and is captained. With actual minutes
Haaland leads by 0.297 and is captained. **Minutes alone move the gap by
-0.759 and flip the armband.**

**What is NOT fixable, established first so it cannot be chased later.**
Haaland's remaining +1.59 pts/gw is finishing over-performance, and finishing
does not persist: corr(goals/xG season N, N+1) = **+0.0353** over 160
consecutive player-seasons, against controls of +0.6787 for xG/90 and +0.5085
for minutes. Top decile 2.108 -> **1.061** the next season; bottom decile 0.522
-> 1.051; population 0.996. Haaland himself 2.06 -> **0.91**. A per-player
conversion factor would fit pure noise. **`CONVERT` stays per-position. This
branch is closed.** Two other candidates also died in diagnosis: the rate cap
does not bind for him (predicted xg/90 0.740 vs cap 0.827) and the top
predicted-xp decile is not under-predicted at all (ratio 1.037).

**The defect.** `minutes.RECENCY_K` blends toward `minutes_r3` with
`w = n/(n+K)`, which counts appearances and ignores how CONSISTENT they were.
Haaland's last five were 90,90,90,90,90 -- zero variance, total certainty --
and he is still weighted 0.64 at n=7, leaving the model's 65-ish prediction a
36% say. A rotated player with the same n gets the identical weight.

**The change.** `K_eff = RECENCY_K * (F + (1-F) * min(sd5/45, 1))` where sd5 is
the strictly lagged rolling sd of minutes over 5 appearances and F is
`RECENCY_SPREAD_FLOOR`. Consistent minutes shrink K and raise w; erratic
minutes leave today's behaviour. **F = 1.0 reproduces the shipped model
exactly.**

**GATE (R24, R27, R2, R26, R30, R31).**

1. **PRECONDITION.** `minutes_sd_r5` must pass truncation equivalence at
   < 1e-9. Nothing is measured if it leaks.
2. **CORRECTNESS.** F = 1.0 must reproduce 2143 / 2420 / 2149 exactly.
3. **MECHANISM, named in advance.** Haaland's predicted live-season minutes
   must rise toward 88.4 AND the live Bruno/Haaland ranking must flip. If the
   minutes rise without the flip, that is recorded as mechanism-only.
4. **DECISION, the gate proper.** Season total vs the SHIPPED K=4.0 baseline
   (not K=0): **>= +100 pts/season**, 3/3 -> SHIP; 2/3 -> REAL BUT NOT
   ESTABLISHED; <=1/3 -> REFUTED.
5. **CONTROL (R26).** Shuffle `minutes_sd_r5` within (gameweek, position).
   Gate 4 must then fail.
6. **SWEEP.** F in {1.0, 0.5, 0.25, 0.1}, whole curve reported, nothing
   selected on the run.

**Declared in advance, because R31 was written today after its fourth
occurrence:** improving Haaland's minutes is a COMPONENT result. If gate 3
passes and gate 4 fails, the conclusion is **mechanism confirmed, does not
pay** -- the same verdict as F28, not a partial success. The prior here is
poor: F20 tested this exact mechanism and reached only 2/3.

---

## F29 / S31 — S20 is a COLD-START error, and the recency blend cannot reach it

`fpl2/run_o41.py`.

| gate | result |
|---|---|
| 1 PRECONDITION leak | **PASS** -- `minutes_sd_r5` max_abs_diff 0.0 |
| 2 CORRECTNESS | **PASS** -- F=1.0 reproduces 2143 / 2420 / 2149 exactly |
| 3 MECHANISM | **FAIL** -- Haaland 71.9 -> **72.2** minutes at F=0.1. Armband stays on Bruno at every dose. |
| 4 DECISION | **REFUTED** -- -8.0 / -20.7 / -185.7, 1/3, 1/3, 0/3 |
| 5 CONTROL | PASSES -- shuffled sd -37.3, 1/3 |

**S31 -- the whole of Haaland's minutes error is COLD START, and `w = n/(n+K)`
is structurally incapable of touching it.**

| gw | actual | F=1.0 | F=0.1 | n |
|---|---|---|---|---|
| 1 | 90 | 59.1 | **59.1** | **0** |
| 2 | 90 | 57.4 | 57.1 | 1 |
| 3 | 90 | 62.4 | 62.0 | 2 |
| 4 | 90 | 80.2 | 80.3 | 3 |
| 7 | 90 | 80.1 | 81.8 | 6 |

At gameweek 1, `n = 0`, so `w = 0` for ANY `k_eff` -- the blend returns the raw
model by construction. By gameweek 4 the model and the recency signal already
agree within a couple of minutes, so there is nothing left to correct. The
error lives at 59/57/62 against 90 in the first three weeks and the mechanism
has no reach there. **This is not a tuning failure; the functional form cannot
express the fix.** F=0.1 also costs -185.7 pts/season by over-trusting three
appearances for everyone else.

**What this says about S20, which is now substantially CLOSED as unfixable by
this route.** Haaland's 45% gap decomposes into: cold-start minutes (no
mechanism in this family can reach it), ~10% xG rate, and ~41% finishing
over-performance that is PROVEN non-persistent (corr +0.0353 against controls
of +0.6787 and +0.5085; top decile 2.108 -> 1.061). The only remaining route is
a genuine cold-start prior -- carrying LAST season's minutes forward explicitly
rather than letting rolling windows that span the season boundary dilute it.
The model already holds `minutes_career` and cross-season `minutes_r5` and
evidently discounts them. **O42, unmeasured.**

**Kept, shipped OFF.** `RECENCY_SPREAD_FLOOR = 1.0` is the default and is the
shipped behaviour exactly, as gate 2 proves. `minutes_sd_r5` is computed but
deliberately NOT in the model feature list (still 100), so the fitted model is
untouched.

**Six pre-registered hypotheses refuted in one day: F24, F25, F26, F27, F28,
F29.** Three of them (F26, F28, F29) had a confirmed or partially confirmed
mechanism and still failed the decision gate -- see R31.

---

## F30 — O42 refuted in DIAGNOSIS, and S20 is now closed, 2026-10-22

First hypothesis killed today before any code was written. The premise was
"carry last season's minutes forward explicitly, because the rolling windows
dilute them at the season boundary". Measured, the premise is backwards.

**Haaland's window really was poisoned, and that part of S31 holds.** His last
five gameweeks of 2025-26 were `[180, 90, 90, 90, 0]`. The single trailing zero
pins `minutes_r3` at **60.00** through 2026-27 GW1, GW2 AND GW3 -- it only
clears once three new appearances roll it out at GW4, which is exactly where
the prediction jumps 62.4 -> 80.2. So the model reasoned correctly from bad
evidence, not stubbornly from good evidence.

**But the general case runs the other way, and that kills the fix.** Grouped by
LAST season's minutes-per-appearance -- fixed before the season starts, so
membership cannot move with the treatment (R29):

| 80+ band | n | actual | mean_r3 | ratio |
|---|---|---|---|---|
| GW1-3 | 443 | **68.2** | 75.4 | **1.105** |
| GW4-8 | 706 | 62.5 | 65.7 | 1.051 |
| GW9+ | 3229 | 60.3 | 60.6 | 1.005 |

Established starters genuinely play FEWER minutes in the opening three
gameweeks than any trailing statistic implies. **The model's cold-start
discount is correct on average.** Candidate replacements, ratio to actual at
GW1-3 in the 80+ band:

| statistic | ratio |
|---|---|
| `mean_r3` (what the model already uses) | **1.105** |
| median r3 (robust to a stale zero) | 1.147 |
| `mean_r5` | 1.186 |
| median r5 | 1.224 |
| **last season's mean -- O42's proposal** | **1.253** |

Every alternative is further from the truth, and `prev_season` degrades as the
season runs (1.253 -> 1.367 -> 1.415). **Nothing was built.**

### S20 is CLOSED. Six routes, all measured, all dead.

| route | why it fails |
|---|---|
| rate ceiling truncates him | does not bind: predicted xG/90 0.740 vs cap 0.827 |
| general elite under-prediction | does not exist: top predicted-xp decile ratio 1.037 |
| per-player finishing factor | finishing does not persist: corr **+0.0353** vs controls +0.6787, +0.5085 |
| variance-aware recency (F29) | `w = n/(n+K)` is 0 at gameweek 1 by construction |
| robust/median window | worse than `mean_r3` at cold start (1.147 vs 1.105) |
| last season's minutes prior (F30) | worst of all (1.253), and degrades with time |

**The residual is irreducible with the information in this pipeline.** Haaland
is under-rated because one stale zero poisoned his window, because he
out-scored his own xG by 41% in a way that provably does not persist, and
because established starters really are rested in August -- he just is not one
of them. The first is bad luck, the second is unpredictable, and the third is
the model being right about a population to which he is an exception.

**That exception is exactly the information gap S29 named.** A human knows
Haaland is not rotated. The model applies a population-average discount and
structurally cannot know he is exempt. This is the same wall from a third
direction, and it is now well enough mapped to stop walking into it.

---

## Coverage mask restored (M20 / F31) — SHIPPED on correctness, 2026-10-22

**A genuine data-integrity BUG, not a modelling idea.** Found by investigating
why the model over-predicts B.Fernandes's assists.

`apply_coverage_mask` deliberately NaNs expected-stats for 2022-23 GW1-32,
where xG/xA were never collected -- in `raw`, **45.2%** of 60+ minute rows in
that window are NaN against **0%** after GW32. `build_panel` then aggregates
with `.sum()`, and **pandas returns 0.0 for an all-NaN group**. The mask is
silently undone: in the panel, NaN is **0.0%** and ZERO is **53.1%** in GW1-32
against 23.2% after. Absent data re-enters as a genuine-looking zero.

This is the SAME failure already fixed once on this project, in
`external.py` (all-NaN sum -> 0.0, fixed with `min_count=1`). It recurred.

**Two distinct harms.**
1. `conversion_factors` divides real goals/assists by xG/xA that includes those
   false zeros, inflating every factor: MID xA->assists **1.459 -> 1.339**,
   MID xG->goals **1.124 -> 1.025**, FWD xG->goals **1.072 -> 0.988**, DEF
   xA->assists 1.328 -> 1.252. Unevenly by position, so it distorts exactly the
   cross-position comparison the squad optimiser runs on.
2. Worse: those false zeros are TRAINING TARGETS for the xG/xA rate models.
   2022-23 GW1-32 is roughly 21% of all training rows, teaching the model that
   players who generated xG generated none.

**The fix.** `min_count=1` on the summed columns in `build_panel`, so an
all-NaN group stays NaN.

**GATE -- and the decision rule is stated FIRST because it differs from every
other gate today.**

1. **CORRECTNESS is the ship criterion, not points.** The mask must survive
   into the panel: NaN share in 2022-23 GW1-32 must go from 0.0% to ~45%, and
   the GW33-38 share must stay 0%. Training on data known to be absent, as
   zeros, is a defect by definition. **This ships if correctness passes, even
   if the season total falls.**
2. **Points effect is REPORTED, not gated.** Three holdout seasons, against the
   current (buggy) baseline of 2143 / 2420 / 2149.
3. **A declared alarm, so it cannot be rationalised afterwards.** If the fix
   costs MORE than the noise floor (mean worse than -150/season), that is a
   surprising result and gets recorded as **UNEXPLAINED**, blocking the
   wildcard rebuild until understood. Removing false zeros should not cost 150
   points; if it does, something else is wrong.
4. **Known consequence, accepted in advance.** Every season total recorded
   before today shifts, and `policy="greedy"` will no longer reproduce
   2143 / 2420 / 2149. That invariant was protecting against ACCIDENTAL change;
   this change is deliberate. The new numbers become the baseline and the
   ledger says so.

---

## M20 — the coverage mask was being undone THREE times, 2026-10-22

**Found by the user challenging two squad picks**, not by any sweep. He asked
why the optimiser wanted Donnarumma over Verbruggen and why it kept buying
B.Fernandes. The second question led here.

`apply_coverage_mask` NaNs expected-stats for 2022-23 GW1-32, where xG/xA were
never collected (**45.2%** of 60+ minute rows in raw, against 0% after GW32).
Three separate places then filled those NaNs back in with zero:

| site | what it did | effect |
|---|---|---|
| `data.build_panel` | `.sum()` returns **0.0** for an all-NaN group | mask gone from the panel: NaN 0.0%, ZERO 53.1% in GW1-32 vs 23.2% after |
| `rates.build_targets` | `.fillna(0.0)` on every rate target | xG TRAINING TARGET 0% NaN, **70.1% zero** -- ~21% of all training rows teach that players who generated xG generated none |
| `rates.conversion_factors` | `.fillna(0)` on the denominator | real goals divided by an expected total containing false zeros |
| `rates.rate_caps` | same | sustained-rate ceilings deflated |

**This is the SAME bug already fixed once on this project, in `external.py`**
(all-NaN sum -> 0.0, fixed with `min_count=1`). It recurred because `.sum()`
reads as obviously correct. Fixed in all four places; `saves`, `bonus` and
cards are deliberately still filled with zero, because a missing value there
really is zero.

**Conversion factors, before -> after:**

| | before | after |
|---|---|---|
| MID xG->goals | 1.122 | **1.018** |
| MID xA->assists | 1.461 | **1.343** |
| FWD xG->goals | 1.072 | **0.975** |
| DEF xA->assists | 1.342 | 1.234 |

**F31 -- the points effect. SUPERSEDED by S36: the -121 is not attributable; see M22/S36.**

| | buggy | fixed | delta |
|---|---|---|---|
| 2023-24 | 2143 | 2180 | **+37** |
| 2024-25 | 2420 | 2168 | **-252** |
| 2025-26 | 2149 | 2001 | **-148** |
| mean | | | **-121.0**, 1/3 |

The pre-registered alarm was "worse than -150 mean". It did not fire, so by
the rule written before the numbers existed, **correctness governs and the fix
ships**. But this is R31 for the FIFTH time -- calibration improved markedly
(xp/actual **1.002 / 0.929 / 0.974**, much closer to 1.000) while points fell,
and 2024-25's -252 is well outside the 112-125 single-season floor. The honest
statement: the model was scoring better partly BECAUSE it was over-predicting
attacking returns, and that is not a property worth keeping.

**BASELINE CHANGE, recorded loudly.** Every season total before 2026-10-22 was
produced on the buggy panel. `policy="greedy"` no longer reproduces
2143 / 2420 / 2149; the new baseline is **2180 / 2168 / 2001**. Results
recorded earlier remain valid as comparisons WITHIN their own run, because
both arms shared the bug, but absolute totals are not comparable across the
fix. 81/81 tests pass.

**Consequence for the live wildcard: B.Fernandes drops out.** His xp falls
5.49 -> 5.08 and at GBP 11.8m he no longer earns the spend. Six of fifteen
change. The player all three pipelines over-rated (S26) was partly an artefact
of absent data filled with zeros.

**Still unfixed and still visible in the same squad:** Donnarumma remains the
top-ranked keeper at 4.93 on a 0.614 clean-sheet probability, which is the S18
tail where the model says 0.542 and reality delivers 0.371. S18 is a separate,
measured, open defect.

---

## O44 / O45 — two live wildcard objections raised by the user, 2026-10-22

Both came from the user looking at the GW8 wildcard and saying the picks were
strange. One has already produced a real bug (M20, via the B.Fernandes
question). These two are still open.

### O44 — the squad build ignores the upcoming fixture run

The GW8 wildcard was built on GW7's single-week xp. The justification given was
S29 (opponent identity carries no ranking value one to three weeks out) plus
the absence of any blank or double in GW8-12. **That justification was
over-extended and the user caught it.** S29 measured whether opponent identity
improves a FORECAST's aggregate spearman. A wildcard commits a squad against
five SPECIFIC opponents for five gameweeks. Those are different claims.

The case in point, FPL's own FDR over GW8-12:

| player | club | mean FDR | run |
|---|---|---|---|
| Calvert-Lewin | LEE | **2.80** | SUN(A) BOU(A) TOT(H) CHE(A) COV(H) |
| Evanilson | BOU | 3.00 | replaced him in the corrected build |
| Barry | EVE | **3.40** | ARS(A) NEW(A) COV(H) BRE(A) LIV(H) |

LEE has the joint-softest run in the league; the build dropped him for a player
on the second-hardest. GW7's xp encodes GW7's opponent and nothing about what
is coming.

**Test under way:** rebuild with `project.project` across GW8-12, substituting
each future week's real opponent and venue while freezing player form at GW7 --
the construction whose leak test passes at 0.000000000000 (F27). GW8-12
fixtures are appended to the team-fixture table from the published list with
lagged strength rewound to GW7. **If DCL returns, S29 does not transfer to
squad construction and the horizon must be fixture-aware. If he does not, a
2.80 FDR run genuinely does not outweigh the rate difference.** Either answer
is worth having.

### O45 — Donnarumma is selected by the S18 bias

The corrected model still ranks him the top keeper at xp 4.93 on a clean-sheet
probability of **0.614**. S18 measured that exact tail: predicted 0.542 against
a realised 0.371, ratio 1.46.

| | raw | S18-adjusted |
|---|---|---|
| Donnarumma (MCI, £5.5m, 7.7% owned) | 4.96 | **4.02** |
| Verbruggen (BHA, £4.6m, 22.6% owned) | 3.29 | **3.41** |

The gap collapses from +1.67 to +0.61 for £0.9m. Realised GW1-7: Donnarumma
**4.14/gw**, Verbruggen **5.43/gw**. Verbruggen also gains under the correction
because he sits in the low-probability region where the model UNDER-predicts.

**S18 is measured and unfixed.** Until it is, keeper selection inherits it, and
the practical advice is to override by hand. A principled fix means applying
the measured S18 curve inside `team.TeamModel.predict` rather than the flat
`CONCEDED_SHRINK` that was tried and shipped OFF (F19) -- but F19 is the
cautionary precedent: it improved calibration and cost points, which is R31.
**Not attempted. Would need its own pre-registered gate.**

---

## S32 — legacy's edge is in TRANSFERS, not squad building, 2026-10-22

`fpl2/run_wc_ab2.py`, re-run on the fixed panel (M20). The user asked for this
re-evaluation on the correct grounds: the fix changed OUR model and not legacy
-- legacy is a cached notebook forecast, and actuals were never masked, so the
two arms are no longer on equal footing with the earlier run.

Wildcard built at gameweek 8 of each holdout, played to gameweek 38, same
budget and same simulator.

| | 2023-24 | 2024-25 | 2025-26 | mean | positive |
|---|---|---|---|---|---|
| model - legacy, **opening squad only** (both on model transfers) | **+17** | -65 | -11 | **-19.7** | 1/3 |
| same, BEFORE the M20 fix | -45 | -54 | -30 | -43.0 | 0/3 |
| model - legacy, **end to end** (each on its own signal) | -65 | -146 | -39 | **-83.3** | 0/3 |

**Two results, and the second is the important one.**

1. **M20 narrowed the gap to legacy by +23/season** (-43.0 -> -19.7) even
   though it cost us 121 points in absolute terms (F31). The buggy conversion
   factors were flattering both arms unequally. A fix can cost absolute points
   and still improve standing; the absolute number alone would have read as a
   regression.

2. **Our opening squad is level with legacy's (-19.7, inside the noise floor)
   while the full pipeline loses by 83.** The gap is four times wider end to
   end than at squad construction. Legacy's advantage is therefore in its
   TRANSFER decisions, not in the squad it builds.

This converges with S27 from the opposite direction. S27 showed opening-squad
quality does not propagate to the season total under the greedy policy; S32
shows the surviving difference between two pipelines sits in the transfers. The
transfer policy is the open frontier, and it is where F26 (receding-horizon
MPC) aimed and failed -- S28 priced its missing ingredient at ~+138/season of
multi-week forecast quality, and F27 then showed fixtures are not that
ingredient.

**Caveat on comparability:** both arms run through OUR simulator and OUR
transfer rule in the first row, so "legacy's transfers" in the second row means
legacy's xP driving the same greedy rule, not the old notebook's own policy.
The distinction matters and is not controlled here.

---

## Fixture-aware wildcard (F32) — REFUTED, and the CONTROL beat the treatment, 2026-10-22

The live GW8 build changed 7 of 15 when fixtures were priced in, and the
mechanism checks out: corr(xp change, mean FDR) = **-0.2557**, with easy runs
gaining +0.078 and hard runs losing -0.064, monotone across bands. But a
mechanism that works is not a result (R31, four prior occurrences). Whether it
SCORES more is a separate question and has never been measured.

**The test.** At gameweek 8 of each holdout season, build a wildcard two ways
and play both to gameweek 38 under an identical simulator and budget:

* **single** -- GW7 xp alone, the original construction
* **fixture** -- `project.project` summed over GW8-12, each week with its real
  opponent and venue, player form frozen at GW7

**GATE (R24, R27, R2, R26, R30, R31).**

1. **PRIMARY.** fixture minus single, mean over 3 holdouts, must be
   **>= +100 pts/season**. 3/3 -> SHIP as the default build. 2/3 -> REAL BUT
   NOT ESTABLISHED. <=1/3 -> REFUTED, and the live squad reverts to whichever
   construction is better justified on other grounds.
2. **CONTROL (R26).** A third arm, **scrambled** -- the same projection
   machinery run with the opponent map permuted. If scrambling the opponents
   scores as well as the real map, the gain is the 5-gameweek AVERAGING, not
   the fixtures, and that distinction decides what to keep.
3. **Reported, not gated:** squad overlap between arms, and mean ownership,
   since the live fixture-aware squad came out at 15.8% with two sub-1%-owned
   players.

**Declared in advance:** S29 already found opponent identity carries no ranking
value 1-3 weeks out. If this gate fails, S29 extends to squad construction and
O44 is answered NO -- the user's DCL objection would then be right about the
specific player (he is back in the squad) and wrong about the mechanism paying.
Note the live evidence already hints at this: Calvert-Lewin's own xp moved
**+0.02**; he returned because Man City defenders fell 1.4-1.7, not because his
soft run was rewarded.

---

## F32 / S33 — fixtures are worth nothing; the five-week AVERAGING is worth ~+85

`fpl2/run_o44.py`, against the gate pre-registered above.

| | 2023-24 | 2024-25 | 2025-26 | mean | positive |
|---|---|---|---|---|---|
| fixture - single | -11 | +61 | -5 | **+15.0** | **1/3** |
| **scrambled** - single | +93 | +62 | +101 | **+85.3** | **3/3** |

**F32 REFUTED.** Required >= +100 and 3/3; got +15.0 and 1/3.

**S33 -- the control outscored the treatment, 3/3 against 1/3.** Permuting the
opponent map and running the identical five-gameweek projection machinery beats
the single-week build in every season and beats the REAL fixture map on the
mean by 70 points. So the benefit is variance reduction from averaging five
forecasts, and real opponent identity contributes nothing -- possibly less than
nothing.

This is S29 from a second, independent direction. S29 measured scrambled
opponents forecasting very slightly BETTER on spearman (0.6456 vs 0.6446) and
that looked like a rounding curiosity; here the same sign appears in realised
season points across three seasons. Two different instruments, same answer:
**this model's opponent-strength features carry no usable fixture signal.**

It also explains the stability problem found in the live audit -- only 3 of 15
players survived rebuilding on GW5/GW6/GW7 forecasts. A single gameweek's xp is
a noisy estimate, and averaging several is simply a better one.

**DISCIPLINE NOTE, recorded because the temptation here is obvious.** The
scrambled arm is a CONTROL. It scored +85.3 at 3/3, which would clear this
project's consistency bar but not its +100 magnitude bar. **It is not promoted
to a shipped recommendation on this run.** "Project forward five weeks with the
opponent neutralised" is now a hypothesis generated by a control, and promoting
a control after seeing its number is exactly the post-hoc reasoning that R30
and the six refutations of 2026-10-22 exist to prevent. It needs its own
pre-registered gate, with the averaging and the neutralisation separated so the
contribution of each is measured rather than assumed. **O46, unmeasured.**

**Consequence for the live GW8 wildcard:** there is now NO measured basis for
preferring the fixture-aware squad to the single-week one -- +15.0 at 1/3 is
noise. The two squads differ by 7 of 15. The user's two objections are both
satisfied by the fixture-aware build (Calvert-Lewin returns, Donnarumma drops
out), but the reason is not the one assumed: DCL's own xp moved **+0.02** and
he returned because Man City defenders fell 1.4-1.7. Choose on grounds other
than measured superiority, and say so.

---

## Transfer threshold (F33) — REFUTED on the gate; the ASYMMETRY is the finding, 2026-10-22

**Directly motivated by S32.** Our opening squad is level with legacy (-19.7,
inside the floor) while the full pipeline loses by **-83.3 (0/3)**. The gap is
four times wider end to end than at squad construction, so the loss is in the
weekly transfer decisions. `simulate_season` commits a swap whenever the
predicted gain clears `min_gain = 0.30` xp -- a threshold that has never been
measured, only assumed.

If the transfers are what bleed, being more conservative should help, and the
question has a directly actionable answer for a human following the model.

**The sweep.** `min_gain` over {0.30 (current), 0.5, 1.0, 1.5, 2.0} plus a
**NEVER-TRANSFER** arm (the opening squad held for 38 gameweeks). Whole curve
reported; no value selected on the run.

**The never-transfer arm is the informative one.** It is a control, not a
candidate: if holding the opening squad all season beats our actual policy,
then every transfer we make is destroying value, and the honest instruction to
a human is "build well and sit still".

**GATE (R24, R27, R2, R30, R31).**

1. **PRIMARY.** Best threshold must beat `min_gain = 0.30` by **>= +100
   pts/season** on the 3-season mean. 3/3 -> SHIP as the new default.
   2/3 -> REAL BUT NOT ESTABLISHED. <=1/3 -> REFUTED, keep 0.30.
2. **Monotonicity is reported, not required.** A single threshold winning with
   neighbours losing is noise (the lesson of S27's jagged bench_weight curve,
   roughness 1.56-2.30).
3. **The never-transfer control is reported against ALL arms.** If it wins,
   that is recorded as the headline regardless of what the sweep does.

**Declared in advance:** this is a POLICY parameter, not a model change, so a
win here does not mean the forecast improved. It would mean the forecast is too
noisy to act on weekly -- which is the same conclusion S27 (squad construction
does not propagate), S29 and S33 (fixtures carry no signal) keep pointing at
from different directions.

---

## M21 — the O48 design did not isolate what it claimed, 2026-10-22

**Correction recorded first.** F31 claimed the M20 fix "improved calibration
markedly", citing post-fix holdout ratios of 1.002 / 0.929 / 0.974. **No
pre-fix baseline was ever measured, so that claim was unfounded.** Measured
properly on 2026-27 GW1-7, buggy against fixed:

| | buggy | fixed |
|---|---|---|
| \|calibration - 1\| | **0.0388** | 0.0418 |
| spearman, all | **0.6656** | 0.6653 |
| GK / DEF / MID / FWD | .7124 / .5898 / .6820 / .6854 | .7155 / .5843 / .6828 / .6833 |

**M20 changes nothing measurable about forecast quality.** It costs 121
pts/season and moves the model from +22 ahead of legacy (F20) to **-99 behind**
it, and from -20 to **-141** against the FPL average proxy. The only surviving
argument for it is correctness in principle.

**M20 did TWO things at once and they were never separated:**
1. cleaned the xG/xA TRAINING TARGETS (~45% of 2022-23 rows dropped from the
   rate fit)
2. deflated the CONVERSION FACTORS (MID xG->goals 1.122 -> 1.018, etc.)

Either could carry the loss, and they imply opposite remedies. (1) losing is a
DATA-VOLUME problem; (2) losing means the inflated factors were compensating
for a real under-prediction elsewhere.

**Four arms, on a panel that always preserves NaN (`min_count=1` stays -- that
part is not in question):**

| arm | targets | conversion |
|---|---|---|
| `both_old` | polluted | inflated | (pre-M20 behaviour) |
| `both_new` | clean | honest | (current) |
| `targets_only` | clean | inflated | |
| `conversion_only` | polluted | honest | |

**GATE.**
1. The arm that recovers **>= +100** of the 121 points while keeping the panel's
   NaN handling correct is adopted, 3/3 required to ship, 2/3 recorded as real
   but not established.
2. If `both_old` is the only winner, M20's two halves are not separable and the
   decision returns to the user as a straight correctness-versus-points call.
3. **Declared in advance:** recovering points by restoring a known-false input
   is NOT a success. If `targets_only` or `conversion_only` wins, the ledger
   records WHICH false input is doing the work, because that is a finding about
   a compensating defect, not a fix.

---

## F33 / S34 — our forecast is too noisy to act on weekly; legacy's is not

`fpl2/run_o47.py`.

| min_gain | MODEL mean | vs 0.30 | pos | LEGACY mean | vs 0.30 | pos |
|---|---|---|---|---|---|---|
| **0.30** (default) | **2116** | - | - | **2215** | - | - |
| 0.5 | 2116 | +0.0 | 0/3 | 2176 | -39.3 | 0/3 |
| 1.0 | 2129 | +12.3 | 2/3 | 2187 | -28.3 | 1/3 |
| **1.5** | **2176** | **+59.3** | 2/3 | 2090 | -125.7 | 0/3 |
| 2.0 | 2169 | +52.7 | 2/3 | 2108 | -107.3 | 0/3 |
| NEVER | 1941 | **-175.3** | 1/3 | 2135 | -80.3 | 1/3 |

**F33 REFUTED** on the pre-registered gate: best arm is +59.3 against a +100
bar, and 2024-25 alone contributes +170 of it (-8 / +170 / +16), which is the
single-season-driver pattern that has misled this project repeatedly.

**The NEVER-TRANSFER control answers its question cleanly.** Holding the
opening squad for 38 gameweeks costs **-175.3**. Transfers add real value; the
weekly policy is not destroying it. That kills the tempting "build well and sit
still" reading of S27.

**S34 -- the two signals want OPPOSITE thresholds, and that is diagnostic.**
Legacy is best at 0.30 and degrades monotonically as the threshold rises; our
model is worst at 0.30 and improves to 1.5. Same simulator, same constraints,
same transfer rule -- only the forecast differs.

A forecast you can act on at a 0.30 xp difference is one whose small
differences carry signal. Ours does not: acting below ~1.5 xp is acting on
noise. **This is the same conclusion as S27 (squad construction does not
propagate), S29 and S33 (no fixture signal), and the live audit (3 of 15
players stable across rebuilds), reached a fifth time by a different route --
our xp is a decent RANKING of the big differences and noise at the small ones.**

It also sharpens S32. Legacy beats us by -83 end to end and only -19.7 on the
opening squad; the extra loss is our noisier forecast being acted on 36 times a
season at a threshold tuned for a better signal.

**Practical, and the one piece of today's work that is directly followable:**
anyone acting on this model weekly should require a LARGE predicted gain before
transferring. Not shipped as a default (it failed the gate), but recorded.

---

## M21 / S35 — the M20 loss is in the FEATURE encoding, not the targets

**M21: the O48 experiment was mis-designed and its own control proved it.**
The four arms were meant to separate M20's two halves by flipping
`rates.MASK_TARGETS` and `rates.MASK_CONVERSION`. But `data.build_panel` keeps
`min_count=1` in every arm, so the preserved NaN also reaches the ROLLING
FEATURES (`expected_goals_r3` and the rest), which neither switch controls.

The tell was built in: `both_old` was supposed to reproduce the pre-M20 model
and scored **2075** against the recorded **2237**. A 162-point discrepancy in
an arm whose only job was to reproduce a known number.

| arm | 2023-24 | 2024-25 | 2025-26 | mean | calib | spearman |
|---|---|---|---|---|---|---|
| both_old | 2090 | 2089 | 2047 | **2075** | 1.016 | 0.689 |
| both_new (current) | 2180 | 2168 | 2001 | 2116 | 0.968 | 0.689 |
| conversion_only | 2230 | 2043 | 2113 | 2129 | 0.956 | 0.688 |
| targets_only | 2137 | 2260 | 2019 | 2139 | 1.033 | 0.690 |
| legacy | | | | **2215** | | |

**Lesson, and the reason an arm that merely reproduces a known number is worth
its compute:** without `both_old` as a reproduction check, the other three arms
would have been read as a clean decomposition of M20 and they are not. Spearman
is **0.688-0.690 across all four** -- every arm forecasts identically well, and
only the realised total moves, which is R31's signature yet again.

**S35 -- decomposition by difference:**

| | mean | change |
|---|---|---|
| true pre-M20 (zeros everywhere) | 2237 | |
| both_old (NaN reaches features) | 2075 | **-162** |
| both_new (+ targets and conversion masked) | 2116 | **+41** |

**The half of M20 that was deliberately fixed is worth +41.** The -162 is an
unintended side effect: NaN reaching the features, where the GLM's
`SimpleImputer(strategy="median")` substitutes a MIDDLING attacking threat for
a player whose threat was never recorded. Zero is also wrong, but it is a far
better stand-in for "nothing recorded" than the median of players who were
measured.

**O48b (running):** keep targets and conversion correctly masked, and zero-fill
the absent expected-stat FEATURES only. If that recovers the 162 while keeping
the +41, the correct configuration is mask-the-targets, zero-the-features, and
the model lands near 2278 -- which would put it ahead of legacy's 2215 for the
first time honestly. **If it does not, S35's attribution is wrong too and the
whole M20 question goes back to the user as a correctness-versus-points call.**

---

## M22 / S36 — the instrument cannot resolve the M20 question, 2026-10-22

**M22: the same design error twice in an hour.** O48's switches did not control
the FEATURES; O48b's zero-fill ran UPSTREAM of the targets, so
`MASK_TARGETS=True` had nothing left to mask and that arm silently reproduced
plain pre-M20 -- **2143 / 2420 / 2149, bit-identical**. Both times the
experiment measured something other than what was claimed. Both times the only
thing that caught it was an arm whose job was to reproduce a KNOWN number and
didn't. **Every decomposition design gets a reproduction arm from now on.**

**O48c finally ran the intended configuration** -- targets and conversion
correctly masked (verified: MID xG->goals 1.018, the honest value), rolling
features zero-filled (raw stays 45.1% NaN, rolling drops to 3.6%).

| configuration | mean | spearman |
|---|---|---|
| zeros everywhere (pre-M20, known-false) | **2237** | 0.689 |
| legacy | 2215 | - |
| NaN everywhere (M20 as shipped) | 2116 | 0.689 |
| masked targets + zeroed features (O48c) | **2068** | 0.689 |

**The more correct the data handling, the lower the score, monotonically.
There is no third option.**

**S36 -- and the reason that ranking must not be acted on.** Spearman is
**0.689 in every variant**; per-position 0.680 / 0.675 / 0.711 against 0.689.
**These configurations are indistinguishable as forecasts.** Only the simulated
totals move, and they move chaotically:

| | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|
| pre-M20 | 2143 | 2420 | 2149 |
| M20 | 2180 | 2168 | 2001 |
| O48c | 2166 | 2095 | 1944 |
| **range** | **37** | **325** | **205** |

2023-24 barely moves; the other two swing 325 and 205 from an encoding choice,
against a 112-125 single-season floor. This is S27 again (bench_weight produced
a 338-point range within one season). With n = 3 and swings of that size, a
mean difference of 169 is not attributable to the encoding -- it is mostly
which arm forked well in 2024-25.

**CONCLUSION, and it decides M20.** The decision metric cannot resolve this
question, and selecting the configuration with the best season total would be
selecting on noise -- the exact failure R24, R30 and six refutations today were
built to prevent. **So the choice is made on correctness, because that is the
only axis on which these variants actually differ.** M20 stays shipped as
`both_new` (NaN preserved end to end, `ZERO_FILL_ABSENT_EXPECTED = False`).

**The honest cost statement: we do not know whether M20 costs points.** F31's
"-121 pts/season" is now withdrawn as unattributable. What is established is
that the pre-M20 model's apparent +22 edge over legacy rested on an encoding
the instrument cannot evaluate, so **no claim that this model beats legacy
survives** -- in either direction.

---

## S37 — a calibrated measuring instrument, 2026-10-22

`fpl2/fpl2/measure.py` + `fpl2/run_calibrate.py`. Built because S36 established
that the project's only decision metric could not resolve the questions being
asked of it.

**The diagnosis.** Season total through the simulator mixes four things:
the effect under test, the opening squad each arm happens to draw, the 38-week
transfer path forking from it, and the interaction of all three. Only the first
is wanted. Measured consequences: a data-ENCODING choice moved one season 325
points (S36) and `bench_weight` moved one season 338 (S27), against a
single-season floor of 112-125 and a 3-season-mean sd of 39-49.

**Metric 1, path-free.** Twelve neutral squads per season -- built from
prior-form points with multiplicative jitter, so they are realistic, diverse,
and IDENTICAL for every arm, meaning no arm is measured on a squad that
favours it. Each arm then makes only the XI and captain choices from those
fixed squads, scored on realised points. The squad never changes, so nothing
forks. n = 12 x 38 x 3 = **1,368 paired observations** instead of 3.

It is still a DECISION metric -- picking an XI and an armband is a real
decision scored on real points, not a component statistic like spearman -- so
it satisfies R31 while being affordable, which R31 and R24 together have never
been on this project.

**CALIBRATION, and one of the three tests was worthless.**

| test | result |
|---|---|
| NULL (arm vs ITSELF) | 0.000 exactly -- **degenerate**, `A vs A.copy()` is zero by construction. It validates the pairing and merge plumbing and NOTHING about noise. The MDE of 0.0 and the "123 billion x improvement" printed from it are artefacts and are discarded. |
| NULL' (same arm, DIFFERENT squads) | 2124 vs 2134 season-equivalent: squad choice alone moves **11 pts**, against 325 on the old instrument |
| SIGNAL (forecast degraded with noise) | 10% -> **+23.1**, 25% -> **+34.1**, 50% -> **+82.4** season-equivalent, t = 4.74 / 4.93 / 9.47. Monotone, all detected, correct sign. |

**Resolution, computed from a REAL comparison rather than the degenerate null:**

| | MDE (season-equivalent pts) |
|---|---|
| old instrument (3 season totals) | **123** |
| **new instrument (metric 1)** | **13.6** |
| improvement | **9.0x** |

**What it CANNOT see, by construction.** It fixes the squad and the path, so
squad-construction changes (F25 `bench_weight`, F32 fixture-aware) and
transfer-policy changes (F26 MPC, F33 threshold) are invisible to it. Those
need metric 2 -- `measure.restart_score`, season totals averaged over K
opening squads -- which is written but **NOT YET CALIBRATED and must not be
used until it is.**

**Lesson for every future instrument: a null test comparing a thing to a copy
of itself proves nothing.** The informative null is the same arm measured
through an independent draw of the nuisance variable, which is what NULL'
does. Write that one first.

---

## S38 — metric 2 calibrated; the old instrument's restart sd was 102 points

`fpl2/run_calibrate2.py`. Season totals WITH transfers, 20 restarts x 3
seasons, paired on restart id.

| test | result |
|---|---|
| NULL' (same arm, different restart squads) | 2143 vs 2130 -> restart choice alone moves **14 pts** |
| **within-arm sd across restarts** | **102 pts** |
| SIGNAL 10% noise | +58.4, CI [+32.7, +84.0], t 4.45 |
| SIGNAL 25% noise | +110.3, CI [+82.2, +138.5], t 7.69 |
| SIGNAL 50% noise | +233.9, CI [+191.2, +276.5], t 10.75 |

**MDE 36.7 pts/season** -- worse than metric 1's 13.6 (n = 60 against 1,368)
but **3.3x finer than the old 123**, and it SEES TRANSFERS, which metric 1
cannot. F25, F32, F26 and F33 are re-testable on it.

**The 102 is the number that explains this project's history.** Every season
total ever recorded on the old instrument was a SINGLE DRAW from a distribution
with that spread. Two arms differing by 100 points were routinely two draws of
the same thing. R1's "noise floor 112-125" was measuring exactly this and the
project treated it as a threshold to clear rather than as evidence the
instrument needed rebuilding.

## S39 — XGBoost is level with the GLM; S12's -194 was ~9x inflated

`fpl2/run_xgb.py`, metric 1 (MDE 13.6), identical fixed squads, n = 1,368
paired.

| arm | mean/gw | vs GLM, season-equivalent | 95% CI | t |
|---|---|---|---|---|
| glm | 55.887 | reference | | |
| hgb (shipped GBDT config) | 55.326 | **-21.3** | [-32.6, -10.0] | **-3.70** |
| xgb (depth 5) | 55.779 | -4.1 | [-14.7, +6.5] | -0.76 |
| xgb_shallow (depth 3, reg_lambda 5) | 55.992 | **+4.0** | [-5.2, +13.2] | 0.85 |

**S12 is right in DIRECTION and wrong in MAGNITUDE by about 9x.** It recorded
"the GBDT is actively hurting; a Poisson GLM beats it by ~194 pts/season". The
real figure for that configuration is **-21.3** -- statistically solid
(CI excludes zero) and practically small.

**And it is not a property of boosted trees.** XGBoost at depth 5 is
indistinguishable from the GLM, and the shallow heavily-regularised variant is
nominally ahead. The mechanism S12 proposed (the GBDT overfits these features)
is supported -- deeper is worse, shallower is better, monotonically -- but the
remedy is regularisation, not a change of model family.

**What this cost.** The project moved to a GLM on the strength of a 194-point
number that was mostly instrument noise. Nothing was gained or lost by the
move, because the families are equivalent once regularised; but it was decided
on a measurement that could not support the decision. **No action is proposed
on this today -- the arms are inside each other's intervals, so switching back
would be selecting on noise exactly as the original switch was.**

---

## S40 — nine refuted experiments re-tested on calibrated instruments, 2026-10-22

Every experiment below was REFUTED by the old instrument (MDE 123). Re-run on
metric 1 (MDE 13.6, path-free, n=1368) or metric 2 (MDE 36.7, with transfers,
n=24-60), all paired.

### Forecast arms, metric 1 — six nulls and one marginal

| arm | season-eq | 95% CI | t | verdict |
|---|---|---|---|---|
| F28 market signals | +6.1 | [-1.0, +13.2] | 1.68 | undecided -> see below |
| F29 spread 0.25 | -0.6 | [-2.1, +0.9] | -0.80 | null |
| F29 spread 0.10 | -0.4 | [-2.3, +1.5] | -0.40 | null |
| M20 zeros | **+0.4** | [-6.4, +7.3] | 0.12 | **null** |
| recency off | -2.1 | [-8.7, +4.5] | -0.63 | null |
| recency K=2 | -2.6 | [-4.5, -0.8] | -2.80 | worse (multiplicity: ~0.4 expected false positives over 8 arms) |
| recency K=8 | -0.7 | [-3.3, +1.9] | -0.52 | null |

**M20 at +0.4, bounded to +/-7.** The encoding question that consumed O48,
O48b and O48c and produced a 169-point spread was ZERO. S36 inferred fork
noise; this confirms it. Keeping M20 on correctness grounds was right.

### Squad and policy arms, metric 2

| test | old verdict | metric 2 | CI | t |
|---|---|---|---|---|
| F33 threshold | refuted, best +59 | **+3.4** | [-9.4, +16.1] | 0.52 |
| F25 bench_weight | refuted | -13.9 to -21.8 | all straddle 0 | |
| F32 fixture-aware | refuted, control +85 | -32.7 / -22.1 | both straddle 0 | |
| **F26 MPC H=2** | **refuted, -64.3** | **+78.5** | **[+29.5, +127.5]** | **3.14** |
| F26 MPC H=4 | | +56.7 | [+10.7, +102.7] | 2.42 |

### S41 — F28 resolved at higher power: the market signal is REAL

K=40 squads, n=4560. **+5.7 season-equivalent, CI [+1.7, +9.7], t=2.77.** The
point estimate barely moved from K=12 (+6.1); the interval tightened enough to
exclude zero. Small, but it is the one proxy with any history for what S29
isolated as the entire remaining headroom. The old instrument scored it -113.

### S42 — model vs legacy: THE GAP DOES NOT EXIST

| comparison | model - legacy | CI | t |
|---|---|---|---|
| shared squads | **+0.7** | [-28.1, +29.6] | 0.05 |
| each builds its own | **+7.8** | [-22.2, +37.8] | 0.51 |

Both null, model nominally ahead in both. **S32 is RETRACTED** -- "legacy beats
us -83 end to end, the gap is in the transfers" was a single draw from a
distribution with a 102-point restart sd. There is no gap to locate.

### S43 — CHIPS ARE WORTH +196.5, NOT +87

**+196.5, CI [+178.1, +214.8], t = 20.98.** The strongest result this project
has produced. Not marginal, not multiplicity-vulnerable, not carried by one
season. S25's +87 was less than half the truth.

**This is the only number today that grew under better measurement.** Everything
else shrank or vanished. For practical play it dominates: squad construction,
transfer thresholds, bench weighting, fixture awareness, model family and the
encoding question are collectively worth 0-25 points; chips are worth ~200.

## M23 — FOUR findings retracted today, all the same shape

| retracted | was | is |
|---|---|---|
| S12 magnitude | GBDT -194 | **-21.3** (direction right, 9x inflated) |
| S32 | legacy +83 end to end | **no gap** |
| S33 | scrambled fixtures +85.3, 3/3 | **-22.1**, null |
| S34 | legacy wants 0.30, ours 1.5 | **threshold has no effect** |

Every one was a single draw from a 102-point distribution, reported as a
measurement. S33 and S34 were called "the sharpest result of the day" and "the
single most valuable result" when recorded. **R1's "noise floor 112-125" was
measuring this exact sd and the project treated it as a bar to clear instead of
as evidence the instrument was unfit.** That single misreading shaped the whole
experimental program.

---

## S44 — F26 confirmed at K=20, but 2/3 not 3/3, 2026-10-22

`fpl2/run_f26_confirm.py`. The sign flip from the original -64.3 had TWO
possible causes -- the instrument, and my changing the squad construction to
shared neutral squads. Only the first is a legitimate improvement, so both
regimes were run at K=20.

| regime | arm | mean | 95% CI | t |
|---|---|---|---|---|
| neutral | mpc_H2 | **+52.3** | [+23.2, +81.4] | 3.52 |
| neutral | mpc_H4 | +36.9 | [+11.3, +62.6] | 2.82 |
| **model_xp** (original design) | mpc_H2 | **+30.9** | [+5.0, +56.7] | 2.34 |

**The design-change hypothesis is dead: BOTH regimes show MPC ahead.** The flip
is the instrument, not my test setup.

**But the per-season breakdown is the single-season-driver pattern again:**

| | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|
| neutral | +13.8 | **+145.8** | -2.8 |
| model_xp | -13.7 | **+95.3** | +10.9 |

2024-25 carries it in both regimes; the other two seasons are at or below zero.
**2/3 by R2's standard, not 3/3.** That is the same shape as F33's +59, S33's
+85 and S34's asymmetry -- all retracted today. The pooled t is significant
because n=60 pools three seasons of which one has the effect.

The point estimate also shrank with power: **+78.5 at K=8 -> +52.3 at K=20**,
regression toward truth as more restarts are added.

**Verdict: REAL BUT NOT ESTABLISHED. Not shipped.** `policy="greedy"` remains
the default. What would settle it is more SEASONS, not more restarts -- the
variance that matters now is between seasons, and restarts cannot reduce it.
**O49: extend the holdout set beyond three seasons. That is the binding
constraint on every remaining question.**

---

## M24 / S45 — today's t-values ignored SEASON CLUSTERING. Two rescues withdrawn.

**M24, and it applies to everything reported on 2026-10-22.** Every paired test
today treated its observations as independent: n=60 for metric 2, n=1368 for
metric 1. They are not independent -- they are clustered in **three seasons**,
and F26's own per-season breakdown (+145.8 / +13.8 / -2.8) proved the effect
varies by cluster. The naive standard error is therefore too small and t too
large.

Cluster-robust variance for a mean over G clusters:
`Var = (G/(G-1)) * (1/n^2) * sum_g [sum_{i in g}(d_i - dbar)]^2`, evaluated on
**G-1 = 2 degrees of freedom**, where |t| must exceed **4.30** for p < 0.05.

| result | mean | naive t | **clustered t** | p | cluster 95% CI |
|---|---|---|---|---|---|
| **CHIPS (S43)** | **196.5** | 20.98 | **7.81** | **0.016** | [+88.2, +304.8] |
| model-legacy shared | 0.7 | 0.05 | 0.03 | 0.981 | [-114.0, +115.5] |
| model-legacy own | 7.8 | 0.51 | 0.16 | 0.885 | [-197.0, +212.6] |
| F26 mpc_H2 neutral | 52.3 | 3.52 | **1.11** | **0.382** | [-149.9, +254.4] |
| F26 mpc_H2 model_xp | 30.9 | 2.34 | **0.93** | **0.449** | [-111.1, +172.8] |
| F28 market K=40 | 5.7 | 2.77 | **1.38** | **0.302** | [-12.0, +23.3] |

**S44 and S41 are WITHDRAWN.** F26 and F28 were reported as rescued on naive
t-values. Clustered, neither is established. S44 already flagged F26 as 2/3 by
season; clustering is the formal statement of that same objection, and it
applies to F28 too.

**S43 (chips) SURVIVES** at p = 0.016. It is the ONLY finding of 2026-10-22
that withstands proper analysis.

**S42 (no model-legacy gap) is unaffected** -- it was null naively and is null
clustered, now with an honest interval.

### What this says about the instrument work

Metric 1 and metric 2 did exactly what they were built for: they removed the
WITHIN-season fork noise, taking the MDE from 123 to 13.6. That is real and the
calibration holds. **But pairing cannot manufacture between-season
replication.** With G = 3 the bar is |t| > 4.30, which only an enormous effect
clears -- and chips is the only candidate we have found.

**O49 is therefore not merely the next priority, it is the binding constraint
on every remaining question**, and it is BLOCKED: xG/xA begin at 2022-23 GW16
(64.1% of that season, 0% before it), and the rate model targets those
quantities. Extending the holdout set backwards requires an expected-stats
source for older seasons, not just more FPL rows.

**Standing rule, R32: report cluster-robust statistics whenever observations
are grouped by season, which on this project is always.** The naive SE answers
"would this arm win again on these three seasons", which nobody is asking. The
clustered one answers "would it win on a new season", which is the only
question that matters.

---

## S46 / M25 — ten seasons of data, and three process failures getting there, 2026-10-22

### S46 — the holdout set goes from 3 to 9

`fpl2/fpl2/legacy_seasons.py` now adapts **2016-17 .. 2021-22** (140,308 rows,
six seasons, each with 20 teams, 38 gameweeks and **zero null join keys**) from
the public FPL exports, joined by an EXACT `element -> code` map out of
`players_raw.csv`. No name matching, so the Mendy class of error cannot occur
on this path. With the shipped file that is **ten seasons, 2016-17 .. 2025-26**.

| | before | after |
|---|---|---|
| seasons | 4 | **10** |
| usable holdouts | 3 | **9** |
| cluster bar (R32) | \|t\| > 4.30 | **~2.31** |

That is the constraint M24/S45 identified as binding on every remaining
question, substantially relieved.

**Understat** supplies per-match xG back to 2014-15 and is joined by
`fpl2/fpl2/understat.py`: same date, same club, assignment not lookup, with
minutes agreement in the score. Measured on 2022-23 where both sources exist
(n=7,217): **xG ratio 1.030, spearman 0.9910** -- the same quantity. **xA ratio
1.231, spearman 0.690 -- a DIFFERENT quantity**, stable across three
independent join constructions, so it is the source and not the matching.

**FBref is NOT a usable alternative.** `soccerdata` exposes only
`standard / keeper / shooting / playing_time / misc` for it, there is no
`passing` table, and even `shooting` returns zero xG columns. The Expected
groups are unreachable. The only working FBref path on this project remains the
manual paste in `external.py`.

### M25 — three process failures in one stretch

**1. A 404 written to disk and parsed as data.** `teams.csv` does not exist for
2016-17..2018-19. `curl -sS -o file URL` wrote the body -- the literal string
`404: Not Found` -- into each file, which pandas then parsed as a one-column
frame. The adapter failed with `KeyError: 'id'`, naming nothing useful.
**A file existing is not a file being real: check the status, not the path.**
Resolved by dropping teams.csv entirely and taking team identity from
`players_raw`'s `team_code`, which is stable across seasons.

**2. Module state used as a loop variable.** The validation script assigned
`ls.SEASONS = (s,)` inside a loop, so a later `load_all()` saw a 1-tuple.

**3. A self-comparison that returned ratio 1.0000, correlation 1.0000.** The
FBref check auto-detected "xG-like" columns and matched `opta_xg` -- a column
of MY OWN aggregate -- so it compared FPL against FPL. This is R14 again. It
was caught only because perfect agreement between two providers is impossible;
the diagnostic had already printed `relevant cols: ['team','player']`, stating
plainly that no FBref xG column existed, and that signal was walked past.
**When a sanity line says the thing you need is absent, stop.**

### Deliberately NOT guessed

Six `team_code`s (25, 38, 80, 88, 97, 110) belong to clubs relegated before
2022-23 and never promoted since, so no source we hold names them. Season
patterns narrow them to pairs -- 25 and 88 both appear in 2016-17 only -- but
do not resolve them. They are labelled **`T<code>`** rather than guessed. The
label only has to be consistent; the Understat join matches clubs by
fixture-date pattern, not by name.

---

## S47 — Understat features vs FPL features: NO EFFECT, and the reason matters

`fpl2/run_us_features.py`. The best-powered test this project has run.

| | today's earlier tests | this one |
|---|---|---|
| holdout seasons (G) | 3 | **8** |
| cluster bar (R32) | \|t\| > 4.30 | **2.36** |
| paired observations | 1,368 | **3,030** |

Arms differ ONLY in the feature source. `fpl` uses Opta `expected_goals`;
`understat` uses `us_xg` across all ten seasons plus shots, key passes,
`xg_chain` and `xg_buildup`, which have no FPL equivalent (100 features
against 128).

| | mean | t | p | 95% CI |
|---|---|---|---|---|
| naive | +1.6 | 0.68 | | |
| **clustered** | **+1.6** | **0.70** | **0.509** | **[-3.7, +6.8]** |

**Not significant, and bounded.** This is not a failure to detect: the effect is
confined to a few points either way.

**Per season, and the pattern is the one that has burned this project all day:**

| | | | |
|---|---|---|---|
| 2018-19 | -0.8 | 2022-23 | +2.4 |
| 2019-20 | -1.9 | **2023-24** | **+16.5** |
| 2020-21 | +1.6 | 2024-25 | -3.1 |
| 2021-22 | -1.5 | 2025-26 | -0.7 |

Seven seasons inside +/-3.1 and one at +16.5. **At G=3 that single season could
have produced a "significant" result; at G=8 it cannot.** This is the first
occasion on this project where the extra clusters EARNED their keep by
preventing a false positive rather than by retracting one after the fact.

### The genuinely informative part

**In 2018-19 .. 2021-22 the FPL arm has NO xG AT ALL** -- Opta figures begin in
2022-23. Four seasons where that arm is entirely blind on the attacking side,
and the differences are **-0.8, -1.9, +1.6, -1.5**.

**Giving the model xG where it previously had none is worth nothing
measurable.** That is a far stronger statement than "Understat approximates
Opta well", and it converges with everything else measured: S27 (squad
construction does not propagate), S29 and S33 (fixtures carry no ranking
signal), and the seven forecast nulls of the re-run. **The xG component -- the
quantity this entire component architecture is organised around -- contributes
less than the noise floor.**

`USE_UNDERSTAT_FEATURES` stays **False**. The Understat pipeline is kept: it is
what makes ten seasons usable, and ten seasons is what made this test decisive.

---

## S48 — COMPONENT ABLATION: the model is a minutes predictor, 2026-10-22

`fpl2/run_ablation.py`. Metric 1, 8 holdout seasons, season-clustered (bar
|t| > 2.36). Each component replaced by the per-position mean of the REALISED
quantity over the TRAINING seasons only, so an ablation cannot leak.

| pooled arm | mean | t | p | 95% CI | verdict |
|---|---|---|---|---|---|
| no_goals | **+14.6** | 1.49 | 0.180 | [-8.6, +37.9] | no effect |
| no_assists | +2.4 | 0.88 | 0.406 | [-4.0, +8.8] | no effect |
| no_bonus | -4.6 | -3.17 | 0.016 | [-8.0, -1.2] | suggestive |
| no_saves | -0.3 | -0.34 | 0.745 | [-2.7, +2.0] | no effect |
| no_cleansheet | -9.0 | -1.10 | 0.308 | [-28.2, +10.3] | no effect |
| no_defret | -0.1 | -1.00 | 0.350 | [-0.5, +0.2] | no effect |
| no_minutes | -19.2 | -0.91 | 0.392 | [-69.0, +30.6] | no effect |
| **minutes_only** | **+0.3** | **0.02** | **0.988** | [-44.2, +44.8] | **no effect** |

**`minutes_only` is the headline. Flatten goals, assists, saves, clean sheets,
bonus and defensive returns -- keep ONLY the minutes hierarchy -- and the model
performs identically (+0.3, p = 0.99).** The entire rate architecture
contributes nothing measurable to the decision.

### Per position -- and this is what rescued the result

| component | GK | DEF | MID | FWD |
|---|---|---|---|---|
| no_goals | -0.1 | -1.4 | +4.6 | **+19.1** |
| no_assists | -0.1 | -1.7 | +1.8 | +3.4 |
| no_bonus | -0.1 | -1.7 | -1.3 | +1.2 |
| no_saves | -0.3 | 0.0 | 0.0 | 0.0 |
| no_cleansheet | +5.0 | -11.0 | -1.4 | 0.0 |
| no_defret | 0.0 | -1.2 | +0.5 | +0.7 |
| **no_minutes** | **+12.1** | -5.7 | **-37.8*** | **-31.1*** |

**The pooled `no_minutes` arm reads "no effect" at p=0.39 and that is an
AGGREGATION ARTEFACT.** Minutes matter enormously for midfielders and forwards
(-37.8 and -31.1, both significant) and NEGATIVELY for goalkeepers (+12.1) --
keepers play 90 or 0, so there is nothing to predict and modelling it adds
noise. Averaging those gives zero. **The per-position split was the user's
suggestion; without it the single real component in the model would have been
recorded as contributing nothing.**

**The goal model is ACTIVELY HARMFUL for forwards: +19.1 from removing it.**
Replacing each forward's predicted goal rate with the position average IMPROVES
XI selection. This is S47 from a second direction -- the xG component is not
merely neutral, it is counterproductive exactly where it is meant to work
hardest. S20's Haaland case was a symptom of this, not an exception to it.

**Multiplicity, stated plainly:** 28 per-position cells at alpha=0.05 expect
~1.4 false positives; two are starred. Both are the SAME component in the two
positions where minutes genuinely vary, which is a coherent mechanism rather
than scattered noise. `no_bonus` at p=0.016 does NOT survive Bonferroni across
8 pooled tests (0.05/8 = 0.006) and is recorded as suggestive only.

### What this means for the project

Months went into the rate layer -- conversion factors, ceilings, Poisson GLM
versus GBDT, per-position fits, xG sourcing. **S48 says none of it reaches the
decision.** It also explains every null of 2026-10-22: F24-F33 were tuning
components that do not propagate, which is why mechanisms kept confirming while
points never moved (R31, five occurrences).

The honest reconstruction of what this model is: **a minutes predictor for
midfielders and forwards, plus FPL's scoring rules.** Everything else is
decoration, and the goal-rate layer is worse than decoration for forwards.

**O50:** build the minimal model -- minutes hierarchy plus position-mean rates
-- and test it against the full one on metric 1 and metric 2. If it matches, it
is simpler, faster, and has far less surface area for the kind of data bug M20
turned out to be.

---

## S49 — the rate layer matters for TRANSFERS, not for line-ups, 2026-10-22

`fpl2/run_o50.py`. `ComponentModel(mode=...)` -- "full" (shipped), "minimal"
(rates -> per-position constants, team model kept), "strict" (clean sheets
flattened too, == S48's `minutes_only`). The rate layer is NEVER FITTED in the
latter two, which is a stronger construction than S48's ablation of a trained
model's outputs.

| | metric 1 (XI+captain, path-free) | metric 2 (with transfers) |
|---|---|---|
| minimal | **-11.8**, t -0.65, p 0.535 | **-61.2**, t -2.41, **p 0.047** |
| strict | **-1.4**, t -0.08, p 0.942 | **-66.1**, t -2.45, **p 0.044** |
| fit time | 210s -> 78s, **2.7x faster** | |

**The two instruments disagree, and the disagreement is the finding.** Metric 1
fixes the squad and measures only XI and captain choice; metric 2 lets the
model make 38 gameweeks of transfer decisions. **The rate layer contributes
nothing to picking an XI from a given squad, and does contribute to deciding
which players to own.**

**S48's headline is therefore QUALIFIED.** "The model is a minutes predictor;
the rate layer contributes nothing" is true FOR LINE-UP SELECTION and false for
transfers. S48 only ever tested metric 1 -- a limitation stated when it was
built, and the reason metric 2 was included here.

It also re-reads S48's most striking cell. "+19.1 from deleting the forward
goal model" is a statement about XI CHOICE. The forward goal model may still
earn its place when deciding who to buy, which is a different decision and the
one metric 2 covers.

**Significance, stated honestly:** p = 0.047 and 0.044 against alpha = 0.05,
with CIs reaching -1.2 and -2.4. Marginal. Neither survives Bonferroni across
the four tests here (0.05/4 = 0.0125). Recorded as **SUGGESTIVE, not
established**.

**NOT SHIPPED.** `mode="full"` remains the default. The 2.7x speedup and the
reduced surface area are real -- `rates.py` is where M20 hid in three separate
places -- but a consistent -61 to -66 on the instrument that includes transfers
is the wrong thing to trade for convenience.

---

## S50 — chips CONFIRMED on eight seasons, and S43's magnitude corrected, 2026-10-23

`fpl2/run_chips10.py`. Metric 2, 15 restarts x 8 seasons, paired on squad,
season-clustered.

| | mean | t | p | 95% CI |
|---|---|---|---|---|
| naive | +119.8 | 10.26 | | |
| **clustered** | **+119.8** | **3.04** | **0.0188** | **[+26.6, +213.0]** |

Bar at G=8 is |t| > 2.36. **CONFIRMED** -- the first finding this project has
established on eight independent seasons.

| | | | |
|---|---|---|---|
| 2018-19 | +44.1 | 2022-23 | +14.9 |
| 2019-20 | **-49.9** | 2023-24 | +244.3 |
| 2020-21 | +139.5 | 2024-25 | +259.4 |
| 2021-22 | +197.0 | 2025-26 | +109.2 |

**S43's +196.5 is CORRECTED to +119.8, a 40% reduction.** S43 measured three
seasons which happened to include two of the strongest in the record (+244 and
+259). This is the single-season-driver pattern once more -- and it appeared in
the ONE finding that had survived everything else, the number on which every
practical recommendation rested. **Re-measuring it was the right call and it
moved the answer.**

The interval is wide: chips are worth **somewhere between +27 and +213 a
season, best estimate ~120**. Seven of eight seasons positive, one negative.

**It remains an order of magnitude larger than every other lever measured on
2026-10-22**, all of which fell between -5 and +6.

### Closing position, 2026-10-22/23

ONE finding survives two days of testing: **play the chips.** Everything else
measured null on instruments that can now resolve 14 points where the old one
needed 123.

The rest of the value is negative knowledge, and it is worth having: nine
refuted experiments confirmed as genuine nulls rather than measurement
failures; four published findings retracted (S12 magnitude, S32, S33, S34); two
rescues withdrawn under clustering (F26, F28); one headline qualified (S48 by
S49); and one correction to the sole survivor (S43 by S50).

**For the live season:** four chips expire at GW19, none played, worth ~120
points at the clustered estimate -- which alone covers the user's -79 deficit.

---

## F34 / S51 — ownership weighting REFUTED, and it fails backwards, 2026-10-23

`fpl2/run_template.py`. Prompted by the user's objection, which was correct:
chips at +120/season are ~3.2 pts/gw, so over his seven gameweeks they explain
~22 of a 79-point deficit. The rest is captain picks and a squad that does not
hit -- and **our model shares that blind spot exactly**, having captained
B.Fernandes in GW4-6 for 1,1,2 while he captained the same player in the same
weeks for 2,2,1 and legacy did it in 3 of 4.

This is the first test on this project to measure RANK rather than points:
`beat%` = share of gameweeks above the crowd's template score.

| lambda | total | beat% | weekly sd | squad own% |
|---|---|---|---|---|
| 0.0 | 1873.8 | **0.323** | **16.77** | 13.9 |
| 2.0 | 1872.8 | 0.310 | 16.81 | 19.4 |
| 5.0 | 1890.8 | 0.303 | 17.10 | 23.8 |
| 10.0 | 1894.7 | 0.311 | 17.45 | 27.3 |
| 20.0 | 1892.6 | 0.298 | 17.49 | 29.2 |

| vs lambda=0 | d total | p | d beat% | p | d sd |
|---|---|---|---|---|---|
| 5.0 | +17.0 | 0.329 | **-2.03** | **0.042** | +0.33 |
| 20.0 | +18.8 | 0.546 | -2.54 | 0.189 | +0.71 |

**The mechanism works and every predicted benefit fails to appear.** Mean squad
ownership moves 13.9% -> 29.2%, so the objective is doing what it was told.
Season total does not move significantly. **beat% gets WORSE** -- the only cell
near significance is -2.03 at p=0.042, in the wrong direction. **And sd RISES**
(16.77 -> 17.49) when the entire argument was that template squads reduce
variance.

**S26's reading is therefore wrong as a prescription.** Our squads being
high-variance and low-ownership is real; buying ownership does not fix it.
Whatever makes the crowd's score stable is not reproduced by holding the
players the crowd holds.

**R33: a mechanism confirmed is not a benefit delivered -- and it may deliver
the opposite.** This is the sixth occurrence of the R31 pattern, and the first
where the targeted metric moved AGAINST the hypothesis while the mechanism
plainly worked.

### Two-day closing position

Fifteen things measured on calibrated instruments. **One survives: chips,
+120/season [+27, +213], 7/8 seasons, p = 0.019.**

For the user's actual deficit -- captaincy and squad -- **this project has
found nothing that helps.** Not the forecast (S47, S48), fixtures (S29, S33),
planning (F26), model family (S39), nor ownership (S51). The only thing that
beat our model, legacy and the user simultaneously was the crowd captaining
Haaland every week, which is not a model insight.

---

## M26 — the model forecast ZERO goals and ZERO assists on 5 of 8 folds, 2026-10-23

Found while scoping a from-scratch rebuild (O51). Not a tuning question: a
whole component of the architecture was silently absent from most of the
evidence this project has been deciding on.

**The mechanism.** `rates.RATE_TARGETS` maps `xg_per90 -> expected_goals` and
`xa_per90 -> expected_assists`. FPL publishes neither before 2022-23. A
holdout fold trains on seasons STRICTLY BEFORE the holdout, so:

| holdout | train rows (mins>=20) | xG rows | xA rows | us_xg rows | goal / assist model |
|---|---|---|---|---|---|
| 2018-19 | 18,069 | **0** | **0** | 17,425 | **no / no** |
| 2019-20 | 27,083 | **0** | **0** | 26,327 | **no / no** |
| 2020-21 | 36,292 | **0** | **0** | 35,396 | **no / no** |
| 2021-22 | 45,080 | **0** | **0** | 44,047 | **no / no** |
| 2022-23 | 53,730 | **0** | **0** | 52,536 | **no / no** |
| 2023-24 | 62,902 | 5,766 | 5,766 | 61,586 | YES / YES |
| 2024-25 | 72,196 | 15,060 | 15,060 | 70,771 | YES / YES |
| 2025-26 | 81,718 | 24,582 | 24,582 | 80,207 | YES / YES |

`RateModel.fit` skips a target at `ok.sum() < 200`; `predict` initialises every
output column to `0.0` and only overwrites it when a model exists. **So on
2018-19..2022-23 the shipped model predicts that every player in the league
scores no goals and makes no assists.** `pred_goals_per90 = pred_xg_per90 x
conversion = 0`. Verified by fitting folds 2019-20 and 2022-23 and listing
`RateModel.models`: `{GK: [bonus, saves, yellow], DEF/MID/FWD: [bonus,
yellow]}`. Fold 2024-25 has `xg_per90` and `xa_per90` in all four positions.

### What this does to S48

**S48 ablated components that were already zero on 5 of its 8 folds.**
Replacing an all-zero goal forecast with the per-position training mean is not
an ablation, it is an UPGRADE -- which is the straightforward reading of
`no_goals = +14.6` pooled and `no_goals@FWD = +19.1`. The headline "the model
is a minutes predictor wearing expensive decoration" is literally true of the
code as it ran, and it is a statement about MISSING DATA, not about whether a
rate layer can help. **S48's interpretation is withdrawn; its measurements
stand.** S49 inherits the same qualification.

### What this does to S47

S47 swapped the xG SOURCE to `us_xg` and measured +1.6 [-3.7, +6.8]. It did
not swap the xA source, because there was no `_xa_source` to swap -- so the
Understat arm ALSO had no assist model on those five folds. S47 bounds the
value of adding xG alone; it never tested adding xA. Its conclusion is
narrowed, not retracted: the CI is tight and nothing here predicts a large
effect.

### The fix

`rates._xa_source()` added as the twin of `_xg_source()`, plus `_targets()`,
`_convert()`, `_ceiling_sources()` and `_is_expected()` so the resolved source
reaches the TARGETS, the CONVERSION denominators and the CEILINGS -- three
places that each hardcoded `expected_assists`. Gated on the existing
`features.USE_UNDERSTAT_FEATURES`, so the default path is unchanged: verified
by re-fitting folds 2019-20 and 2022-23 with the flag off and getting the
identical (broken) model list, and 81/81 tests green.

With the flag ON, all eight folds now fit `xg_per90` AND `xa_per90` in all
four positions.

**How it hid for so long.** The project's holdouts were 2023-24..2025-26 until
2026-10-22 -- the three folds that DO have Opta data. Extending to ten seasons
(S46) is what put five blind folds into every measurement, and the ablation
built to interrogate the rate layer (S48) was the first thing to run on them.
**The extra seasons did not just add power; they changed what was being
measured, and nothing checked that.**

### Confirmed at the OUTPUT, on every fold, with a positive control

The table above is an argument from training-row counts. This is the model's
actual output. `run_bakeoff.py` prints the largest predicted goal and assist
rate anywhere in the holdout, every fold:

| holdout | max pred_goals/90 | max pred_assists/90 |
|---|---|---|
| 2018-19 | **0.0000** | **0.0000** |
| 2019-20 | **0.0000** | **0.0000** |
| 2020-21 | **0.0000** | **0.0000** |
| 2021-22 | **0.0000** | **0.0000** |
| 2022-23 | **0.0000** | **0.0000** |
| 2023-24 | 0.6057 | 0.3489 |

Not "small". Not "poorly fitted". **Exactly zero for every player in the
league, across 21,134 rows in the first fold alone** -- and then nonzero the
moment Opta data enters the training window. The check can fail and does fail,
in precisely the five folds predicted and no others, which is what separates a
control from a reassurance (R26).

### The same audit on the other components -- clean

Checked, because a defect of this shape is rarely alone:

| component | column absent early | behaviour |
|---|---|---|
| `minutes.MinutesModel` | `starts` (none before 2022-23) | **handled.** Detects it, sets `p_start = None`, falls back to `p_sixty`, and says so in a comment. |
| `team.TeamModel` | `expected_goals_conceded` | **handled.** `use_xgc = notna().mean() > 0.5`, else the target falls back to realised `conceded`. |
| `rates.RateModel` defensive | `defensive_contribution` (2025-26 only) | **handled by design.** Refit within-season on strictly earlier gameweeks. |
| `rates.RateModel` xG/xA | `expected_goals`/`expected_assists` | **NOT handled.** `continue`, leaving the prediction at its 0.0 initialisation. |

Three of the four fall back to a weaker but sensible quantity and one of them
documents it. The fourth falls back to a number that is not merely weaker but
false. **The difference is not sophistication, it is that `continue` inside a
loop over targets produces no object to inspect and no message.**

**R34: when a fold's training window moves, re-check which models actually
fit.** `usable()` filters features per fold and `fit` skips targets per fold.
Both are silent, both are correct in isolation, and together they can empty an
entire component without raising anything. A fitted model's `.models` keys are
one line to print and were never printed.

---

## M27 — S47 is RETRACTED. The ceiling zeroed the predictions it was capping.

S47 published **+1.6 [-3.7, +6.8], p 0.509** for Understat features and
concluded "giving the model xG where it previously had none is worth nothing
measurable". `USE_UNDERSTAT_FEATURES` was left **False** on that basis.

**Re-measured on the identical instrument -- K=10, the same 8 holdouts, seed=1
neutral squads, season-clustered -- the answer is +47.7 [+14.2, +81.2],
p 0.0119, winning 8 of 8 seasons.**

Two independent routes agree to four significant figures:

| | mean | t | p | 95% CI |
|---|---|---|---|---|
| S47 as published | +1.6 | 0.70 | 0.509 | [-3.7, +6.8] |
| fresh re-implementation (`run_bakeoff.py` family `s47`) | **+47.7** | 3.37 | 0.0119 | [+14.2, +81.2] |
| **S47's OWN script, re-run today** | **+47.7** | **3.37** | **0.0119** | **[+14.2, +81.2]** |

The harness was never the problem, which is why reading it found nothing.

### The defect, located and verified

`rate_caps` read the module constant `CEILING_SOURCES`, which hardcodes
`xg_per90 -> "expected_goals"`. With `USE_UNDERSTAT_FEATURES` on, the goal
model was FITTED on `us_xg` and then CLIPPED to a ceiling measured from
`expected_goals` -- a column that is entirely NaN before 2022-23. The ceiling
came out **0.0**, and `predict` applies it:

```python
ceiling = cap.get(name)
if ceiling is not None and np.isfinite(ceiling):
    pred = np.minimum(pred, ceiling)        # 0.0 is finite
```

Measured directly:

| fold | position | xG ceiling, pre-M26 | post-M26 |
|---|---|---|---|
| 2019-20 | MID | **0.0** | 0.495 |
| 2019-20 | FWD | **0.0** | 0.830 |
| 2022-23 | MID | **0.0** | 0.612 |
| 2022-23 | FWD | **0.0** | 0.863 |
| 2024-25 | MID | 0.426 | 0.628 |
| 2024-25 | FWD | 0.771 | 0.989 |

**S47's Understat arm fitted a working goal model and then multiplied its
output by zero.** Both arms predicted no goals for anybody, the comparison was
between two blind models, and +1.6 is what two blind models differ by.

Note the last two rows: even on folds WITH Opta data the old code capped a
`us_xg`-trained model at an `expected_goals`-derived ceiling, ~32% too low.
The bug bit on every fold; it was merely catastrophic on five.

### Same root cause, third location

M26 found expected-stat columns read by name without checking fold
availability in the TARGETS. This is the same mistake in the CEILINGS, and
`conversion_factors` was the third. All three are fixed by the same patch
(`_targets`, `_convert`, `_ceiling_sources`, `_is_expected`), which is why the
Understat arm went from worthless to +47.7 without any change to the model.

**R35: a clip is a silent multiply-by-zero waiting to happen.** `np.minimum`
with a ceiling of 0.0 destroys a prediction and raises nothing; `0.0` is
finite, so the guard admitted it. Any bound derived from data must assert that
the data it was derived FROM exists.

**R36: when a result cannot be reproduced, run the original artefact before
theorising.** Two careful readings of S47's harness against a fresh
implementation found nothing, because the harness was correct. Running it took
ten minutes and localised the defect immediately.

---

## S52 — THE BAKE-OFF: twenty arms, and the architecture question is a null, 2026-10-23

`fpl2/run_bakeoff.py` (fit, 5 families x 8 seasons), `run_bakeoff_score.py`
(score). Prompted by the user: "with the new seasons we have, can we start
over and try to build a new model and lets see how it competes?"

**Pre-registered before any arm was scored:** metric 1, neutral fixed squads,
K=10, 8 holdouts, paired on (season, squad, gameweek), season-clustered
(G=8, bar |t| > 2.36). BETTER requires the clustered CI to exclude 0 AND the
point estimate to exceed +13.6, the instrument's MDE at this n (R24).

### The arms

Full factorial over architecture, learner, data, position handling, objective:

| family | arms |
|---|---|
| baselines | `b_price`, `b_form5`, `b_form10` |
| component | `m_full_us` (shipped after M27), `m_full_fpl` (pre-flip), `m_gbdt_fpl`, `m_gbdt_us`, `m_pool_us`, `m_s47_xgonly` |
| direct | `d_glm`, `d_hgb`, `d_xgb`, `d_hurdle`, `d_per90`, `d_xgb_q80`, `d_xgb_pos` |
| ensembles | `e_rank_mdl_xgb`, `e_rank_fpl_us`, `e_rank_mean_q` |

### THE RESULT, against the repaired reference `m_full_us`

| arm | mean | t | p | MDE | seasons | verdict |
|---|---|---|---|---|---|---|
| e_rank_mean_q | +8.0 | 0.96 | 0.371 | 23.5 | 5/8 | equivalent |
| d_xgb | +7.3 | 1.04 | 0.332 | 19.7 | 5/8 | equivalent |
| d_xgb_fpl | +4.9 | 0.65 | 0.538 | 21.3 | 4/8 | equivalent |
| d_hgb | +4.4 | 0.48 | 0.645 | 25.8 | 4/8 | equivalent |
| d_hurdle | +3.8 | 0.42 | 0.685 | 25.4 | 5/8 | equivalent |
| d_per90 | -0.5 | -0.07 | 0.949 | 20.7 | 4/8 | equivalent |
| m_gbdt_us | -2.4 | -0.32 | 0.762 | 21.3 | 3/8 | equivalent |
| d_xgb_q80 | -2.9 | -0.36 | 0.732 | 23.0 | 3/8 | equivalent |
| m_s47_xgonly | -7.8 | -2.97 | 0.021 | 7.4 | 0/8 | — |
| d_xgb_pos | -11.5 | -1.41 | 0.201 | 22.8 | 2/8 | equivalent |
| **m_pool_us** | **-17.4** | **-4.62** | **0.002** | **10.5** | **0/8** | **WORSE** |
| d_glm | -19.1 | -2.93 | 0.022 | 18.3 | 1/8 | WORSE |
| e_rank_mdl_xgb | -21.0 | -1.83 | 0.110 | 32.2 | 2/8 | — |
| e_rank_fpl_us | -36.6 | -3.40 | 0.011 | 30.1 | 1/8 | WORSE |
| **m_full_fpl** | **-55.5** | -3.56 | 0.009 | 43.7 | 0/8 | **WORSE** |
| m_gbdt_fpl | -56.4 | -4.12 | 0.004 | 38.4 | 0/8 | WORSE |
| b_price | -61.8 | -6.65 | 0.000 | 26.0 | 0/8 | WORSE |
| b_form10 | -61.8 | -2.13 | 0.071 | 81.2 | 1/8 | — |
| b_form5 | -69.3 | -2.18 | 0.066 | 89.2 | 1/8 | — |

**NOT ONE ARM OF NINETEEN BEATS THE REPAIRED COMPONENT MODEL.** Zero BETTER
verdicts. The best challenger is +8.0 at p = 0.371, inside the gate and inside
the noise. **The component architecture is not the problem and never was.**

### Four results worth keeping

1. **Per-position rates earn their place.** `m_pool_us` -- identical model,
   rate layer POOLED -- is **-17.4, p 0.002, MDE 10.5, 0/8 seasons**. One of
   the best-powered positive findings on this project, and it validates a
   design choice (`RateModel._groups`) that had never been tested. Splitting
   the DIRECT model does not help either (`d_xgb_pos` -11.5), so the answer is
   position-specific to the architecture, not a general rule. **F7's rejection
   of per-position models stands for direct models and is INVERTED for the
   component rate layer.**

2. **GLM vs GBDT is a null on repaired data.** `m_gbdt_us` -2.4, p 0.762,
   MDE 21.3. S12 claimed +194; S39 corrected it to ~21; on data where the rate
   layer actually exists it is zero. **S12 is superseded. The question is
   closed.** Note `m_gbdt_fpl` measured -0.9 earlier against the broken
   reference: the estimator choice was never going to matter while the targets
   were absent.

3. **The xA half of M26 is real and tightly measured.** `m_s47_xgonly`
   (us_xg but FPL xA) is **-7.8 with MDE 7.4** -- wiring `us_xa` is worth
   +7.8 on top of the xG fix.

4. **The baselines are -62 to -69.** Against the BROKEN reference the same
   arms measured "equivalent" (b_price +1.0, p 0.98). Same data, same
   instrument, reference repaired. That inversion is the clearest single
   statement of what M26/M27 cost.

### Where the loss lived: MID and FWD, through the ARMBAND

Measured against the pre-flip reference, two decompositions agree.

**Per position** (arm's signal for one position, reference for the other three):

| arm | GK | DEF | MID | FWD |
|---|---|---|---|---|
| d_per90 | -0.4 | +0.1 | **+47.8*** | +26.4 |
| d_xgb | +0.7 | +0.2 | **+45.5*** | **+32.6*** |
| m_full_us | +0.7 | -1.9 | **+41.5*** | +28.2 |
| m_gbdt_fpl (control, same broken data) | +2.2 | -2.1 | -0.6 | -2.5 |
| b_price | **-25.6*** | **-59.7*** | +21.4 | -2.7 |

Nothing at GK, nothing at DEF, everything at MID and FWD -- exactly the two
positions that score goals and assists, and exactly what a missing
goal-and-assist model predicts. The clean-sheet model was never broken and GK
and DEF read +-2.6 across every arm.

**Captain vs XI** (XI held on the reference, armband only; and the reverse):

| arm | captain | XI | sum | actual total |
|---|---|---|---|---|
| d_xgb | +49.7 | +12.4 | 62.1 | 62.9 |
| d_xgb_fpl | +49.9 | +10.7 | 60.6 | 60.5 |
| m_full_us | +49.3 | +6.5 | 55.8 | 55.5 |
| e_rank_fpl_us | +15.7 (t 4.44) | +3.3 | 19.0 | 19.0 |

Additive, which is a soundness check on the measurement. **Every repaired arm's
captain term sits in +45 to +50 regardless of architecture** -- component,
XGBoost, hurdle, per-90, quantile, per-position. That band is what fixing the
attacking layer buys through the armband, and no architecture exceeds it.
Three arms (`d_glm`, `m_pool_us`, `b_price`) pick a WORSE eleven and still win
overall on the armband alone.

**This closes the user's standing objection from 2026-10-22** -- "my captain
pick and my general squad were not hitting". The squad half was approximately
right; the captain half was broken, and the cause was M26.

### The per-season split, which is the whole interpretation

Pre-registered before the run: if M26 explains the gap, it must shrink on the
three folds where the component model HAS a rate layer.

| arm | 2018-19..2022-23 (no goal/assist layer) | 2023-24..2025-26 (layer present) |
|---|---|---|
| d_xgb | +95.5 | **+9.1** |
| d_xgb_fpl | +94.2 | **+4.9** |
| e_rank_mean_q | +95.5 | +10.9 |
| d_hgb | +89.1 | +11.8 |
| m_full_us | +84.8 | +7.3 |
| d_per90 | +81.1 | +12.0 |

An 8-9x collapse; several arms go NEGATIVE on 2024-25. **The +55 to +63
headline was five crippled folds averaged with three honest ones.** The
prediction was made before the numbers and it held.

---

### WHAT M26/M27 DID **NOT** BREAK — the live model

Stated loudly because every headline above overstates the practical gain.

| holdout | max pred_goals/90 | max pred_assists/90 |
|---|---|---|
| 2018-19 .. 2022-23 | **0.0000** | **0.0000** |
| 2023-24 | 0.6057 | 0.3489 |
| 2024-25 | 0.7308 | 0.3976 |
| 2025-26 | **0.7417** | **0.4106** |

**The live model has always had a working goal and assist layer.** A model
trained for 2026-27 sees Opta xG from 2022-23 onward -- roughly 33,000 rows.
M26 and M27 destroyed the HISTORICAL EVALUATION on five of eight folds; they
never touched the forecast actually being used to pick a team.

So the honest value of turning Understat on, per season:

| | gain |
|---|---|
| 5 broken folds | +84.8 |
| **3 repaired folds** | **+7.3** |
| trend within them | +15.6 -> +2.7 -> +3.5, **declining as Opta data accumulates** |

**+55.5 is the average over folds the user will never play. The live number is
~+7 and falling toward ~+3.** The flip is still right -- it is free and
positive -- but it is not worth 55 points this season and must never be quoted
as if it were.

**This also means M26 does NOT explain the live underperformance.** The
2026-27 model captained Bruno over Haaland in GW4-6 *with a working rate
layer*. S20 stands, and is now a sharper question: not "the model cannot see
goals" but "the model can see goals and still gets this wrong".

**R40: a defect's blast radius is not its headline number.** The same bug was
worth +85/season on folds nobody plays and +3 on the fold that matters.
Always report the effect on the configuration that will actually run.

---

## S53 — THE OPTIMISER RACE: transfers null, bench weight null, 2026-10-23

`fpl2/run_opt_bakeoff.py`, `run_opt_bakeoff2.py`, `run_opt_bakeoff2_combine.py`.
User: "also we should look at ways we can improve the optimizer the same way we
are doing for the model." Signal held FIXED across every arm, so nothing here
is confounded with S52.

### Captaincy -- and the reason to re-run it on a repaired signal

| rule | on BROKEN signal | on REPAIRED signal |
|---|---|---|
| cap_xgb_mean | +49.7, p 0.010, 7/8 | **+0.6, p 0.909, 5/8** |
| cap_q80 | +46.6, p 0.014, 6/8 | **-2.6, p 0.582** (MDE 12.5) |
| cap_upside_05 | +17.1, p 0.022 | **-5.8, p 0.133** |
| cap_upside_03 | +11.2, p 0.023 | **-3.3, p 0.347** |
| cap_upside_10 | +12.5, ns | **-16.5, p 0.018, 0/8** |
| cap_premium | +29.8, p 0.095 | **-20.3, p 0.024, 1/8** |
| cap_form5 | +21.9, ns | **-27.6, p 0.022, 1/8** |
| cap_safe | -1.7 | -5.6, p 0.054 |

**`argmax(xp)` is the correct captain rule. All seven alternatives are null or
significantly worse.**

`cap_xgb_mean` collapsing +49.7 -> +0.6 is the whole story: once the signal has
a goal layer, choosing the armband with XGBoost is worth nothing over choosing
it with the component model. The +49.7 was a MODEL difference measured through
the armband, because that is where `argmax` concentrates it.

**F9 / O12 is CLOSED.** The ledger flagged variance-adjusted captaincy as
"highest-value reopen on the board". On the broken signal it looked alive at
p ~ 0.022; on the repaired signal both doses are null and the strongest
(lambda=1.0) is significantly NEGATIVE at 0/8. F9's original rejection was
right.

**`cap_premium` is REFUTED** (-20.3, 1/8). "Captain the most expensive player"
is worse than argmax. The crowd captaining Haaland was not a better heuristic;
our model simply could not see he would score, because it predicted nobody
would.

**The control earned its keep.** `cap_q80` changes BOTH the model and the
functional; `cap_xgb_mean` was added as the same-model control. `cap_q80 -
cap_xgb_mean = -3.1`: **the tail objective is worth nothing.** Without that
control a tail-aware armband would have been reported as a finding.

**R37: measure an optimiser change only on a signal you have verified.** Seven
captain rules were tested against a model that forecast zero goals for every
player on five of eight folds. Three came back at p < 0.025. Every one
inverted on the repaired signal. A decision rule cannot be evaluated through a
broken input, and the failure is not conservative -- it manufactures
significant results in the wrong direction.

### Transfers -- a genuine null with real power

| arm | mean | p | MDE | seasons |
|---|---|---|---|---|
| gain_0.0 | +1.5 | 0.422 | **5.0** | 1/8 |
| gain_1.0 | -0.1 | 0.987 | **10.0** | 1/8 |
| gain_2.0 | +5.2 | 0.504 | 20.9 | 4/8 |
| mpc_h2 | +13.6 | 0.655 | **81.4** | 3/8 |
| mpc_h4 | +18.5 | 0.483 | **70.0** | 4/8 |

The transfer threshold `min_gain` is **settled null** -- MDE 5-10 against
effects of 0-5. The MPC arms are **NOT null, they are UNMEASURABLE**: point
estimates +13.6 / +18.5 against MDEs of 70-81, because MPC reroutes the whole
transfer path and the variance explodes. Consistent with F26's withdrawal
under clustering. Reported as unresolved, not as no effect (R31).

### Squad building

`bench_weight` null at every value (0.0, 0.25, 0.5; MDE 70-132). **This retires
S27's "bench_weight swings 338 points"** -- that was fork variance, as
suspected when the calibrated instrument was built.

Useful anchor from the absolutes: a squad held all season averages ~1850;
with transfers ~2095. **Transfers are worth ~245 pts/season.**

---

## M28 — the opening-horizon sweep is F10's leak, reproduced at larger magnitude

`fpl2/run_opt_horizon.py`. The only non-null in the optimiser race was
`opening_horizon`, so it was swept at eleven doses, path-free:

| h | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 15 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean | -159 | -130 | -107 | -68 | ref | +30 | +83 | +137 | +204 | +234 | **+313** |
| seasons | 1/8 | 1/8 | 2/8 | 2/8 | — | 5/8 | 5/8 | 7/8 | 6/8 | 7/8 | **8/8** |

h=19: t 4.52, p 0.003, CI [+149, +477], 8/8 -- the largest effect ever
measured on this project, perfectly monotone across all eleven doses.

**It is a leak, and the ledger already recorded it.** F10: *"Longer decision
horizon (H = 2..12). Leaky version said +196/season. Leakage-free: H=2-10 all
worse."* The appendix even labels the scripts "(F10, M1 — both leak)".

`simulate._opening_squad` averages the signal over gameweeks 1..h. But the
walk-forward xp for gameweek 15 is computed from form data through gameweek
14; at the gameweek-1 deadline that forecast cannot exist. Longer horizon =
more future information. **A perfectly monotone dose-response with no
saturation is the signature of a leak, not of a strategy** -- a real strategic
parameter has an optimum.

Caught before it was shipped or acted on, but it was presented to the user as
a finding first and retracted ten minutes later.

**R38: a dose-response that never saturates is a leak until proven otherwise.**
Real effects have optima; monotone-to-the-boundary means the treatment is
correlated with something that keeps increasing, and "how much of the future
can I see" is the usual candidate. R26 demands a negative control for the
information; this is its dose-response twin.

**R39: search the ledger for the PARAMETER, not just the hypothesis.** F10
tested this exact parameter and recorded it as a leak. The sweep was written
and run without looking, because it arrived as "the only non-null in the
optimiser race" rather than as "let us test the decision horizon".

---

## S54 — the rate CEILING is innocent. Mechanism confirmed, benefit zero, 2026-10-23

`fpl2/run_ceiling.py`. Hypothesis, formed from M27: `rate_caps` clips every
per-90 prediction at the 99th percentile of sustained player-season rates, so
it must bind precisely on the players who ARE the 99th percentile -- the same
set worth captaining. Proposed as the live-model explanation for S20.

### The mechanism is real and it is dramatic

Diagnostic on holdout 2024-25:

| player | weeks clipped | raw xG/90 | cap | lost goals/90 |
|---|---|---|---|---|
| **Mohamed Salah** | **32 of 32** | 0.904 | 0.628 (MID) | **-0.278** |
| Erling Haaland | 8 | 1.087 | 0.989 (FWD) | -0.090 |

Salah is clipped in **every gameweek of the season**; the model wants to rate
him 44% higher and is not allowed to. The ceiling fires on 0.38% of played
rows and those rows are the two best attackers in the league.

It is also doing necessary work: on holdout 2025-26 it caught the GLM
extrapolating to **2478 xG/90** for a player with almost no history. So the
test is a dose-response on `WINSOR_QUANTILE`, not a deletion.

### The benefit is zero, and the measurement is the tightest on this project

Metric 1, 8 holdouts, K=10, season-clustered, reference `ceil_p99` (shipped).
All five arms share ONE fit per season with only the caps swapped, and the
`ceil_p99` arm reproduced `m_full_us` **bit-identically on all 8 seasons**.

| arm | mean | t | p | 95% CI | **MDE** | seasons |
|---|---|---|---|---|---|---|
| ceil_p95 | -3.9 | -1.61 | 0.151 | [-9.7, +1.8] | 6.8 | 4/8 |
| ceil_p99 | reference | | | | | |
| ceil_p995 | +0.1 | 0.15 | 0.883 | [-0.9, +1.0] | **1.1** | 4/8 |
| ceil_p999 | +0.6 | 1.25 | 0.251 | [-0.6, +1.8] | **1.4** | 5/8 |
| **ceil_none** | **+0.8** | 1.55 | 0.164 | [-0.4, +2.0] | **1.4** | 5/8 |

**Deleting the ceiling entirely is worth +0.8 pts/season against an MDE of
1.4.** Not "too small to detect" -- detectable to within +-1.4 and measured at
+0.8. No inverted U, no dose-response, nothing. REFUTED.

### Why it does not matter, which is the transferable part

The ceiling changes a player's LEVEL. Every decision in this pipeline --
`best_xi`, `pick_captain` -- is `argmax`, which depends only on RANK. Clipping
Salah from 0.904 to 0.628 removes 0.278 goals/90 from his xP and leaves him
exactly where he was: the highest-rated midfielder in the league. The armband
does not move. A per-position cap compresses the top of each position
together, and the ordering within that compressed group is almost entirely
preserved.

**R41: a level effect cannot move a rank decision.** Before testing a change
that shifts predicted VALUES, ask whether it reorders anything. Clipping,
shrinkage, calibration and scaling are all level transforms; `argmax` is blind
to every one of them unless they cross two players over. This retires a whole
family of candidate explanations at once -- and it is why S48 found the entire
rate layer contributing nothing to line-up choice while S49 found it mattering
for transfers, where the budget constraint makes levels bind.

**Seventh occurrence of R31** (mechanism confirmed, benefit absent), and the
most clear-cut: the mechanism here is not subtle or marginal, it is Salah
clipped 32 weeks out of 32, and it is worth +0.8 +- 1.4 points a season.

**S20 remains open with a seventh route closed.** The live model has a working
goal layer (S52), its predictions are not being suppressed by the ceiling
(here), and it still preferred Bruno to Haaland in 2026-27 GW4-6.

---

## S55 — IS THERE ANYTHING TO EXPLAIN? A -79 start is a 1-in-10 event, 2026-10-23

The question that was never asked. Months of this project have hunted for a
defect to explain the user sitting ~79 points behind the FPL average after 7
gameweeks of 2026-27. Nobody measured how much a 7-gameweek stretch varies
when NOTHING is wrong. It was listed as an untried angle in the 2026-10-21
handover ("Variance check -- two bad weeks in seven is not obviously
abnormal") and never run.

Measured from `metric1_m_full_us.pkl`: 2,550 seven-gameweek windows, each one
a FIXED squad scored by a FIXED model. No transfers, no chips, no manager
decisions, no model error differences. Pure outcome noise.

| 7-gameweek total | |
|---|---|
| mean | 401.7 |
| **sd** | **60.0** |
| 10th percentile | 324.0 (**-77.7** vs mean) |
| 25th | 363.0 (-38.7) |
| 75th | 441.8 (+40.0) |
| 90th | 475.0 (+73.3) |

**A -79 point shortfall over 7 gameweeks is 1.32 sd. It occurs in 9.5% of
windows with the same squad and the same model** -- and +79 occurs in 8.4%,
so the distribution is near-symmetric and this is not a tail artefact.

And the directly relevant comparison, two squads over the SAME seven
gameweeks with the SAME model -- i.e. you against a rival manager:

| \|difference\| | median | 75th | 90th | 95th |
|---|---|---|---|---|
| | 40.0 | 70.0 | 105.0 | 128.0 |

**One pair of managers in five is 79 or more points apart after seven
gameweeks, from squad composition alone.** These squads make no transfers and
play no chips, so real managers diverge at least this much.

### What this means

**The premise of the entire investigation may be false.** There is very likely
nothing to explain. A -79 start is an ordinary draw, not evidence of a defect,
and every mechanism hunted for it -- S20's captaincy, S29/S33's fixtures,
S48's components, S51's ownership, S54's ceiling -- was searching for a cause
of something that happens one time in ten by chance.

It also explains the shape of the last three days: **mechanisms kept
confirming and benefits kept vanishing** (R31, seven occurrences) because the
thing being explained was noise, and noise has no mechanism.

**R42: before explaining a deficit, measure the distribution it was drawn
from.** The cost of skipping it here was months. One query against data
already on disk answered it in seconds. If an observation sits inside the
inter-decile range of its own null distribution, there is no finding to chase
-- and chasing it manufactures exactly the reactive decisions that turn
ordinary variance into real losses (S51: chasing ownership moved beat%
BACKWARDS).

**Caveat, stated:** these windows are fixed-squad model scores, not the user's
realised points against the live FPL average. They measure how much a
7-gameweek stretch varies in this competition, which is the quantity that
decides whether -79 is unusual. They do not prove the user's particular -79
contained no error -- only that it needs no error to occur.

---

## S56 — CONDITIONAL BONUS: a 122x mechanism that reorders nothing, 2026-10-23

`fpl2/run_bonus.py`. **The user's hypothesis, from watching the live season**,
and the sharpest one anyone has brought to this project: B.Fernandes was
forecast 5.7-7.1 every week while returning 2,23,2,2,2,1,1, and the model
RAISED him to 7.1 after the single haul. His attacking rates are not the
problem -- 2.66 xG and 1.50 xA for 3 goals and 1 assist is performing TO xG.
The gap is BONUS: predicted 1.055/90 against an actual ~0.43/90.

And bonus is overwhelmingly conditional. 105,659 played player-gameweeks, ten
seasons:

| condition | P(bonus>0) | E[bonus] |
|---|---|---|
| points (ex-bonus) <= 5 | 0.024 | 0.038 |
| points (ex-bonus) > 5 | 0.523 | **1.099** |
| no goal or assist | 0.041 | 0.071 |
| 1+ goal or assist | 0.509 | **1.089** |

Per position, per 90, from training data only:

| | GK | DEF | MID | FWD |
|---|---|---|---|---|
| with a return | 0.768 | 0.686 | **1.095** | **1.457** |
| without | 0.091 | 0.013 | **0.009** | 0.000 |

**A midfielder who returns earns 1.095 bonus per 90; one who does not earns
0.009. A ratio of 122.** `rates` fits a single flat marginal rate, so the model
predicts "probably blanks" and "collects bonus anyway" for the same player in
the same fixture.

### Measured: nothing

Metric 1, 8 holdouts, K=10, season-clustered, reference `bon_flat` (shipped),
which reproduced `m_full_us` bit-identically on every season.

| arm | mean | t | p | **MDE** | seasons |
|---|---|---|---|---|---|
| bon_cond | -1.8 | -1.49 | 0.179 | **3.3** | 2/8 |
| bon_scale | -1.9 | -2.31 | 0.054 | **2.3** | 1/8 |
| bon_hard | -6.9 | -2.80 | 0.026 | 6.9 | 1/8 |

All three NEGATIVE, with a dose-response pointing the wrong way: the harder
the conditioning, the worse. MDEs of 2.3-6.9 -- this is not a failure to
detect.

### Why, and it generalises R41

| arm | rank rho vs bon_flat | mean rank shift | top-1 changed |
|---|---|---|---|
| bon_cond | **0.9998** | 2.4 places | 5% |
| bon_scale | 0.9995 | 4.3 | 18% |
| bon_hard | 0.9998 | 2.7 | 0% |

**P(return) is a monotone function of predicted goals + assists, which is
already the dominant term in xp.** Multiplying bonus by P(return) is therefore
approximately a monotone transform of the score being ranked, and `argmax` is
blind to monotone transforms. The correction is real, the arithmetic is
right, and it moves the ordering by 2.4 places out of ~600.

**R43 (strengthens R41): a correction that is MONOTONE IN THE EXISTING SCORE
cannot reorder it, however large the correction is.** R41 covered level
effects -- clipping, shrinkage, scaling. This is the wider class: any
adjustment whose size tracks the thing already driving the ranking. Before
building such a fix, compute the rank correlation between the old and new
signal. It costs one line and would have retired both S54 and S56 before they
were built.

**Eighth occurrence of R31**, and the second today where a mechanism measured
at better than 100x delivers nothing.

**The over-prediction is still REAL.** Bruno's bonus forecast is wrong and the
fix makes it less wrong. It simply does not change which player the armband
goes on, because everyone's bonus moves the same way at once. **An error that
is shared across the pool is invisible to a decision that only compares within
it.**

---

## S57 — the opening horizon, measured honestly: SHORTER is better, 2026-10-23

`fpl2/run_horizon_honest.py`. M28 found the leaky sweep monotone to +312.8 at
h=19 and identified it as F10's leak. This is the leak-free version, with the
leaky arm kept alongside as a control.

  leaky_h   mean of walk-forward xp over GW1..h -- week k's forecast used form
            data through week k-1, which the GW1 deadline cannot have
  honest_h  mean of `project.project(from_gw=1, to_gw=k)` -- form FROZEN at
            GW1, opponent and venue from the published list, opponent strength
            rewound to GW1 by `_strength_as_of`

| h | leaky | honest |
|---|---|---|
| 1 | — | **+65.5** |
| 2 | — | +38.4 |
| 4 | — | +10.4 |
| 5 | reference | reference |
| 10 | +75.9 | -47.9 |
| 12 | +134.7 | **-69.1** (p 0.044) |
| 15 | +160.3 (p 0.013) | -64.1 |
| 19 | **+235.4, t 6.23, p 0.000, 8/8** | -49.7 |

Absolute season totals:

| | h=1 | h=5 | h=19 |
|---|---|---|---|
| leaky | **1690** | 1906 | **2141** |
| honest | **1690** | 1625 | **1576** |

**The two arms are IDENTICAL at h=1 -- 1690 and 1690 -- because one week needs
no projection, and then diverge in opposite directions.** Same code, same
squads, same scoring; the only difference is whether week k's signal was
allowed to know about weeks 1..k-1. **The leak is worth +235 points a season
and the honest effect is the reverse.**

F10 replicates exactly: "Leakage-free: H=2-10 all worse."

**`opening_horizon = 5` is probably too long.** The best honest arm is h=1
(+65.5), though at MDE 157 no single row clears the bar; the TREND is
monotone and unambiguous where the individual rows are not. This also
vindicates `build_wildcard.py`, which collapsed the horizon to one week on
S29's argument -- that choice is now supported by measurement.

**Caveat, pre-registered and still pointing the same way:** the honest arm's
target fixture frame comes from the realised season, so it knows about doubles
and blanks in GW1..h, which FAVOURS long horizons. It loses anyway.

---

## M29 — FOUR defects in the LIVE path that no backtest could see, 2026-10-23

Found because the user asked a question nobody had asked: *"are we sure that
every time we run this, we pull updated data from the past gws, or are we
still basing this on data from the first gws?"*

The holdout path and the live path are different code. Eight seasons of
clustered evaluation say nothing about the second.

| # | defect | severity |
|---|---|---|
| 1 | `run_live_glm.py:50` — `if LIVE.exists(): read_pickle(LIVE)`, **no freshness check ever** | latent; correct that day only because GW7 was still the last finished. One gameweek from serving stale data for ever |
| 2 | trained on **four** seasons — `data.load_raw()` without `extended=True` | real, cost unmeasured |
| 3 | live season joined **ZERO** Understat rows | **real and active** |
| 4 | `understat.attach` returns `panel.assign(us_*=NaN)` when a season joins nothing, **wiping columns already present** | real |

**Defect 3 is the serious one and it was caused by that morning's own fix.**
`load_raw` derives `kickoff` from `kickoff_time`; the live frame comes straight
from the API and never passes through it, so after the concat every live row
has `kickoff = NaT`. `build_panel` folds that into `kickoff_first`, and
`understat.join_season` matches teams BY FIXTURE DATE -- so the live season
matched nothing, silently. Defect 4 then turned that silence into damage.

**Consequence: turning `USE_UNDERSTAT_FEATURES` on -- worth +7/season on
holdouts -- would have made the LIVE forecast WORSE.** The model trains on ten
seasons carrying 28 Understat features and would have predicted a live season
where all 28 were NaN, imputed to the training median: every player handed
identical attacking-threat inputs. Coverage after the fix: **0.0% -> 99.4%**.

**R44: a backtest cannot validate the live path, because it does not run it.**
Every result in this ledger is computed by `model.walk_forward` over a panel
assembled by `data.build_panel(data.load_raw(...))`. The live forecast is
assembled differently -- API rows concatenated onto processed history -- and
that seam is invisible to all of it. Any switch that changes what the model
consumes must be re-verified ON THE LIVE PATH, by printing coverage of the
columns it touches, before it is trusted for a real decision.

**R45: cache by CONTENT, never by existence.** `if path.exists()` is not a
cache policy. The fix asks the API which gameweeks are finished and rebuilds
whenever the cache disagrees. Three lines, and it turns a silent staleness bug
into a printed line.

### A fifth, in the decision layer

`plan_gw.py` filtered the TRANSFER POOL for availability and not the squad
already owned, so Calvert-Lewin -- status `d`, 50%, foot injury -- kept his
full 3.98 xp and was selected in the XI. The user caught it. Now every held
player's xp is scaled by chance of playing; he reads 1.99 and benches, and the
no-transfer XI fell from an inflated 50.74 to 50.30.

**R46: availability is a property of a PLAYER, not of a transfer.** The model's
`p_appear` is learned from minutes history and cannot know today's news, so the
news has to be applied wherever a player is scored -- including the ones you
already own.

---

## S58 — CLUB CHANGE: the model carries a player's old club with him, 2026-10-23

**The user's hypothesis**, from his own squad: Elliot Anderson is recommended
on 180 points scored at Nottingham Forest and now plays for Manchester City;
Enzo Fernandez carries 9,812 Chelsea minutes into the same move. Six of his
fifteen players had changed club.

### The bias is real and it lives in the RECOMMENDATIONS

Ranking players by xp within each gameweek, 8 holdout seasons, bias = actual
minus predicted:

| xp rank | moved | stayed | gap | t | seasons |
|---|---|---|---|---|---|
| **top 10** | **-0.851** | -0.208 | **-0.818** | **-2.36** | **7/8** |
| 11-30 | -0.355 | -0.166 | -0.239 | -0.88 | 5/8 |
| 31-60 | -0.257 | -0.071 | -0.212 | -1.85 | 5/8 |
| 61+ | +0.081 | +0.045 | +0.028 | +1.06 | 2/8 |

**A top-10-ranked player who changed club is over-predicted by 0.85 points a
gameweek -- about 32 points a season.** The gradient across rank bands is
monotone and REVERSES SIGN outside the top 60: the bias exists precisely in
the players the model tells you to own, and nowhere else.

### The axis is TENURE, not the calendar -- which is why it was never found

The user asked whether the effect decays through the season. By calendar
gameweek it does not, cleanly:

| gameweeks | 1-5 | 6-10 | 11-19 | 20-29 | 30-38 |
|---|---|---|---|---|---|
| gap | -1.248 | -0.640 | -0.474 | **-1.161** | -1.091 |

It falls then returns, because **January signings reset the clock**. Keyed on
tenure instead it is clean and monotone:

| fixtures at current club | 0-9 | 10-24 | 25-49 | 50+ |
|---|---|---|---|---|
| bias | **-0.952** | -0.516 | -0.474 | **-0.140** |

**A 7x decay from fresh signing to settled player.** n = 2,283 in the settled
band, so the floor is well measured.

**R47: when an effect decays, find the clock it decays on.** Season-long
averages dilute this one ~7x, and the obvious diagnostic cut -- early season
versus late -- actively HIDES it, because the natural covariate (calendar
week) is a noisy proxy for the causal one (tenure). Two separate reasons this
survived eight seasons of clustered testing.

### Why the model cannot see it

`features.py` carries three club-aware columns -- `changed_club`,
`history_at_current_club_r5`/`_r10`, `fixtures_at_current_club` -- but after
about five games at a new club `history_at_current_club_r5` reads 1.0 and
stops flagging anything, while the **21 `*_career` columns carry the old
club's output undiscounted for ever**.

### Manager changes: UNTESTED, not refuted

The user extended the hypothesis to managerial change, which alters the
starting XI and the style. No manager data exists in the panel and
`soccerdata`'s readers do not provide appointments. A proxy -- rolling churn
in a team's starting XI -- came back null (high minus low churn **-0.139,
t -0.91, p 0.395, 5/8**) and is too blunt to settle anything: mean churn is
0.325, so a third of the XI changes every match from ordinary rotation, which
swamps the ~10 genuine upheavals a season. **Testing it needs real appointment
dates**, after which `fixtures_under_current_manager` is the same construction
as `fixtures_at_current_club`.

### Scope and honesty

The "top 10" cut was chosen AFTER seeing the 150+-points result, so it is
post-hoc; t = -2.36 sits exactly on the bar and p = 0.050 is as marginal as a
result gets. The stayers are over-predicted too (-0.208), so part of this is
general top-end optimism rather than a pure transfer effect.

**The treatment is tested separately (O57) and a diagnosis is not a fix** --
S54 and S56 both had confirmed mechanisms far larger than this one and
delivered nothing.

---

## S59 / O57 — the tenure shrink WORKS and is worth +1.2 pts/season, 2026-10-23

`fpl2/run_tenure.py`. The treatment for S58: shrink all 21 `*_career` features
toward their TRAINING position mean by tenure at the current club,
`w = n/(n+K)` with `n = fixtures_at_current_club`. K=0 is the shipped model
and reproduced `m_full_us` bit-identically on all 8 seasons.

### The gate — NULL

Metric 1, 8 holdouts, K=10 squads, season-clustered, bar |t| > 2.36:

| arm | mean | t | p | 95% CI | MDE | seasons |
|---|---|---|---|---|---|---|
| ten_K5 | +1.2 | 0.34 | 0.745 | [-6.9, +9.2] | 9.5 | 4/8 |
| ten_K10 | +1.0 | 0.40 | 0.704 | [-4.9, +6.8] | **6.9** | 5/8 |
| ten_K20 | +1.3 | 0.36 | 0.731 | [-7.1, +9.6] | 9.9 | 5/8 |
| ten_K40 | +1.5 | 0.47 | 0.650 | [-6.0, +9.0] | 8.9 | 5/8 |

Captain-only is the same story: -1.6 to +1.4 against MDEs of 4.2-6.0.

### The treatment DOES work -- and my first measurement of that said the opposite

| arm | moved | stayed | gap |
|---|---|---|---|
| ten_K0 (shipped) | -1.019 | -0.201 | **-0.818** |
| ten_K5 | -0.943 | -0.245 | -0.697 |
| ten_K10 | -0.919 | -0.264 | -0.655 |
| ten_K20 | -0.891 | -0.301 | -0.590 |
| ten_K40 | -0.861 | -0.336 | **-0.525** |

**Monotone in K, closing the targeted gap by 36%**, movers improving and
stayers giving a little back -- exactly the designed trade.

**The first version of this table showed the gap WIDENING, and it was an R29
violation of my own.** It ranked the top 10 by EACH ARM'S OWN xp, so the
shrink demoted movers out of the band and the arms were scored on different
populations:

| arm | movers in the top 10 |
|---|---|
| ten_K0 | 195 (6.4%) |
| ten_K40 | 138 (4.6%) |

Fixing the population on the shipped arm's ranking reverses the conclusion.
**R29 is in this ledger, written on 2026-10-21, and I broke it anyway** -- the
rule is easy to state and easy to miss, because ranking by the arm under test
is the obvious way to write the query.

### Why a working fix is worth 1.2 points

Top-10 movers are 6.4% of top-10 rows -- about 0.6 players per gameweek across
the whole league, so a given manager owns well under one at a time. Removing
36% of a -0.818 bias from a fraction of a player is worth roughly a point a
season, and the gate measures +1.0 to +1.5 against an MDE of 6.9. **The
arithmetic agrees with the measurement.**

**Eighth occurrence of R31, and a distinct variety of it.** S54's ceiling and
S56's conditional bonus were mechanisms that did not convert -- they moved the
ranking by two places out of six hundred. This one converts correctly: it
reorders, it demotes exactly the players it should, it closes the gap it
targeted. **The quantity is simply small.** "Confirmed mechanism, correct
treatment, measurable improvement in the targeted diagnostic, no points" is a
failure mode worth naming separately.

**R48: size the affected POPULATION before building the fix.** The bias was
-0.85 pts/gameweek, which sounds material, and the population carrying it was
0.6 players per gameweek league-wide. The product was always going to be ~1
point. One query -- how many rows does this actually touch -- would have
priced the ceiling before the experiment rather than after.

**NOT SHIPPED.** `features.py` is unchanged. The diagnosis (S58) stands and is
worth keeping: it explains why a top-10 recommendation who just moved clubs
underperforms his forecast, which is useful for a human reading the output
even when it is not worth encoding.

---

## S60 / O58 — TEAM PLAYING STYLE is worth nothing, and the shuffle proves it

`fpl2/style.py`, `fpl2/run_style.py`. **The user's observation**: every team
feature the model carries is an OUTCOME -- `team_scored_r5`,
`team_conceded_r10`, the opponent twins. None describes HOW a side plays. A
possession team working the ball into the box and a counter-attacking side
crossing from deep can score at identical rates and offer their forwards
completely different chances.

36 features from match-level FBref: possession, shots, shots on target, SoT%,
**goals per shot (quality separated from volume)**, crosses, interceptions,
tackles won, fouls -- at r5 and r10, for the team and the opponent.

| comparison | mean | t | p | 95% CI | MDE | seasons |
|---|---|---|---|---|---|---|
| style vs shipped | -3.1 | -0.70 | 0.507 | [-13.6, +7.4] | 12.4 | 3/8 |
| SHUFFLED vs shipped | -3.2 | -1.14 | 0.293 | [-9.7, +3.4] | 7.8 | 3/8 |
| **style vs SHUFFLED** | **+0.1** | **0.02** | **0.988** | [-9.1, +9.3] | 10.9 | 4/8 |

**The real features and the same features scrambled between teams perform
identically.** The information is worth +0.1 against an MDE of 10.9.

### Why the control is the whole result

Both arms lose the SAME -3.1 / -3.2 against the shipped model. That is the
price of adding 36 columns to the fit, paid equally whether or not they mean
anything. **Without the shuffle this reads as "team style costs 3 points a
season", which is true of the columns and false of the information.** R26
exists for exactly this and it has now earned its keep twice -- F16's
Championship priors and this.

### What was verified before the gate

* **Leak test with a working negative control.** Every style value equals the
  mean of STRICTLY EARLIER matches: 1,060 rows, 0 mismatches, while a
  deliberately unshifted version fails 1,034 times.
* **`sty_off` reproduces `m_full_us` bit-identically**, so the arms differ
  only in features.
* **Team mapping audited by eye**, all 20 names for 2024-25, after two earlier
  approaches failed (below).

### Three data defects found getting here

1. **FBref returns `team` as NaN for this entire league.** The game id reads
   `"2024-08-16 Manchester Utd-nan"`, the nan always being the row's own team.
   Recovered by pairing each fixture's two rows on reversed scoreline and
   reading each side's team off the other's `opponent`. Validated at 20 teams
   x exactly 38 matches for all ten completed seasons.
2. **Date-overlap team matching fails here.** Within a season all twenty clubs
   play on roughly the same Saturdays, so the sets barely discriminate -- the
   attempt mapped **Everton to Sunderland** at 0.62. The Understat join works
   on dates because PLAYERS have distinctive date sets; teams do not.
3. **`kickoff_first` is 0% non-null from 2022-23 onward**, so any date-based
   join silently drops four seasons. The style join uses match ORDER within a
   team-season instead.

**R49: a join key that works for one entity can be useless for another.**
Fixture dates identify a player and cannot identify a club, because every club
in the league shares a calendar. The method was carried over from the
Understat join without asking whether the discriminating power survived the
change of entity.

**Coverage, stated:** 95.2% of panel rows; 99-100% from 2019-20, but 76-89%
in 2016-19 because the legacy adapter left placeholder names (`Team 110`,
`Team 25`...) for six club-seasons it could not resolve. Those seasons are
training-only for the gate's holdouts.

**NOT SHIPPED.** `features.py` unchanged. `style.py` is kept: the puller, the
team recovery and the leak-tested rolling are reusable if a better style
question comes along.

---

# Appendix — script index

| Script | Answers |
|---|---|
| `baseline_showdown.py` | Does model xP beat dumb heuristics end-to-end? (S1) |
| `squad_drag.py` | What does a weak opening squad cost? (W8 — construct is flawed) |
| `horizon_sweep.py` / `_clean.py` | How far ahead should decisions look? (F10, M1 — both leak) |
| `price_wildcard.py` / `_horizon.py` | What is a wildcard worth? |
| `training_window_sweep.py` | Best training window X and feature windows (F4) |
| `understat_features.py` | Does PPDA/np-xG team context help? (F6) |
| `per_position_models.py` | Pooled vs per-position models (F7) |
| `selection_experiments.py` | Fixture cap, captain rules (F8, F9) |
| `validate_on_real_folds.py` | Candidate layers on the notebook's own cached folds |
| `validate_chip_timing.py` | Chip policy vs random (S4) |
| `team_value_model.py` | Price prediction and value-aware play (S5, F3) |
| `bonus_model.py` | Rank-based bonus (F2) |
| `deadline_check.py` | Pre-deadline re-run, diff and availability check |
| `make_charts.py` | Diagnostic charts |
| `fpl2/run_g9.py` | Do manual Championship priors fix the new-to-PL blind spot? (F16, G9) |
| `fpl2/run_g9_detail.py` | Matched vs unmatched split — is the effect diluted? (S9) |
| `fpl2/run_g9_xi.py` | Decision metric: unconstrained best-legal-XI points (F16, instrument 1) |
| `fpl2/run_g9_sim.py` | Decision metric: budget-constrained squad + SHUFFLED negative control (F16, instrument 2, R26) |
| `fpl2/fpl2/external.py::load_manual` | Reads hand-exported FBref tables from `data/external_manual/` |
| `fpl2/fpl2/simple.py` | The no-training tier model (S10, G10) |
| `fpl2/fpl2/chips.py` | SHIPPED chip policy: rolling 4-gameweek horizon (S25, M17, M18) |
| `fpl2/run_chips.py` | What chips are worth; calendar vs hindsight (S23) |
| `fpl2/run_chip_policy.py` | Secretary-rule timing (F21, F22) |
| `fpl2/run_chip3.py` | Four timing policies compared (S24, withdrawn by M17) |
| `fpl2/run_chip_honest.py` | Timing without future fixture knowledge (S25) |
| `fpl2/run_s1_check.py` | Does S1 reproduce? Simulator handicap (S22) |
| `fpl2/run_template.py` | Most-owned XI as a stand-in for the average |
| `fpl2/run_minutes_fix.py` | Minutes recency blend vs legacy vs average (F20) |
| `fpl2/run_o28.py` | Decision-conditioned gate for the minutes fix (O29) |
| `fpl2/diag_minutes_coldstart.py` | Minutes over-shrinkage by start rate (S19) |
| `fpl2/diag_bruno.py` | Why Bruno outranks Haaland (S20) |
| `fpl2/run_simple_gate.py` | Simple model vs legacy on the unmodified G1 gate |
| `fpl2/run_decompose.py` | Where legacy's ~300 pts come from: opening squad vs rest (S11) |
| `fpl2/run_opening.py` | Opening-squad horizon sweep + hindsight bound |
| `fpl2/run_shrink_sweep.py` | Empirical-Bayes shrinkage dose-response (O15) |
| `fpl2/run_captain_floor.py` | Noise floor for ARMBAND points (R27) |
| `fpl2/run_captain_rules.py` | 13 captain arms, disagreement rates, paired inference (F18) |
| `fpl2/run_captain_control.py` | Shuffled-ceiling negative control (R26) |
| `fpl2/run_captain_squad.py` | Captain rule inside a budget-constrained squad |
| `fpl2/run_ceiling_gate.py` | Ceiling features A/B + G1 (F17) |
| `fpl2/run_ceiling_minperiods.py` | min_periods robustness for the ceiling signal |
