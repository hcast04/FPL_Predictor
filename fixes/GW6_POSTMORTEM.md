---
document: GW6 post-mortem — what the notebook recommended vs what happened
version: v2.6.1
as_of: 2026-10-13
status: provisional — FPL has not run its final data check on GW6
---

# GW6 post-mortem

## Verdict

**The recommendation was bad luck, not a bad decision. The season is being lost
somewhere else entirely — and that somewhere else is fixable.**

The committed GW6 line-up scored **28** against a league average of **55**. But
it still beat both alternatives available at the deadline. The real damage is a
six-week structural pattern the single-week result almost hid.

## 1. What was recommended, and what it scored

Two different recommendations were produced for GW6:

| | generated | transfers | captain | projected |
|---|---|---|---|---|
| Thursday | 08 Oct 08:22Z | Palmer → Saka; Dunk → Hall | Saka | 51.15 |
| **Friday (committed)** | **09 Oct 14:50Z** | **João Pedro → Wissa; N.Williams → Gabriel** | **B.Fernandes** | **51.32** |

Both reported `decision_confidence: Low`. The Friday run — the scheduled
pre-deadline re-run — replaced *every* decision from Thursday: both transfers
and the captain. Two runs 30 hours apart, projections 0.17 points apart,
recommending completely disjoint actions. That is what `Low` confidence looks
like in practice, and it is the honest reading: at this horizon the model cannot
separate these options.

Scored against actuals, with FPL autosub rules applied:

| path | actual |
|---|---|
| Thursday's recommendation (dropped) | 23 |
| no transfers at all (roll both) | ~29 |
| **Friday's recommendation (committed)** | **30** |

So the committed path was the best of the three. Re-running on Friday was worth
**+7** over Thursday's answer. The transfer machinery did its job.

> The official score currently reads **28**, not 30, because GW6 shows
> `finished: false` / `data_checked: false` — FPL has not applied autosubs yet.
> Szoboszlai played 0 minutes, so Havertz (2) should come on, taking it to 30.
> All figures here are provisional until the data check runs.

## 2. Why it still scored 28

Two appearance-model failures in opposite directions, in the same week:

| player | model said | actually |
|---|---|---|
| Szoboszlai | p(appear) **0.94**, 76.6 expected minutes | **0 minutes** |
| João Pedro | p(appear) **0.33**, 18.7 expected minutes | **71 minutes, 12 points** |

Neither carried an injury flag. These came from the minutes layer's own
rotation read, and both were wrong. The João Pedro miss is the more serious of
the two because the Friday transfer *sold him* on the strength of that 0.33 —
and he outscored the entire rest of the squad.

On top of that, correlated fixture exposure landed badly: three defenders
(Botman, Thiaw, Virgil) in two matches that produced COV 4-1 NEW and LIV 0-3 MCI.
**The pre-deadline verification explicitly flagged the Newcastle defender
concentration and it was not acted on.** That is the one process failure in the
week — the warning was generated and ignored.

Captain B.Fernandes returned 1 point. Thursday's Saka would have returned 2.
Neither choice was available to be "right"; the armband was a coin flip between
two blanks.

## 3. Is this week evidence the model is broken? No.

Measured on the notebook's own 114 out-of-sample fold gameweeks, the standard
deviation of (actual − projected) for a best XI is **18.2 points**. A −21 miss
is inside the ordinary spread of this model's weekly error.

One week is far inside the noise band and **must not be used to re-tune
anything**. That is the project's own standing rule: the live sample is a shift
detector, not a tuning set. Nothing in this post-mortem changes a model
coefficient, a gate, or a flag.

The same fold data flags a separate calibration point worth noting: across those
114 gameweeks the model *under*-projects the top end by a mean of +12.6 points.
That is the original audit finding — high-minutes, high-points players have the
worst xP error — still present, and still unfixed.

## 4. Where the season is actually being lost

This is the part that is not noise. Six gameweeks, not one.

### Leak 1 — the squad carries two starting goalkeepers

| GW | started | pts | benched | pts | min |
|---|---|---:|---|---:|---:|
| 1 | Lammens | 1 | Verbruggen | 6 | 90 |
| 2 | Lammens | 2 | Verbruggen | 0 | 90 |
| 3 | Verbruggen | 3 | Lammens | 2 | 90 |
| 4 | Verbruggen | 8 | Lammens | 2 | 90 |
| 5 | Lammens | 4 | Verbruggen | 6 | 90 |
| 6 | Lammens | 2 | Verbruggen | 9 | 90 |

Both keepers have started all six matches. Only one can ever be fielded, so the
other's points are structurally unreachable — **25 of the 62 bench points this
season**.

Worse, the model is picking the wrong one. Started keepers scored **20**;
always starting Verbruggen would have scored **32**. That is **+12 in six
weeks (~76 a season)** from a single line-up decision, and Verbruggen is both
the better scorer (32 vs 13) and the cheaper asset (£4.5m vs £4.9m).

The xP gap the model is deciding on was 3.39 vs 3.13 — a coin flip it has now
lost four times out of six.

### Leak 2 — 62 points left on the bench in six weeks

| GW | 1 | 2 | 3 | 4 | 5 | 6 | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| bench pts | 11 | 3 | 8 | 16 | 10 | 14 | **62** |

**10.3 points per gameweek**, equal to 21% of everything scored. GW4 is the
clearest case: João Pedro scored 12 on the bench while the captain returned 2.

### Leak 3 — the armband is returning almost nothing

| GW | captain | pts | best in squad | pts |
|---|---|---:|---|---:|
| 1 | Havertz | 6 | Palmer | 13 |
| 2 | Maguire | 4 | B.Fernandes | 23 |
| 3 | Virgil | 6 | Tavernier | 10 |
| 4 | B.Fernandes | 2 | Dunk | 12 |
| 5 | B.Fernandes | 2 | Virgil | 8 |
| 6 | B.Fernandes | 1 | Verbruggen | 9 |

Extra points from the armband: **21 in six weeks**. Captaining Haaland blindly
every week would have returned **45**. Perfect hindsight was 75.

Captaining a defender (Virgil) and a £4.5m centre-back (Maguire) are decisions
worth questioning on their own terms. The captain model is choosing from a squad
with no genuinely high-ceiling asset in it, which is a squad-construction
problem the optimiser cannot solve at line-up time.

### The trajectory

| GW | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| you | 54 | 67 | 47 | 63 | 39 | 28 |
| average | 50 | 81 | 51 | 69 | 48 | 55 |
| vs avg | +4 | −14 | −4 | −6 | −9 | **−27** |

**298 points, 56 behind average, overall rank 3.2M → 8.1M.** Five consecutive
gameweeks below average is no longer a variance story.

## 5. What to change

Nothing in the model. Three things around it.

1. **Fix the goalkeeper slot.** Start Verbruggen, not Lammens, until the
   evidence reverses. Better: sell Lammens for a genuine non-playing £4.0–4.5m
   backup and put the money into the XI. Worth roughly 76 points a season on
   current form, and it is the single most certain gain available.
2. **Make the captain gap a hard check.** The card already flags near-ties under
   0.35 xP. Three straight B.Fernandes captaincies returning 2, 2, 1 against a
   squad maximum of 12, 8 and 9 says the armband is being assigned on a margin
   the model cannot actually resolve.
3. **Act on the correlated-exposure warning.** It fired before this deadline,
   named the Newcastle defenders, and was correct. It should block or at least
   force an acknowledgement, not print alongside forty other diagnostics.

## 6. What this post-mortem does not claim

- GW6 is not final; autosubs and the data check are outstanding.
- The "no transfers" counterfactual (~29) reconstructs an XI the model was never
  asked to produce. Treat committed-vs-nothing as a wash, not as a measured +1.
- The 114-gameweek fold distribution uses an unconstrained best XI drawn from the
  whole player pool, so its levels do not transfer to a real budget-constrained
  squad. Only the spread (σ ≈ 18) is used here.
- One gameweek cannot confirm or refute any modelling change, in either
  direction.
