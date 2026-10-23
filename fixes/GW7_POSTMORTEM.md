---
document: GW7 post-mortem + minutes/xP investigation
version: v2.6.1
as_of: 2026-10-19
status: provisional — GW7 not data-checked; TOT v COV unplayed (no squad exposure)
---

# GW7 post-mortem

**37 points against an average of 55. 23 on the bench. The card was executed
exactly.** Verbruggen started, Anderson came in, Palmer took the armband.

## 1. The week

| | pts |
|---|---|
| what was fielded | **37** |
| model's exact card (B.Fernandes C) | 36 |
| card + Groß instead of Anderson | 41 |
| card + Gabriel captain | 43 |
| best legal XI from the **same 15** | **62** |

The squad was good enough to beat the average by 7. Selection delivered 18 below.

Newcastle triple-up (NEW 0-1 AVL): Botman 3, Thiaw 1, Wissa 1 — **5 points from
three starters**. Flagged before GW6 and again before GW7; it has now cost two
weeks running. The optimiser still applies no penalty for single-fixture
concentration.

Bench: Tavernier 10, Dunk 6, Virgil 4, Lammens 3. **Running total 85 benched
points in 7 gameweeks.**

## 2. Correction: the minutes model is not broken

After GW7 I claimed three consecutive weeks of minutes-layer failure
(Szoboszlai, João Pedro, Anderson). **That claim was wrong.** Measured across
GW4–GW7, 1,394 player-gameweeks:

| | value |
|---|---|
| mean predicted minutes | 29.26 |
| mean actual minutes | 30.30 |
| bias | **−1.04** |
| correlation | 0.820 |

Of players predicted ≥70 minutes, **5.8% played zero** and 9.6% played under 30.
With eleven starters that is roughly one near-blank per gameweek *as the
expected rate*. Szoboszlai and Anderson were that base rate arriving, not a
defect. The stated appearance probabilities are close to honest: 0.95 claimed
against 0.942 realised in that bucket.

**One genuine weakness did show up.** Against a naive "average of the last three
gameweeks" baseline:

| | MAE | corr | bias |
|---|---:|---:|---:|
| model | 13.26 | **0.820** | −1.04 |
| naive last-3 | **12.77** | 0.787 | +1.61 |

The model wins on ranking and loses on absolute error. For team selection
ranking is what matters, so this is not a reason to replace the layer — but a
bespoke minutes model that cannot beat a three-game average on MAE is not
earning its complexity either.

## 3. The xP model under-predicts its own top end

On 86,755 out-of-sample player-gameweeks from the walk-forward fold cache:

| xP bucket | n | predicted | actual | bias |
|---|---:|---:|---:|---:|
| 3.0–3.5 | 5,924 | 3.22 | 3.46 | −0.24 |
| 3.5–4.0 | 2,057 | 3.69 | 4.33 | **−0.64** |
| 4.0–4.5 | 364 | 4.17 | 4.78 | −0.61 |
| 4.5–5.0 | 87 | 4.70 | **6.53** | **−1.83** |
| 5.0+ | 28 | 5.38 | 7.25 | −1.87 |

The model is **conservative where it matters most**, by 30–40% in the captaincy
region. This is the project's original audit finding — high-minutes, high-points
players carry the worst xP error — confirmed at scale, and still unfixed.

The ranking itself is sound: the model's **#1 pick averages 6.07 actual points**,
against 4.94 for its top ten and 0.00 for the league median. When it says
someone is the best captain in the game, it is usually right.

(An apparent *over*-prediction at 4.5–5.0 in the GW4–7 live sample was n=8 —
noise in the opposite direction. It does not survive the larger sample.)

## 4. The real finding: the ceiling is the squad, not the model

If the model ranks well and under-rates its own top end, the captaincy losses
cannot be a captaincy-logic problem. They are an ownership problem:

| GW | model's global #1 | actual | best in this squad | actual | gap |
|---|---|---:|---|---:|---:|
| 4 | B.Fernandes (4.22) | 2 | B.Fernandes (4.22) | 2 | 0 |
| 5 | Haaland (4.76) | 6 | B.Fernandes (4.26) | 2 | −4 |
| 6 | B.Fernandes (5.02) | 1 | B.Fernandes (5.02) | 1 | 0 |
| 7 | **Haaland (5.63)** | **13** | B.Fernandes (4.78) | 1 | **−12** |

Captaining the model's global #1 across GW4–7 would have returned **44 armband
points**. The best available inside this squad returned **12**.

**−32 points in four gameweeks, purely from not owning the top of the ranking.**

The model identified Haaland as the best captain in the game in both GW5 and
GW7. It cannot act on that, because the optimiser only ever chooses from the
fifteen players already owned.

## 5. What this changes

The three leaks from the GW6 post-mortem are now better understood:

1. **Goalkeeper** — fixed itself in GW7 (Verbruggen started, 6 pts). Lammens is
   still £4.9m of unusable bench.
2. **Bench waste** — 85 points in 7 weeks, unaddressed.
3. **Captaincy** — *not* a model defect. A squad-construction ceiling worth
   roughly 8 points per gameweek.

That reframes the wildcard question. The case does not rest on a bad week; it
rests on a measured −32 over four gameweeks from a structural gap the weekly
optimiser has no instrument to close.

## 6. Not checked

- Whether the Man City rotation in GW7 (Cherki 0 min, Semenyo 13, Anderson 13)
  was detectable pre-deadline from rotation history.
- Whether the minutes layer is specifically worse on newly-transferred-in
  players — the live archive only covers GW4–7, too few transfers to test.
- GW7 is not data-checked and bonus is provisional, though no squad player is in
  the unplayed TOT v COV fixture.
- Whether a wildcard now actually beats holding, which needs the optimiser run
  against the full pool rather than this squad.
