# fpl2 — the rebuild

A rewrite of the notebook engine as a Python package, built around one idea:
**a result is only as good as the instrument that measured it.**

The notebook shipped findings worth 100+ points a season that turned out to be
noise. `fpl2` exists because the measuring apparatus was the problem, not the
model. Everything is gated on a calibrated instrument with a stated
resolution, over **ten seasons (2016-17 .. 2025-26)** with **season-clustered**
standard errors (G=8, so the bar is |t| > 2.36).

## What the measurements say

| | |
|---|---|
| chip timing (`chips.py`) | **+119.8 pts/season**, 7/8 seasons |
| making transfers at all | **~+245/season** vs holding a squad |
| per-position rate layer | **+17.4** vs pooling it (p 0.002) |
| ten seasons of training data | **~+5/season** vs the best shorter window |
| Understat features | ~+7/season on recent data, falling as Opta accumulates |
| **everything else tested** | **zero, or below the instrument's resolution** |

That last row is the point, so here is what is in it:

- **25 model architectures** — Poisson GLM, HistGradientBoosting, XGBoost, a
  hurdle, per-90 × minutes, an 80th-percentile quantile head, per-position
  splits, rank ensembles. **None beats the component model.** Best challenger
  +8.0, p 0.371.
- **8 captaincy rules** against `argmax(xp)` — form blending at every λ,
  premium, quantile, upside, minutes-safety. **All lost.**
- **the rate ceiling** — clips Salah in 32 of 32 gameweeks; removing it
  entirely is worth **+0.8 against an MDE of 1.4**.
- **conditional bonus** — bonus is 122× more likely after a return, and
  conditioning on it measures **−1.8**, because it reorders by 2.4 places in
  600.
- **team playing style** — possession, shot quality, crossing, pressing from
  match-level FBref. **+0.1 against its own shuffled control.**
- transfer thresholds, bench weight, fixture concentration, ownership
  weighting, club-tenure shrinkage — all null.

**The two biggest effects ever found here were defects, not features**: a rate
layer that forecast zero goals and assists on 5 of 8 folds, and a ceiling
derived from a missing column that multiplied predictions by zero. That is why
the harness below exists.

## Layout

```
fpl2/            the package
  model.py       composes minutes × rates × team strength into xP
  rates.py       per-90 scoring rates, fitted per position
  minutes.py     the appearance hierarchy — the component that matters most
  team.py        goals conceded -> clean-sheet probability
  measure.py     THE INSTRUMENT: fixed-squad scoring with a stated MDE
  optimise.py    exact MILP for squad and XI
  chips.py       chip timing policy (the one confirmed lever)
  project.py     leak-free forward projection
  style.py       FBref team-style features (measured null; kept for reuse)
  verify.py      the ledger's rules as executable checks
tests/           94 tests, including leak tests with their own negative controls
```

## Weekly use

Set your FPL team id once — the number in `fantasy.premierleague.com/entry/<id>/`:

```bash
export FPL_ENTRY=1234567
```

```bash
python fpl2/run_live_chip.py       # rebuild the forecast, ask the chip policy
python fpl2/run_live_project.py    # project the next four gameweeks
python fpl2/plan_gw.py             # XI, captain, vice, best transfer
WC_BUDGET=<selling value + bank> WC_VARIANT="max 2/club" \
  python fpl2/build_wildcard_now.py
```

Three things that are easy to get wrong and are handled here:

- **`WC_BUDGET` must come from selling prices**, not the API's
  `entry_history.value`, which is the purchase-price valuation and overstates
  a wildcard budget by ~1%.
- **The tools forecast the gameweek being decided**, not the last one played.
  They used to do the latter, which priced every recommendation against the
  previous week's opponents.
- **Run `run_live_project.py` before the chip policy.** It needs four
  gameweeks of visibility (`chips.KNOWN`), and a live grid holds only
  finished ones.

## Verification

`verify.py` turns the ledger's hard-won rules into functions, because a rule
that lives only in prose gets broken by its own author — four were, in a
single day.

| | |
|---|---|
| `reference_identical` | an arm that should reproduce the shipped model must do so bit-identically (pin `OMP_NUM_THREADS`: HistGradientBoosting is thread-count dependent) |
| `reorders` | rank correlation *before* building a fix — `argmax` is blind to monotone transforms |
| `stable_population` | every arm scored on the same rows |
| `dose_response` | monotone to the boundary ⇒ suspect a leak |
| `price_ceiling` | size the affected population before building anything |
| `strictly_lagged` | a leak test that also proves it can fail |
| `clustered` | refuses G=1 instead of dividing by zero; always reports the MDE |

Its 13 tests each replay a real defect with the real numbers, so the checks
are proven to fire rather than merely to pass.

## The ledger

`../fixes/EXPERIMENT_LEDGER.md` — 4,800+ lines recording every experiment,
including the refuted ones and the retractions. **Read it before proposing an
idea; most have been tested.** Standing rules include:

- **R26** a negative control separates information from perturbation
- **R30** a live-season table is a hypothesis generator, never evidence
- **R32** cluster by season, always — the naive SE answers the wrong question
- **R41/R43** a correction monotone in the existing score cannot reorder it
- **R44** a backtest cannot validate the live path, because it does not run it
- **R47** when an effect decays, find the clock it decays on
- **R49** a join key that works for one entity can be useless for another
- **R50** a rule that lives only in prose will be broken by its own author

Data (`data/`) is gitignored except the seed `historical_fpl.csv.gz`: the rest
is ~541 MB of FPL exports, Understat cache and fold caches, all regenerable.
