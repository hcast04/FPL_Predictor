# fpl2 — the rebuild

A rewrite of the notebook engine as a Python package, built around one idea:
**a result is only as good as the instrument that measured it.**

The notebook version shipped findings worth 100+ points a season that later
turned out to be noise. `fpl2` exists because the measuring apparatus was the
problem, not the model. Everything here is gated on a calibrated instrument
with a stated resolution, over **ten seasons (2016-17 .. 2025-26)** with
**season-clustered** standard errors.

## What the measurements say

| | |
|---|---|
| chip timing (`chips.py`) | **+119.8 pts/season**, 7/8 seasons |
| making transfers at all | **~+245/season** vs holding a squad |
| per-position rate layer | **+17.4** vs pooling it (p 0.002) |
| Understat features | ~+7/season on recent data, falling as Opta accumulates |
| **everything else tested** | **zero** |

That last row is the point. Twenty-five model architectures were raced —
Poisson GLM, HistGradientBoosting, XGBoost, a hurdle, per-90 × minutes, an
80th-percentile quantile head, per-position splits, rank ensembles — and
**none beat the component model**. Seven captaincy rules were tested against
`argmax(xp)` and all lost. Transfer thresholds, bench weight and
fixture-concentration all measured null.

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
tests/           81 tests, including truncation-equivalence leak tests
```

## Weekly use

Set your FPL team id once — it is the number in
`fantasy.premierleague.com/entry/<id>/`:

```bash
export FPL_ENTRY=1234567
```

```bash
python fpl2/run_live_chip.py       # rebuild the forecast, ask the chip policy
python fpl2/run_live_project.py    # project the next four gameweeks
python fpl2/plan_gw.py             # XI, captain, best transfer
WC_BUDGET=<selling value + bank> WC_VARIANT="max 2/club" \
  python fpl2/build_wildcard_now.py
```

`WC_BUDGET` must come from **selling** prices, not the API's `entry_history.value`,
which is the purchase-price valuation and overstates a wildcard budget.

## The ledger

`../fixes/EXPERIMENT_LEDGER.md` — 4,500+ lines recording every experiment,
including the refuted ones and the retractions. Read it before proposing an
idea; most have been tested. It carries the standing rules, among them:

- **R32** cluster by season, always — the naive SE answers the wrong question
- **R41/R43** a correction monotone in the existing score cannot reorder it,
  however large it is; `argmax` is blind to monotone transforms
- **R38** a dose-response that never saturates is a leak until proven otherwise
- **R44** a backtest cannot validate the live path, because it does not run it

Data (`data/`) is gitignored: ~541 MB of FPL exports, Understat cache and
walk-forward fold caches, all regenerable by the collectors.
