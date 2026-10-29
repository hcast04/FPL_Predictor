# fpl2 — rebuild

A clean rebuild of the FPL predictor. The old pipeline
(`FPL_Predictor_v2.6.ipynb`) is untouched and keeps producing weekly
recommendations until this replaces it.

## The one rule

**The ruler was built before anything it measures.** `run_noise_floor.py`
establishes how much a season total moves when the signal is perturbed with
information-free noise. Nothing smaller than that is a result.

### Measured 2026-10-20, on the legacy signal

| jitter | single-season range | sd of the cross-season mean |
|---|---:|---:|
| 0.1% | up to **125 pts** | 12.9 |
| 1% | up to **268 pts** | 49.5 |
| 5% | up to 388 pts | 37.8 |

> **A season-total improvement under ~150 points is not believable.**
> Independently reproduced: a separate implementation gave ~150 on 2026-10-19.

Re-measure whenever the signal or the policy changes.

## Layout

| module | responsibility |
|---|---|
| `config.py` | FPL rules and paths. No logic. |
| `data.py` | the canonical panel: one row per (season, gameweek, player_code) |
| `optimise.py` | exact MILP squad; exact best legal XI; captain |
| `simulate.py` | season under real constraints — decisions on `signal`, points on `actual` |
| `noise.py` | the noise-floor harness |
| `legacy.py` | **temporary** bridge to the old pipeline's cached forecasts |

## Design decisions, and the bug each one prevents

| decision | prevents |
|---|---|
| `player_code` is the only join key | "Palmer" is two different players |
| one row per player-gameweek, `fixtures` recorded | losing the second fixture of a double |
| `with_blanks()` materialises absent rows | a blank gameweek read as "no information" instead of "scores nothing" |
| `signal` and `actual` are always separate columns | a forecast scored against itself |
| `best_xi` tested against brute force over all C(15,11) | a plausible-looking but non-optimal line-up |
| noise floor before any comparison | believing a reshuffle |

## Running

```bash
.venv/bin/python -m pytest fpl2/tests -q      # 46 tests
.venv/bin/python fpl2/run_noise_floor.py      # ~50s
```

## Status

Done: data layer, optimiser, simulator, noise floor, tests.
Next: the forecasting model. It must beat the legacy signal by more than the
noise floor, in all three holdout seasons, against a gate declared in
`fixes/EXPERIMENT_LEDGER.md` before the run.
