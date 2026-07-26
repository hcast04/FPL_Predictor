# FPL Predictor v1.0.1

A data-driven forecasting and decision-support system for Fantasy Premier League.

The project uses live FPL data, historical player-fixture records, statistical forecasting and constrained optimisation to recommend:

- the initial 15-player squad;
- transfers or rolling a free transfer;
- the starting XI and formation;
- bench order;
- captain and vice-captain;
- whether to use a chip;
- alternative actions and their projected cost.

The analysis is rerun before every Gameweek deadline. Only the immediate recommendation is intended to be executed; later-Gameweek plans provide context and are recalculated when new information becomes available.

> **Status:** production-ready decision-support notebook.  
> It is designed to support FPL decisions, not to guarantee points or rank.

---

## Contents

1. [Quick start](#quick-start)
2. [Installation](#installation)
3. [Project structure](#project-structure)
4. [Weekly use](#weekly-use)
5. [Outputs](#outputs)
6. [How the system works](#how-the-system-works)
7. [Modelling and optimisation choices](#modelling-and-optimisation-choices)
8. [Validation](#validation)
9. [Production safeguards](#production-safeguards)
10. [Limitations](#limitations)
11. [Data sources](#data-sources)
12. [Version history](#version-history)

---

## Quick start

### First complete run

1. Create and activate a Python environment.
2. Install the required packages.
3. Open `fpl_predictor.ipynb`.
4. Run the notebook from the beginning.
5. Run the full walk-forward validation in Section 11 at least once.
6. Use Section 12 for the final Gameweek recommendation.

The first run downloads and prepares historical data, fits the forecasting models, produces live projections and creates local cache files.

### Normal weekly run

Before each deadline:

1. Restart the kernel.
2. Run the notebook through **Section 10: Chip evaluation and timing**.
3. Update the live team state in Section 12.
4. Run Section 12.
5. Review the primary action, alternatives, warnings and chip-readiness decision.
6. Execute only the current Gameweek recommendation.

Sections 10A–11 are validation sections and do not need to be rerun every week unless the production policy is being recalibrated.

---

## Installation

### Requirements

- Python 3.10 or later
- Jupyter Notebook or JupyterLab
- Internet access for live FPL data and initial historical-data downloads

Install the core dependencies:

```bash
pip install numpy pandas requests scipy scikit-learn jupyter
```

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
fpl_predictor.ipynb
```

The notebook includes an environment preflight check and raises a clear error when a required package or incompatible runtime is detected.

---

## Project structure

The project deliberately keeps the number of files small.

```text
fpl_predictor/
├── fpl_predictor.ipynb
├── README.md
├── data/
│   ├── historical_fpl.csv.gz
│   └── fpl_weekly_snapshots.json.gz
└── outputs/
    └── latest_weekly_recommendation.json
```

### Files

`fpl_predictor.ipynb`  
Contains the complete data, modelling, optimisation, validation and weekly recommendation workflow.

`data/historical_fpl.csv.gz`  
Local compressed cache of the merged historical player-fixture dataset. It avoids downloading and rebuilding the same historical archive on every run.

`data/fpl_weekly_snapshots.json.gz`  
Compressed pre-deadline snapshots used to preserve what the model knew at the time of each live recommendation.

`outputs/latest_weekly_recommendation.json`  
The latest consolidated recommendation. It is overwritten rather than creating a new output file every week.

---

## Weekly use

Section 12 is the production interface.

### Current squad

After Gameweek 1, enter exactly 15 players:

```python
WEEKLY_CURRENT_SQUAD = [
    # Official FPL player IDs or exact player names
]
```

Players can be entered using:

- official FPL player IDs;
- exact `web_name` values;
- exact full names from the current player table.

Before Gameweek 1, leave the list empty. The notebook will select an initial legal squad instead of treating the action as a transfer.

### Money and free transfers

```python
WEEKLY_MONEY_IN_BANK = 0.0
WEEKLY_FREE_TRANSFERS = 1
```

Amounts are expressed in millions of pounds, following FPL notation.

### Purchase and selling prices

```python
WEEKLY_PURCHASE_PRICES = {
    # "Player name": 7.5,
}

WEEKLY_SELLING_PRICES = {
    # "Player name": 7.7,
}
```

Explicit selling prices take precedence.

When purchase prices are supplied, the notebook reconstructs the FPL selling price where possible. Omitting both values makes the current market price the fallback.

### Available chips

Update the active half-season chip set after using a chip:

```python
WEEKLY_AVAILABLE_CHIPS = {
    "triple_captain": True,
    "bench_boost": True,
    "free_hit": True,
    "wildcard": True,
}

WEEKLY_LAST_FREE_HIT_GAMEWEEK = None
```

### Policy selection

```python
WEEKLY_POLICY_PROFILE = "auto"
WEEKLY_POLICY_FALLBACK = "rolling_6gw_no_hits"
```

With `auto`:

- Section 12 uses the policy selected by the completed Section 11 validation when that result exists in memory;
- otherwise it uses the conservative six-Gameweek, no-hit fallback.

The currently validated historical selection is the one-Gameweek, no-hit policy. The wider fixture horizon remains visible as strategic context even when the immediate action is chosen primarily from the next Gameweek.

Available policy profiles are:

```text
greedy_1gw_no_hits
rolling_3gw_no_hits
rolling_6gw_no_hits
rolling_6gw_hits
```

### Late information overrides

Reliable information may appear after the FPL API was last updated, such as a confirmed injury or credible press-conference statement.

Use multiplicative projection adjustments:

```python
WEEKLY_CURRENT_GW_PROJECTION_MULTIPLIERS = {
    # "Player name": 0.0,  # confirmed unavailable this Gameweek
}

WEEKLY_HORIZON_PROJECTION_MULTIPLIERS = {
    # "Player name": 0.5,  # reduced expected availability over the horizon
}
```

Values must be between 0 and 1.

Use these overrides sparingly and only for reliable information.

---

## Outputs

Section 12 produces one consolidated decision report.

### Primary recommendation

The dashboard shows:

- initial squad selection, transfer or roll;
- transfer(s) in and out;
- expected gain versus the best roll/no-transfer path;
- starting XI and formation;
- captain and vice-captain;
- bench order;
- expected current-Gameweek score;
- chip recommendation;
- decision confidence;
- warnings and assumptions.

### Alternative actions

The best alternative immediate actions are displayed with their projected objective values. This makes it possible to see whether the recommendation is decisive or marginal.

A small difference between the top two actions should be treated differently from a clear expected-points advantage.

### Future context

The notebook also shows a provisional transfer and chip outlook across the active planning horizon.

This is not a commitment. The future plan is recalculated before the next deadline using updated:

- fixtures;
- injuries;
- prices;
- recent performances;
- minutes;
- team strength;
- available transfers;
- chip state.

### JSON output

The latest recommendation is saved to:

```text
outputs/latest_weekly_recommendation.json
```

The file includes:

- resolved policy and its source;
- current team state;
- recommendation;
- transfer alternatives;
- XI, captaincy and bench;
- final chip decision;
- raw chip candidate before the readiness gate;
- freshness audit;
- warnings;
- generation timestamp.

---

## How the system works

```text
Official live FPL data
        +
Historical player-fixture data
        ↓
Leakage-safe feature engineering
        ↓
Minutes and rotation model
        ↓
Team-strength and player-event models
        ↓
Component-level expected FPL points
        ↓
Squad, line-up and captain optimisation
        ↓
Transfer-path search
        ↓
Chip evaluation and readiness gate
        ↓
Weekly recommendation dashboard
```

Prediction and decision-making are deliberately separate.

The forecasting layer estimates player and team outcomes. The optimisation layer decides what to do with those estimates under FPL constraints.

This separation makes the project easier to:

- interpret;
- debug;
- backtest;
- recalibrate;
- extend when FPL rules change.

---

## Modelling and optimisation choices

### Player-fixture modelling unit

The fundamental row is one player in one fixture.

This is necessary because a player can have:

- one fixture in a Gameweek;
- no fixture in a blank Gameweek;
- multiple fixtures in a double Gameweek.

Fixture predictions are later aggregated to player-Gameweek projections.

### Expected minutes and rotation

Minutes are modelled as related components rather than one opaque regression:

```text
P(appearance)
P(start)
P(60+ minutes)
Expected minutes if starting
Expected minutes if appearing as a substitute
```

The final expectation is:

```text
Expected minutes
= P(start) × E(minutes | start)
+ P(substitute appearance) × E(minutes | substitute appearance)
```

The probabilities are reconciled so that:

```text
P(start) ≤ P(appearance)
P(60+ minutes) ≤ P(start)
P(no appearance) + P(start) + P(sub appearance) = 1
```

Historical features use only information available before the relevant Gameweek.

### Team strength

Poisson models estimate:

- expected team goals;
- expected goals conceded;
- clean-sheet probability.

Inputs include rolling team attacking and defensive performance, opponent strength and home advantage.

### Player event rates

Separate position-aware models estimate relevant event rates, including:

- goals;
- assists;
- goalkeeper saves;
- bonus;
- cards.

Goalkeepers do not receive ordinary attacking projections in the baseline.

Defenders, midfielders and forwards use separate attacking models to avoid excessive shrinkage toward a shared average.

### Component-level expected points

Expected points are calculated through transparent FPL components:

```text
Appearance
+ goals
+ assists
+ clean sheets
+ saves
+ defensive contributions
+ bonus
− goals-conceded deductions
− cards and rare-event deductions
```

Threshold-based events use expected probabilities rather than rounded averages.

Examples include:

- save points at 3, 6 and 9 saves;
- goals-conceded deductions at 2, 4 and 6 goals;
- defensive-contribution thresholds.

### Defensive contributions

Defensive-contribution probabilities combine:

- expected action volume;
- shifted historical qualification rates;
- position-level priors;
- expected minutes.

This prevents the threshold from becoming almost automatic for centre-backs while preserving genuine value for high-volume defenders.

### Squad and line-up optimisation

The optimiser enforces:

```text
15 total players
2 goalkeepers
5 defenders
5 midfielders
3 forwards
Maximum 3 players per club under normal circumstances
£100.0m initial budget
```

The starting XI must contain:

```text
1 goalkeeper
3–5 defenders
2–5 midfielders
1–3 forwards
11 starters
```

The optimiser selects:

- squad;
- starting XI;
- bench order;
- captain;
- vice-captain.

Automatic-substitution scenarios and captain fallback are included in expected scoring.

A real-world player transfer can temporarily leave an owned squad above the three-player club limit. The notebook permits that transitional state but requires subsequent transfers to reduce the excess.

### Transfer planning

The full multi-Gameweek transfer space is too large to enumerate exhaustively.

The notebook therefore uses beam search:

1. generate legal transfer actions;
2. score the resulting current-Gameweek squad;
3. estimate the remaining planning horizon;
4. retain the strongest distinct squad states;
5. repeat for later projected Gameweeks.

The transfer engine handles:

- rolling;
- one or two transfers;
- banked free transfers;
- selling prices;
- money in the bank;
- transfer-hit deductions when the selected policy permits them;
- terminal squad value;
- club and positional constraints.

Future prices are held constant inside a planning horizon and refreshed at the next real deadline.

### Chip evaluation

Every chip is evaluated through its incremental expected value.

#### Triple Captain

Adds one extra expected copy of the captain’s score beyond normal captaincy.

#### Bench Boost

Compares:

```text
Bench Boost uplift
= E(all 15 players counting)
- E(normal score with automatic substitutions)
```

#### Free Hit

Compares the best temporary one-Gameweek squad with the best normal path, including avoided transfer costs.

#### Wildcard

Compares the best permanent rebuilt squad over the remaining horizon with the best path without using the chip.

### Chip readiness gate

A positive or locally maximal uplift does not automatically trigger a chip.

In conservative mode, the use-now recommendation also requires structural evidence such as:

- a blank or double Gameweek;
- chip-expiry pressure;
- exceptional uplift;
- a strong squad-restructuring reason for a Wildcard;
- avoided paid transfers or other clear strategic value.

The raw chip candidate remains visible even when the production gate recommends saving the chip.

---

## Validation

### Leakage prevention

Historical forecasts are frozen at the simulated deadline.

For each forecast origin:

- player form and minutes use only earlier Gameweeks;
- team strength uses only completed earlier matches;
- future outcomes cannot update an earlier transfer horizon;
- actual points are accessed only after the simulated decision is fixed.

The notebook includes assertions that stop execution if future information enters historical features.

### Walk-forward folds

The expected-points system is evaluated using expanding chronological folds:

| Holdout season | Training seasons |
|---|---|
| 2023/24 | 2022/23 |
| 2024/25 | 2022/23–2023/24 |
| 2025/26 | 2022/23–2024/25 |

The component xP model outperformed its rolling-points baseline on mean absolute error in all three folds and produced stronger rank correlation in each fold.

Observed holdout results:

| Season | Component xP MAE | Rolling MAE | Component Spearman | Rolling Spearman |
|---|---:|---:|---:|---:|
| 2023/24 | 0.937 | 1.008 | 0.680 | 0.673 |
| 2024/25 | 1.007 | 1.057 | 0.703 | 0.683 |
| 2025/26 | 0.976 | 1.056 | 0.718 | 0.706 |

### Policy comparison

The transfer policy is selected by comparing:

```text
1-Gameweek, no hits
3-Gameweek, no hits
6-Gameweek, no hits
6-Gameweek, hits allowed
```

In the current three-season comparison, the one-Gameweek no-hit profile produced the strongest average realised points.

This result is treated as a production policy selection, not as proof that longer planning horizons are intrinsically inferior. Future fixtures remain useful context, and the policy should be recalibrated as additional seasons and live decisions become available.

### Interpretation

Historical backtests are evidence of internal consistency and comparative value. They are not expected future points or rank guarantees.

A static opening squad is a weak benchmark. Strategy gains against it should not be interpreted as gains over an expert FPL manager.

---

## Production safeguards

### Freshness audit

Before generating a weekly recommendation, Section 12 compares the in-memory projections with the latest official FPL API state.

It checks:

- next Gameweek;
- player prices;
- availability status;
- player news;
- fixture assignments;
- fixture dates;
- blank and double-Gameweek changes.

With strict freshness enabled, the notebook stops instead of recommending from stale projections.

```python
WEEKLY_STRICT_FRESHNESS_CHECK = True
```

### Input validation

The production section validates:

- squad size;
- unique player references;
- position composition;
- player availability in the current pool;
- money in the bank;
- free-transfer range;
- chip-state values;
- price mappings;
- late-news multipliers;
- selected policy profile.

### Numerical stability

Event-rate predictions are bounded and checked for:

- missing values;
- infinite values;
- implausible projections;
- invalid probability relationships.

The notebook stops if numerical instability reappears.

### Decision confidence

The dashboard reports the projected gap between the best and second-best immediate actions.

This helps distinguish:

- a clear recommendation;
- a marginal decision that may be overturned by late team news or a small modelling error.

---

## Limitations

The notebook is production-ready as a decision-support system, but several limitations remain.

### Rotation and availability

The live model does not automatically collect all:

- press-conference information;
- predicted line-ups;
- European match minutes;
- domestic-cup minutes;
- reliable recovery dates;
- detailed competition for each squad position.

Manual projection multipliers may therefore be needed.

### Absolute calibration

Player ranking is more reliable than the exact displayed team-score total.

Historical results show some underprediction of weekly team scores. This matters particularly when interpreting:

- exact expected scores;
- small transfer gains;
- chip-uplift thresholds.

### Search approximation

Beam search controls runtime but can miss the global optimum.

Increasing candidate and beam widths may improve recall at the cost of substantially longer execution.

### Future prices

Future player prices are not forecast inside the planning horizon.

Current prices and selling prices are refreshed before the next deadline.

### Chip timing

Chip decisions are strongest when confirmed blanks, doubles or expiry pressure enter the explicit forecast horizon.

Long-range opportunities outside the modelled horizon remain uncertain.

### Historical data coverage

Historical defensive-action fields are concentrated in recent seasons. Earlier defensive-contribution estimates rely more heavily on priors.

The external historical archive can contain duplicate or incomplete records. The loader resolves known duplicate patterns and validates uniqueness after cleaning.

### Structural changes

FPL rules, scoring, APIs and data fields can change between seasons.

The notebook includes schema checks, but major rule changes may require model and optimisation updates.

---

## Data sources

### Official Fantasy Premier League API

Used for current-season:

- players;
- teams;
- prices;
- ownership;
- availability;
- player news;
- fixtures;
- Gameweek deadlines;
- live and completed Gameweek statistics.

The official API is treated as the source of truth for the live weekly state.

### Historical FPL archive

Vaastav’s Fantasy Premier League repository is used for completed historical player-fixture records, player metadata and team information.

The notebook:

- downloads selected seasons;
- standardises fields;
- removes rows without valid fixtures;
- resolves repeated player-fixture records;
- excludes known leakage-prone fields;
- merges seasons;
- stores one local compressed cache.

---

## Reproducibility

Every live recommendation should preserve:

- the current FPL data snapshot;
- the team state;
- the resolved policy;
- the generated recommendation;
- the generation timestamp.

This allows later evaluation of what the model knew at the deadline rather than judging decisions using information that appeared afterwards.

Set a fixed random seed where applicable and avoid modifying historical caches during an active evaluation without recording the change.

---

## Recommended operating discipline

Before acting on a recommendation:

1. Run the freshness check.
2. Review FPL player news.
3. Review reliable late injury and press-conference information.
4. Apply manual multipliers only when justified.
5. Inspect the top alternative action.
6. Check decision confidence.
7. Treat a chip recommendation more conservatively than a normal transfer.
8. Execute only the immediate action.
9. Save the resulting weekly state for the next deadline.

---

## Version history

### v1.0.1

- Added handling for temporary club-limit violations caused by real-world player transfers.
- Finalised the live weekly dashboard.
- Added automatic production-policy resolution.
- Added conservative chip-readiness gating.
- Added current-state freshness checks.
- Added Gameweek 1 initial-squad handling.
- Added input and numerical-stability validation.
- Added full walk-forward policy comparison.
- Added deadline-frozen historical forecasting.
- Corrected position-level attacking-rate shrinkage.
- Recalibrated defensive-contribution projections.
- Added recommendation JSON output.

### v1.0.0

- First complete end-to-end forecasting and optimisation pipeline.
- Added historical data preparation.
- Added minutes and rotation models.
- Added component expected-points models.
- Added squad, line-up, captain and bench optimisation.
- Added transfer beam search.
- Added chip evaluation.
- Added sequential and walk-forward backtesting.

---

## Disclaimer

This project is independent and is not affiliated with, endorsed by or sponsored by the Premier League or Fantasy Premier League.

Forecasts are uncertain. Injuries, line-ups, postponements, tactical changes, transfers and random match events can materially change realised outcomes.
