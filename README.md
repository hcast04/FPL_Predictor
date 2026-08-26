# FPL Predictor v2.0.2 — Autonomous Manager

A full-season Fantasy Premier League forecasting and decision engine.

The notebook is designed to make one authoritative decision before every FPL deadline:

- initial 15-player squad;
- transfers or rolling;
- whether a points hit is justified;
- starting XI and formation;
- bench order;
- captain and vice-captain;
- Wildcard, Free Hit, Bench Boost and Triple Captain use;
- future transfer planning as context for the immediate decision.

The intended operating rule is:

> **If Section 12 completes without a hard error, execute its primary current-Gameweek recommendation.**

Future transfer paths are provisional. The notebook is rerun before every deadline and re-solves the problem using the latest available information.

The objective is to maximise **expected season FPL points**. Football outcomes remain stochastic, so no model can guarantee the maximum realised score.

---

## Contents

1. [Quick start](#quick-start)
2. [Installation](#installation)
3. [Project structure](#project-structure)
4. [Autonomous operating model](#autonomous-operating-model)
5. [First-time setup](#first-time-setup)
6. [Normal weekly workflow](#normal-weekly-workflow)
7. [Outputs](#outputs)
8. [Forecasting pipeline](#forecasting-pipeline)
9. [New and returning players](#new-and-returning-players)
10. [Expected-points calibration](#expected-points-calibration)
11. [Optimisation](#optimisation)
12. [Chip strategy](#chip-strategy)
13. [Historical validation](#historical-validation)
14. [Production safeguards](#production-safeguards)
15. [Common runtime messages](#common-runtime-messages)
16. [Limitations](#limitations)
17. [Data sources](#data-sources)
18. [Version history](#version-history)

---

## Quick start

Open:

```text
fpl_predictor_v2_0_2_autonomous.ipynb
```

For normal deadline use:

```python
AUTONOMOUS_MODE = True

RUN_SEQUENTIAL_BACKTEST = False
RUN_FULL_POLICY_BACKTEST = False
```

Then:

```text
Restart kernel
    ↓
Run Sections 1–10
    ↓
Skip expensive historical validation
    ↓
Run Section 12
    ↓
Execute the primary recommendation
```

The production xP calibration and preferred transfer policy are loaded from versioned cache files created during the first validation run.

Do **not** routinely run the long historical backtests immediately before an FPL deadline. They are validation tools, not weekly inputs.

---

## Installation

Requirements:

- Python 3.10+
- Jupyter Notebook or JupyterLab
- internet access

Install:

```bash
pip install numpy pandas requests scipy scikit-learn soccerdata jupyter
```

Start Jupyter:

```bash
jupyter notebook
```

The notebook performs an environment preflight and stops early when a required package is unavailable.

---

## Project structure

```text
FPL_Predictor/
├── fpl_predictor_v2_0_2_autonomous.ipynb
├── README.md
│
├── data/
│   ├── historical_fpl.csv.gz
│   ├── fpl_weekly_snapshots.json.gz
│   ├── external_player_priors_v1_2.csv.gz
│   ├── xp_calibration_v2.json
│   ├── walk_forward_policy_scorecard_v2.csv.gz
│   ├── sequential_backtest_v2.pkl.gz
│   ├── autonomous_fpl_state_v2.json
│   ├── soccerdata_fbref/
│   └── soccerdata_understat/
│
└── outputs/
    └── latest_weekly_recommendation.json
```

Not every cache exists before the first complete run.

### `historical_fpl.csv.gz`

Compressed historical player-fixture dataset used for model training and chronological validation.

### `fpl_weekly_snapshots.json.gz`

Pre-deadline snapshots of the official FPL API. These make future live-decision backtests reproducible.

### `external_player_priors_v1_2.csv.gz`

Processed previous-season player priors used especially for new and returning Premier League players.

### `xp_calibration_v2.json`

Versioned out-of-sample expected-points calibration fitted in Section 11. Autonomous production requires this by default.

### `walk_forward_policy_scorecard_v2.csv.gz`

Cached comparison of transfer policies so the expensive policy backtest does not need to be rerun weekly.

### `sequential_backtest_v2.pkl.gz`

Optional cached full sequential strategy/chip backtest.

### `autonomous_fpl_state_v2.json`

Persistent season ledger containing the model's committed team state and its next pending recommendation.

### `latest_weekly_recommendation.json`

The latest consolidated production recommendation.

---

## Autonomous operating model

The real FPL team is represented through a persistent ledger.

### Committed state

The last state verified after a completed deadline:

```text
permanent squad
purchase prices
bank
banked free transfers
available chips
last Free Hit Gameweek
```

### Pending state

The recommendation the model expects you to execute for the next deadline.

For example:

```text
GW8
Transfer: Player A → Player B
Captain: Player C
Chip: None
```

After the deadline is completed, the next run compares the public FPL team with the pending recommendation.

If they match, the pending state becomes committed.

If they do not match and strict reconciliation is enabled, the notebook stops rather than silently optimising from the wrong team.

---

## First-time setup

### 1. Autonomous configuration

Keep:

```python
AUTONOMOUS_MODE = True
```

Before your FPL team exists:

```python
FPL_TEAM_ID = None
```

Once your entry exists, set the numeric team ID once:

```python
FPL_TEAM_ID = 1234567
```

Leave it set for the season.

### 2. Generate the calibration and policy caches

For the initial v2 validation:

```python
RUN_SEQUENTIAL_BACKTEST = False
RUN_FULL_POLICY_BACKTEST = True
WALK_FORWARD_MAX_GAMEWEEKS = None
```

Run the notebook through **Section 11**.

This creates or refreshes:

```text
data/xp_calibration_v2.json
data/walk_forward_policy_scorecard_v2.csv.gz
```

Section 11 can take a long time because it performs chronological forecast and policy validation across completed seasons.

### 3. Restart for production

After Section 11 finishes:

```python
RUN_FULL_POLICY_BACKTEST = False
RUN_SEQUENTIAL_BACKTEST = False
```

Restart the kernel.

Run Sections 1–10 and then Section 12.

### Optional sequential validation

After the xP calibration exists, the complete sequential strategy/chip backtest can be generated with:

```python
RUN_SEQUENTIAL_BACKTEST = True
```

This can be slow and is **not** needed for weekly production.

After it finishes:

```python
RUN_SEQUENTIAL_BACKTEST = False
```

The result is cached.

---

## Normal weekly workflow

Before each deadline:

```text
1. Restart the kernel.
2. Keep both historical-backtest flags False.
3. Run Sections 1–10.
4. Run Section 12.
5. If there is no hard error, execute the primary recommendation.
6. Ignore later provisional transfer-path actions.
7. Rerun the notebook before the next deadline.
```

Recommended production settings:

```python
AUTONOMOUS_MODE = True

AUTONOMOUS_STRICT_RECONCILIATION = True
AUTONOMOUS_REQUIRE_CALIBRATION_CACHE = True
AUTONOMOUS_ALLOW_MANUAL_PROJECTION_OVERRIDES = False

RUN_SEQUENTIAL_BACKTEST = False
RUN_FULL_POLICY_BACKTEST = False

WEEKLY_STRICT_FRESHNESS_CHECK = True
WEEKLY_POLICY_PROFILE = "auto"
```

The model uses a rolling/receding horizon. New information is incorporated every week instead of committing to a fixed season-long plan.

---

## Outputs

Section 12 is the production interface.

It reports:

- initial squad or transfer action;
- transfer(s) in and out;
- hit cost if applicable;
- starting XI;
- formation;
- captain;
- vice-captain;
- bench order;
- expected current-GW score;
- chip decision;
- decision confidence;
- future transfer path;
- warnings and audits.

### Alternatives

The strongest alternative actions are displayed with their path objectives.

The decision-confidence label is based on the objective gap between the best and second-best immediate actions.

### Selected-vs-omitted audit

The notebook compares selected players with strong omitted alternatives.

This is especially useful when a more expensive player has similar immediate xP to a cheaper alternative: the audit shows whether the selection is justified by the wider planning horizon and squad structure.

### New-player audit

The weekly output records:

- external-prior availability;
- matching method;
- source league;
- reliability;
- translated xG/xA/save rates;
- sparse market-prior use;
- important unmatched new players.

### JSON output

Saved to:

```text
outputs/latest_weekly_recommendation.json
```

The payload includes:

```text
summary
autonomous pending state
selected-vs-omitted audit
starting XI
bench
transfer alternatives
rolling plan
chip comparison
raw chip recommendation
chip readiness gate
best chip opportunities
freshness audit
projection overrides
external-prior audit
priority unmatched new players
```

---

## Forecasting pipeline

```text
Official live FPL data
        +
Historical FPL player-fixture data
        +
Previous-season external player priors
        ↓
Leakage-safe feature engineering
        ↓
Expected minutes / rotation
        ↓
Team attack and defence forecasts
        ↓
Player event-rate forecasts
        ↓
Raw component expected FPL points
        ↓
Sparse-player fallback if needed
        ↓
Out-of-sample xP calibration
        ↓
Calibrated Gameweek projections
        ↓
Squad / XI / captain optimiser
        ↓
Rolling transfer search
        ↓
Chip evaluation
        ↓
Autonomous current-GW instruction
```

The fundamental forecasting unit is one player in one fixture.

That naturally handles:

```text
normal GW  → one fixture
blank GW   → zero fixtures
double GW  → multiple fixtures
```

Fixture predictions are aggregated into player-Gameweek projections.

---

## Expected minutes and rotation

The model estimates:

```text
P(appearance)
P(start)
P(60+ minutes)
expected minutes if starting
expected minutes if used as a substitute
```

Conceptually:

```text
Expected minutes
= P(start) × E(minutes | start)
+ P(substitute appearance) × E(minutes | substitute appearance)
```

Probability relationships are reconciled so that, for example:

```text
P(start) <= P(appearance)
P(60+) <= P(start)
```

Minutes feed every scoring component and are particularly important for rotation, clean sheets, captaincy fallback and automatic substitutions.

---

## Component expected points

Expected FPL points are built from interpretable components:

```text
appearance
goals
assists
clean sheets
goalkeeper saves
defensive contributions
bonus
goals-conceded deductions
cards
rare deductions
```

The event layer includes numerical safeguards such as:

```text
finite-value checks
non-negative rates
event-rate caps
exact zero ordinary xG/xA for goalkeepers
fixture-level xP sanity bounds
```

---

## New and returning players

The hierarchy is:

```text
1. Existing Premier League history
       ↓
2. Previous-season external prior
       ↓
3. Weak position/price market prior
       ↓
4. Generic model priors
```

### FBref

Used primarily for:

```text
minutes
appearances
starts
role
goals
assists
goalkeeper saves
```

### Understat

Used primarily for previous-season:

```text
xG
xA
```

Expected attacking-stat precedence is:

```text
Understat xG/xA
    ↓
direct FBref per-90 xG/xAG
    ↓
FBref shooting per-90
    ↓
FBref total xG/xA divided by minutes
```

### League translation

Foreign-league rates are translated relative to source-league and Premier League positional baselines and shrunk by sample size.

### Fade-out

Explicit external-prior influence fades as current-season Premier League evidence accumulates.

### Default automatic league coverage

```text
Premier League
La Liga
Bundesliga
Serie A
Ligue 1
```

Additional SoccerData leagues can be configured where supported.

### Sparse-player market prior

If a player has almost no Premier League history and no matched external prior, v2 applies a weak empirical prior based on position and FPL price.

Defaults:

```python
SPARSE_PLAYER_MARKET_PRIOR_WEIGHT = 0.30
SPARSE_PLAYER_MAX_PRIOR_FIXTURES = 1
```

This prevents an expensive unmatched signing from being treated exactly like an unknown minimum-price player.

---

## Expected-points calibration

Raw component xP can rank players well while still being systematically compressed or biased.

v2 adds an out-of-sample calibration layer.

```text
raw component xP
        ↓
monotone position-level calibration
        ↓
shrunk price-band residual correction
        ↓
bounded adjustment
        ↓
production xP
```

The live tables retain:

```text
raw_expected_points
xp_calibration_adjustment
expected_points
```

so the calibration is auditable.

Default safeguards:

```python
XP_CALIBRATION_BLEND = 0.85
XP_CALIBRATION_MIN_ROWS_PER_POSITION = 500
XP_CALIBRATION_MAX_ABS_ADJUSTMENT = 1.50
XP_CALIBRATION_MAX_REL_ADJUSTMENT = 0.40
```

The calibration is fitted from historical out-of-sample predictions, not from future information.

Calibration matters because absolute xP affects:

```text
transfer-hit decisions
captaincy
Bench Boost uplift
Triple Captain uplift
Free Hit value
Wildcard value
```

---

## Optimisation

### Squad structure

The optimiser enforces the configured FPL structure:

```text
15 players
2 GK
5 DEF
5 MID
3 FWD
budget constraint
club limit
```

The XI must satisfy legal formation rules.

### XI, bench and captaincy

For each squad it selects:

```text
starting XI
bench order
captain
vice-captain
```

Expected automatic substitutions and captain fallback are incorporated.

### Transfers

The system evaluates:

```text
roll
one transfer
multiple free transfers
paid transfers when the selected policy allows them
```

It accounts for:

```text
banked free transfers
money in the bank
purchase/selling prices
positions
club limits
hit costs
future squad value
```

### Rolling horizon

The full transfer tree is too large to enumerate exhaustively, so the notebook uses beam search.

Conceptually:

```text
Current action value
= current expected points
+ future projected points
- hit costs
+ terminal value
```

Only the first action is executed.

### Policy profiles

Section 11 compares:

```text
greedy_1gw_no_hits
rolling_3gw_no_hits
rolling_6gw_no_hits
rolling_6gw_hits
```

With:

```python
WEEKLY_POLICY_PROFILE = "auto"
```

production loads the historically selected cached policy.

The v2 fallback is:

```text
rolling_6gw_hits
```

but the fallback is not a substitute for completing the validation run.

---

## Chip strategy

The model evaluates:

```text
Triple Captain
Bench Boost
Free Hit
Wildcard
```

### Triple Captain

Incremental value is the extra captain copy beyond normal captaincy.

### Bench Boost

Compares all 15 players counting with expected normal scoring including automatic substitutions.

### Free Hit

Compares the best temporary one-GW squad with the best normal path while preserving the permanent squad state.

### Wildcard

Compares a permanent rebuild with the best no-Wildcard path over the planning horizon.

### Readiness gate

A locally best chip uplift does not automatically trigger chip use.

The production gate can require structural evidence such as:

```text
blank GW
double GW
chip expiry pressure
major squad restructuring
avoided paid transfers
exceptional current uplift
```

The raw chip candidate remains visible for audit.

### Half-season handling

The autonomous ledger resets the new chip set when crossing into GW20 while preserving state required for Free Hit restrictions.

---

## Historical validation

### Walk-forward forecast validation

Chronological holdouts:

```text
2023/24
2024/25
2025/26
```

Each holdout is trained only on configured earlier seasons.

Metrics include:

```text
MAE
RMSE
Brier score
AUC
Spearman correlation
prediction bias
calibration diagnostics
position diagnostics
```

### Deadline-frozen decisions

Historical decisions are generated using only information available before the simulated deadline.

Assertions guard against leakage in:

```text
player history
team results
external-prior season
forecast origin
future GW outcomes
```

### Policy validation

Controlled by:

```python
RUN_FULL_POLICY_BACKTEST
```

Normal weekly value:

```python
False
```

The resulting scorecard is cached.

### Sequential strategy validation

Controlled by:

```python
RUN_SEQUENTIAL_BACKTEST
```

Normal weekly value:

```python
False
```

It simulates:

```text
opening squad
transfers
hits
XI
bench
captaincy
chips
realised historical points
```

The sequential backtest requires the v2 calibration cache first.

Backtests are evidence of comparative historical performance, not guarantees of future rank.

---

## Production safeguards

A **hard error means do not make an FPL move from the current output** until the technical/data issue is resolved.

### Freshness check

Before producing a recommendation, Section 12 compares the in-memory forecast state with a fresh official FPL API request.

Relevant player checks include:

```text
price
status
chance of playing next round
news
```

Fixture changes are also checked.

With:

```python
WEEKLY_STRICT_FRESHNESS_CHECK = True
```

stale projections cause the notebook to stop.

### Team-state reconciliation

With:

```python
AUTONOMOUS_STRICT_RECONCILIATION = True
```

the notebook stops if the public FPL team does not match the previous pending recommendation.

### Calibration requirement

With:

```python
AUTONOMOUS_REQUIRE_CALIBRATION_CACHE = True
```

autonomous Section 12 requires:

```text
data/xp_calibration_v2.json
```

### Manual overrides

By default:

```python
AUTONOMOUS_ALLOW_MANUAL_PROJECTION_OVERRIDES = False
```

This prevents subjective projection changes from silently altering a supposedly autonomous season.

---

## Common runtime messages

### Official FPL API changed since projections were run

Meaning:

```text
The projections in memory are stale.
```

Action:

```text
Rerun Sections 1–10
then rerun Section 12
```

Do not normally disable the freshness check.

### Missing xP calibration cache

Action:

```text
Run Section 11 once
restart kernel
rerun Sections 1–10
run Section 12
```

### Public FPL picks do not match the previous recommendation

Possible causes:

```text
previous recommendation was not executed
a manual transfer was made
wrong FPL_TEAM_ID
ledger belongs to a different state
```

Resolve the mismatch before continuing.

### External-prior warning

An external data source may be unavailable or a player may be unmatched.

The notebook can fall back to Premier League history, the sparse position/price prior or generic priors. Important unmatched players are surfaced in the weekly audit.

---

## Limitations

### No guarantee of realised points

Expected value is not realised value.

Good decisions can lose points because of:

```text
red cards
penalty misses
unexpected benchings
injuries
finishing variance
late goals
bonus variance
```

The system should be evaluated over repeated decisions, not one transfer.

### Team news

The notebook does not automatically collect every:

```text
press conference
predicted lineup
European match minute
domestic-cup minute
training report
manager quote
```

Running close to the deadline reduces, but does not eliminate, this information gap.

### External league coverage

Unsupported competitions can leave some new players dependent on the sparse price/position prior.

### Future prices

Future FPL price changes are not perfectly forecast within the planning horizon.

### Beam-search approximation

Beam search makes long-horizon transfer optimisation tractable but does not guarantee that every possible squad path has been evaluated.

### Absolute xP remains uncertain

Calibration improves historical bias but does not make a displayed expected-points value a literal promise.

### Long-range chip uncertainty

Future blank and double Gameweeks may not yet be confirmed inside the active horizon.

### Rule/API changes

Major FPL scoring, chip or API changes may require code updates.

---

## Data sources

### Official Fantasy Premier League API

Used as the source of truth for the live season:

```text
players
teams
prices
availability
news
fixtures
Gameweeks
deadlines
live/completed statistics
public team verification
```

### Historical FPL archive

Used for completed player-fixture history and chronological validation.

The loader standardises schemas, resolves duplicate player-fixture rows, keeps stable identifiers where possible and excludes known leakage-prone fields.

### FBref via SoccerData

Used for previous-season new-player information including role, minutes and goalkeeper saves.

### Understat via SoccerData

Used primarily for previous-season xG and xA.

External data are used as priors; they are not future current-season information.

---

## Recommended season operating rule

```text
IF Section 12 completes successfully:
    execute exactly the primary current-GW recommendation

IF Section 12 raises a hard error:
    execute nothing
    resolve the data/state problem first

NEVER:
    execute the provisional later-GW transfer path without rerunning
    rerun expensive historical backtests before every deadline
    silently override the autonomous ledger
```

This keeps football decisions model-led while treating technical failures separately from football judgement.

---

## Version history

### v2.0.2 — Autonomous Manager

Current version.

Major additions and fixes:

- persistent autonomous team ledger;
- public post-deadline state reconciliation;
- pending-versus-committed team state;
- automatic bank/free-transfer/chip continuity;
- half-season chip-set handling;
- out-of-sample xP calibration;
- raw-versus-calibrated xP audit;
- price-band calibration adjustment;
- sparse new-player price/position prior;
- selected-vs-omitted audit;
- improved freshness diagnostics;
- cacheable sequential backtest;
- cacheable policy backtest;
- autonomous calibration requirement;
- model-led production mode;
- robust position normalisation for historical `GKP`, numeric element types and `model_position`.

### v1.2.x

- Understat xG/xA integration;
- robust FBref expected-stat fallbacks;
- external-stat coverage audits;
- goalkeeper xG/xA positional safeguard;
- cache-backed Section 11.

### v1.1.x

- external new-signing priors;
- player matching and overrides;
- league-to-PL translation;
- reliability weighting;
- prior fade-out as PL evidence arrives.

### v1.0.x

- live FPL ingestion;
- historical archive;
- player-fixture modelling;
- minutes forecasts;
- team-strength models;
- component xP;
- squad/XI/captain optimisation;
- transfer search;
- chips;
- deadline-frozen backtesting;
- walk-forward validation;
- weekly recommendation dashboard.

---

## Development principle

Every substantial change should satisfy:

```text
1. It is leakage-safe.
2. It improves or protects decision quality in historical validation.
3. Production uses the same logic that was validated.
```

The goal is not maximum model complexity.

The priority is reliable full-season decision quality through:

```text
better information
better calibration
better state handling
better optimisation
strict validation
strict production safeguards
```
