---
document: end-to-end pipeline audit
version: v2.6.1
as_of: 2026-10-06
---

# Pipeline audit — can you just execute what it says?

**Short answer: yes, after one validation run — and the main thing that was
missing was not capability but presentation.**

## Scope of this audit

Audited in depth: the expected-points layer, the minutes layer, the chip
decision path, the squad/line-up/captain optimiser, and the Section 12 weekly
output. Verified against four seasons of archived data where a claim was
testable.

**Not** audited line by line: the data-ingest layer, the external-prior
matching, the FBref enrichment, the transfer beam search internals, and the
price/ledger reconciliation. Those were read where they touched a decision, not
exhaustively. A 36,000-line notebook has not been exhaustively verified by
anyone here, and claiming otherwise would be false.

## Finding: the decision data was already complete

Section 12 already computes everything needed to execute a gameweek:

| Decision | Variable |
|---|---|
| chip | `weekly_chip_recommendation` (`chip`, `action`, `reason`) |
| transfers | `weekly_primary_action`, `weekly_transfer_moves`, `weekly_hit_cost` |
| starting XI | `weekly_starting_xi` |
| bench, in autosub order | `weekly_bench` |
| captain / vice | `weekly_captain`, `weekly_vice_captain` |
| everything, as JSON | `weekly_recommendation_summary` |

The gap was that this arrives as roughly forty diagnostic print statements and
several DataFrames spread through the section, so the instruction has to be
reassembled by hand each week. That is where execution errors come from.

## Fix: the execution card (P13)

A new cell at the end of Section 12 consolidates it into one block:

```
======================================================================
  GAMEWEEK 9 — EXECUTE THIS  —  deadline Sat 10 Oct 18:30
======================================================================
  1. CHIP       : NONE
                  Save all chips
                  why: no double/blank in the explicit horizon
  2. TRANSFERS  : 1 transfer: Maguire -> Dunk
                  OUT Maguire (4.5)   ->   IN Dunk (4.6)
                  points hit: 0
  3. STARTING XI:
                  GK   Raya
                  DEF  Gabriel, Van Dijk, Dunk
                  MID  Salah (C), Saka, Palmer, Semenyo, Rogers
                  FWD  Haaland (V), Watkins
                  formation 3-5-2
  4. BENCH ORDER: (autosubs happen in this order)
                  1. Mbeumo (FWD)   2. Hall (DEF)   3. Richards (DEF)   4. Sels (GK)
  5. CAPTAIN    : Salah
     VICE       : Haaland
----------------------------------------------------------------------
  Projected points for GW9: 62.4
  Captain confidence: MEDIUM (xP gap +0.31)  <- near-tie, treat as a coin flip
======================================================================
```

Every lookup is defensive. If a variable is missing the card prints what is
missing rather than silently emitting a partial instruction — a half-printed
team is worse than a loud gap. Tested against both a complete run and a
degraded one.

It also surfaces two things that were previously buried: whether the captain
choice is a near-tie (gap < 0.35 xP), and whether the v2.6 assist conversion is
active without its gate having been confirmed.

## Fix: no more "Validation artifacts changed" restarts (P15)

Section 11 refits and rewrites the xP-calibration and premium-attacker
artifacts *after* Section 7 has already built the live forecast. Section 12 then
compares artifact ids and raises:

```
RuntimeError: Validation artifacts changed after the live projections were
generated. Restart the kernel and rerun Sections 1-10 before Section 12.
```

The notebook's own comment accepted this:

> *"Make the new artifacts visible in this kernel for inspection. Live forecasts
> generated earlier in the notebook still need a restart/rerun before Section 12."*

It fires on **every** run that rebuilds an artifact — which is every first run
after a signature change, so the v2.6 signature update guarantees it.

**The guard is correct and is not suppressed.** Acting on projections built from
stale artifacts is exactly the bug it exists to catch. Instead, P15 rebuilds the
live forecast in place from `live_xp_components` through the same post-layer
chain Section 7 uses, with the freshly validated payloads, so the projections
genuinely do match the current artifacts. Every input is already in memory, so
this costs seconds rather than a full rerun.

The notebook already did this for the v2.4 and v2.5 layers; it simply never did
it for the calibration and premium artifacts.

One subtlety that makes the naive fix wrong: neither
`apply_premium_attacker_adjustment` nor `apply_xp_calibration_payload` is
idempotent. Each captures its own `pre_*` / `raw_*` baseline from whatever frame
it is handed, so re-applying on top of an already-corrected forecast compounds
the correction. P15 therefore rebuilds from the raw components rather than
re-applying. Verified: the refreshed forecast matches a single fresh
application and **not** a compounded one, on a test where the two differ by up
to 1.16 points.

If `V26_ENABLED = False`, P15 is inert and the original restart behaviour
returns.

## Fix: materiality-aware freshness gate (P16)

The second blocker is a different kind of problem:

```
RuntimeError: The official FPL API has changed since the live projection
cells were run. Rerun the notebook through Section 10, then run Section 12.
```

`is_fresh` requires `changed_player_rows == 0` across **every player in the
game**. FPL re-prices players nightly and updates injury news continuously, so
any Run-All long enough to reach Section 12 will almost always trip on someone
irrelevant. A real example: five changed rows, of which three were £0.1m price
ticks (two on players you would never own) and two were availability changes on
fringe players.

Unlike the artifact guard, this is **not** an internal inconsistency — the
outside world really did change — so the gate is kept, and made *material*:

| Change | Handling |
|---|---|
| gameweek rollover | still blocks outright |
| fixture schedule changed | still blocks outright |
| price-only move | **refreshed into the live frame**, does not block |
| availability change, player projected ≥ 30 min | **blocks** |
| availability change, player projected < 30 min | warns, does not block |

Price moves are refreshed rather than ignored so affordability stays correct. A
£0.1m move cannot cross a calibration price band (5.0 / 6.5 / 8.0 / 10.0), so
the already-applied calibration stays valid.

Verified against the real payload above:

```
fringe players changed            -> PROCEED, 3 prices refreshed (Saka 9.5 -> 9.6)
a projected starter's availability -> BLOCK, names the player
```

**What this trades away:** a player who becomes *newly available* and is
projected under 30 minutes will not trigger a rerun, so a late transfer target
can be missed. That is a missed opportunity rather than a decision made on
stale data, which is the right side to err on. Set
`WEEKLY_STRICT_FRESHNESS_CHECK = False` to disable the gate entirely, or
`V26_ENABLED = False` to restore the original all-or-nothing behaviour.

## Before you rely on it

1. **Run the notebook once and check the F1 gate** (`AUDIT_RESPONSE.md` §9).
   v2.6.1 has never been through the real pipeline. If it fails, set
   `V26_ENABLED = False` — that reverts to exact v2.5 behaviour.
2. **Confirm the chip rule fires.** The default output is "Save all chips", and
   the failure mode that costs real points is deferral to zero (−92/season), not
   bad timing (±5 vs random). Expiry pressure first triggers at GW16.
3. **Treat near-ties as ties.** The card now flags them.

## What is still open, in order of points at stake

| Item | Points | Status |
|---|---|---|
| Wildcard timing | ~99/season | **shipped (P14)** — structural gate dropped for the wildcard, worth ~+40/season |
| Policy scorecard | unknown | selected with `WALK_FORWARD_INCLUDE_CHIPS = False`, and v2.6 changes the forecast it depends on |
| Set-piece / penalty roles | ~20–30 per wrong call | `LIVE_SET_PIECE_PENALTY_XG_ADDITIVE = False`; a *new* penalty taker is not recognised until history accumulates |
| Chip rule in the first half | a few | adds nothing before GW19; no double/blank exists to trigger on, and predicted chip value scores below random there |
| Bonus model | measured | rank-based model **rejected** — recovers dispersion, ranks worse (`MODELLING_EXPERIMENTS.md`) |
| Team value growth | measured | value-aware tiebreak **rejected** — unstable across folds; the price model is kept as information only |

The largest genuinely open item is now the **policy scorecard**: `greedy_1gw_no_hits`
was selected with chips disabled, and v2.6 changes the forecast it was selected
against. Re-running it is expensive, so it is a deliberate deferral rather than
an oversight — but it should not be described as "validated for v2.6 semantics".

Second is **set-piece roles**, which is cheap to fix and never measured.

## Honest limits

- I cannot run your live pipeline: it needs your FPL team id, the local ledger,
  and live endpoints. The execution card is verified against synthetic data in
  both complete and degraded states, not against a real deadline.
- The validation harness throughout is an independent re-implementation. It is
  good for comparing variants; it is not your model, and absolute levels do not
  transfer.
- Two chronological folds is the project's minimum standard and all of this
  meets it — no more. Several results disagreed sharply between the two.
