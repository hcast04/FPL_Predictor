"""
Feature construction, with leakage made testable.

Every feature here is a STRICTLY LAGGED aggregate: the value on row t is
computed from rows < t only. That is enforced by construction (`.shift(1)`
before every rolling window) and verified by `tests/test_leakage.py`, which
recomputes a sample of rows from a truncated panel and demands the same answer.

The reason for the ceremony: in the previous codebase a "leakage-free" script
leaked three different ways and still carried the word in its docstring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

# per-fixture quantities that get rolled into form features
# O48: how an ABSENT expected-stat should enter the FEATURES.
#
# `data.apply_coverage_mask` marks 2022-23 GW1-32 absent and `build_panel` now
# preserves that as NaN (M20). The NaN then flows into the rolling features,
# where the GLM's SimpleImputer replaces it with the column MEDIAN -- a
# middling attacking threat for a player whose threat was never recorded.
# Before M20 those rows carried 0.0, which is also wrong but is a far better
# stand-in for "nothing recorded". Measured at -162 pts/season (O48).
#
# True  = fill absent expected-stat features with 0.0 (pre-M20 behaviour)
# False = leave NaN and let the estimator's imputer decide (post-M20)
#
# This is a FEATURE-encoding choice and is independent of whether the TRAINING
# TARGETS and CONVERSION FACTORS treat absence correctly -- those stay masked.
ZERO_FILL_ABSENT_EXPECTED = False

ROLL_COLS = ["minutes", "points", "goals_scored", "assists", "clean_sheets",
             "goals_conceded", "saves", "bonus", "bps", "expected_goals",
             "expected_assists", "starts", "defensive_contribution"]
# rolled in addition to the above when USE_UNDERSTAT_FEATURES is on
US_ROLL = ["us_xg", "us_shots", "us_key_passes", "us_xg_chain", "us_xg_buildup"]
WINDOWS = (3, 5, 10)


def _lagged_roll(g: pd.Series, window: int) -> pd.Series:
    """Mean of the previous `window` values. Row t never sees row t."""
    return g.shift(1).rolling(window, min_periods=1).mean()


def _lagged_expanding(g: pd.Series) -> pd.Series:
    return g.shift(1).expanding().mean()


def _lagged_roll_sd(g: pd.Series, window: int) -> pd.Series:
    """
    Sd of the previous `window` values. Row t never sees row t.

    Needed because `w = n/(n+K)` in `minutes.py` counts HOW MANY appearances a
    player has made and ignores how consistent they were. 90,90,90,90,90 and
    90,20,0,75,90 carry the same weight at the same n, although the first is
    certainty and the second is a coin flip (O41, S20).
    """
    return g.shift(1).rolling(window, min_periods=2).std()


def _history_at_club(s: pd.Series, w: int) -> pd.Series:
    """
    Share of the previous `w` appearances played at THIS row's club.

    Strictly lagged by construction: row i reads arr[i-w:i] only, never arr[i]
    and never beyond. A player with no history yet scores 1.0 -- all of his
    (empty) history is at his current club, which is the neutral value.
    """
    arr = s.to_numpy()
    out = np.empty(len(arr), dtype=float)
    for i in range(len(arr)):
        prev = arr[max(0, i - w):i]
        out[i] = float((prev == arr[i]).mean()) if len(prev) else 1.0
    return pd.Series(out, index=s.index)


# Where the UNDERLYING attacking quantities come from (S47).
#
# FPL publishes Opta xG/xA from 2022-23 only, so six of our ten seasons have
# none. Understat covers all ten with its own model. Measured on the four
# overlapping seasons (n=7,217 in 2022-23 alone):
#
#     xG spearman 0.9884 / 0.9890 / 0.9876 / 0.9875   -- the same ORDERING
#     xG ratio    1.030  / 1.092  / 1.124  / 1.107    -- a drifting LEVEL
#
# The ranking is reproducible across sources; the level is not, and it drifts
# even within the Opta era. That is the argument for taking the features from
# ONE source across all ten seasons rather than splicing Understat onto the
# older half -- a spliced model inherits that drift exactly at the boundary
# between its training and holdout data.
#
# FPL keeps what it alone owns and what the scoring rules are defined on:
# goals, assists, clean sheets, defensive contributions, minutes, bonus, cards.
# Understat adds `xg_chain` and `xg_buildup`, which have no FPL equivalent --
# they credit every player in the possession leading to a shot, which is the
# closest thing to creativity that does not depend on an assist being awarded.
#
# NOT taken from Understat: xA. Its ratio to Opta is 1.23 and its spearman only
# 0.69, stable across three independent joins, so it is a DIFFERENT quantity
# rather than a noisier measure of the same one.
#
# --- 2026-10-23: TURNED ON. S47 RETRACTED (M27). -----------------------------
#
# S47 measured this switch at +1.6 [-3.7, +6.8] and that is why it was False.
# S47 was wrong, for a reason outside this file: `rates.rate_caps` derived the
# xG CEILING from `expected_goals` while the model was being FITTED on
# `us_xg`, and `expected_goals` does not exist before 2022-23. The ceiling came
# out 0.0 and clipped every goal prediction to zero, so S47 compared two blind
# models. Re-measured after M26/M27, on the same instrument:
#
#     xG only  (us_xg, FPL xA)   +47.7  [+14.2, +81.2]  p 0.012   8/8 seasons
#     xG + xA  (us_xg, us_xa)    +55.5  [+18.6, +92.4]  p 0.009   8/8 seasons
#
# Confirmed three ways: a fresh re-implementation, S47's own script re-run, and
# the per-fold ceiling values printed directly. See ledger M27.
#
# ON THE xA RESERVATION ABOVE, which this switch now overrides. It stands as a
# statement about measurement: us_xa IS a different quantity from Opta xA. But
# five of the eight folds have NO Opta xA at all, and a different quantity
# beats no quantity -- that gap is most of the +7.8 that xA adds. On the three
# folds that DO have Opta xA the margin is +0.7 / +1.9 / +0.9, i.e. nearly
# nothing. So this is not a finding that Understat xA is better; it is a
# finding that it is available. If FPL ever backfills xA, re-test.
USE_UNDERSTAT_FEATURES = True

# Market-signal features (O40). OFF by default: F23/M19 already cost -83
# pts/season with ownership features built from a POST-HOC snapshot, so this
# family starts from a position of suspicion even though the construction here
# is different -- strictly lagged, from recorded per-gameweek history.
#
# The quantity being proxied is the one S29 isolated as the entire remaining
# headroom: near-future minutes and availability. `chance_of_playing_next_round`
# and `news` would say it directly but exist only as a current snapshot, with
# no history to validate against. What the crowd DOES is recorded: when a player
# is injured, suspended or dropped, his owners sell him before the deadline.
#
# Everything is shift(1) or longer. It could not be established whether
# `transfers_in[N]` accumulates before or during gameweek N, and lagging makes
# the question moot rather than answered.
MARKET = False


def _market_features(d: pd.DataFrame, g) -> list:
    """Strictly lagged transfer-market signals. Returns the names created."""
    made = []
    sel = pd.to_numeric(d.get("selected"), errors="coerce")
    if sel is None or sel.notna().sum() == 0:
        return made
    denom = sel.replace(0, np.nan)
    for name, col in (("sold_share", "transfers_out"),
                      ("bought_share", "transfers_in"),
                      ("net_share", "transfers_balance")):
        v = pd.to_numeric(d.get(col), errors="coerce") / denom
        d["_tmp"] = v
        for w in (1, 3):
            nm = f"{name}_r{w}"
            d[nm] = g["_tmp"].transform(lambda x, w=w: _lagged_roll(x, w))
            made.append(nm)
    # ownership as a SHARE of that gameweek's total, so the season-long growth
    # in the player base (8.90M -> 11.06M entries inside 2026-27 alone) is not
    # read as everybody becoming more popular
    tot = d.groupby(["season", "gameweek"])["selected"].transform("sum")
    d["_own"] = sel / tot.replace(0, np.nan)
    d["owned_share_r1"] = g["_own"].transform(lambda x: _lagged_roll(x, 1))
    d["owned_change_r1"] = g["_own"].transform(
        lambda x: x.shift(1) - x.shift(2))
    made += ["owned_share_r1", "owned_change_r1"]
    d.drop(columns=["_tmp", "_own"], inplace=True, errors="ignore")
    return made


def player_form(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-player rolling form. Input must be sorted by (player, t)."""
    d = panel.sort_values([C.PLAYER_KEY, "t"]).copy()
    roll_cols = list(ROLL_COLS)
    if USE_UNDERSTAT_FEATURES:
        # `us_xg` REPLACES `expected_goals` as the rolled quantity: both measure
        # the same thing and keeping both would hand the model two collinear
        # versions of one signal, differing only by a drifting constant.
        roll_cols = [c for c in roll_cols if c != "expected_goals"]
        roll_cols += [c for c in US_ROLL if c in d.columns]
    # Zero-fill ONLY for the rolling computation, then put the NaN back.
    #
    # M22: the first version filled in place, and because `player_form` runs
    # upstream of everything, the zeros reached `rates.build_targets` and
    # `conversion_factors` as well -- so `MASK_TARGETS=True` had nothing left
    # to mask and the arm silently reproduced plain pre-M20 (2143/2420/2149,
    # bit-identical). Restoring the originals afterwards is what makes
    # "zeroed features, masked targets" actually expressible.
    _restore = {}
    if ZERO_FILL_ABSENT_EXPECTED:
        from . import data as _d
        for c in _d.EXPECTED_COLS:
            if c in d.columns:
                _restore[c] = d[c].copy()
                d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    g = d.groupby(C.PLAYER_KEY, sort=False)
    made = []
    for col in roll_cols:
        if col not in d.columns:
            continue
        for w in WINDOWS:
            name = f"{col}_r{w}"
            d[name] = g[col].transform(lambda s, w=w: _lagged_roll(s, w))
            made.append(name)
        name = f"{col}_career"
        d[name] = g[col].transform(_lagged_expanding)
        made.append(name)
    # --- transfer context -----------------------------------------------
    # A player who changed club carries rolling form earned at his OLD team,
    # while the team-strength features describe his NEW one. The model has no
    # way to know the two disagree, and movers are the worst-ranked group in
    # every season measured (spearman 0.45-0.62 vs 0.58-0.68 for stayers) and
    # consistently OVER-predicted (ratio 1.02-1.09). These flags let it learn
    # to discount history that was not earned where the player now plays.
    prev_team = g["team"].shift(1)
    d["changed_club"] = (prev_team.notna() & prev_team.ne(d["team"])).astype(float)
    # how much of the recent history was actually earned at the current club
    #
    # The previous implementation compared each row's history against
    # `s.iloc[-1]` -- the player's LAST team in the whole series, i.e. the club
    # he ends up at. At gameweek 10 that is a club he may not sign for until
    # gameweek 25, so the feature read the future and `test_leakage.py` caught
    # it (max abs diff 1.0 at 2025-26 GW20). The non-leaky expression was
    # computed on the line above and discarded unused.
    #
    # A rolling window cannot express this: the reference team changes per row,
    # so each row must compare its own window against its own current club.
    for w in (5, 10):
        d[f"history_at_current_club_r{w}"] = g["team"].transform(
            lambda s, w=w: _history_at_club(s, w))
    # gameweeks since the most recent club change, capped: the staleness decays
    chg = d.groupby(C.PLAYER_KEY, sort=False)["changed_club"].cumsum()
    d["fixtures_at_current_club"] = (
        d.groupby([C.PLAYER_KEY, chg], sort=False).cumcount().astype(float))

    d["appearances_so_far"] = g["minutes"].transform(
        lambda s: (s.shift(1) > 0).cumsum())
    # WITHIN-SEASON evidence count. Distinct from the career count above, and
    # it is the quantity that matters for cold start: at gameweek 1 every
    # player has zero of it regardless of how long a career he has had.
    d["season_appearances_so_far"] = (
        d.sort_values([C.PLAYER_KEY, "t"])
         .groupby([C.PLAYER_KEY, "season"], sort=False)["minutes"]
         .transform(lambda s: (s.shift(1) > 0).cumsum()))
    d["minutes_sd_r5"] = g["minutes"].transform(
        lambda s: _lagged_roll_sd(s, 5))
    d["starts_so_far"] = g["starts"].transform(lambda s: s.shift(1).cumsum())
    d["fixtures_so_far"] = g["minutes"].transform(lambda s: s.shift(1).notna().cumsum())
    # `minutes_sd_r5` is deliberately NOT added to `made`. It is an input to the
    # recency blend in `minutes.py`, not a model feature. Adding it would change
    # the fitted model and destroy the only check that matters here -- that
    # RECENCY_SPREAD_FLOOR = 1.0 reproduces the shipped season totals exactly.
    made += ["appearances_so_far", "starts_so_far", "fixtures_so_far",
             "season_appearances_so_far", "changed_club",
             "history_at_current_club_r5", "history_at_current_club_r10",
             "fixtures_at_current_club"]
    if MARKET:
        made += _market_features(d, g)
    # rate features: only meaningful once a player has minutes on the clock
    for w in WINDOWS:
        mins = d[f"minutes_r{w}"].replace(0.0, np.nan)
        for col in (("goals_scored", "assists", "expected_goals",
                     "expected_assists", "saves", "bonus", "bps")
                    + (tuple(US_ROLL) if USE_UNDERSTAT_FEATURES else ())):
            if f"{col}_r{w}" not in d.columns:
                continue
            name = f"{col}_per90_r{w}"
            d[name] = 90.0 * d[f"{col}_r{w}"] / mins
            made.append(name)
    for c, orig in _restore.items():      # raw columns go back to NaN
        d[c] = orig
    return d, made


def team_strength(panel: pd.DataFrame,
                  fixtures: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """
    Rolling team form for the player's OWN team AND the opponent he faces,
    plus home/away. All strictly lagged.

    The opponent half was missing in the first version while this docstring
    already claimed it was here -- the model was fixture-blind, which inflated
    goalkeeper saves (no idea whether the opposition attacks) and flattened
    attacking returns (no idea whether the defence is weak). `fixtures` is the
    team-fixture table from `team.team_fixtures`, which carries the opponent
    and venue that the player panel does not.
    """
    d = panel.copy()
    team_gw = (d[d["fixtures"] > 0]
               .groupby(["season", "gameweek", "team"], as_index=False)
               .agg(scored=("goals_scored", "sum"),
                    conceded=("goals_conceded", "max"),
                    t=("t", "first")))
    team_gw = team_gw.sort_values(["team", "t"])
    g = team_gw.groupby("team", sort=False)
    for col in ("scored", "conceded"):
        team_gw[f"team_{col}_r5"] = g[col].transform(lambda s: _lagged_roll(s, 5))
        team_gw[f"team_{col}_r10"] = g[col].transform(lambda s: _lagged_roll(s, 10))
        team_gw[f"team_{col}_career"] = g[col].transform(_lagged_expanding)
    strength_cols = [c for c in team_gw.columns if c.startswith("team_")]
    own = team_gw[["season", "gameweek", "team"] + strength_cols]
    out = d.merge(own, on=["season", "gameweek", "team"], how="left")
    made = list(strength_cols)

    if fixtures is not None:
        # who each team faces, and whether at home. A double gameweek is
        # averaged: two opponents means the player faces both.
        fx = fixtures[["season", "gameweek", "team", "opponent", "is_home"]].copy()
        opp = own.rename(columns={"team": "opponent",
                                  **{c: c.replace("team_", "opp_") for c in strength_cols}})
        fx = fx.merge(opp, on=["season", "gameweek", "opponent"], how="left")
        opp_cols = [c for c in fx.columns if c.startswith("opp_")]
        agg = (fx.groupby(["season", "gameweek", "team"], as_index=False)
                 .agg({**{c: "mean" for c in opp_cols}, "is_home": "mean"}))
        out = out.merge(agg, on=["season", "gameweek", "team"], how="left")
        made += opp_cols + ["is_home"]
        # relative strength: what actually drives a fixture
        out["att_vs_opp_def"] = out["team_scored_r5"] - out["opp_conceded_r5"]
        out["def_vs_opp_att"] = out["opp_scored_r5"] - out["team_conceded_r5"]
        made += ["att_vs_opp_def", "def_vs_opp_att"]
    return out, made


def usable(train: pd.DataFrame, names: list[str],
           min_unique: int = 2) -> list[str]:
    """
    Features that carry information in THIS training set.

    A column can be entirely NaN or constant in one fold and informative in
    another -- external priors do not exist for the earliest season, and the
    defensive-contribution columns are absent before 2025-26. Handing such a
    column to a histogram binner raises "window shape cannot be larger than
    input array shape" from deep inside sklearn, far from the cause.

    Filtering per fold keeps every model honest about what it actually had.
    """
    keep = []
    for c in names:
        if c not in train.columns:
            continue
        v = pd.to_numeric(train[c], errors="coerce")
        if v.notna().sum() == 0:
            continue
        if v.nunique(dropna=True) < min_unique:
            continue
        keep.append(c)
    return keep


def build(panel: pd.DataFrame,
          fixtures: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """The full feature frame plus the list of model-usable feature names."""
    d, player_cols = player_form(panel)
    d, team_cols = team_strength(d, fixtures)
    d["pos_code"] = d["position"].map(
        {p: i for i, p in enumerate(C.POSITIONS)}).astype(float)
    # Season stage. Without it the model cannot tell gameweek 1 (where
    # season_appearances_so_far is 0 for EVERYONE) from gameweek 20 (where 0
    # means "never selected"). It averaged the two and under-predicted
    # appearances at cold start by 18%.
    d["gw"] = pd.to_numeric(d["gameweek"], errors="coerce")
    d["price"] = pd.to_numeric(d["cost"], errors="coerce")
    d["n_fixtures"] = pd.to_numeric(d["fixtures"], errors="coerce")
    extra = ["pos_code", "price", "n_fixtures", "gw"]
    feats = player_cols + team_cols + extra
    feats = [f for f in feats if f in d.columns]
    return d, feats


def truncation_equivalence(panel: pd.DataFrame, season: str, gameweek: int,
                           feats: list[str] | None = None,
                           fixtures: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    THE leakage test, at panel scale.

    Build features twice: once on the whole panel, once on a panel truncated so
    that nothing AFTER (season, gameweek) exists. For rows AT that gameweek the
    two must agree exactly -- a feature that differs is reading the future.

    Returns one row per feature with the worst absolute disagreement.
    """
    full, names = build(panel, fixtures)
    feats = feats or names
    target = full[(full["season"].eq(season)) & (full["gameweek"].eq(gameweek))]
    if target.empty:
        raise ValueError(f"no rows at {season} GW{gameweek}")
    t = int(target["t"].iloc[0])

    trunc_input = panel[panel["t"] <= t].copy()
    trunc, _ = build(trunc_input, fixtures)
    trunc = trunc[trunc["t"].eq(t)]

    a = target.set_index(C.PLAYER_KEY)[feats].sort_index()
    b = trunc.set_index(C.PLAYER_KEY)[feats].sort_index()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    diff = (a - b).abs()
    out = pd.DataFrame({
        "feature": feats,
        "max_abs_diff": [float(np.nanmax(diff[f].to_numpy())) if len(diff) else np.nan
                         for f in feats],
        "n_rows": len(common)})
    return out.sort_values("max_abs_diff", ascending=False).reset_index(drop=True)
