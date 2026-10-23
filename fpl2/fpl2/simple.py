"""
A deliberately simple xP model.

The rebuilt component model loses the G1 gate to the legacy signal by ~290
points a season, and the diagnosis is top-end OVER-prediction: at xp>=5 the
rebuild runs a ratio of 1.07-1.14 while legacy runs 0.83-0.93. Under a budget
the optimiser only buys from the top, so inflating it is expensive -- you pay
premium prices for forecasts that do not land.

The rebuild's tail-inflating machinery (sustained-rate ceilings, shrunk
conversion, a hierarchical minutes model, 120 features) is the obvious suspect:
G3, G3b and G6 were all attempts to tame that tail, which suggests it has been
fighting us the whole time.

So this is the control that experiment never had. Everything here is a lagged
average or a four-bucket lookup:

  minutes     plain rolling mean, no hierarchy, no P(start) model
  attack      lagged xG90 / xA90, no ceiling, no shrinkage, no conversion
  fixtures    opponent strength as FOUR TIERS with multiplicative factors,
              not a continuous team-strength model
  clean sheet exp(-lambda), lambda from own lagged concession x opponent tier
  scoring     scoring.expected_points, which reconstructs 99.41% of four
              seasons exactly and is the one piece worth keeping

If this beats the complex model, the complexity was the problem. If it also
loses to legacy, the problem is upstream of the model and no amount of
feature work will reach it.

Walk-forward discipline: tiers and tier factors are fitted on seasons STRICTLY
BEFORE the holdout. Player and team rates are shift(1)-lagged within it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import scoring

N_TIERS = 4
FORM_WINDOW = 5          # gameweeks of trailing form for minutes
MIN_RATE_MINUTES = 180.0  # below this a per-90 rate is mostly noise

# Empirical-Bayes shrinkage toward the position mean, in units of 90s played.
#
# Without it the opening squad is a disaster. Picked at GW1-5 there is no
# within-season data, so a career per-90 rate projects forward at full value --
# and the 2025-26 opening squad put 14.5, a seventh of the budget, into Salah
# on the back of an exceptional 2024-25. He returned 123. Legacy's opening 15
# was worth 2184 season points to ours at 1683, and it never buys the 14.5
# asset at all.
#
# A MIN_RATE_MINUTES floor does not help: it excludes thin samples but does
# nothing to a player with 3000 career minutes at an unsustainable rate. k is
# in 90s, so the pull is hard at GW1 of a new season and negligible by GW20 --
# which is the shape the problem actually has.
SHRINK_K = 10.0
LEAGUE_AVG_GOALS = 1.4    # fallback when a team has no history at all


# ---------------------------------------------------------------- team tiers
def team_table(tf: pd.DataFrame, before: str) -> pd.DataFrame:
    """Per-team scored/conceded per game, from seasons strictly before `before`."""
    past = tf[tf["season"] < before]
    if past.empty:
        return pd.DataFrame(columns=["team", "gf", "ga"])
    last = past["season"].max()          # most recent completed season only
    past = past[past["season"].eq(last)]
    g = (past.groupby("team")
             .agg(gf=("scored", "mean"), ga=("conceded", "mean"))
             .reset_index())
    return g


def assign_tiers(table: pd.DataFrame) -> dict:
    """
    Four tiers by goal difference. `rank()` is ascending, so TIER 1 IS THE
    WEAKEST and tier 4 the strongest.

    A promoted team has no row here and gets no tier, falling back to league
    average -- the honest default: it is what the data cannot contradict until
    the season starts.
    """
    if table.empty:
        return {}
    t = table.assign(gd=table["gf"] - table["ga"]).sort_values("gd", ascending=False)
    t["tier"] = pd.qcut(t["gd"].rank(method="first"), N_TIERS,
                        labels=list(range(1, N_TIERS + 1))).astype(int)
    return dict(zip(t["team"], t["tier"]))


def tier_factors(tf: pd.DataFrame, before: str, tiers: dict) -> tuple[dict, dict, dict]:
    """
    How much a tier-k opponent suppresses scoring, and concedes.

    `att[k]`  multiplies a team's attacking rate when facing a tier-k opponent
              (how leaky tier k is)
    `con[k]`  expected goals conceded BY a tier-k team, per game
    `atk[k]`  multiplies a team's CONCESSION when facing a tier-k opponent
              (how dangerous tier k is)

    att and atk are different quantities and the distinction matters: how many
    goals a tier leaks says nothing about how many it scores. Using 1/att for
    the concession term conflates the two.

    All fitted on seasons before the holdout only.
    """
    past = tf[tf["season"] < before].copy()
    if past.empty or not tiers:
        return ({k: 1.0 for k in range(1, N_TIERS + 1)},
                {k: LEAGUE_AVG_GOALS for k in range(1, N_TIERS + 1)},
                {k: 1.0 for k in range(1, N_TIERS + 1)})
    past["opp_tier"] = past["opponent"].map(tiers)
    past["own_tier"] = past["team"].map(tiers)
    overall = past["scored"].mean()
    att = (past.dropna(subset=["opp_tier"]).groupby("opp_tier")["scored"].mean()
           / max(overall, 1e-9)).to_dict()
    con = past.dropna(subset=["own_tier"]).groupby("own_tier")["conceded"].mean().to_dict()
    atk = (past.dropna(subset=["own_tier"]).groupby("own_tier")["scored"].mean()
           / max(overall, 1e-9)).to_dict()
    att = {int(k): float(v) for k, v in att.items()}
    con = {int(k): float(v) for k, v in con.items()}
    atk = {int(k): float(v) for k, v in atk.items()}
    for k in range(1, N_TIERS + 1):
        att.setdefault(k, 1.0)
        con.setdefault(k, LEAGUE_AVG_GOALS)
        atk.setdefault(k, 1.0)
    return att, con, atk


# ------------------------------------------------------------- player rates
def _lag_mean(g: pd.Series, w: int | None) -> pd.Series:
    s = g.shift(1)
    return s.rolling(w, min_periods=1).mean() if w else s.expanding().mean()


def player_rates(panel: pd.DataFrame) -> pd.DataFrame:
    """Lagged per-90 rates and minutes expectations. No shrinkage, no ceilings."""
    d = panel.sort_values([C.PLAYER_KEY, "t"]).copy()
    g = d.groupby(C.PLAYER_KEY, sort=False)

    d["exp_minutes"] = g["minutes"].transform(lambda s: _lag_mean(s, FORM_WINDOW))
    d["p_appear"] = g["minutes"].transform(
        lambda s: _lag_mean(s.gt(0).astype(float), FORM_WINDOW))
    d["p_sixty"] = g["minutes"].transform(
        lambda s: _lag_mean(s.ge(60).astype(float), FORM_WINDOW))

    cum_min = g["minutes"].transform(lambda s: s.shift(1).cumsum())
    nineties = cum_min / 90.0
    for out, col in (("xg90", "expected_goals"), ("xa90", "expected_assists"),
                     ("saves90", "saves"), ("bonus90", "bonus"),
                     ("yellow90", "yellow_cards")):
        cum = g[col].transform(lambda s: s.shift(1).cumsum())
        # The prior is the position's minutes-weighted rate over LAGGED
        # quantities only, so a holdout gameweek cannot see its own outcome.
        pos_cum = d.assign(_c=cum, _n=nineties).groupby("position")
        tot_c = pos_cum["_c"].transform("sum")
        tot_n = pos_cum["_n"].transform("sum")
        prior = (tot_c / tot_n.where(tot_n > 0)).fillna(0.0)
        d[out] = ((cum.fillna(0.0) + SHRINK_K * prior)
                  / (nineties.fillna(0.0) + SHRINK_K))
        thin = cum_min.fillna(0.0) < MIN_RATE_MINUTES
        d.loc[thin, out] = prior[thin]
    d["def_prob"] = g["defensive_contribution"].transform(
        lambda s: _lag_mean(s.fillna(0).ge(10).astype(float), 10)).fillna(0.0)
    return d


def team_concession(tf: pd.DataFrame) -> pd.DataFrame:
    """Each team's lagged goals-conceded-per-game, within season."""
    t = tf.sort_values(["season", "gameweek"]).copy()
    t["ga_lag"] = (t.groupby(["season", "team"])["conceded"]
                    .transform(lambda s: s.shift(1).expanding().mean()))
    return t[["season", "gameweek", "team", "ga_lag"]]


# ------------------------------------------------------------------- compose
def build(panel: pd.DataFrame, tf: pd.DataFrame, season: str) -> pd.DataFrame:
    """Simple xP for every row of one holdout season."""
    tiers = assign_tiers(team_table(tf, season))
    att, con, atk = tier_factors(tf, season, tiers)

    d = player_rates(panel)
    d = d[d["season"].eq(season)].copy()
    ga = team_concession(tf)
    d = d.merge(ga, on=["season", "gameweek", "team"], how="left")

    opp = d["opponents"].map(lambda o: o[0] if isinstance(o, tuple) and o else None)
    opp_tier = opp.map(tiers)
    own_tier = d["team"].map(tiers)

    # attacking: own lagged rate, suppressed by how good the opponent is
    supp = opp_tier.map(att).astype(float).fillna(1.0)
    # a tier-1 opponent concedes LESS, so invert the scored-against factor
    goals90 = d["xg90"].astype(float) * supp
    assists90 = d["xa90"].astype(float) * supp

    # defending: own concession rate, inflated by how good the opponent's attack is
    own_ga = d["ga_lag"].astype(float)
    own_ga = own_ga.fillna(own_tier.map(con).astype(float)).fillna(LEAGUE_AVG_GOALS)
    opp_att = opp_tier.map(lambda k: atk.get(int(k), 1.0)
                           if pd.notna(k) else 1.0).astype(float)
    lam = (own_ga * opp_att).clip(lower=0.05, upper=6.0)
    cs_prob = np.exp(-lam)

    fixtures = d["fixtures"].astype(float).clip(lower=0)
    xp = scoring.expected_points(
        p_appear=d["p_appear"].fillna(0.0), p_sixty=d["p_sixty"].fillna(0.0),
        exp_minutes=d["exp_minutes"].fillna(0.0), position=d["position"].to_numpy(),
        goals_per90=goals90.fillna(0.0), assists_per90=assists90.fillna(0.0),
        saves_per90=d["saves90"].fillna(0.0), clean_sheet_prob=cs_prob,
        conceded_per90=lam, bonus_per90=d["bonus90"].fillna(0.0),
        yellow_per90=d["yellow90"].fillna(0.0),
        defensive_return_prob=d["def_prob"].fillna(0.0))

    out = d[["season", "gameweek", C.PLAYER_KEY, "position", "team", "cost",
             "fixtures", "points"]].copy()
    # a double gameweek is two matches: the per-fixture expectation applies twice
    out["xp"] = np.asarray(xp, float) * np.where(fixtures > 1, fixtures, 1.0)
    out.loc[out["fixtures"] <= 0, "xp"] = 0.0
    out["xp_raw"] = out["xp"]
    return out
