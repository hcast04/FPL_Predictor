"""
Team-level model: goals conceded, and from it the clean-sheet probability.

Clean sheets are a TEAM outcome that every player in that team shares, so
modelling them per player would both waste data and let eleven correlated
predictions drift apart. One model per team-fixture, joined back to players.

The clean-sheet probability is derived from the conceded rate under a Poisson
assumption, P(0 conceded) = exp(-lambda), rather than fitted separately. That
keeps the two quantities mutually consistent -- a team cannot be predicted to
concede 2.0 goals and keep a clean sheet half the time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config as C

FEATS = ["att_r5", "att_r10", "att_career", "def_r5", "def_r10", "def_career",
         "opp_att_r5", "opp_att_r10", "opp_att_career",
         "opp_def_r5", "opp_def_r10", "opp_def_career",
         "xgc_r5", "xgc_r10", "xgc_career",
         "opp_xgc_r5", "opp_xgc_r10", "opp_xgc_career",
         "is_home", "gameweek",
         # the quantity that actually decides a clean sheet is the MISMATCH
         # between this defence and that attack, not either in isolation
         "def_minus_opp_att", "att_minus_opp_def"]


def team_fixtures(raw: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (season, gameweek, team, fixture) with lagged strength.

    Carries xGC as well as goals conceded. xGC is the smoother underlying
    quantity (sd/mean 0.68 vs 0.85) and a better clean-sheet predictor from
    trailing form (AUC 0.574 vs 0.546) -- the same argument that made xG the
    right attacking target. Taken as the median over players who played nearly
    the whole match, since FPL reports xGC per player while they were on.
    """
    r = raw.copy()
    r["_xgc"] = pd.to_numeric(r.get("expected_goals_conceded"), errors="coerce")
    tf = (r.groupby(["season", "gameweek", "fixture_id", "team_short_name",
                     "opponent_short_name", "was_home"], as_index=False)
            .agg(scored=("team_goals", "first"),
                 conceded=("opponent_goals", "first"),
                 xgc=("_xgc", lambda x: x[r.loc[x.index, "minutes"] >= 85].median())))
    tf = tf.rename(columns={"team_short_name": "team",
                            "opponent_short_name": "opponent"})
    seasons = sorted(tf["season"].unique())
    order = {s: i for i, s in enumerate(seasons)}
    tf["t"] = tf["season"].map(order) * 38 + tf["gameweek"]
    tf = tf.sort_values(["team", "t", "fixture_id"]).reset_index(drop=True)
    g = tf.groupby("team", sort=False)
    # strictly lagged: a fixture never sees its own result
    tf["att_r5"] = g["scored"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    tf["def_r5"] = g["conceded"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    tf["att_r10"] = g["scored"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    tf["def_r10"] = g["conceded"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    tf["xgc_r5"] = g["xgc"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    tf["xgc_r10"] = g["xgc"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    tf["xgc_career"] = g["xgc"].transform(lambda s: s.shift(1).expanding().mean())
    tf["att_career"] = g["scored"].transform(lambda s: s.shift(1).expanding().mean())
    tf["def_career"] = g["conceded"].transform(lambda s: s.shift(1).expanding().mean())
    src = ["att_r5", "att_r10", "att_career", "def_r5", "def_r10", "def_career",
           "xgc_r5", "xgc_r10", "xgc_career"]
    opp = tf[["season", "gameweek", "fixture_id", "team"] + src].rename(
        columns={"team": "opponent", **{c: "opp_" + c for c in src}})
    tf = tf.merge(opp, on=["season", "gameweek", "fixture_id", "opponent"], how="left")
    tf["is_home"] = tf["was_home"].astype(float)
    tf["def_minus_opp_att"] = tf["def_r5"] - tf["opp_att_r5"]
    tf["att_minus_opp_def"] = tf["att_r5"] - tf["opp_def_r5"]
    return tf


# Shrink predicted goals-conceded toward the training mean before taking
# exp(-lambda). Measured over 4 seasons (`fpl2/run_cs_calib.py`, per player
# among those who played 60+, which is what clean_sheet_prob means), the
# model's clean-sheet probabilities are well calibrated in the middle and badly
# OVER-stated at the strong-defence end:
#
#   predicted 0.111 -> realised 0.144   (ratio 0.77)
#   predicted 0.294 -> realised 0.302   (ratio 0.97)
#   predicted 0.542 -> realised 0.371   (ratio 1.46)
#
# Converting back, lambda = -ln(CS): we predict 0.61 for the best defences when
# the truth is 0.99, and 2.20 for the worst when the truth is 1.94. The lambda
# predictions are OVER-DISPERSED. Goals-conceded is genuinely Poisson
# (var/mean 1.078, actual P(0) 0.242 vs Poisson 0.227), so exp(-lambda) is the
# right functional form -- it is lambda that is not shrunk enough.
#
# 0.0 = no shrinkage, i.e. the behaviour before this was added.
CONCEDED_SHRINK = 0.0


@dataclass
class TeamModel:
    conceded: object = None
    scale: float = 1.0
    prior: float = None          # training-set mean conceded, the shrink target
    feats: list = None           # FEATS usable in THIS fold; see fit()

    def fit(self, train: pd.DataFrame) -> "TeamModel":
        """
        Fit on xGC where available, falling back to goals conceded.

        Predicting the expected quantity rather than the realised count is the
        same change made on the attacking side: xGC is smoother and forecasts
        better. The fitted `scale` converts predicted xGC back onto the goals
        scale so the Poisson clean-sheet derivation stays correct.
        """
        d = train.dropna(subset=["conceded"])
        use_xgc = "xgc" in d.columns and d["xgc"].notna().mean() > 0.5
        target = "xgc" if use_xgc else "conceded"
        fit_on = d.dropna(subset=[target])
        self.scale = (float(fit_on["conceded"].sum() / fit_on["xgc"].sum())
                      if use_xgc and fit_on["xgc"].sum() > 1e-6 else 1.0)
        self.prior = float(np.clip(
            pd.to_numeric(fit_on["conceded"], errors="coerce").mean(), 1e-6, None))
        # Keep only the features that carry information in THIS fold -- the
        # same guard `features.usable` applies to the player models, which this
        # one never had because until 2026-10-22 every training fold happened to
        # contain every column.
        #
        # It stops being true the moment older seasons are added: xGC does not
        # exist before 2022-23, so `xgc_r5`/`xgc_r10`/`xgc_career` and their
        # `opp_` twins are entirely NaN when training on 2020-22, and an
        # all-NaN column reaches the histogram binner as "window shape cannot
        # be larger than input array shape" -- an error from deep inside
        # sklearn that names nothing useful.
        from . import features as _f
        self.feats = _f.usable(fit_on, list(FEATS))
        if not self.feats:
            raise ValueError("TeamModel: no usable features in this fold")
        self.conceded = HistGradientBoostingRegressor(
            loss="poisson", max_iter=400, learning_rate=0.05, max_depth=5,
            min_samples_leaf=20, l2_regularization=0.5, random_state=0
        ).fit(fit_on[self.feats].to_numpy("float32"),
              np.clip(fit_on[target].to_numpy(float), 0, None))
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        cols = self.feats or list(FEATS)      # `or` keeps old pickles working
        lam = np.clip(self.conceded.predict(frame[cols].to_numpy("float32")),
                      1e-6, None) * self.scale
        if CONCEDED_SHRINK > 0.0 and self.prior is not None:
            lam = np.clip((1.0 - CONCEDED_SHRINK) * lam
                          + CONCEDED_SHRINK * self.prior, 1e-6, None)
        return pd.DataFrame({
            "expected_conceded": lam,
            # Poisson P(0), so the two can never contradict each other
            "clean_sheet_prob": np.exp(-lam),
        }, index=frame.index)


def walk_forward(tf: pd.DataFrame, holdout: str) -> pd.DataFrame:
    train = tf[tf["season"] < holdout]
    test = tf[tf["season"].eq(holdout)]
    m = TeamModel().fit(train)
    return test.join(m.predict(test))


def attach_to_players(panel: pd.DataFrame, tf_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Join team predictions onto players.

    A double gameweek gives a player two fixtures: the conceded rates add and
    the clean-sheet probabilities multiply (he needs BOTH to be shut-outs to
    collect twice... but FPL pays per fixture, so the EXPECTED number of clean
    sheets is the sum). Summing is correct for an expectation.
    """
    agg = (tf_pred.groupby(["season", "gameweek", "team"], as_index=False)
           .agg(expected_conceded=("expected_conceded", "sum"),
                clean_sheet_prob=("clean_sheet_prob", "sum"),
                team_fixtures=("fixture_id", "nunique")))
    out = panel.merge(agg, on=["season", "gameweek", "team"], how="left")
    out["expected_conceded"] = out["expected_conceded"].fillna(0.0)
    out["clean_sheet_prob"] = out["clean_sheet_prob"].fillna(0.0).clip(0, 2)
    return out
