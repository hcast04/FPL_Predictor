"""
The component model: compose minutes, rates and team strength into xP.

    xP = f(P(appear), P(60+), E[minutes], per-90 rates, clean-sheet prob)

where f is `scoring.expected_points` -- the FPL rules, verified to reconstruct
99.4% of four seasons of real points exactly. Nothing here learns the scoring
system; it is known, so it is applied rather than fitted. That is the whole
argument for a component model over direct regression to points.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from . import minutes as minutes_mod
from . import rates as rates_mod
from . import scoring
from . import shrink as shrink_mod
from . import team as team_mod


# Which rate quantities a minimal model replaces with a per-position constant,
# and the realised column each constant is computed from.
_FLAT_RATES = {"goals_per90": "goals_scored", "assists_per90": "assists",
               "saves_per90": "saves", "bonus_per90": "bonus",
               "yellow_per90": "yellow_cards"}


@dataclass
class ComponentModel:
    feats: list
    minutes: minutes_mod.MinutesModel = None
    rates: rates_mod.RateModel = None
    team: team_mod.TeamModel = None
    prior: "shrink_mod.ShrinkagePrior" = None
    # O50. S48 ablated every component on 8 holdout seasons and found that
    # flattening the ENTIRE rate layer -- goals, assists, saves, bonus, clean
    # sheets, defensive returns -- costs +0.3 points a season (p = 0.99), while
    # the minutes hierarchy is the only part that matters, and only for
    # midfielders (-37.8) and forwards (-31.1). The goal model is ACTIVELY
    # HARMFUL for forwards (+19.1 from deleting it).
    #
    #   "full"    everything, the shipped model
    #   "minimal" minutes + team/clean-sheet model, rates -> position means
    #   "strict"  minutes only; clean sheets flattened too (== S48's
    #             `minutes_only` arm, the one that measured +0.3)
    #
    # Default "full" so every prior result reproduces unchanged.
    mode: str = "full"
    flat: dict = None            # position -> {rate name: constant}

    def _fit_flat(self, train: pd.DataFrame):
        """Per-position per-90 constants from TRAINING realised values only."""
        out = {}
        mins = train.groupby("position")["minutes"].sum().clip(lower=1)
        for name, col in _FLAT_RATES.items():
            if col not in train.columns:
                continue
            per90 = 90.0 * train.groupby("position")[col].sum() / mins
            for pos, v in per90.items():
                out.setdefault(pos, {})[name] = float(v)
        if "clean_sheets" in train.columns:
            cs = train.groupby("position")["clean_sheets"].mean()
            for pos, v in cs.items():
                out.setdefault(pos, {})["clean_sheet_prob"] = float(v)
        return out

    def fit(self, train_players: pd.DataFrame, train_teams: pd.DataFrame):
        self.minutes = minutes_mod.MinutesModel(feats=self.feats).fit(train_players)
        if self.mode == "full":
            self.rates = rates_mod.RateModel(feats=self.feats).fit(train_players)
        else:
            # the rate layer is not fitted at all -- that is most of the cost
            self.flat = self._fit_flat(train_players[train_players["fixtures"] > 0])
        self.team = team_mod.TeamModel().fit(train_teams)
        return self

    def refit_defensive(self, history: pd.DataFrame):
        """Refit only the defensive component on within-season history."""
        if self.rates is not None:
            self.rates.fit_defensive_within_season(history)
        return self

    def predict(self, players: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
        mins = self.minutes.predict(players)
        if self.mode == "full":
            rts = self.rates.predict(players)
        else:
            P = rates_mod.RateModel.PREFIX
            pos = players["position"]
            rts = pd.DataFrame(index=players.index)
            for name in _FLAT_RATES:
                rts[P + name] = pos.map(
                    lambda q: (self.flat.get(q) or {}).get(name, 0.0)).astype(float)
            rts[P + "xg_per90"] = 0.0
            rts[P + "xa_per90"] = 0.0
            rts[P + "defensive_return_prob"] = 0.0
            rts.loc[pos.ne("GK"), P + "saves_per90"] = 0.0
        out = players.join(mins).join(rts)

        tp = teams.join(self.team.predict(teams))
        out = team_mod.attach_to_players(out, tp)
        if self.mode == "strict":
            # S48's `minutes_only`: the clean-sheet model is flattened too
            out["clean_sheet_prob"] = out["position"].map(
                lambda q: (self.flat.get(q) or {}).get("clean_sheet_prob", 0.0)
            ).astype(float)

        # per-fixture conceded rate; a double gameweek already has both summed
        fx = pd.to_numeric(out["fixtures"], errors="coerce").fillna(0).clip(lower=1)
        conceded_per90 = out["expected_conceded"].to_numpy(float) / fx.to_numpy(float)

        P = rates_mod.RateModel.PREFIX
        out["xp"] = scoring.expected_points(
            p_appear=out["p_appear"], p_sixty=out["p_sixty"],
            exp_minutes=out["expected_minutes"], position=out["position"],
            goals_per90=out[P + "goals_per90"],
            assists_per90=out[P + "assists_per90"],
            saves_per90=out[P + "saves_per90"],
            clean_sheet_prob=out["clean_sheet_prob"],
            conceded_per90=conceded_per90,
            bonus_per90=out[P + "bonus_per90"],
            yellow_per90=out[P + "yellow_per90"],
            defensive_return_prob=out[P + "defensive_return_prob"],
        )
        out["xp_raw"] = out["xp"].fillna(0.0).clip(lower=0.0)
        if self.prior is not None:
            out["xp"] = np.clip(
                self.prior.apply(out, out["xp_raw"].to_numpy(float)), 0.0, None)
        else:
            out["xp"] = out["xp_raw"]
        return out


def walk_forward(features: pd.DataFrame, team_fixtures: pd.DataFrame,
                 feats: list, holdout: str, mode: str = "full") -> pd.DataFrame:
    """
    Train on every season strictly before `holdout`; forecast `holdout`.

    """
    tr_p = features[features["season"] < holdout]
    te_p = features[features["season"].eq(holdout)]
    tr_t = team_fixtures[team_fixtures["season"] < holdout]
    te_t = team_fixtures[team_fixtures["season"].eq(holdout)]
    if tr_p.empty or te_p.empty:
        raise ValueError(f"no data for holdout {holdout}")

    # Shrinkage toward a position x price prior was tested and REJECTED
    # (no k won all three seasons; see fpl2/shrink.py). The pipeline runs
    # unshrunk, so xp == xp_raw.
    m = ComponentModel(feats=feats, mode=mode).fit(tr_p, tr_t)

    # The defensive-contribution inputs do not exist before 2025-26, so that
    # one component is refit gameweek by gameweek on the holdout season's own
    # earlier weeks. Strictly expanding: week k is predicted from weeks < k.
    out = []
    for gw in sorted(te_p["gameweek"].unique()):
        hist = te_p[te_p["gameweek"] < gw]
        if len(hist) > 500:
            m.refit_defensive(hist)
        out.append(m.predict(te_p[te_p["gameweek"].eq(gw)],
                             te_t[te_t["gameweek"].eq(gw)]))
    return pd.concat(out, ignore_index=True)


