"""
Forward projection: forecast week t+k using only what is known at week t.

S28 priced this. Receding-horizon planning (`mpc.py`) converts forecast quality
into points at a steady, monotone rate -- shuffled -143.3, flat projection
-64.3, true four-week forecasts +73.7 pts/season against the greedy policy.
The planner is sound. What it is fed is not: `mpc.honest_frames` reuses week
t's xp scaled by week t+k's fixture COUNT, discarding the opponent entirely,
although the fixture list is published about four weeks ahead and the model
already carries `opp_def_r5`, `opp_att_r5`, `att_vs_opp_def` and `is_home`.

WHY THIS DOES NOT LEAK, stated precisely, because M17 was exactly this class of
mistake and three separate leaks were found on this project (M14, M17, M19):

  * every player-form feature is a strictly lagged aggregate -- `features.py`
    enforces `.shift(1)` before every window and `test_leakage.py` verifies it
    by truncation -- so its week-t value is legitimately known at week t. It is
    FROZEN, not recomputed.
  * the opponent each team faces in week t+k, the venue, and the number of
    fixtures come from the PUBLISHED FIXTURE LIST. That is public at week t.
    `mpc.HORIZON_MAX = 4` bounds how far that claim is safe.
  * the opponent's STRENGTH is taken at week t, not at week t+k. Using week
    t+k's `opp_def_r5` would read results from weeks t..t+k-1. This is the one
    place the leak would hide and it is the reason for `_strength_as_of`.

Nothing is refitted. The same fitted `ComponentModel` is applied to a frame
describing a known future fixture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import team as team_mod

# team-level lagged columns that must be rewound to week t
TEAM_LAG = ["att_r5", "att_r10", "att_career", "def_r5", "def_r10", "def_career",
            "xgc_r5", "xgc_r10", "xgc_career"]


def _strength_as_of(tf: pd.DataFrame, season: str, gw: int) -> pd.DataFrame:
    """
    Each team's lagged strength as it stood at `gw`.

    The LAST fixture at or before `gw` -- not the fixture AT `gw`, because a
    team on a blank has no row that week and would otherwise vanish from the
    projection.
    """
    hist = tf[tf["season"].eq(season) & tf["gameweek"].le(gw)]
    if hist.empty:
        return pd.DataFrame(columns=["team"] + TEAM_LAG)
    return (hist.sort_values(["team", "gameweek", "fixture_id"])
                .groupby("team", as_index=False)[TEAM_LAG].last())


def _player_strength_as_of(feats: pd.DataFrame, season: str,
                           gw: int) -> pd.DataFrame:
    """Each team's `team_*` feature block as it stood at `gw` (see above)."""
    # numeric only: the panel also carries `team_name`, a string, which the
    # "team_" prefix would otherwise sweep into a groupby().mean()
    cols = [c for c in feats.columns
            if c.startswith("team_") and c != "team_fixtures"
            and pd.api.types.is_numeric_dtype(feats[c])]
    hist = feats[feats["season"].eq(season) & feats["gameweek"].le(gw)]
    if hist.empty or not cols:
        return pd.DataFrame(columns=["team"] + cols), cols
    out = (hist.sort_values(["team", "gameweek"])
               .groupby("team", as_index=False)[cols].last())
    return out, cols


def project(model, feats: pd.DataFrame, tf: pd.DataFrame, season: str,
            from_gw: int, to_gw: int) -> pd.DataFrame:
    """
    Forecast `to_gw` using only information available at `from_gw`.

    Returns the week-`from_gw` player rows with `gameweek` set to `to_gw` and
    `xp` replaced by the projection. `to_gw == from_gw` returns the ordinary
    forecast unchanged, so the caller needs no special case for k = 0.
    """
    base = feats[feats["season"].eq(season) & feats["gameweek"].eq(from_gw)].copy()
    if base.empty:
        return base
    tgt_tf = tf[tf["season"].eq(season) & tf["gameweek"].eq(to_gw)]
    if to_gw == from_gw or tgt_tf.empty:
        return model.predict(base, tf[tf["season"].eq(season)
                                      & tf["gameweek"].eq(from_gw)])

    # ---- the team frame: target fixtures, week-`from_gw` strength -----------
    as_of = _strength_as_of(tf, season, from_gw)
    t_frame = tgt_tf.copy()
    for c in TEAM_LAG:
        t_frame[c] = t_frame["team"].map(as_of.set_index("team")[c])
        t_frame["opp_" + c] = t_frame["opponent"].map(as_of.set_index("team")[c])
    t_frame["def_minus_opp_att"] = t_frame["def_r5"] - t_frame["opp_att_r5"]
    t_frame["att_minus_opp_def"] = t_frame["att_r5"] - t_frame["opp_def_r5"]

    # ---- the player frame: frozen form, substituted fixture context --------
    out = base.copy()
    out["gameweek"] = to_gw
    out["gw"] = float(to_gw)

    # how many fixtures each team has in the target week, and the venue mix
    fx = (tgt_tf.groupby("team", as_index=False)
                .agg(_n=("fixture_id", "nunique"), _home=("is_home", "mean")))
    fx = fx.set_index("team")
    out["fixtures"] = out["team"].map(fx["_n"]).fillna(0.0).to_numpy()
    out["n_fixtures"] = out["fixtures"].astype(float)
    out["is_home"] = out["team"].map(fx["_home"]).fillna(0.5).to_numpy()

    # own-team strength: explicitly re-pinned to `from_gw`. It is already the
    # from_gw value because `base` is the from_gw frame, but a future caller
    # passing a different base would silently leak without this.
    own_sf, team_cols = _player_strength_as_of(feats, season, from_gw)
    if team_cols:
        oi = own_sf.set_index("team")
        for c in team_cols:
            out[c] = out["team"].map(oi[c]).to_numpy()

    # opponent strength: the team faced in `to_gw`, valued as at `from_gw`
    opp_map = (tgt_tf.groupby(["team", "opponent"], as_index=False).size()
               .merge(own_sf.rename(columns={"team": "opponent"}),
                      on="opponent", how="left"))
    for c in team_cols:
        agg = opp_map.groupby("team")[c].mean()
        out[c.replace("team_", "opp_")] = out["team"].map(agg).to_numpy()

    if "att_vs_opp_def" in out.columns:
        out["att_vs_opp_def"] = out.get("team_scored_r5") - out.get("opp_conceded_r5")
    if "def_vs_opp_att" in out.columns:
        out["def_vs_opp_att"] = out.get("opp_scored_r5") - out.get("team_conceded_r5")

    pred = model.predict(out, t_frame)
    pred.loc[pd.to_numeric(pred["fixtures"], errors="coerce").fillna(0) <= 0,
             "xp"] = 0.0
    return pred


def frames(model, feats: pd.DataFrame, tf: pd.DataFrame, season: str,
           gw: int, horizon: int) -> list[pd.DataFrame]:
    """
    Planning frames for `mpc.plan`, drop-in for `mpc.honest_frames`.

    Week 0 is the ordinary forecast; weeks 1..H-1 are projections made from
    week 0. Each carries `xp_plan`.
    """
    gws = [g for g in sorted(feats[feats["season"].eq(season)]["gameweek"].unique())
           if gw <= g < gw + horizon]
    out = []
    for g in gws:
        fr = project(model, feats, tf, season, gw, int(g))
        if fr.empty:
            continue
        fr = fr.copy()
        fr["xp_plan"] = pd.to_numeric(fr["xp"], errors="coerce").fillna(0.0)
        out.append(fr)
    return out
