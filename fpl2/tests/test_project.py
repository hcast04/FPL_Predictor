"""
Forward projection (O39).

The whole module exists to forecast week t+k at week t, which is the exact
shape of M17 -- a policy reading a calendar it does not have. So the leak test
is the point of this file, and it carries its own negative control: the same
mutation run through the ORACLE path MUST move the forecast, or the test is
passing because nothing happened rather than because nothing leaked.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpl2 import config as C                      # noqa: E402
from fpl2 import data, features, model, project, team   # noqa: E402

SEASON, FROM, TO = "2025-26", 10, 13
ID_F = {"season", "gameweek", "player_code", "player_name", "position", "team", "t"}
ID_T = {"season", "gameweek", "fixture_id", "team", "opponent", "was_home",
        "is_home", "t"}


@pytest.fixture(scope="module")
def fitted():
    raw = data.load_raw()
    panel = data.build_panel(raw)
    tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)
    m = model.ComponentModel(feats=names).fit(feats[feats.season < SEASON],
                                              tf[tf.season < SEASON])
    return m, feats, tf


def _scramble(df, idcols, mask, seed):
    rng = np.random.default_rng(seed)
    d = df.copy()
    for c in d.columns:
        if c in idcols or not pd.api.types.is_numeric_dtype(d[c]):
            continue
        v = d.loc[mask, c].to_numpy(float)
        if len(v):
            d.loc[mask, c] = rng.permutation(v) * 3.0 + 1.0
    return d


def test_projection_cannot_see_the_future(fitted):
    """
    Scramble every numeric value after week FROM -- player rows and team rows
    alike, keeping only fixture identity intact -- and the week-FROM projection
    of week TO must not move by one float.
    """
    m, feats, tf = fitted
    mf = feats["season"].eq(SEASON) & feats["gameweek"].gt(FROM)
    mt = tf["season"].eq(SEASON) & tf["gameweek"].gt(FROM)
    assert mf.sum() > 1000 and mt.sum() > 100, "mutation would be too small to detect"

    a = project.project(m, feats, tf, SEASON, FROM, TO)
    b = project.project(m, _scramble(feats, ID_F, mf, 7),
                        _scramble(tf, ID_T, mt, 7), SEASON, FROM, TO)
    A = a.set_index(C.PLAYER_KEY)["xp"].sort_index()
    B = b.set_index(C.PLAYER_KEY)["xp"].sort_index()
    idx = A.index.intersection(B.index)
    assert len(idx) > 100
    worst = float(np.nanmax(np.abs(A.loc[idx].to_numpy() - B.loc[idx].to_numpy())))
    assert worst < 1e-9, f"projection reads the future: max diff {worst}"


def test_the_leak_test_is_not_vacuous(fitted):
    """NEGATIVE CONTROL for the test above: the oracle path MUST react."""
    m, feats, tf = fitted
    mf = feats["season"].eq(SEASON) & feats["gameweek"].gt(FROM)
    mt = tf["season"].eq(SEASON) & tf["gameweek"].gt(FROM)
    fm, tm = _scramble(feats, ID_F, mf, 7), _scramble(tf, ID_T, mt, 7)
    sel = lambda f, t: m.predict(f[f.season.eq(SEASON) & f.gameweek.eq(TO)],
                                 t[t.season.eq(SEASON) & t.gameweek.eq(TO)])
    A = sel(feats, tf).set_index(C.PLAYER_KEY)["xp"].sort_index()
    B = sel(fm, tm).set_index(C.PLAYER_KEY)["xp"].sort_index()
    idx = A.index.intersection(B.index)
    worst = float(np.nanmax(np.abs(A.loc[idx].to_numpy() - B.loc[idx].to_numpy())))
    assert worst > 1e-6, "the mutation changed nothing, so the leak test proves nothing"


def test_projecting_to_the_same_week_is_the_ordinary_forecast(fitted):
    m, feats, tf = fitted
    a = project.project(m, feats, tf, SEASON, FROM, FROM)
    b = m.predict(feats[feats.season.eq(SEASON) & feats.gameweek.eq(FROM)],
                  tf[tf.season.eq(SEASON) & tf.gameweek.eq(FROM)])
    assert float(np.nanmax(np.abs(a["xp"].to_numpy() - b["xp"].to_numpy()))) < 1e-12


def test_the_opponent_actually_reaches_the_forecast(fitted):
    """
    NEGATIVE CONTROL on the module's reason to exist. If scrambling WHICH
    opponent each team faces leaves the projection unchanged, then the
    projection is not using opponent information and O39 is pointless.
    """
    m, feats, tf = fitted
    bad = tf.copy()
    msk = bad.season.eq(SEASON)
    rng = np.random.default_rng(5)
    bad.loc[msk, "opponent"] = rng.permutation(bad.loc[msk, "opponent"].to_numpy())
    a = project.project(m, feats, tf, SEASON, FROM, TO)
    b = project.project(m, feats, bad, SEASON, FROM, TO)
    A = a.set_index(C.PLAYER_KEY)["xp"].sort_index()
    B = b.set_index(C.PLAYER_KEY)["xp"].sort_index()
    idx = A.index.intersection(B.index)
    assert float(np.nanmax(np.abs(A.loc[idx].to_numpy()
                                  - B.loc[idx].to_numpy()))) > 1e-6


def test_blank_gameweeks_score_zero(fitted):
    m, feats, tf = fitted
    pr = project.project(m, feats, tf, SEASON, FROM, TO)
    blank = pd.to_numeric(pr["fixtures"], errors="coerce").fillna(0) <= 0
    if blank.any():
        assert float(pr.loc[blank, "xp"].abs().max()) == 0.0


def test_frames_cover_the_horizon_and_carry_xp_plan(fitted):
    m, feats, tf = fitted
    fr = project.frames(m, feats, tf, SEASON, FROM, 4)
    assert 1 <= len(fr) <= 4
    assert [int(f["gameweek"].iloc[0]) for f in fr] == list(range(FROM, FROM + len(fr)))
    for f in fr:
        assert "xp_plan" in f.columns and f["xp_plan"].notna().all()
