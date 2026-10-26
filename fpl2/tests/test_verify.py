"""
Every check must FIRE on the defect it was written for.

A verification harness that has only ever passed is worth nothing, so each
test replays the real case from 2026-10-23 and asserts the check catches it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpl2 import verify as V     # noqa: E402


# --- reference_identical ---------------------------------------------------
def test_reference_identical_passes_on_an_exact_copy():
    a = np.linspace(0, 5, 500)
    assert V.reference_identical(a, a.copy()) == 0.0


def test_reference_identical_catches_the_cross_arm_state_leak():
    """The real case: 0.0016 max difference from a defensive model carried
    between arms. Small enough to look like rounding, large enough to matter."""
    a = np.linspace(0, 5, 500)
    b = a.copy(); b[17] += 0.0015939820
    with pytest.raises(V.VerificationError, match="does not"):
        V.reference_identical(b, a)


# --- reorders (R43) --------------------------------------------------------
def test_reorders_flags_a_monotone_transform_as_not_reordering():
    """S56: bonus scaled by P(return), which is monotone in xp. rho 0.9998."""
    rng = np.random.default_rng(0)
    xp = rng.gamma(2.0, 1.5, 600)
    scaled = xp * (0.3 + 0.7 * xp / xp.max())      # monotone in xp
    r = V.reorders(scaled, xp, by=np.zeros(600))
    assert r["rank_rho"] > 0.999
    assert r["reorders"] is False


def test_reorders_detects_a_genuine_reordering():
    """S59's tenure shrink keys off a variable orthogonal to xp, so it does
    reorder -- which is why it was worth testing even though it measured +1."""
    rng = np.random.default_rng(1)
    xp = rng.gamma(2.0, 1.5, 600)
    tenure = rng.permutation(600) / 600.0          # orthogonal to xp
    r = V.reorders(xp * (0.5 + tenure), xp, by=np.zeros(600))
    assert r["reorders"] is True
    assert r["mean_rank_shift"] > 5


# --- stable_population (R29) ----------------------------------------------
def test_stable_population_catches_arms_scored_on_different_rows():
    """The real case: 'top 10 by xp' ranked by EACH ARM'S own xp, so the
    shrink demoted movers out of the band and the arms were not comparable."""
    with pytest.raises(V.VerificationError, match="DIFFERENT populations"):
        V.stable_population({"K0": np.array([True] * 195 + [False] * 5),
                             "K40": np.array([True] * 138 + [False] * 62)})


def test_stable_population_passes_when_the_rows_are_fixed():
    m = np.array([True] * 50 + [False] * 50)
    assert V.stable_population({"a": m, "b": m.copy()}) == {"a": 50, "b": 50}


# --- dose_response (R38) ---------------------------------------------------
def test_dose_response_flags_the_horizon_leak():
    """The real numbers: monotone to +312.8 at h=19, which was F10's leak."""
    r = V.dose_response([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 19],
                        [-159, -130, -107, -68, 0, 30, 83, 137, 204, 234, 313])
    assert r["suspect_leak"] is True
    assert "LEAK" in r["warning"]


def test_dose_response_accepts_an_interior_optimum():
    r = V.dose_response([1, 2, 5, 10, 20], [-5, 2, 11, 4, -8])
    assert r["suspect_leak"] is False


# --- price_ceiling (R48) ---------------------------------------------------
def test_price_ceiling_would_have_predicted_the_tenure_result():
    """-0.85 pts/gameweek on 6.4% of top-10 rows. The gate measured +1.0."""
    r = V.price_ceiling(per_unit_effect=-0.85, affected_rows=195,
                        total_rows=3030, slots=11)
    assert r["ceiling_per_season"] < 30


def test_price_ceiling_respects_the_number_of_slots():
    """A vice-captain rule touches ONE place, not an XI. The first version of
    this function hardcoded 11 and priced it at 223 pts/season instead of 20."""
    one = V.price_ceiling(per_unit_effect=12.84, affected_rows=126,
                          total_rows=3030, slots=1)
    eleven = V.price_ceiling(per_unit_effect=12.84, affected_rows=126,
                             total_rows=3030, slots=11)
    assert 15 < one["ceiling_per_season"] < 25
    assert eleven["ceiling_per_season"] == pytest.approx(
        one["ceiling_per_season"] * 11)


# --- strictly_lagged -------------------------------------------------------
def _series(leaky: bool):
    rng = np.random.default_rng(3)
    rows = []
    for team in range(6):
        v = rng.normal(50, 8, 40)
        s = pd.Series(v)
        f = s.rolling(5, min_periods=2).mean() if leaky else \
            s.shift(1).rolling(5, min_periods=2).mean()
        rows.append(pd.DataFrame({"team": team, "i": np.arange(40),
                                  "poss": v, "feat": f.to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def test_strictly_lagged_passes_on_a_shifted_feature():
    r = V.strictly_lagged(_series(False), "team", "poss", "feat", 5, order="i")
    assert r["mismatches"] == 0
    assert r["control_failures"] > 0        # the test can fail


def test_strictly_lagged_catches_an_unshifted_feature():
    with pytest.raises(V.VerificationError, match="READ THE FUTURE"):
        V.strictly_lagged(_series(True), "team", "poss", "feat", 5, order="i")


# --- clustered (R32) -------------------------------------------------------
def test_clustered_refuses_a_single_cluster():
    """A fanned-out per-season process has G=1 and must defer, not divide
    by zero -- which is what run_opt_bakeoff2 did on all eight jobs."""
    with pytest.raises(V.VerificationError, match="at least two"):
        V.clustered(np.ones(38), np.array(["2024-25"] * 38))


def test_clustered_reports_whether_the_effect_is_resolvable():
    rng = np.random.default_rng(5)
    g = np.repeat([f"s{i}" for i in range(8)], 100)
    tiny = rng.normal(0.2, 10, 800)
    r = V.clustered(tiny, g)
    assert r["G"] == 8
    assert r["mde"] > abs(r["mean"])
    assert r["resolvable"] is False
