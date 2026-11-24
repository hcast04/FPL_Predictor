"""
M37. FPL sets `finished` only after the final bonus check, days after the
football. From Sunday evening until roughly Wednesday a played gameweek reads
`finished=False, is_current=True`, so `max(finished)` returns the week BEFORE
it -- and the squad fetched is the one from before last week's transfers.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plan_gw                        # noqa: E402
import build_wildcard_now             # noqa: E402

MODULES = [plan_gw, build_wildcard_now]


def _boot(current_provisional: bool):
    """GW1-10 finished; GW11 played but awaiting its bonus check."""
    return {"events": [{"id": i, "finished": True, "is_current": False}
                       for i in range(1, 11)]
            + [{"id": 11, "finished": False, "is_current": True}]
            + [{"id": 12, "finished": False, "is_current": False}]}, \
           (lambda url: [{"finished_provisional": current_provisional}] * 10)


@pytest.mark.parametrize("mod", MODULES, ids=lambda m: m.__name__)
def test_a_played_but_unchecked_week_counts_as_played(mod):
    boot, get = _boot(True)
    assert mod.last_played(boot, get) == 11, (
        "GW11 is played -- all fixtures finished_provisional -- so the squad "
        "must be read from GW11, not GW10")


@pytest.mark.parametrize("mod", MODULES, ids=lambda m: m.__name__)
def test_an_unplayed_current_week_does_not_count(mod):
    """NEGATIVE CONTROL: mid-gameweek, GW11 must NOT be treated as played."""
    boot, get = _boot(False)
    assert mod.last_played(boot, get) == 10


@pytest.mark.parametrize("mod", MODULES, ids=lambda m: m.__name__)
def test_the_naive_rule_would_have_been_wrong(mod):
    """
    The bug, stated as a test: `max(finished)` gives 10 where the right
    answer is 11. If these ever agree the fixture no longer reproduces the
    defect and the test above proves nothing.
    """
    boot, get = _boot(True)
    naive = max(e["id"] for e in boot["events"] if e.get("finished"))
    assert naive == 10
    assert mod.last_played(boot, get) != naive


@pytest.mark.parametrize("mod", MODULES, ids=lambda m: m.__name__)
def test_no_current_week_falls_back_cleanly(mod):
    boot = {"events": [{"id": i, "finished": True, "is_current": False}
                       for i in range(1, 11)]}
    assert mod.last_played(boot, lambda url: []) == 10
