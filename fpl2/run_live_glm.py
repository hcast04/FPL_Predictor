#!/usr/bin/env python3
"""
2026-27 to date with the GLM, against the user's actual season and the average.

Uses the Poisson-GLM rate component (S12) -- the best signal measured, and the
only one statistically indistinguishable from legacy (S16).

CAVEATS, both material:
  * prices are CURRENT, not historical (`fpl2.live`), so budget-constrained
    selection is mildly optimistic -- a player who has risen is cheap in the
    early gameweeks of this simulation in a way he was not in reality.
  * the simulator starts from a cold optimised squad and takes one free
    transfer a week with no hits, no chips. A real manager had a wildcard and
    made hits. It is a measuring instrument, not a replay of the season.
"""
import sys, time, json, pickle, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, live, model, rates, simulate, team  # noqa: E402
from fpl2 import chips as chips_mod   # noqa: E402

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor

SEASON = "2026-27"
LIVE = C.cache("fpl2_live2627.pkl")


class _WeightedPipe:
    def __init__(self, est):
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.est = est

    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.est.predict(self.pre.transform(X))


def api(path):
    return json.load(urllib.request.urlopen(
        f"https://fantasy.premierleague.com/api/{path}", timeout=30))


def _entry():
    if not ENTRY:
        raise SystemExit(
            "set your FPL team id first:  export FPL_ENTRY=1234567\n"
            "(it is the number in fantasy.premierleague.com/entry/<id>/)")
    return ENTRY


def main():
    t0 = time.time()
    if LIVE.exists():
        cur = pd.read_pickle(LIVE)
    else:
        cur = live.build(SEASON); cur.to_pickle(LIVE)
    print(f"[{time.time()-t0:4.0f}s] live panel {len(cur):,} rows", flush=True)

    boot = api("bootstrap-static/")
    avg = {e["id"]: e.get("average_entry_score", 0) for e in boot["events"]
           if e.get("finished") or e.get("data_checked")}
    hist = api("entry/{_entry()}/history/")
    usr = {e["event"]: e["points"] for e in hist["current"]}

    raw = pd.concat([data.load_raw(), cur], ignore_index=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)
    rates._reg = lambda s=0: _WeightedPipe(PoissonRegressor(alpha=1.0, max_iter=500))
    out = model.walk_forward(feats, tf, names, SEASON)
    print(f"[{time.time()-t0:4.0f}s] forecast done", flush=True)

    t = out[out.fixtures > 0]
    print(f"  ratio {t.xp.sum()/t.points.sum():.3f}   "
          f"spearman {t.xp.corr(t.points, method='spearman'):.4f}", flush=True)

    grid = data.with_blanks(panel, SEASON).merge(
        out[["season", "gameweek", C.PLAYER_KEY, "xp"]],
        on=["season", "gameweek", C.PLAYER_KEY], how="left")
    grid["xp"] = grid["xp"].fillna(0.0)
    grid.loc[grid["fixtures"] <= 0, "xp"] = 0.0
    r = simulate.simulate_season(grid, "xp",
                                 chip_policy=chips_mod.rolling_policy(grid, "xp"))

    gws = sorted(grid.gameweek.unique())
    print(f"\n{'gw':>3}{'model':>8}{'you':>7}{'avg':>7}{'vs you':>9}{'vs avg':>9}", flush=True)
    cm = cu = ca = 0
    for i, gw in enumerate(gws):
        m = r.per_gameweek[i] if i < len(r.per_gameweek) else float("nan")
        u, a = usr.get(gw), avg.get(gw)
        if u is None or a is None or not np.isfinite(m):
            continue
        cm += m; cu += u; ca += a
        print(f"{gw:>3}{m:>8.0f}{u:>7}{a:>7}{m-u:>+9.0f}{m-a:>+9.0f}", flush=True)
    print(f"\n{'TOT':>3}{cm:>8.0f}{cu:>7}{ca:>7}{cm-cu:>+9.0f}{cm-ca:>+9.0f}", flush=True)
    n = len([g for g in gws if g in usr and g in avg])
    print(f"\n  per gameweek: model {cm/n:.1f}, you {cu/n:.1f}, average {ca/n:.1f}", flush=True)
    print(f"  transfers made: {r.transfers}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
