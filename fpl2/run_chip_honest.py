#!/usr/bin/env python3
"""
Chip timing WITHOUT reading the future fixture schedule.

Policies A (calendar), C (learned) and D (assignment) in run_chip3.py all
scanned season-long double/blank counts to choose chip gameweeks. A June
fixture list contains NO doubles and NO blanks -- every one arises later from
cup progression, European conflicts and postponements, and FPL confirms them
only ~2-4 gameweeks ahead. Those policies were reading the future (R4/M1), and
their +69 / +62 / +52 are not achievable. Measured: doubles occur in just 3-10
gameweeks a season, first appearing at GW19 / GW7 / GW24 / GW26.

Only B (multiweek, 5-gameweek lookahead) was defensible, at +24.

Here every policy sees the schedule ONLY to gw + KNOWN. Beyond that it must
assume the default: one fixture each, no doubles, no blanks.

  E rolling-fixture   fixture-shaped benefit within the visible window, with a
                      deadline forcing a play rather than wasting the chip
  B multiweek         as before, for reference (already honest)
  A* leaky-calendar   the old season-long policy, reported ONLY to show the
                      size of the leak
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, model, optimise, rates, simulate, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor

KNOWN = 4        # gameweeks of fixture schedule visible at any deadline


class _WP:
    def __init__(self, est):
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.est = est

    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight); return self

    def predict(self, X):
        return self.est.predict(self.pre.transform(X))


def leaky_calendar(grid):
    f = grid.groupby("gameweek").agg(
        d=("fixtures", lambda x: float((x > 1).sum())),
        b=("fixtures", lambda x: float((x <= 0).sum())))
    out = {}
    for lo, hi in ((2, 19), (20, 38)):
        w = f[(f.index >= lo) & (f.index <= hi)]
        if w.empty:
            continue
        out[int(lo + 7)] = "wildcard"
        out.setdefault(int(w.d.idxmax()), "bboost")
        for g in w.d.sort_values(ascending=False).index:
            if int(g) not in out:
                out[int(g)] = "3xc"; break
        if w.b.max() > 0:
            for g in w.b.sort_values(ascending=False).index:
                if int(g) not in out:
                    out[int(g)] = "freehit"; break
    return out


def rolling_policy(grid, signal):
    """E: only gameweeks <= gw+KNOWN are visible. Deadline forces a play."""
    gws = sorted(grid["gameweek"].unique())
    seen = {}

    def policy(gw, squad, week, avail):
        visible = [g for g in gws if gw <= g <= gw + KNOWN]
        vis = grid[grid.gameweek.isin(visible)]
        held_now = week[week[C.PLAYER_KEY].isin(squad)]
        if held_now.empty:
            return None
        xi = optimise.best_xi(held_now, signal)
        if xi is None:
            return None
        mine = vis[vis[C.PLAYER_KEY].isin(squad)]
        # how this week looks for MY squad, within what is actually knowable
        my_dbl = float((week[week[C.PLAYER_KEY].isin(squad)].fixtures > 1).sum())
        my_blank = float((week[week[C.PLAYER_KEY].isin(squad)].fixtures <= 0).sum())

        cands = []
        for chip in avail:
            win = next((w for w in simulate.CHIP_WINDOWS[chip] if w[0] <= gw <= w[1]), None)
            if win is None:
                continue
            if chip == "3xc":
                b = float(xi[signal].max()) * (1.6 if my_dbl else 1.0)
            elif chip == "bboost":
                bench = held_now[~held_now[C.PLAYER_KEY].isin(xi[C.PLAYER_KEY])]
                b = float(bench[signal].sum()) * (1.6 if my_dbl else 1.0)
            elif chip == "freehit":
                pool = week[week.fixtures > 0]
                bx = optimise.best_xi(pool, signal)
                b = (float(bx[signal].sum() - xi[signal].sum()) if bx is not None else 0.0)
                b += 2.0 * my_blank
            else:
                allp = vis[vis.fixtures > 0].groupby(C.PLAYER_KEY)[signal].mean()
                mn = mine[mine.fixtures > 0].groupby(C.PLAYER_KEY)[signal].mean()
                b = float(allp.nlargest(15).sum() - mn.nlargest(15).sum())
            cands.append((chip, win, b))
        if not cands:
            return None
        hi = cands[0][1][1]
        left = len([g for g in gws if gw <= g <= hi])
        forced = left <= len([c for c in cands if c[1][1] == hi])
        best = None
        for chip, win, b in cands:
            span = [g for g in gws if win[0] <= g <= win[1]]
            key = (chip, win)
            cut = span[int(len(span) * 0.37)] if len(span) > 1 else span[0]
            if gw <= cut and not forced:
                seen[key] = max(seen.get(key, 0.0), b); continue
            if (b > seen.get(key, 0.0) or gw >= span[-1] or forced) \
               and (best is None or b > best[1]):
                best = (chip, b)
        return best[0] if best else None
    return policy


def main():
    t0 = time.time()
    cur = pd.read_pickle(C.cache("fpl2_live2627.pkl"))
    raw = pd.concat([data.load_raw(), cur], ignore_index=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    print(f"[{time.time()-t0:4.0f}s] setup\n", flush=True)

    print(f"{'season':10s}{'none':>8}{'E rolling':>11}{'gain':>7}"
          f"{'A* leaky':>10}{'leak':>7}", flush=True)
    print("-" * 53, flush=True)
    e_g, a_g = [], []
    for s in C.HOLDOUT_SEASONS:
        out = model.walk_forward(feats, tf, names, s)
        g = data.with_blanks(panel, s).merge(
            out[["season", "gameweek", C.PLAYER_KEY, "xp"]],
            on=["season", "gameweek", C.PLAYER_KEY], how="left")
        g["xp"] = g["xp"].fillna(0.0); g.loc[g.fixtures <= 0, "xp"] = 0.0
        base = float(simulate.simulate_season(g, "xp").total)
        e = float(simulate.simulate_season(g, "xp",
                                           chip_policy=rolling_policy(g, "xp")).total)
        a = float(simulate.simulate_season(g, "xp", chips=leaky_calendar(g)).total)
        e_g.append(e - base); a_g.append(a - base)
        print(f"{s:10s}{base:>8.0f}{e:>11.0f}{e-base:>+7.0f}{a:>10.0f}"
              f"{a-e:>+7.0f}", flush=True)
    print(f"\n  E rolling (honest)  mean {np.mean(e_g):+.0f}  "
          f"{sum(1 for x in e_g if x>0)}/3 seasons", flush=True)
    print(f"  A* leaky            mean {np.mean(a_g):+.0f}  "
          f"{sum(1 for x in a_g if x>0)}/3", flush=True)
    print(f"  SIZE OF THE LEAK    {np.mean(a_g) - np.mean(e_g):+.0f} pts/season", flush=True)
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
