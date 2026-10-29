#!/usr/bin/env python3
"""
Optimal-stopping chip timing. The calendar policy claims +47 of a +194 ceiling
(S23); ~147 points sit in timing alone.

CONSTRAINT: a policy may use only what is knowable at the deadline -- this
week's signal, the current squad, and the published fixture list. It may NOT
read a future gameweek's forecast; those were produced with data from their own
week and reading them at time g is R4's leak, which invalidated an entire
horizon study (M1).

So each chip's benefit is estimated FOR THIS WEEK ONLY:
  3xc       the captain's xP -- what doubling him again is worth
  bboost    the bench's xP -- what the four extra players add
  freehit   best available XI xP minus my own XI's xP, this week
  wildcard  same gap, but measured over the next `horizon` gameweeks, because
            a wildcard squad is kept (M16: optimising it for one week cost
            81 pts/season)

Then the classic secretary rule: observe the first `LOOK` fraction of the
window recording the best benefit seen, then take the first gameweek that beats
it; if none does, play on the last legal gameweek rather than waste the chip
(S4: deferral is the expensive failure, and S23 confirms it is worth +47-194).

Compared against: no chips, the calendar policy, and the hindsight ceiling.
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

LOOK = 0.37          # secretary-problem observation fraction
HORIZON = 5


class _WP:
    def __init__(self, est):
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.est = est

    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight); return self

    def predict(self, X):
        return self.est.predict(self.pre.transform(X))


def make_policy(panel, signal):
    """Secretary-rule chip timing. Closes over the fixture list only."""
    gws = sorted(panel["gameweek"].unique())
    seen = {}        # (chip, window) -> best benefit observed so far

    def benefit(chip, gw, squad, week):
        held = week[week[C.PLAYER_KEY].isin(squad)]
        if held.empty:
            return 0.0
        xi = optimise.best_xi(held, signal)
        if xi is None:
            return 0.0
        if chip == "3xc":
            return float(xi[signal].max())
        if chip == "bboost":
            bench = held[~held[C.PLAYER_KEY].isin(xi[C.PLAYER_KEY])]
            return float(bench[signal].sum())
        if chip == "freehit":
            pool = week[week["fixtures"] > 0]
            best = optimise.best_xi(pool, signal)
            return float(best[signal].sum() - xi[signal].sum()) if best is not None else 0.0
        if chip == "wildcard":
            fut = [g for g in gws if gw <= g < gw + HORIZON]
            w = panel[panel["gameweek"].isin(fut) & (panel["fixtures"] > 0)]
            mine = w[w[C.PLAYER_KEY].isin(squad)].groupby(C.PLAYER_KEY)[signal].mean()
            allp = w.groupby(C.PLAYER_KEY)[signal].mean()
            return float(allp.nlargest(15).sum() - mine.nlargest(15).sum())
        return 0.0

    def policy(gw, squad, week, avail):
        # FPL allows ONE chip per gameweek, so this is a SCHEDULING problem,
        # not a ranking one. The first version checked chips in a fixed order
        # and returned the first match, which let a non-urgent chip claim the
        # deadline gameweek and strand the rest: 2 of 8 went unplayed in
        # 2024-25 (F21), and deferral is the expensive failure (S4, S23).
        cands = []
        for chip in avail:
            win = next((w for w in simulate.CHIP_WINDOWS[chip]
                        if w[0] <= gw <= w[1]), None)
            if win is None:
                continue
            cands.append((chip, win, benefit(chip, gw, squad, week)))
        if not cands:
            return None

        # how many gameweeks are left in this window, and how many chips still
        # need one. If they are equal, something must fire EVERY week from here.
        hi = cands[0][1][1]
        left = len([g for g in gws if gw <= g <= hi])
        need = len([c for c in cands if c[1][1] == hi])
        forced = left <= need

        best = None
        for chip, win, b in cands:
            lo, whi = win
            span = [g for g in gws if lo <= g <= whi]
            if not span:
                continue
            key = (chip, win)
            cutoff = span[int(len(span) * LOOK)] if len(span) > 1 else span[0]
            if gw <= cutoff and not forced:
                seen[key] = max(seen.get(key, 0.0), b)
                continue
            ready = b > seen.get(key, 0.0) or gw >= span[-1] or forced
            if ready and (best is None or b > best[1]):
                best = (chip, b)
        return best[0] if best else None
    return policy


def calendar_chips(grid):
    fx = (grid[grid.fixtures > 0].groupby("gameweek")
            .agg(doubles=("fixtures", lambda x: (x > 1).sum())))
    blanks = grid.groupby("gameweek").agg(blank=("fixtures", lambda x: (x <= 0).sum()))
    out = {}
    for lo, hi in ((2, 19), (20, 38)):
        w = fx[(fx.index >= lo) & (fx.index <= hi)]
        b = blanks[(blanks.index >= lo) & (blanks.index <= hi)]
        if w.empty:
            continue
        out[int(lo + 7)] = "wildcard"
        out[int(w.doubles.idxmax())] = "bboost"
        tc = int(w.doubles.sort_values(ascending=False).index[1]) if len(w) > 1 else int(w.doubles.idxmax())
        out.setdefault(tc, "3xc")
        if not b.empty and b.blank.max() > 0:
            out.setdefault(int(b.blank.idxmax()), "freehit")
    return out


def main():
    t0 = time.time()
    cur = pd.read_pickle(C.cache("fpl2_live2627.pkl"))
    raw = pd.concat([data.load_raw(), cur], ignore_index=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    print(f"[{time.time()-t0:4.0f}s] setup\n", flush=True)

    print(f"{'season':10s}{'none':>8}{'calendar':>10}{'secretary':>11}"
          f"{'gain':>7}{'ceiling':>9}{'captured':>10}", flush=True)
    print("-" * 65, flush=True)
    rows = []
    for s in C.HOLDOUT_SEASONS:
        out = model.walk_forward(feats, tf, names, s)
        grid = data.with_blanks(panel, s).merge(
            out[["season", "gameweek", C.PLAYER_KEY, "xp"]],
            on=["season", "gameweek", C.PLAYER_KEY], how="left")
        grid["xp"] = grid["xp"].fillna(0.0)
        grid.loc[grid["fixtures"] <= 0, "xp"] = 0.0

        base = float(simulate.simulate_season(grid, "xp").total)
        cal = float(simulate.simulate_season(grid, "xp",
                                             chips=calendar_chips(grid)).total)
        r = simulate.simulate_season(grid, "xp",
                                     chip_policy=make_policy(grid, "xp"))
        sec = float(r.total)
        ceil = {"2023-24": 2491, "2024-25": 2545, "2025-26": 2258}[s]
        cap = (sec - base) / max(ceil - base, 1e-9)
        rows.append((base, cal, sec, ceil))
        print(f"{s:10s}{base:>8.0f}{cal:>10.0f}{sec:>11.0f}{sec-base:>+7.0f}"
              f"{ceil:>9.0f}{cap:>9.0%}", flush=True)
        print(f"           chips: {r.chips_played}", flush=True)

    b = np.mean([x[0] for x in rows]); c = np.mean([x[1] for x in rows])
    sc = np.mean([x[2] for x in rows]); ce = np.mean([x[3] for x in rows])
    print(f"\n  mean: none {b:.0f}  calendar {c:.0f} ({c-b:+.0f})  "
          f"secretary {sc:.0f} ({sc-b:+.0f})  ceiling {ce:.0f} ({ce-b:+.0f})", flush=True)
    print(f"  FPL average equivalent 2257 -> secretary is {sc-2257:+.0f}", flush=True)
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
