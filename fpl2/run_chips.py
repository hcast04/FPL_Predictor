#!/usr/bin/env python3
"""
What are the chips worth? S22 prices the simulator's omissions at ~287/season.

REGRESSION CHECK FIRST: with chips=None the season totals must reproduce the
shipped model exactly (2143 / 2420 / 2149). If they do not, the chip code
changed behaviour it should not have touched.

Then three arms:
  none      no chips (the simulator as it has always been)
  calendar  a FORWARD-LOOKING policy using only the fixture list, known in
            advance: wildcard at a fixed gameweek in each half; bench boost and
            triple captain on the gameweek with the most DOUBLE fixtures in
            each half; free hit on the gameweek with the most BLANKS.
  hindsight each chip placed at its best gameweek in that half, chosen on
            realised points. Unattainable -- it bounds what chips can be worth
            so we do not chase a ceiling that is mostly variance.

S4 recorded chip deferral at ~-92/season with timing worth only +/-5, so the
calendar policy should capture most of the hindsight value. If it does not,
S4's timing claim does not survive contact with this simulator.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, model, rates, simulate, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor

EXPECTED = {"2023-24": 2143, "2024-25": 2420, "2025-26": 2149}


class _WP:
    def __init__(self, est):
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.est = est

    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight); return self

    def predict(self, X):
        return self.est.predict(self.pre.transform(X))


def calendar_chips(grid):
    """Chip timing from the FIXTURE LIST only -- known before the season starts."""
    fx = (grid[grid.fixtures > 0].groupby("gameweek")
            .agg(doubles=("fixtures", lambda x: (x > 1).sum()),
                 playing=("fixtures", "size")))
    blanks = (grid.groupby("gameweek")
                .agg(blank=("fixtures", lambda x: (x <= 0).sum())))
    out = {}
    for lo, hi in ((2, 19), (20, 38)):
        w = fx[(fx.index >= lo) & (fx.index <= hi)]
        b = blanks[(blanks.index >= lo) & (blanks.index <= hi)]
        if w.empty:
            continue
        best_dbl = int(w.doubles.idxmax())
        out[int(lo + 7)] = "wildcard"                 # fixed: early in each half
        out[best_dbl] = "bboost"
        tc = int(w.doubles.sort_values(ascending=False).index[1]) if len(w) > 1 else best_dbl
        if tc not in out:
            out[tc] = "3xc"
        if not b.empty:
            fh = int(b.blank.idxmax())
            if fh not in out and b.blank.max() > 0:
                out[fh] = "freehit"
    return out


def main():
    t0 = time.time()
    cur = pd.read_pickle(C.cache("fpl2_live2627.pkl"))
    raw = pd.concat([data.load_raw(), cur], ignore_index=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    print(f"[{time.time()-t0:4.0f}s] setup\n", flush=True)

    print(f"{'season':10s}{'none':>8}{'expect':>8}{'regress':>9}{'calendar':>10}"
          f"{'gain':>7}{'hindsight':>11}{'gain':>7}", flush=True)
    print("-" * 70, flush=True)
    for s in C.HOLDOUT_SEASONS:
        out = model.walk_forward(feats, tf, names, s)
        grid = data.with_blanks(panel, s).merge(
            out[["season", "gameweek", C.PLAYER_KEY, "xp"]],
            on=["season", "gameweek", C.PLAYER_KEY], how="left")
        grid["xp"] = grid["xp"].fillna(0.0)
        grid.loc[grid["fixtures"] <= 0, "xp"] = 0.0

        base = float(simulate.simulate_season(grid, "xp").total)
        ok = "OK" if abs(base - EXPECTED[s]) < 0.5 else f"BROKE({base:.0f})"

        cal = calendar_chips(grid)
        c_tot = float(simulate.simulate_season(grid, "xp", chips=cal).total)

        # hindsight: each chip at its best gameweek in that half, chosen alone
        best = {}
        for chip, windows in simulate.CHIP_WINDOWS.items():
            for lo, hi in windows:
                cands = [g for g in sorted(grid.gameweek.unique())
                         if lo <= g <= hi and g not in best]
                scored = []
                for g in cands[::3]:        # every 3rd gameweek, for runtime
                    r = simulate.simulate_season(grid, "xp", chips={**best, g: chip})
                    if r:
                        scored.append((float(r.total), g))
                if scored:
                    best[max(scored)[1]] = chip
        h_tot = float(simulate.simulate_season(grid, "xp", chips=best).total)

        print(f"{s:10s}{base:>8.0f}{EXPECTED[s]:>8}{ok:>9}{c_tot:>10.0f}"
              f"{c_tot-base:>+7.0f}{h_tot:>11.0f}{h_tot-base:>+7.0f}", flush=True)
        print(f"           calendar: {cal}", flush=True)
    print(f"\n[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
