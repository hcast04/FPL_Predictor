#!/usr/bin/env python3
"""
O33: three ideas for chip timing, against the calendar heuristic (+47 of +194).

A  calendar    fixture list only -- the incumbent, and currently the best
B  multiweek   benefit estimated over the next H gameweeks instead of one,
               using THIS week's xP as the per-gameweek rate and the published
               fixture count. No future forecast is read (R4).
C  learned     fit gain ~ features on PRIOR seasons' realised chip outcomes,
               then play each chip at the window's predicted argmax.
D  assignment  solve 8 chips -> gameweeks ONCE before the season, on fixture
               features alone, as an optimal assignment rather than greedily.
               Motivated by the incumbent already ignoring the forecast.

A shared precompute gives the realised gain of every (gameweek, chip) pair,
sampled every 2nd gameweek; it is C's training data and the ceiling.
C trains ONLY on seasons strictly before the holdout.
"""
import sys, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, model, optimise, rates, simulate, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor, Ridge

H = 5
CHIPS = ("wildcard", "freehit", "bboost", "3xc")


class _WP:
    def __init__(self, est):
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.est = est

    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight); return self

    def predict(self, X):
        return self.est.predict(self.pre.transform(X))


def fixture_feats(grid):
    """Per-gameweek features from the FIXTURE LIST only -- known in advance."""
    f = grid.groupby("gameweek").agg(
        doubles=("fixtures", lambda x: float((x > 1).sum())),
        blanks=("fixtures", lambda x: float((x <= 0).sum())),
        playing=("fixtures", lambda x: float((x > 0).sum())))
    return f


def calendar_chips(grid):
    f = fixture_feats(grid)
    out = {}
    for lo, hi in ((2, 19), (20, 38)):
        w = f[(f.index >= lo) & (f.index <= hi)]
        if w.empty:
            continue
        out[int(lo + 7)] = "wildcard"
        out.setdefault(int(w.doubles.idxmax()), "bboost")
        srt = w.doubles.sort_values(ascending=False)
        for g in srt.index:
            if int(g) not in out:
                out[int(g)] = "3xc"; break
        if w.blanks.max() > 0:
            for g in w.blanks.sort_values(ascending=False).index:
                if int(g) not in out:
                    out[int(g)] = "freehit"; break
    return out


def assignment_chips(grid):
    """D: optimal 8-chip assignment on fixture features, no collisions."""
    f = fixture_feats(grid)
    # value of each chip at each gameweek, from fixtures only
    val = {}
    for g, r in f.iterrows():
        val[("bboost", int(g))] = r.doubles * 2.0
        val[("3xc", int(g))] = r.doubles * 1.5
        val[("freehit", int(g))] = r.blanks * 1.0
        val[("wildcard", int(g))] = r.doubles * 0.5 + r.playing * 0.02
    best, best_v = None, -1e18
    for lo, hi in ((2, 19), (20, 38)):
        span = [int(g) for g in f.index if lo <= g <= hi]
        if len(span) < 4:
            continue
        local, lv = None, -1e18
        for combo in itertools.permutations(span, 4):
            v = sum(val.get((c, g), 0.0) for c, g in zip(CHIPS, combo))
            if v > lv:
                lv, local = v, dict(zip(combo, CHIPS))
        if local:
            best = {**(best or {}), **local}
    return best or {}


def multiweek_policy(grid, signal):
    """B: benefit over the next H gameweeks, using this week's xP as the rate."""
    gws = sorted(grid["gameweek"].unique())
    fx = grid[grid.fixtures > 0].groupby([C.PLAYER_KEY, "gameweek"]).fixtures.max()
    seen = {}

    def policy(gw, squad, week, avail):
        held = week[week[C.PLAYER_KEY].isin(squad)]
        if held.empty:
            return None
        xi = optimise.best_xi(held, signal)
        if xi is None:
            return None
        fut = [g for g in gws if gw <= g < gw + H]
        cands = []
        for chip in avail:
            win = next((w for w in simulate.CHIP_WINDOWS[chip] if w[0] <= gw <= w[1]), None)
            if win is None:
                continue
            if chip == "3xc":
                b = float(xi[signal].max())
            elif chip == "bboost":
                bench = held[~held[C.PLAYER_KEY].isin(xi[C.PLAYER_KEY])]
                b = float(bench[signal].sum())
            elif chip == "freehit":
                pool = week[week.fixtures > 0]
                bx = optimise.best_xi(pool, signal)
                b = float(bx[signal].sum() - xi[signal].sum()) if bx is not None else 0.0
            else:   # wildcard: this week's rate x future fixture counts
                w = grid[grid.gameweek.isin(fut) & (grid.fixtures > 0)]
                rate = week.set_index(C.PLAYER_KEY)[signal]
                cnt = w.groupby(C.PLAYER_KEY).fixtures.sum()
                proj = (cnt * rate.reindex(cnt.index).fillna(0.0)).dropna()
                b = float(proj.nlargest(15).sum()
                          - proj.reindex(squad).fillna(0.0).sum())
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

    grids, bases, gains = {}, {}, {}
    for s in C.HOLDOUT_SEASONS:
        out = model.walk_forward(feats, tf, names, s)
        g = data.with_blanks(panel, s).merge(
            out[["season", "gameweek", C.PLAYER_KEY, "xp"]],
            on=["season", "gameweek", C.PLAYER_KEY], how="left")
        g["xp"] = g["xp"].fillna(0.0); g.loc[g.fixtures <= 0, "xp"] = 0.0
        grids[s] = g
        bases[s] = float(simulate.simulate_season(g, "xp").total)
        # precompute: realised gain of each single chip at each sampled gameweek
        rows = []
        f = fixture_feats(g)
        for chip in CHIPS:
            for lo, hi in simulate.CHIP_WINDOWS[chip]:
                for gw in [x for x in sorted(g.gameweek.unique()) if lo <= x <= hi][::2]:
                    r = simulate.simulate_season(g, "xp", chips={int(gw): chip})
                    if r:
                        rows.append({"chip": chip, "gw": int(gw),
                                     "gain": float(r.total) - bases[s],
                                     "doubles": f.doubles.get(gw, 0.0),
                                     "blanks": f.blanks.get(gw, 0.0),
                                     "playing": f.playing.get(gw, 0.0),
                                     "frac": gw / 38.0})
        gains[s] = pd.DataFrame(rows)
        print(f"[{time.time()-t0:4.0f}s] {s}: base {bases[s]:.0f}, "
              f"{len(gains[s])} chip-gameweek pairs", flush=True)

    print(f"\n{'season':10s}{'none':>8}{'A cal':>8}{'B multi':>9}{'C learn':>9}"
          f"{'D assign':>10}", flush=True)
    print("-" * 54, flush=True)
    agg = {k: [] for k in "ABCD"}
    for s in C.HOLDOUT_SEASONS:
        g, base = grids[s], bases[s]
        a = float(simulate.simulate_season(g, "xp", chips=calendar_chips(g)).total)
        b = float(simulate.simulate_season(g, "xp",
                                           chip_policy=multiweek_policy(g, "xp")).total)
        # C: learn on PRIOR seasons only
        prior = [gains[p] for p in C.HOLDOUT_SEASONS if p < s]
        if prior:
            tr = pd.concat(prior, ignore_index=True)
            X = ["doubles", "blanks", "playing", "frac"]
            picks = {}
            for chip in CHIPS:
                sub = tr[tr.chip.eq(chip)]
                if len(sub) < 8:
                    continue
                m = make_pipeline(SimpleImputer(strategy="median"),
                                  StandardScaler(), Ridge(alpha=1.0)).fit(
                                      sub[X], sub.gain)
                f = fixture_feats(g).reset_index().rename(columns={"gameweek": "gw"})
                f["frac"] = f.gw / 38.0
                f["pred"] = m.predict(f[X])
                for lo, hi in simulate.CHIP_WINDOWS[chip]:
                    w = f[(f.gw >= lo) & (f.gw <= hi)].sort_values("pred", ascending=False)
                    for _, r in w.iterrows():
                        if int(r.gw) not in picks:
                            picks[int(r.gw)] = chip; break
            c = float(simulate.simulate_season(g, "xp", chips=picks).total)
        else:
            c = float("nan")
        d = float(simulate.simulate_season(g, "xp", chips=assignment_chips(g)).total)
        for k, v in zip("ABCD", (a, b, c, d)):
            agg[k].append(v - base)
        print(f"{s:10s}{base:>8.0f}{a:>8.0f}{b:>9.0f}"
              f"{c if np.isfinite(c) else float('nan'):>9.0f}{d:>10.0f}", flush=True)

    print(f"\n{'policy':12s}{'mean gain':>11}{'seasons+':>10}", flush=True)
    for k, lbl in zip("ABCD", ("A calendar", "B multiweek", "C learned", "D assignment")):
        v = [x for x in agg[k] if np.isfinite(x)]
        print(f"{lbl:12s}{np.mean(v):>+11.0f}{sum(1 for x in v if x>0)}/{len(v):>9}",
              flush=True)
    print(f"  ceiling +194.  FPL average equivalent 2257.", flush=True)
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
