#!/usr/bin/env python3
"""
O28: re-test the minutes fix with a gate that counts DECISIONS, not seasons.

F20 failed 3/3-seasons at 2/3 and that verdict stands. But S21 showed the two
losing/winning seasons are not comparable events: in 2025-26 the fix changed
the captain in 30 of 38 gameweeks (+219); in 2024-25 it changed 2 (-33, pure
variance). A gate counting SEASONS cannot distinguish "acted and worked" from
"did nothing and the noise went the wrong way".

So condition on the treatment actually being applied. ~40-60 changed decisions
across four seasons instead of 3 season totals.

GATE, declared before the run:
  1. on gameweeks where the fix CHANGES the captain, the mean realised
     difference must be positive with |t| >= 2;
  2. CONTROL -- on gameweeks where it changes nothing, the difference must be
     indistinguishable from zero (|t| < 2). If the control moves, the arms have
     drifted apart through divergent squads and the comparison is contaminated,
     not measuring the fix.
Both clauses must hold. Clause 2 failing invalidates clause 1.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, model, rates, simulate, team  # noqa: E402
from fpl2 import minutes as minutes_mod   # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor

SEASONS = list(C.HOLDOUT_SEASONS) + ["2026-27"]
K = 4.0


class _WP:
    def __init__(self, est):
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.est = est

    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight); return self

    def predict(self, X):
        return self.est.predict(self.pre.transform(X))


def main():
    t0 = time.time()
    cur = pd.read_pickle(C.cache("fpl2_live2627.pkl"))
    raw = pd.concat([data.load_raw(), cur], ignore_index=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    print(f"[{time.time()-t0:4.0f}s] setup\n", flush=True)

    rows = []
    for s in SEASONS:
        arms = {}
        for k in (0.0, K):
            minutes_mod.RECENCY_K = k
            out = model.walk_forward(feats, tf, names, s)
            g = data.with_blanks(panel, s).merge(
                out[["season", "gameweek", C.PLAYER_KEY, "xp"]],
                on=["season", "gameweek", C.PLAYER_KEY], how="left")
            g["xp"] = g["xp"].fillna(0.0)
            g.loc[g["fixtures"] <= 0, "xp"] = 0.0
            arms[k] = simulate.simulate_season(g, "xp")
        minutes_mod.RECENCY_K = 0.0
        a, b = arms[0.0], arms[K]
        la = {gw: (set(xi), cap, val) for gw, xi, cap, val in a.squad_log}
        lb = {gw: (set(xi), cap, val) for gw, xi, cap, val in b.squad_log}
        for gw in sorted(set(la) & set(lb)):
            xa, ca, va = la[gw]; xb, cb, vb = lb[gw]
            rows.append({"season": s, "gw": gw, "diff": vb - va,
                         "cap_changed": ca != cb,
                         "xi_changed": xa != xb})
        print(f"  {s}: captain changed {sum(1 for gw in set(la)&set(lb) if la[gw][1]!=lb[gw][1])}"
              f"/{len(set(la)&set(lb))}  season {a.total:.0f} -> {b.total:.0f}", flush=True)

    d = pd.DataFrame(rows)

    def stat(sub, label):
        if len(sub) < 2:
            print(f"  {label:34s} n={len(sub):3d}  (too few)", flush=True)
            return None
        x = sub["diff"].to_numpy(float)
        se = x.std(ddof=1) / np.sqrt(len(x))
        t = x.mean() / se if se > 0 else np.nan
        wins = int((x > 0).sum()); losses = int((x < 0).sum())
        print(f"  {label:34s} n={len(x):3d}  mean {x.mean():+7.2f}  se {se:5.2f}  "
              f"t {t:+5.2f}  {wins}W/{losses}L", flush=True)
        return t

    print("\nPER-GAMEWEEK PAIRED DIFFERENCE  (K=4 minus K=0)", flush=True)
    stat(d, "ALL gameweeks")
    t_cap = stat(d[d.cap_changed], "captain CHANGED")
    t_none = stat(d[~d.cap_changed & ~d.xi_changed], "CONTROL: nothing changed")
    stat(d[~d.cap_changed & d.xi_changed], "XI changed, captain same")

    print("\nGATE", flush=True)
    c1 = t_cap is not None and t_cap >= 2
    c2 = t_none is None or abs(t_none) < 2
    print(f"  1. captain-changed gameweeks gain, |t|>=2 : "
          f"{'PASS' if c1 else 'FAIL'}", flush=True)
    print(f"  2. CONTROL indistinguishable from zero    : "
          f"{'PASS' if c2 else 'FAIL'}", flush=True)
    print(f"  => {'PASS' if (c1 and c2) else 'FAIL'}", flush=True)
    if not c2:
        print("  control moved -- the arms drifted apart, so clause 1 is not", flush=True)
        print("  measuring the fix and must not be read as if it were.", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
