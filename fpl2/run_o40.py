#!/usr/bin/env python3
"""
O40: market signals as a proxy for team news, against the gate pre-registered
2026-10-22. Gate 1 (truncation equivalence) PASSED as a precondition: all 8 new
features at max_abs_diff 0.000000000000.

  2. MECHANISM: minutes correlation must improve 3/3 (the S29 claim is about
     minutes specifically, so a points-only gain does NOT confirm it)
  3. DECISION: season total >= +100 pts/season, 3/3 to ship
  4. R26 control: shuffle the new features within (gameweek, position);
     gates 2 and 3 must then fail
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, minutes as minutes_mod, model, rates, simulate, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor

NEW = ['sold_share_r1','sold_share_r3','bought_share_r1','bought_share_r3',
       'net_share_r1','net_share_r3','owned_share_r1','owned_change_r1']


class _WP:
    def __init__(self, est):
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()); self.est = est
    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight); return self
    def predict(self, X): return self.est.predict(self.pre.transform(X))


def build(market: bool, scramble: bool = False):
    features.MARKET = market
    raw = data.load_raw()
    cur = pd.read_pickle(C.cache("fpl2_live2627.pkl"))
    raw = pd.concat([raw, cur], ignore_index=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    f, n = features.build(panel, tf)
    if scramble:
        rng = np.random.default_rng(11)
        for c in NEW:
            if c in f.columns:
                f[c] = (f.groupby(["gameweek", "position"])[c]
                          .transform(lambda v: rng.permutation(v.to_numpy())))
    return panel, tf, f, n


def main():
    t0 = time.time()
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    arms = {}
    for label, (mk, sc) in {"base": (False, False), "market": (True, False),
                            "shuffled": (True, True)}.items():
        panel, tf, f, n = build(mk, sc)
        arms[label] = (panel, tf, f, n)
        print(f"[{time.time()-t0:5.0f}s] built {label}: {len(n)} features", flush=True)

    print("\n" + "="*72)
    print("GATE 2 -- MECHANISM: predicted vs actual MINUTES")
    print("="*72)
    mins = {}
    for label in ("base", "market", "shuffled"):
        panel, tf, f, n = arms[label]
        mins[label] = {}
        for s in C.HOLDOUT_SEASONS:
            out = minutes_mod.walk_forward(f, n, s)
            t = out[out.fixtures > 0]
            mins[label][s] = float(t["expected_minutes"].corr(t["minutes"]))
        print(f"  {label:<9} " + "  ".join(f"{s} {mins[label][s]:.4f}"
                                           for s in C.HOLDOUT_SEASONS), flush=True)
    d2 = [mins["market"][s] - mins["base"][s] for s in C.HOLDOUT_SEASONS]
    dsh = [mins["shuffled"][s] - mins["base"][s] for s in C.HOLDOUT_SEASONS]
    print(f"\n  market   - base: {['%+.4f'%x for x in d2]}  positive {sum(x>0 for x in d2)}/3")
    print(f"  shuffled - base: {['%+.4f'%x for x in dsh]}  positive {sum(x>0 for x in dsh)}/3")
    print(f"  -> gate 2 {'PASS' if all(x>0 for x in d2) else 'FAIL'} (needs 3/3)")

    print("\n" + "="*72)
    print("GATE 3 -- DECISION: season total")
    print("="*72)
    tot = {}
    for label in ("base", "market", "shuffled"):
        panel, tf, f, n = arms[label]
        tot[label] = {}
        for s in C.HOLDOUT_SEASONS:
            out = model.walk_forward(f, tf, n, s)
            g = data.with_blanks(panel, s).merge(
                out[["season","gameweek",C.PLAYER_KEY,"xp"]],
                on=["season","gameweek",C.PLAYER_KEY], how="left")
            g["xp"] = g["xp"].fillna(0.0)
            g.loc[g["fixtures"] <= 0, "xp"] = 0.0
            tot[label][s] = simulate.simulate_season(g, "xp").total
        print(f"  {label:<9} " + "  ".join(f"{s} {tot[label][s]:7.0f}"
                                           for s in C.HOLDOUT_SEASONS), flush=True)
    d3 = np.array([tot["market"][s] - tot["base"][s] for s in C.HOLDOUT_SEASONS])
    dsh3 = np.array([tot["shuffled"][s] - tot["base"][s] for s in C.HOLDOUT_SEASONS])
    npos = int((d3 > 0).sum())
    verdict = ("SHIP" if d3.mean() >= 100 and npos == 3 else
               "REAL BUT NOT ESTABLISHED" if d3.mean() >= 100 and npos == 2 else "REFUTED")
    print(f"\n  market   - base: {['%+.0f'%x for x in d3]}  mean {d3.mean():+.1f}"
          f"  positive {npos}/3  -> {verdict}")
    print(f"  shuffled - base: {['%+.0f'%x for x in dsh3]}  mean {dsh3.mean():+.1f}"
          f"  positive {int((dsh3>0).sum())}/3   [GATE 4 control]")
    ctl_ok = not (dsh3.mean() >= 100 and (dsh3 > 0).sum() >= 2)
    print(f"  -> gate 4 control {'PASSES (scrambled signals do not pay)' if ctl_ok else 'FAILS'}")
    pickle.dump({"minutes": mins, "totals": tot}, open(C.cache("fpl2_o40.pkl"),"wb"))
    print(f"\n[{time.time()-t0:.0f}s] cached -> ~/fpl2_o40.pkl")


if __name__ == "__main__":
    main()
