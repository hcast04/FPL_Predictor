#!/usr/bin/env python3
"""
Minutes over-shrinkage fix (S19/S20), measured on the DECISION metric.

Four arms per season: ours +/- the fix, legacy +/- the same fix applied
post-hoc (legacy is a cached forecast, so its minutes model cannot be re-run;
xp is near-linear in minutes, so scaling by blended/unblended is a fair
approximation). For 2026-27 the real FPL global average is included.

GATE, declared before the run, and on the DECISION metric only:
  the fix must improve mean season total vs legacy in at least 3 of 3 holdout
  seasons. Predicted/actual minutes ratios are NOT the gate -- calibration
  fixes went 0 for 3 on 2026-10-21 (W13, F19, and R9 predates both), each time
  improving the thing they targeted and degrading points.

K is swept and the WHOLE curve reported; no value is selected on the run.
"""
import sys, time, json, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, legacy, model, rates, simulate, team  # noqa: E402
from fpl2 import minutes as minutes_mod   # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor

GRID = (0.0, 2.0, 4.0, 8.0)


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

    boot = json.load(urllib.request.urlopen(
        "https://fantasy.premierleague.com/api/bootstrap-static/", timeout=30))
    avg7 = sum(e.get("average_entry_score", 0) for e in boot["events"]
               if e.get("finished") or e.get("data_checked"))

    seasons = list(C.HOLDOUT_SEASONS) + ["2026-27"]
    base_minutes = {}
    print(f"[{time.time()-t0:4.0f}s] setup; 2026-27 global average = {avg7}\n", flush=True)

    res = {}
    for k in GRID:
        minutes_mod.RECENCY_K = k
        row = {}
        for s in seasons:
            out = model.walk_forward(feats, tf, names, s)
            grid = data.with_blanks(panel, s)
            g = grid.merge(out[["season", "gameweek", C.PLAYER_KEY, "xp",
                                "expected_minutes"]],
                           on=["season", "gameweek", C.PLAYER_KEY], how="left")
            g["xp"] = g["xp"].fillna(0.0)
            g["expected_minutes"] = g["expected_minutes"].fillna(0.0)
            g.loc[g["fixtures"] <= 0, "xp"] = 0.0
            if k == 0.0:
                base_minutes[s] = g.set_index(
                    ["gameweek", C.PLAYER_KEY])["expected_minutes"]
            lg = legacy.attach(data.with_blanks(panel, s)).rename(
                columns={"xp": "xp_leg"}) if s != "2026-27" else None
            if lg is not None:
                g = g.merge(lg[["season", "gameweek", C.PLAYER_KEY, "xp_leg"]],
                            on=["season", "gameweek", C.PLAYER_KEY], how="left")
                g["xp_leg"] = g["xp_leg"].fillna(0.0)
                g.loc[g["fixtures"] <= 0, "xp_leg"] = 0.0
                # same correction applied post-hoc to legacy
                b = g.set_index(["gameweek", C.PLAYER_KEY]).index.map(
                    base_minutes[s]).to_numpy(float)
                scale = np.where(b > 1e-6, g.expected_minutes.to_numpy(float)
                                 / np.maximum(b, 1e-6), 1.0)
                g["xp_leg_fix"] = g["xp_leg"] * np.clip(scale, 0.5, 2.0)
                row[f"{s}_leg"] = float(simulate.simulate_season(g, "xp_leg").total)
                row[f"{s}_legfix"] = float(simulate.simulate_season(g, "xp_leg_fix").total)
            row[s] = float(simulate.simulate_season(g, "xp").total)
        res[k] = row
        print(f"K={k:.0f}  " + "  ".join(f"{s}:{row[s]:.0f}" for s in seasons), flush=True)
    minutes_mod.RECENCY_K = 0.0

    print(f"\n{'':12s}" + "".join(f"{s:>11}" for s in seasons), flush=True)
    print("-" * 56, flush=True)
    for k in GRID:
        print(f"ours K={k:<5.0f}" + "".join(f"{res[k][s]:>11.0f}" for s in seasons),
              flush=True)
    print(f"{'legacy':12s}" + "".join(
        f"{res[0.0].get(s+'_leg', float('nan')):>11.0f}" for s in seasons), flush=True)
    for k in GRID[1:]:
        print(f"legacy+K={k:<4.0f}" + "".join(
            f"{res[k].get(s+'_legfix', float('nan')):>11.0f}" for s in seasons), flush=True)
    print(f"{'FPL average':12s}" + " " * 33 + f"{avg7:>11}", flush=True)

    print(f"\nGATE (holdout seasons only; needs 3/3 improvement vs K=0)", flush=True)
    for k in GRID[1:]:
        w = sum(1 for s in C.HOLDOUT_SEASONS if res[k][s] > res[0.0][s])
        d = np.mean([res[k][s] - res[0.0][s] for s in C.HOLDOUT_SEASONS])
        print(f"  K={k:.0f}  improves {w}/3   mean {d:+.0f}   "
              f"{'PASS' if w == 3 else 'FAIL'}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
