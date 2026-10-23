#!/usr/bin/env python3
"""
Two-factor experiment: how much history should the model train on, and which
rolling windows should it see?

  FACTOR A  training window X -- train only on the last X gameweeks before the
            one being predicted. X = "all" means every prior gameweek.
  FACTOR B  feature windows -- which trailing means (3 / 5 / 10 gameweeks) are
            offered as predictors.

These are different questions and are often conflated. A is about discarding
stale data; B is about letting the model see form at several timescales while
still training on everything.

Protocol is strict walk-forward: to predict (season S, gameweek t) the model
trains only on rows chronologically before it, never on the target gameweek.

The decision metric is BEST LEGAL XI points (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD,
captain doubled) because that is what actually decides a gameweek. Spearman and
top-10 are reported alongside since they can disagree.

A result is only reported as real if it holds in EVERY holdout season.

Run:  .venv/bin/python fixes/training_window_sweep.py
"""
from __future__ import annotations
import pickle, sys, time
from itertools import product
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import xgboost as xgb

SCR = ("/tmp/claude-1000/-home-mdmsystems-acer-Escritorio-pers-fpl/"
       "a909bbf0-dfbe-4cc0-a5f8-534424e1d2b9/scratchpad")
XS = [3, 5, 8, 12, 20, 38, 76, "all"]
WINSETS = {"w3": [3], "w3_5": [3, 5], "w3_5_10": [3, 5, 10], "w10": [10]}
HOLDOUTS = ["2023-24", "2024-25", "2025-26"]
MIN_GW = 6
BOUNDS = {"DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}


def best_xi_points(g: pd.DataFrame, pred: str) -> float:
    """Actual points of the best legal XI chosen by `pred`, captain doubled."""
    by = {p: g[g.position.eq(p)].nlargest(6, pred) for p in ("GK", "DEF", "MID", "FWD")}
    if by["GK"].empty:
        return np.nan
    gk = by["GK"].iloc[0]
    best, best_pred = None, -np.inf
    for dd in range(BOUNDS["DEF"][0], BOUNDS["DEF"][1] + 1):
        for mm in range(BOUNDS["MID"][0], BOUNDS["MID"][1] + 1):
            ff = 10 - dd - mm
            if not (BOUNDS["FWD"][0] <= ff <= BOUNDS["FWD"][1]):
                continue
            if len(by["DEF"]) < dd or len(by["MID"]) < mm or len(by["FWD"]) < ff:
                continue
            xi = pd.concat([by["DEF"].head(dd), by["MID"].head(mm), by["FWD"].head(ff)])
            tot = float(xi[pred].sum()) + float(gk[pred])
            if tot > best_pred:
                best_pred, best = tot, xi
    if best is None:
        return np.nan
    squad = pd.concat([best, gk.to_frame().T])
    cap = squad.loc[squad[pred].astype(float).idxmax()]
    return float(squad["y"].sum()) + float(cap["y"])


def main() -> None:
    d = pd.read_pickle(f"{SCR}/features.pkl")
    spec = pickle.load(open(f"{SCR}/featspec.pkl", "rb"))
    d = d[d.y.notna()].copy()
    rows = []
    t0 = time.time()
    for wname, wins in WINSETS.items():
        feats = spec["base"] + sum((spec["win"][w] for w in wins), [])
        Xall = d[feats].to_numpy("float32")
        yall = d["y"].to_numpy("float32")
        gidx = d["gidx"].to_numpy()
        for X in XS:
            for season in HOLDOUTS:
                sel = d.season.eq(season)
                gws = sorted(d.loc[sel, "gameweek"].unique())
                per_gw = []
                for t in gws:
                    if t < MIN_GW:
                        continue
                    tmask = sel & d.gameweek.eq(t)
                    tg = int(d.loc[tmask, "gidx"].iloc[0])
                    lo = -np.inf if X == "all" else tg - int(X)
                    trm = (gidx < tg) & (gidx >= lo)
                    if trm.sum() < 500 or tmask.sum() < 50:
                        continue
                    m = xgb.XGBRegressor(
                        n_estimators=300, learning_rate=0.06, max_depth=6,
                        min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                        reg_lambda=1.0, tree_method="hist", n_jobs=4, random_state=0)
                    m.fit(Xall[trm], yall[trm])
                    te = d.loc[tmask].copy()
                    te["p"] = m.predict(Xall[tmask.to_numpy()])
                    a = te["y"].to_numpy(float)
                    per_gw.append({
                        "spearman": spearmanr(te["p"], a).statistic,
                        "top10": a[np.argsort(-te["p"].to_numpy())][:10].sum(),
                        "best_xi": best_xi_points(te, "p"),
                        "train_rows": int(trm.sum())})
                if not per_gw:
                    continue
                p = pd.DataFrame(per_gw)
                rows.append({"features": wname, "train_window": str(X), "season": season,
                             "gameweeks": len(p), "spearman": p.spearman.mean(),
                             "top10": p.top10.mean(), "best_xi": p.best_xi.mean(),
                             "mean_train_rows": p.train_rows.mean()})
                print(f"  {wname:8s} X={str(X):4s} {season}  "
                      f"gw={len(p):2d} spearman={p.spearman.mean():.4f} "
                      f"top10={p.top10.mean():6.2f} best_xi={p.best_xi.mean():6.2f} "
                      f"[{time.time()-t0:5.0f}s]", flush=True)
    res = pd.DataFrame(rows)
    res.to_pickle(f"{SCR}/sweep_results.pkl")
    print(f"\ndone in {time.time()-t0:.0f}s -> {SCR}/sweep_results.pkl")


if __name__ == "__main__":
    main()
