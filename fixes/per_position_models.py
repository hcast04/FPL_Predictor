#!/usr/bin/env python3
"""
Per-position models -- the structural version of "per-position fixture rating".

The base model is POOLED: one regressor for all four positions, with position
as a feature. Adham fits a SEPARATE model per position. That is not the same
thing as adding a per-position fixture-difficulty feature:

  * Adding ATT_FDR / DEF_FDR as features gives a gradient-boosted tree
    information it can already derive -- opponent attacking and defensive form
    are already in the base set, and trees learn interactions natively. That is
    why the Understat team-context block failed.
  * Fitting per position changes the MODEL, not the features. Each position
    gets its own tree structure, its own splits on fixture difficulty, and its
    own loss surface. A clean sheet matters to a defender and is noise to a
    forward; a pooled model has to compromise.

This tests the second thing, which has not been tested.

Same strict walk-forward and the same best configuration from the sweep, so
the only change is pooled vs per-position.

Run:  .venv/bin/python fixes/per_position_models.py
"""
from __future__ import annotations
import pickle
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import xgboost as xgb

SCR = ("/tmp/claude-1000/-home-mdmsystems-acer-Escritorio-pers-fpl/"
       "a909bbf0-dfbe-4cc0-a5f8-534424e1d2b9/scratchpad")
HOLDOUTS = ["2023-24", "2024-25", "2025-26"]
MIN_GW = 6
POS = ["GK", "DEF", "MID", "FWD"]


def mk(seed=0, n=300):
    return xgb.XGBRegressor(n_estimators=n, learning_rate=0.06, max_depth=6,
                            min_child_weight=20, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=1.0,
                            tree_method="hist", n_jobs=6, random_state=seed)


def best_xi_points(g, pred):
    by = {p: g[g.position.eq(p)].nlargest(6, pred) for p in POS}
    if by["GK"].empty:
        return np.nan
    gk = by["GK"].iloc[0]
    best, bs = None, -np.inf
    for dd in range(3, 6):
        for mm in range(2, 6):
            ff = 10 - dd - mm
            if not (1 <= ff <= 3):
                continue
            if len(by["DEF"]) < dd or len(by["MID"]) < mm or len(by["FWD"]) < ff:
                continue
            xi = pd.concat([by["DEF"].head(dd), by["MID"].head(mm), by["FWD"].head(ff)])
            s = float(xi[pred].sum()) + float(gk[pred])
            if s > bs:
                bs, best = s, xi
    if best is None:
        return np.nan
    sq = pd.concat([best, gk.to_frame().T])
    cap = sq.loc[sq[pred].astype(float).idxmax()]
    return float(sq["y"].sum()) + float(cap["y"])


def main():
    d = pd.read_pickle(f"{SCR}/features.pkl")
    spec = pickle.load(open(f"{SCR}/featspec.pkl", "rb"))
    feats = spec["base"] + spec["win"][3] + spec["win"][5] + spec["win"][10]
    d = d[d.y.notna() & d.position.isin(POS)].copy()
    X = d[feats].to_numpy("float32"); y = d["y"].to_numpy("float32")
    gidx = d["gidx"].to_numpy(); pos = d["position"].to_numpy()

    rows = []
    for variant in ("pooled", "per_position"):
        for season in HOLDOUTS:
            sel = d.season.eq(season)
            per = []
            for t in sorted(d.loc[sel, "gameweek"].unique()):
                if t < MIN_GW:
                    continue
                tm = (sel & d.gameweek.eq(t)).to_numpy()
                tg = int(d.loc[tm, "gidx"].iloc[0])
                tr = gidx < tg
                if tr.sum() < 500 or tm.sum() < 50:
                    continue
                te = d.loc[tm].copy()
                if variant == "pooled":
                    m = mk(); m.fit(X[tr], y[tr])
                    te["p"] = m.predict(X[tm])
                else:
                    pred = np.zeros(tm.sum())
                    for p in POS:
                        trp = tr & (pos == p)
                        tep = (pos[tm] == p)
                        if trp.sum() < 200 or tep.sum() == 0:
                            # too little history for this position: fall back
                            mm_ = mk(); mm_.fit(X[tr], y[tr])
                            pred[tep] = mm_.predict(X[tm][tep])
                            continue
                        mp = mk(); mp.fit(X[trp], y[trp])
                        pred[tep] = mp.predict(X[tm][tep])
                    te["p"] = pred
                a = te["y"].to_numpy(float)
                per.append({"spearman": spearmanr(te["p"], a).statistic,
                            "top10": a[np.argsort(-te["p"].to_numpy())][:10].sum(),
                            "best_xi": best_xi_points(te, "p")})
            pf = pd.DataFrame(per)
            rows.append({"variant": variant, "season": season, "gw": len(pf),
                         "spearman": pf.spearman.mean(), "top10": pf.top10.mean(),
                         "best_xi": pf.best_xi.mean()})
            print(f"  {variant:13s} {season}  gw={len(pf):2d} "
                  f"spearman={pf.spearman.mean():.4f} top10={pf.top10.mean():6.2f} "
                  f"best_xi={pf.best_xi.mean():6.2f}", flush=True)
    r = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("POOLED vs PER-POSITION MODELS")
    print("=" * 70)
    for metric in ("best_xi", "top10", "spearman"):
        piv = r.pivot_table(index="variant", columns="season", values=metric)
        delta = piv.loc["per_position"] - piv.loc["pooled"]
        print(f"\n  {metric}: per-position wins in {int((delta>0).sum())}/3 seasons")
        print("    " + "  ".join(f"{s}: {delta[s]:+.4f}" for s in delta.index))
    print("\n  Counts only if it wins in all three seasons.")


if __name__ == "__main__":
    main()
