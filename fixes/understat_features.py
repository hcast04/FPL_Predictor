#!/usr/bin/env python3
"""
Does Understat team context (PPDA, non-penalty xG, deep completions) improve
the forecast?

The current feature set has NO team-level pressing, possession or
shot-creation signal -- 336 fold-cache features and not one of them is PPDA or
equivalent. The notebook pulls Understat/FBref only for player-level priors on
incoming transfers. This is the gap Adham's model covers and ours does not.

Builds per-team trailing form from Understat match data, attaches both the
player's own team and his opponent, and tests whether adding it to an
otherwise identical XGBoost improves BEST LEGAL XI points.

Protocol is the same strict walk-forward as training_window_sweep.py, with the
best configuration from that sweep (train on all history, 3/5/10 windows) held
fixed so the ONLY change is the Understat block.

A result counts only if it holds in all three holdout seasons.

Run:  .venv/bin/python fixes/understat_features.py
"""
from __future__ import annotations
import pickle
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import xgboost as xgb

SCR = ("/tmp/claude-1000/-home-mdmsystems-acer-Escritorio-pers-fpl/"
       "a909bbf0-dfbe-4cc0-a5f8-534424e1d2b9/scratchpad")
ALIAS = {"Manchester City": "Man City", "Manchester United": "Man Utd",
         "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest",
         "Sheffield United": "Sheffield Utd", "Tottenham": "Spurs",
         "Wolverhampton Wanderers": "Wolves"}
SEASON_MAP = {"2223": "2022-23", "2324": "2023-24", "2425": "2024-25", "2526": "2025-26"}
HOLDOUTS = ["2023-24", "2024-25", "2025-26"]
MIN_GW = 6
BOUNDS = {"DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
US_STATS = ["xg", "np_xg", "ppda", "deep_completions", "goals", "expected_points"]


def build_understat_long() -> pd.DataFrame:
    t = pd.read_pickle(f"{SCR}/understat_team.pkl")
    t["date"] = pd.to_datetime(t["date"]).dt.date
    t["season_fpl"] = t["season"].map(SEASON_MAP)
    out = []
    for side, opp in (("home", "away"), ("away", "home")):
        d = pd.DataFrame({
            "season": t["season_fpl"], "date": t["date"],
            "team": t[f"{side}_team"].replace(ALIAS),
            "opp": t[f"{opp}_team"].replace(ALIAS),
            "is_home": 1.0 if side == "home" else 0.0})
        for s in US_STATS:
            d[f"us_{s}"] = pd.to_numeric(t[f"{side}_{s}"], errors="coerce")
            d[f"us_{s}_against"] = pd.to_numeric(t[f"{opp}_{s}"], errors="coerce")
        out.append(d)
    return pd.concat(out, ignore_index=True).sort_values(["team", "season", "date"])


def rolling_team_form(long: pd.DataFrame, w: int = 5) -> pd.DataFrame:
    g = long.groupby("team", sort=False)
    cols = []
    for s in US_STATS:
        for suf in ("", "_against"):
            src = f"us_{s}{suf}"
            dst = f"tf_{s}{suf}_r{w}"
            long[dst] = g[src].transform(lambda x: x.shift().rolling(w, min_periods=1).mean())
            cols.append(dst)
    return long, cols


def best_xi_points(g: pd.DataFrame, pred: str) -> float:
    by = {p: g[g.position.eq(p)].nlargest(6, pred) for p in ("GK", "DEF", "MID", "FWD")}
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


def main() -> None:
    feats = pd.read_pickle(f"{SCR}/features.pkl")
    spec = pickle.load(open(f"{SCR}/featspec.pkl", "rb"))
    base_feats = spec["base"] + spec["win"][3] + spec["win"][5] + spec["win"][10]

    raw = pd.read_csv("data/historical_fpl.csv.gz", compression="gzip", low_memory=False)
    raw["date"] = pd.to_datetime(raw["kickoff_time"], errors="coerce", utc=True).dt.date
    key = raw[["season", "gameweek", "player_code", "team_name", "opponent_name", "date"]].copy()
    feats = feats.merge(key, on=["season", "gameweek", "player_code"], how="left")

    long, tf_cols = rolling_team_form(build_understat_long())
    own = long[["season", "date", "team"] + tf_cols].rename(
        columns={"team": "team_name", **{c: c + "_own" for c in tf_cols}})
    opp = long[["season", "date", "team"] + tf_cols].rename(
        columns={"team": "opponent_name", **{c: c + "_opp" for c in tf_cols}})
    feats = feats.merge(own, on=["season", "date", "team_name"], how="left")
    feats = feats.merge(opp, on=["season", "date", "opponent_name"], how="left")
    us_feats = [c + "_own" for c in tf_cols] + [c + "_opp" for c in tf_cols]

    cov = feats[us_feats].notna().all(axis=1).mean()
    print(f"Understat join coverage: {100*cov:.1f}% of player-gameweeks")
    print(f"base features {len(base_feats)}  +  understat {len(us_feats)}\n")

    feats = feats[feats.y.notna()].copy()
    rows = []
    for label, fset in (("base", base_feats), ("base+understat", base_feats + us_feats)):
        X = feats[fset].to_numpy("float32")
        y = feats["y"].to_numpy("float32")
        gidx = feats["gidx"].to_numpy()
        for season in HOLDOUTS:
            sel = feats.season.eq(season)
            per = []
            for t in sorted(feats.loc[sel, "gameweek"].unique()):
                if t < MIN_GW:
                    continue
                tm = sel & feats.gameweek.eq(t)
                tg = int(feats.loc[tm, "gidx"].iloc[0])
                tr = gidx < tg
                if tr.sum() < 500 or tm.sum() < 50:
                    continue
                m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.06, max_depth=6,
                                     min_child_weight=20, subsample=0.8,
                                     colsample_bytree=0.8, reg_lambda=1.0,
                                     tree_method="hist", n_jobs=6, random_state=0)
                m.fit(X[tr], y[tr])
                te = feats.loc[tm].copy()
                te["p"] = m.predict(X[tm.to_numpy()])
                a = te["y"].to_numpy(float)
                per.append({"spearman": spearmanr(te["p"], a).statistic,
                            "top10": a[np.argsort(-te["p"].to_numpy())][:10].sum(),
                            "best_xi": best_xi_points(te, "p")})
            p = pd.DataFrame(per)
            rows.append({"variant": label, "season": season, "gw": len(p),
                         "spearman": p.spearman.mean(), "top10": p.top10.mean(),
                         "best_xi": p.best_xi.mean()})
            print(f"  {label:16s} {season}  gw={len(p):2d} "
                  f"spearman={p.spearman.mean():.4f} top10={p.top10.mean():6.2f} "
                  f"best_xi={p.best_xi.mean():6.2f}", flush=True)
    r = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print("DOES UNDERSTAT TEAM CONTEXT HELP?")
    print("=" * 72)
    for metric in ("best_xi", "top10", "spearman"):
        piv = r.pivot_table(index="variant", columns="season", values=metric)
        d = piv.loc["base+understat"] - piv.loc["base"]
        wins = int((d > 0).sum())
        print(f"\n  {metric}: wins in {wins}/3 seasons")
        print("    " + "  ".join(f"{s}: {d[s]:+.4f}" for s in d.index))
    print("\n  A result counts only if it wins in all three seasons.")


if __name__ == "__main__":
    main()
