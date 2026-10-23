#!/usr/bin/env python3
"""
Two selection-layer experiments, run on the notebook's OWN cached out-of-sample
forecasts so any result transfers directly to production.

  E1  FIXTURE CONCENTRATION CAP
      Three GW6 starters were in one match (COV 4-1 NEW) and three GW7 starters
      in another (NEW 0-1 AVL). Both blanked. Does capping the number of XI
      players drawn from a single fixture improve realised points -- or at least
      cut the downside?

  E2  CAPTAIN SELECTION RULE
      The captain is currently argmax(expected_points). Because the armband
      doubles, ceiling matters more than mean. Does picking on a
      variance-adjusted score beat argmax?

Both report the mean AND the 10th percentile, because a variance-reducing
change can be right even when the mean is flat.

Run:  .venv/bin/python fixes/selection_experiments.py
"""
from __future__ import annotations
import glob
import numpy as np, pandas as pd

CACHE = "data/walk_forward_fold_cache_base_v2_4_0"
BOUNDS = {"DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}


def pick_xi(g: pd.DataFrame, score: str, fixture_cap: int | None):
    """Best legal XI by `score`, optionally capped at N players per fixture."""
    best, best_s = None, -np.inf
    for dd in range(3, 6):
        for mm in range(2, 6):
            ff = 10 - dd - mm
            if not (1 <= ff <= 3):
                continue
            need = {"GK": 1, "DEF": dd, "MID": mm, "FWD": ff}
            chosen, used = [], {}
            ok = True
            for pos, k in need.items():
                pool = g[g.model_position.eq(pos)].sort_values(score, ascending=False)
                took = 0
                for _, r in pool.iterrows():
                    if took >= k:
                        break
                    fx = r.get("fixture_id")
                    if fixture_cap is not None and used.get(fx, 0) >= fixture_cap:
                        continue
                    chosen.append(r); used[fx] = used.get(fx, 0) + 1; took += 1
                if took < k:
                    ok = False; break
            if not ok:
                continue
            xi = pd.DataFrame(chosen)
            s = float(xi[score].sum())
            if s > best_s:
                best_s, best = s, xi
    return best


def main() -> None:
    frames = []
    for p in sorted(glob.glob(f"{CACHE}/*.pkl.gz")):
        pl = pd.read_pickle(p, compression="gzip")
        fr = pl["bundle"]["holdout_expected_points"].copy()
        fr["season"] = str(pl["holdout_season"])
        frames.append(fr)
    d = pd.concat(frames, ignore_index=True)
    d["xp"] = pd.to_numeric(d.expected_points, errors="coerce")
    d["y"] = pd.to_numeric(d.target_fpl_points, errors="coerce")
    d = d[d.xp.notna() & d.y.notna()]

    # upside proxy: trailing dispersion of returns, from features already present
    d["upside"] = (pd.to_numeric(d.get("threat_per90_last_5_gws"), errors="coerce").fillna(0.0)
                   / 100.0)

    print("=" * 84)
    print("E1 — FIXTURE CONCENTRATION CAP  (realised XI points, captain doubled)")
    print("=" * 84)
    rows = []
    for season, sd in d.groupby("season"):
        for cap in (None, 3, 2):
            per = []
            for gw, g in sd.groupby("gameweek"):
                if len(g) < 50:
                    continue
                xi = pick_xi(g, "xp", cap)
                if xi is None:
                    continue
                c = xi.loc[xi.xp.idxmax()]
                per.append(float(xi.y.sum()) + float(c.y))
            if per:
                rows.append({"season": season, "cap": "none" if cap is None else cap,
                             "gw": len(per), "mean": np.mean(per),
                             "p10": np.percentile(per, 10), "sd": np.std(per)})
    r1 = pd.DataFrame(rows)
    print(r1.round(2).to_string(index=False))
    print("\n  per-season delta vs uncapped:")
    for season, sd in r1.groupby("season"):
        base = sd[sd.cap.eq("none")]["mean"].iloc[0]
        bp10 = sd[sd.cap.eq("none")]["p10"].iloc[0]
        for _, r in sd[~sd.cap.eq("none")].iterrows():
            print(f"    {season}  cap={r['cap']}  mean {r['mean']-base:+6.2f}   p10 {r['p10']-bp10:+6.2f}")

    print("\n" + "=" * 84)
    print("E2 — CAPTAIN RULE  (points contributed by the armband)")
    print("=" * 84)
    rows = []
    for season, sd in d.groupby("season"):
        per = {"argmax_xp": [], "top3_by_upside": [], "xp_plus_upside": []}
        for gw, g in sd.groupby("gameweek"):
            if len(g) < 50:
                continue
            xi = pick_xi(g, "xp", None)
            if xi is None:
                continue
            per["argmax_xp"].append(float(xi.loc[xi.xp.idxmax()].y))
            t3 = xi.nlargest(3, "xp")
            per["top3_by_upside"].append(float(t3.loc[t3.upside.idxmax()].y))
            sc = xi.xp + 0.5 * xi.upside
            per["xp_plus_upside"].append(float(xi.loc[sc.idxmax()].y))
        for k, v in per.items():
            if v:
                rows.append({"season": season, "rule": k, "gw": len(v),
                             "mean": np.mean(v), "p10": np.percentile(v, 10)})
    r2 = pd.DataFrame(rows)
    print(r2.round(3).to_string(index=False))
    print("\n  delta vs argmax_xp:")
    for season, sd in r2.groupby("season"):
        base = sd[sd.rule.eq("argmax_xp")]["mean"].iloc[0]
        for _, r in sd[~sd.rule.eq("argmax_xp")].iterrows():
            print(f"    {season}  {r['rule']:16s} {r['mean']-base:+.3f} per gameweek")
    print("\n  A rule is only real if it wins in EVERY season.")


if __name__ == "__main__":
    main()
