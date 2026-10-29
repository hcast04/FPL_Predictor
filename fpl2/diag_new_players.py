#!/usr/bin/env python3
"""Is the model blind to players new to the Premier League?"""
import sys, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, model, team   # noqa: E402

CACHE = Path("/tmp/fpl2_xp_2627.pkl")

def main():
    t0 = time.time()
    hist = data.load_raw(); cur = pd.read_pickle("/tmp/fpl2_live2627.pkl")
    raw = pd.concat([hist, cur], ignore_index=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    if CACHE.exists():
        out = pd.read_pickle(CACHE)
        print(f"[{time.time()-t0:4.0f}s] loaded cached 2026-27 forecast", flush=True)
    else:
        feats, names = features.build(panel, tf)
        out = model.walk_forward(feats, tf, names, "2026-27")
        out.to_pickle(CACHE)
        print(f"[{time.time()-t0:4.0f}s] forecast computed and cached", flush=True)

    prior = (panel[panel.season < "2026-27"].groupby("player_code")
             .agg(prior_minutes=("minutes", "sum")).reset_index())
    t = out[out.fixtures > 0].merge(prior, on="player_code", how="left")
    t["prior_minutes"] = t.prior_minutes.fillna(0)
    t["grp"] = np.where(t.prior_minutes <= 0, "NEW to PL",
               np.where(t.prior_minutes < 900, "<900 prior min", "established"))

    print("\nMODEL ACCURACY BY PRIOR PREMIER LEAGUE HISTORY (2026-27 GW1-6)\n")
    g = t.groupby("grp").apply(lambda x: pd.Series({
        "rows": len(x), "players": x.player_code.nunique(),
        "xp_mean": x.xp.mean(), "actual_mean": x.points.mean(),
        "ratio": x.xp.sum() / max(x.points.sum(), 1),
        "spearman": x.xp.corr(x.points, method="spearman")}), include_groups=False)
    print(g.round(3).to_string())

    tot = t.points.sum()
    print("\n  share of all points scored, by group:")
    for k, v in t.groupby("grp").points.sum().items():
        print(f"    {k:16s} {v:7.0f}  ({100*v/tot:4.1f}%)")

    nm = raw.drop_duplicates("player_code").set_index("player_code").player_name
    nw = (t[t.grp == "NEW to PL"].groupby("player_code")
          .agg(pts=("points", "sum"), xp=("xp", "sum"),
               pos=("position", "first")).reset_index())
    nw["name"] = nw.player_code.map(nm)
    print("\n  TOP SCORERS NEW TO THE PL — and what the model predicted:")
    print(nw.nlargest(12, "pts")[["name", "pos", "pts", "xp"]].round(2).to_string(index=False))
    print(f"\n[{time.time()-t0:4.0f}s] done", flush=True)

if __name__ == "__main__":
    main()
