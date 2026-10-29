#!/usr/bin/env python3
"""G8: external priors for new arrivals + mover flags for players who changed club."""
import sys, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, external, features, model, team   # noqa: E402

EXT_CACHE = Path("/tmp/fpl2_ext_all.pkl")
SEASONS = list(C.HOLDOUT_SEASONS) + ["2026-27"]


def external_frame(panel):
    if EXT_CACHE.exists():
        return pd.read_pickle(EXT_CACHE)
    frames = []
    for s in SEASONS:
        e = external.build_for_season(panel, s, include_same_season=True)
        if not e.empty:
            frames.append(e)
            print(f"    {s}: {int(e.ext_available.sum())}/{len(e)} matched", flush=True)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_pickle(EXT_CACHE)
    return out


def main():
    t0 = time.time()
    raw = pd.concat([data.load_raw(), pd.read_pickle("/tmp/fpl2_live2627.pkl")],
                    ignore_index=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    print(f"[{time.time()-t0:4.0f}s] fetching external priors...", flush=True)
    ext = external_frame(panel)

    feats, names = features.build(panel, tf)
    ext_cols = [c for c in ext.columns if c.startswith("ext_")
                and c not in ("ext_match_method", "ext_source", "ext_name",
                              "ext_league")]
    feats = feats.merge(ext[[C.PLAYER_KEY, "season"] + ext_cols],
                        on=[C.PLAYER_KEY, "season"], how="left")
    feats["ext_available"] = feats["ext_available"].fillna(0.0)
    names = names + ext_cols
    print(f"[{time.time()-t0:4.0f}s] features {len(names)} "
          f"({len(ext_cols)} external + 4 mover)", flush=True)

    prior_all = {s: panel[panel.season < s].groupby("player_code").minutes.sum()
                 for s in SEASONS}

    print(f"\nGW1-6 BY TRANSFER STATUS  (ratio<1 = under-predicted)\n", flush=True)
    print(f"{'season':9s}{'group':18s}{'ratio':>8}{'spearman':>10}", flush=True)
    res = {}
    for s in SEASONS:
        out = model.walk_forward(feats, tf, names, s)
        res[s] = out
        t = out[out.fixtures > 0].copy()
        t["pm"] = t.player_code.map(prior_all[s]).fillna(0)
        prev = (panel[panel.season < s].sort_values(["season", "gameweek"])
                .groupby("player_code").agg(pt=("team", "last")).reset_index())
        t = t.merge(prev, on="player_code", how="left")
        t["grp"] = np.where(t.pm <= 0, "new to PL",
                   np.where(t.pt.isna(), "other",
                   np.where(t.pt.eq(t.team), "stayed", "moved in PL")))
        w = t[t.gameweek <= 6]
        for grp in ("stayed", "moved in PL", "new to PL"):
            x = w[w.grp.eq(grp)]
            if len(x) < 40:
                continue
            print(f"{s:9s}{grp:18s}{x.xp.sum()/max(x.points.sum(),1):>8.3f}"
                  f"{x.xp.corr(x.points, method='spearman'):>10.3f}", flush=True)
        print(flush=True)
    pickle.dump({k: v[["season","gameweek","player_code","position","team","cost",
                       "fixtures","points","xp","xp_raw"]]
                 for k, v in res.items() if k in C.HOLDOUT_SEASONS},
                open("/tmp/fpl2_xp.pkl", "wb"))
    res["2026-27"].to_pickle("/tmp/fpl2_xp_2627.pkl")
    print(f"[{time.time()-t0:4.0f}s] done", flush=True)


if __name__ == "__main__":
    main()
