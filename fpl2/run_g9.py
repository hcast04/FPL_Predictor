#!/usr/bin/env python3
"""
G9: do hand-exported Championship priors fix the new-to-PL blind spot?

Gate, declared before the run (2026-27, the live season):
  * new-to-PL calibration ratio must move from 0.847 TOWARD 1.000
  * new-to-PL spearman must not fall below 0.527
  * unmatched players must stay unmatched -- reported, not imputed

Two arms over identical features and folds; only the prior layer differs.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, external, features, model, team   # noqa: E402

SEASONS = list(C.HOLDOUT_SEASONS) + ["2026-27"]
UNDERSTAT_CACHE = Path("/tmp/fpl2_ext_understat.pkl")
TARGET = "2026-27"


def understat_frame(panel):
    if UNDERSTAT_CACHE.exists():
        print("    (understat priors from cache)", flush=True)
        return pd.read_pickle(UNDERSTAT_CACHE)
    frames = []
    for s in SEASONS:
        e = external.build_for_season(panel, s, include_same_season=True)
        if not e.empty:
            frames.append(e)
            print(f"    {s}: {int(e.ext_available.sum())}/{len(e)} matched", flush=True)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_pickle(UNDERSTAT_CACHE)
    return out


def score(out, panel, season, label, through):
    t = out[out.fixtures > 0].copy()
    pm = panel[panel.season < season].groupby("player_code").minutes.sum()
    t["pm"] = t.player_code.map(pm).fillna(0)
    w = t[(t.gameweek <= through) & (t.pm <= 0)]
    ratio = w.xp.sum() / max(w.points.sum(), 1)
    sp = w.xp.corr(w.points, method="spearman")
    print(f"  {label:26s} GW1-{through}  ratio={ratio:.3f}  "
          f"spearman={sp:.3f}  rows={len(w)}  pts={w.points.sum():.0f}", flush=True)
    return ratio, sp


def main():
    t0 = time.time()
    raw = pd.concat([data.load_raw(), pd.read_pickle("/tmp/fpl2_live2627.pkl")],
                    ignore_index=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    print(f"[{time.time()-t0:4.0f}s] panel built", flush=True)

    base_ext = understat_frame(panel)
    print(f"[{time.time()-t0:4.0f}s] understat priors ready", flush=True)

    feats, names = features.build(panel, tf)
    print(f"[{time.time()-t0:4.0f}s] features built ({len(names)})", flush=True)

    results = {}
    for arm in ("understat_only", "understat+manual"):
        ext = base_ext.copy()
        if arm == "understat+manual":
            parts = []
            for s in SEASONS:
                e = ext[ext.season.eq(s)].copy()
                if not e.empty:
                    before = int(e.ext_available.fillna(0).sum())
                    e = external.with_manual(e, panel, s)
                    after = int(e.ext_available.fillna(0).sum())
                    if after != before:
                        print(f"    {s}: manual filled {after-before} players "
                              f"({before} -> {after})", flush=True)
                    parts.append(e)
            ext = pd.concat(parts, ignore_index=True)

        ext_cols = [c for c in ext.columns if c.startswith("ext_")
                    and c not in ("ext_match_method", "ext_source", "ext_name",
                                  "ext_league", "ext_squads")]
        f = feats.merge(ext[[C.PLAYER_KEY, "season"] + ext_cols],
                        on=[C.PLAYER_KEY, "season"], how="left")
        f["ext_available"] = f["ext_available"].fillna(0.0)
        n = names + ext_cols
        print(f"[{time.time()-t0:4.0f}s] ARM {arm}: {len(ext_cols)} ext features", flush=True)

        out = model.walk_forward(f, tf, n, TARGET)
        results[arm] = (score(out, panel, TARGET, arm, 6),
                        score(out, panel, TARGET, arm, 7))
        print(f"[{time.time()-t0:4.0f}s] arm done\n", flush=True)

    print("\nGATE (2026-27 new-to-PL, GW1-6):", flush=True)
    (r0, s0), _ = results["understat_only"]
    (r1, s1), _ = results["understat+manual"]
    print(f"  ratio    {r0:.3f} -> {r1:.3f}   "
          f"{'PASS' if abs(1-r1) < abs(1-r0) else 'FAIL'} (must move toward 1.000)", flush=True)
    print(f"  spearman {s0:.3f} -> {s1:.3f}   "
          f"{'PASS' if s1 >= s0 - 1e-9 else 'FAIL'} (must not fall)", flush=True)
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
