#!/usr/bin/env python3
"""
G9 follow-up: is the manual-prior effect real but diluted?

The headline ratio covers ALL new-to-PL players, but only the ~30% with a
Championship match can possibly be affected. This splits the two groups so a
real gain on the matched players cannot hide behind the unmatched ones, and
saves both arms' forecasts so the split can be re-cut without refitting.
"""
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, external, features, model, team   # noqa: E402

SEASONS = list(C.HOLDOUT_SEASONS) + ["2026-27"]
TARGET = "2026-27"
OUT = Path("/tmp/fpl2_g9_arms.pkl")


def main():
    t0 = time.time()
    raw = pd.concat([data.load_raw(), pd.read_pickle("/tmp/fpl2_live2627.pkl")],
                    ignore_index=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    base_ext = pd.read_pickle("/tmp/fpl2_ext_understat.pkl")
    feats, names = features.build(panel, tf)
    print(f"[{time.time()-t0:4.0f}s] setup done", flush=True)

    arms, matched_codes = {}, None
    for arm in ("understat_only", "understat+manual"):
        ext = base_ext.copy()
        if arm == "understat+manual":
            parts = []
            for s in SEASONS:
                e = ext[ext.season.eq(s)].copy()
                if not e.empty:
                    parts.append(external.with_manual(e, panel, s))
            ext = pd.concat(parts, ignore_index=True)
            matched_codes = set(ext.loc[ext.ext_source.eq("manual"), C.PLAYER_KEY])

        ext_cols = [c for c in ext.columns if c.startswith("ext_")
                    and c not in ("ext_match_method", "ext_source", "ext_name",
                                  "ext_league", "ext_squads")]
        f = feats.merge(ext[[C.PLAYER_KEY, "season"] + ext_cols],
                        on=[C.PLAYER_KEY, "season"], how="left")
        f["ext_available"] = f["ext_available"].fillna(0.0)
        arms[arm] = model.walk_forward(f, tf, names + ext_cols, TARGET)
        print(f"[{time.time()-t0:4.0f}s] {arm} done", flush=True)

    pickle.dump({"arms": arms, "matched": matched_codes}, open(OUT, "wb"))

    pm = panel[panel.season < TARGET].groupby("player_code").minutes.sum()
    print(f"\n{'group':22s}{'n':>5}{'pts':>7}{'ratio_before':>14}{'ratio_after':>13}"
          f"{'sp_before':>11}{'sp_after':>10}", flush=True)
    for label, sel in (("matched (Champ)", lambda c: c in matched_codes),
                       ("unmatched", lambda c: c not in matched_codes)):
        rows = {}
        for arm, out in arms.items():
            t = out[out.fixtures > 0].copy()
            t["pm"] = t.player_code.map(pm).fillna(0)
            w = t[(t.gameweek <= 6) & (t.pm <= 0)]
            w = w[w.player_code.map(sel)]
            rows[arm] = w
        a, b = rows["understat_only"], rows["understat+manual"]
        print(f"{label:22s}{b.player_code.nunique():>5}{b.points.sum():>7.0f}"
              f"{a.xp.sum()/max(a.points.sum(),1):>14.4f}"
              f"{b.xp.sum()/max(b.points.sum(),1):>13.4f}"
              f"{a.xp.corr(a.points, method='spearman'):>11.4f}"
              f"{b.xp.corr(b.points, method='spearman'):>10.4f}", flush=True)

    # exact overall spearman, to see what the 3dp display was hiding
    for lbl, out in arms.items():
        t = out[out.fixtures > 0].copy()
        t["pm"] = t.player_code.map(pm).fillna(0)
        w = t[(t.gameweek <= 6) & (t.pm <= 0)]
        print(f"\n  {lbl:20s} overall spearman = {w.xp.corr(w.points, method='spearman'):.6f}",
              flush=True)
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
