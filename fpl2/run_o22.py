#!/usr/bin/env python3
"""
O22: does the GBDT fail to identify ELITE players, or is it a tie-break artefact?

Measured in S13: the GBDT captains the season top scorer 15/38 (2024-25) and
3/38 (2025-26); the GLM manages 28 and 23, legacy 25 and 22. Two independently
built signals agree and ours is the outlier.

But S13 only looks at the ARGMAX. Two very different defects produce it:

  systematic  the GBDT genuinely ranks elite players lower -> a real defect
  tie-break   it ranks them fine (2nd-3rd) and loses a near-tie at the top
              -> a much narrower problem, fixable without touching the model

So: for each gameweek, the RANK each model assigns to players who finished the
season in the top 10. Lower is better; 1 = this model's top pick that week.

CONTROL, because "ranks everyone worse" would be a different story entirely and
is contradicted by near-identical overall spearman: the same measurement for
players who finished 11-50 and 51-150. The defect must be SPECIFIC to the top.

NOTE: "finished top 10" is hindsight. That is legitimate here -- the question is
whether the model saw the players who turned out to be elite -- but it means
these numbers diagnose, they do not forecast.
"""
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, legacy   # noqa: E402

TIERS = [("top-10", 0, 10), ("11-50", 10, 50), ("51-150", 50, 150)]


def main():
    arms = pickle.load(open(C.cache("fpl2_o19_arms.pkl"), "rb"))
    panel = data.build_panel()
    leg = legacy.load_fold_forecasts().rename(columns={"xp": "xp_leg"})

    print("MEAN WITHIN-GAMEWEEK RANK BY REALISED SEASON TIER  (1 = model's top pick)\n")
    print(f"{'season':10s}{'tier':9s}{'n':>5}{'gbdt':>8}{'glm':>8}{'legacy':>9}"
          f"{'gbdt-glm':>10}", flush=True)
    agg = {}
    for s in C.HOLDOUT_SEASONS:
        p = panel[panel.season.eq(s)]
        order = p.groupby(C.PLAYER_KEY).points.sum().sort_values(ascending=False)
        frames = {}
        for arm in ("gbdt", "glm"):
            t = arms[arm][s]; t = t[t.fixtures > 0][["gameweek", C.PLAYER_KEY, "xp"]]
            t["rank"] = t.groupby("gameweek")["xp"].rank(ascending=False, method="min")
            frames[arm] = t.set_index(["gameweek", C.PLAYER_KEY])["rank"]
        l = leg[leg.season.eq(s)].copy()
        l["rank"] = l.groupby("gameweek")["xp_leg"].rank(ascending=False, method="min")
        frames["legacy"] = l.set_index(["gameweek", C.PLAYER_KEY])["rank"]

        for name, lo, hi in TIERS:
            members = set(order.index[lo:hi])
            row = {}
            for arm, ser in frames.items():
                idx = [i for i in ser.index if i[1] in members]
                row[arm] = float(ser.loc[idx].mean()) if idx else np.nan
            agg.setdefault(name, []).append(row)
            print(f"{s:10s}{name:9s}{len(members):>5}{row['gbdt']:>8.1f}"
                  f"{row['glm']:>8.1f}{row['legacy']:>9.1f}"
                  f"{row['gbdt']-row['glm']:>+10.1f}", flush=True)
        print(flush=True)

    print("MEAN OVER SEASONS", flush=True)
    print(f"{'tier':9s}{'gbdt':>8}{'glm':>8}{'legacy':>9}{'gbdt-glm':>10}", flush=True)
    for name, rows in agg.items():
        g = np.mean([r["gbdt"] for r in rows]); m = np.mean([r["glm"] for r in rows])
        l = np.mean([r["legacy"] for r in rows])
        print(f"{name:9s}{g:>8.1f}{m:>8.1f}{l:>9.1f}{g-m:>+10.1f}", flush=True)

    print("\nHOW OFTEN IS AN ELITE PLAYER IN THE MODEL'S TOP 3 THAT WEEK?", flush=True)
    print(f"{'season':10s}{'gbdt':>9}{'glm':>9}{'legacy':>9}", flush=True)
    for s in C.HOLDOUT_SEASONS:
        p = panel[panel.season.eq(s)]
        order = p.groupby(C.PLAYER_KEY).points.sum().sort_values(ascending=False)
        members = set(order.index[:10])
        out = {}
        for arm in ("gbdt", "glm"):
            t = arms[arm][s]; t = t[t.fixtures > 0]
            top3 = t.loc[t.groupby("gameweek")["xp"].nlargest(3)
                          .index.get_level_values(1)]
            out[arm] = float(top3[C.PLAYER_KEY].isin(members).mean())
        l = leg[leg.season.eq(s)]
        t3 = l.loc[l.groupby("gameweek")["xp_leg"].nlargest(3).index.get_level_values(1)]
        out["legacy"] = float(t3[C.PLAYER_KEY].isin(members).mean())
        print(f"{s:10s}{out['gbdt']:>9.1%}{out['glm']:>9.1%}{out['legacy']:>9.1%}",
              flush=True)
    print("\n  If gbdt is worse ONLY in top-10, the defect is elite identification.", flush=True)
    print("  If it is worse in every tier, it is just poorer ranking overall --", flush=True)
    print("  which the near-identical overall spearman (0.67-0.71) contradicts.", flush=True)


if __name__ == "__main__":
    main()
