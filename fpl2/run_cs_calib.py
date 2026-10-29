#!/usr/bin/env python3
"""
O25: do we under-predict CLEAN SHEETS?

Clean sheets are 34% of the top defenders' points in 2026-27 (120 of 358) --
their largest single source, and the part that is NOT the ~3x xA
over-performance the model is right to ignore.

Clean sheet enters xP as a PROBABILITY (exp(-lambda) from predicted goals
conceded), weighted by P(60+). So it is directly checkable against the realised
rate, with no simulator and no noise floor: predicted 0.30 should mean 30% of
those team-gameweeks ended 0-0 at the back.

Reported per season, overall and by predicted-probability bucket, so an
under-prediction concentrated in the strong-defence tail is visible rather than
averaged away. Also per team for 2026-27, since the specific miss was cheap
defenders at teams keeping 4-5 clean sheets in 7 games.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, model, rates, team   # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor


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

    seasons = list(C.HOLDOUT_SEASONS) + ["2026-27"]
    allrows = []
    for s in seasons:
        out = model.walk_forward(feats, tf, names, s)
        if "clean_sheet_prob" not in out.columns:
            print("  clean_sheet_prob not exposed by predict(); columns:",
                  [c for c in out.columns][:25], flush=True)
            return
        # PER PLAYER, among those who played 60+. That is what the model's
        # clean_sheet_prob means: it enters expected_points weighted by
        # p_sixty. An earlier version aggregated max(clean_sheets) per TEAM,
        # which counts players subbed off before a late goal -- FPL awards them
        # a clean sheet even though the team conceded -- and inflated the
        # realised rate from ~0.24 to ~0.39 against a prediction that means
        # something narrower.
        t = out[(out.fixtures > 0) & (out.minutes >= 60)].copy()
        tg = t[["season", "gameweek", "team", "clean_sheet_prob", "clean_sheets"]].rename(
            columns={"clean_sheet_prob": "p"})
        tg["real"] = (pd.to_numeric(tg["clean_sheets"], errors="coerce") > 0).astype(float)
        allrows.append(tg)
        print(f"[{time.time()-t0:4.0f}s] {s}: predicted {tg.p.mean():.3f}  "
              f"realised {tg.real.mean():.3f}  ratio {tg.p.mean()/max(tg.real.mean(),1e-9):.3f}"
              f"  n={len(tg)}", flush=True)

    d = pd.concat(allrows, ignore_index=True)
    print("\nBY PREDICTED-PROBABILITY BUCKET  (pooled)", flush=True)
    d["bucket"] = pd.cut(d.p, [0, .15, .25, .35, .45, 1.0])
    g = d.groupby("bucket", observed=True).agg(
        n=("p", "size"), predicted=("p", "mean"), realised=("real", "mean"))
    g["ratio"] = g.predicted / g.realised.replace(0, np.nan)
    print(g.round(3).to_string(), flush=True)

    print("\n2026-27 BY TEAM  (sorted by realised clean-sheet rate)", flush=True)
    cs = d[d.season.eq("2026-27")].groupby("team").agg(
        n=("p", "size"), predicted=("p", "mean"), realised=("real", "mean"))
    cs["gap"] = cs.realised - cs.predicted
    print(cs.sort_values("realised", ascending=False).head(10).round(3).to_string(),
          flush=True)
    print("\n  gap > 0 means the team kept MORE clean sheets than predicted.", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
