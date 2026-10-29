#!/usr/bin/env python3
"""Why does the model rank Bruno (11.8m, 33 pts) above Haaland (15.7m, 58 pts)?"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, model, rates, scoring, team  # noqa: E402
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
    cur = pd.read_pickle(C.cache("fpl2_live2627.pkl"))
    raw = pd.concat([data.load_raw(), cur], ignore_index=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    out = model.walk_forward(feats, tf, names, "2026-27")

    nm = panel[panel.season.eq("2026-27")].groupby(C.PLAYER_KEY).player_name.last()
    out["name"] = out[C.PLAYER_KEY].map(nm)
    who = out[out.name.astype(str).str.contains("Bruno Borges|Haaland", na=False)]
    who = who[who.fixtures > 0]
    P = rates.RateModel.PREFIX

    cols = ["p_appear", "p_sixty", "expected_minutes", "clean_sheet_prob",
            P + "goals_per90", P + "assists_per90", P + "bonus_per90",
            P + "defensive_return_prob"]
    g = who.groupby("name")
    print("MEAN COMPONENTS OVER 2026-27 GW1-7\n")
    tab = g[cols + ["xp", "points"]].mean().round(3)
    tab["pos"] = g["position"].first()
    print(tab.T.to_string(), flush=True)

    print("\n\nxP DECOMPOSED INTO FPL POINTS  (mean per gameweek)\n", flush=True)
    rows = {}
    for name, d in g:
        pos = d["position"].iloc[0]
        share = d["expected_minutes"].to_numpy(float) / 90.0
        goal_v = scoring.GOAL.get(pos, 0); cs_v = scoring.CLEAN_SHEET.get(pos, 0)
        parts = {
            "appearance": (scoring.APPEARANCE_SHORT
                           * np.clip(d.p_appear - d.p_sixty, 0, None)
                           + scoring.APPEARANCE_LONG * d.p_sixty).mean(),
            f"goals (x{goal_v})": (d[P + "goals_per90"] * share * goal_v).mean(),
            "assists (x3)": (d[P + "assists_per90"] * share * scoring.ASSIST).mean(),
            f"clean sheet (x{cs_v})": (d.clean_sheet_prob * d.p_sixty * cs_v).mean(),
            "bonus": (d[P + "bonus_per90"] * share).mean(),
            "defensive": (d[P + "defensive_return_prob"]
                          * scoring.DEFENSIVE_POINTS).mean(),
        }
        parts["TOTAL xP"] = d.xp.mean()
        parts["realised"] = d.points.mean()
        rows[f"{name[:18]} ({pos})"] = parts
    print(pd.DataFrame(rows).round(3).to_string(), flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
