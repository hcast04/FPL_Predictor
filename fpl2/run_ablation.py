#!/usr/bin/env python3
"""
S48: what is each component of the model actually contributing?

S47 found that giving the model xG where it had NONE is worth nothing
measurable (-0.8 / -1.9 / +1.6 / -1.5 across four seasons with no Opta data).
That invites the question nobody has asked here: which components matter at
all? The architecture composes

    xP = f(P(appear), P(60+), E[minutes], goals/90, assists/90, saves/90,
           clean-sheet prob, conceded, bonus/90, yellows, defensive return)

and months have gone into tuning pieces of it without knowing which carry the
result.

METHOD. Fit ONCE per holdout season, then recompute xP with one component
replaced by a constant. The constant is the per-position mean of the REALISED
quantity over the TRAINING seasons only -- never the holdout -- so an ablation
cannot leak. The loss relative to the full model is that component's
contribution.

`minutes_only` is the inverse test: flatten everything EXCEPT the minutes
hierarchy. If that arm loses little, the model is a minutes predictor wearing
expensive decoration.

Instrument: metric 1 (MDE 13.6), paired on (squad, gameweek, season), reported
with SEASON-CLUSTERED errors over 8 holdouts (bar |t| > 2.36, R32).
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy import stats
from fpl2 import config as C, data, features, measure, model, rates, scoring, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
K = 10
PANEL = C.cache("fpl2_panel_us.pkl")
HOLDOUTS = ["2018-19", "2019-20", "2020-21", "2021-22",
            "2022-23", "2023-24", "2024-25", "2025-26"]
P = rates.RateModel.PREFIX

# component -> (columns it owns, realised column used for the training constant)
ABLATE = {
    "no_goals":    ([P + "goals_per90"],   "goals_scored"),
    "no_assists":  ([P + "assists_per90"], "assists"),
    "no_bonus":    ([P + "bonus_per90"],   "bonus"),
    "no_saves":    ([P + "saves_per90"],   "saves"),
    "no_cleansheet": (["clean_sheet_prob"], "clean_sheets"),
    "no_defret":   ([P + "defensive_return_prob"], None),
    "no_minutes":  (["expected_minutes", "p_sixty", "p_appear"], None),
}


class _WP:
    def __init__(self, est):
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()); self.est = est
    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight); return self
    def predict(self, X): return self.est.predict(self.pre.transform(X))


def cluster(d, g):
    d = np.asarray(d, float); g = np.asarray(g)
    n = len(d); db = d.mean(); G = len(np.unique(g))
    ss = sum((d[g == k] - db).sum() ** 2 for k in np.unique(g))
    se = np.sqrt((G / (G - 1)) * ss / n ** 2)
    t = db / se if se > 0 else np.nan
    crit = stats.t.ppf(0.975, G - 1)
    return db, t, 2 * (1 - stats.t.cdf(abs(t), G - 1)), db - crit * se, db + crit * se


def recompute(o: pd.DataFrame) -> np.ndarray:
    fx = pd.to_numeric(o["fixtures"], errors="coerce").fillna(0).clip(lower=1)
    v = scoring.expected_points(
        p_appear=o["p_appear"], p_sixty=o["p_sixty"],
        exp_minutes=o["expected_minutes"], position=o["position"],
        goals_per90=o[P + "goals_per90"], assists_per90=o[P + "assists_per90"],
        saves_per90=o[P + "saves_per90"], clean_sheet_prob=o["clean_sheet_prob"],
        conceded_per90=o["expected_conceded"].to_numpy(float) / fx.to_numpy(float),
        bonus_per90=o[P + "bonus_per90"], yellow_per90=o[P + "yellow_per90"],
        defensive_return_prob=o[P + "defensive_return_prob"],
    )
    return np.clip(np.nan_to_num(np.asarray(v, dtype=float), nan=0.0), 0, None)


def main():
    t0 = time.time()
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    panel = pd.read_pickle(PANEL)
    raw = data.load_raw(extended=True)
    tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)
    squads = {s: measure.neutral_squads(data.with_blanks(panel, s), k=K, seed=1)
              for s in HOLDOUTS}
    print(f"[{time.time()-t0:5.0f}s] setup, {len(names)} features", flush=True)

    # Pooled arms, plus one arm per (component, position). A component can be
    # decisive for goalkeepers and irrelevant for the other three -- saves are
    # the obvious case -- and a pooled ablation averages that away to
    # "no effect".
    arms = (["full"] + list(ABLATE) + ["minutes_only"]
            + [f"{a}@{pos}" for a in ABLATE for pos in C.POSITIONS])
    scores = {a: [] for a in arms}
    for s in HOLDOUTS:
        o = model.walk_forward(feats, tf, names, s)
        tr = panel[panel["season"] < s]
        # per-position constants from TRAINING realised values only
        const = {}
        for arm, (cols, real) in ABLATE.items():
            if real and real in tr.columns:
                per90 = (90.0 * tr.groupby("position")[real].sum()
                         / tr.groupby("position")["minutes"].sum().clip(lower=1))
                const[arm] = per90
        for arm in arms:
            w = o.copy()
            if "@" in arm:
                a2, pos = arm.split("@")
                mask = w["position"].eq(pos)
                trp = tr[tr["position"].eq(pos)]
                cols, real = ABLATE[a2]
                if a2 == "no_minutes":
                    w.loc[mask, "expected_minutes"] = float(trp["minutes"].mean())
                    w.loc[mask, "p_sixty"] = float((trp["minutes"] >= 60).mean())
                    w.loc[mask, "p_appear"] = float((trp["minutes"] > 0).mean())
                else:
                    for c in cols:
                        if c not in w.columns:
                            continue
                        if a2 == "no_cleansheet":
                            w.loc[mask, c] = float(trp["clean_sheets"].mean())
                        elif a2 == "no_defret":
                            w.loc[mask, c] = float(w.loc[mask, c].mean())
                        else:
                            w.loc[mask, c] = float(const.get(a2, {}).get(pos, 0.0))
            elif arm in ABLATE:
                cols, real = ABLATE[arm]
                for c in cols:
                    if c not in w.columns: continue
                    if arm == "no_cleansheet":
                        w[c] = w["position"].map(
                            tr.groupby("position")["clean_sheets"].mean()).fillna(0.0)
                    elif arm == "no_minutes":
                        w["expected_minutes"] = w["position"].map(
                            tr.groupby("position")["minutes"].mean()).fillna(0.0)
                        w["p_sixty"] = float((tr["minutes"] >= 60).mean())
                        w["p_appear"] = float((tr["minutes"] > 0).mean())
                    elif arm == "no_defret":
                        w[c] = float(w[c].mean())
                    else:
                        w[c] = w["position"].map(const.get(arm, {})).fillna(0.0)
            elif arm == "minutes_only":
                for a2, (cols, real) in ABLATE.items():
                    if a2 == "no_minutes": continue
                    for c in cols:
                        if c not in w.columns: continue
                        if a2 == "no_cleansheet":
                            w[c] = w["position"].map(
                                tr.groupby("position")["clean_sheets"].mean()).fillna(0.0)
                        elif a2 == "no_defret":
                            w[c] = float(w[c].mean())
                        else:
                            w[c] = w["position"].map(const.get(a2, {})).fillna(0.0)
            w["xp"] = recompute(w)
            g = data.with_blanks(panel, s).merge(
                w[["season", "gameweek", C.PLAYER_KEY, "xp"]],
                on=["season", "gameweek", C.PLAYER_KEY], how="left")
            g["xp"] = g["xp"].fillna(0.0); g.loc[g["fixtures"] <= 0, "xp"] = 0.0
            d = measure.fixed_squad_score(g, squads[s], "xp"); d["season"] = s
            scores[arm].append(d)
        print(f"[{time.time()-t0:5.0f}s] {s} done", flush=True)

    for a in arms: scores[a] = pd.concat(scores[a])
    ON = ["squad", "gameweek", "season"]
    print("\n" + "=" * 86)
    print("S48 -- COMPONENT ABLATION (metric 1, season-eq pts, 8 clusters, bar |t|>2.36)")
    print("=" * 86)
    print(f"{'arm':<16}{'mean':>9}{'t':>8}{'p':>9}{'95% CI':>22}{'verdict':>16}")
    rows = []
    for a in [x for x in arms[1:] if "@" not in x]:
        m = scores[a].merge(scores["full"], on=ON, suffixes=("_a", "_f"))
        d = (m.value_a - m.value_f) * 38
        mu, t, p, lo, hi = cluster(d, m.season)
        v = "MATTERS" if abs(t) > 2.36 and mu < 0 else "no effect"
        print(f"{a:<16}{mu:>9.1f}{t:>8.2f}{p:>9.4f}  [{lo:>+7.1f},{hi:>+7.1f}]{v:>16}")
        rows.append(dict(arm=a, mean=mu, t=t, p=p, lo=lo, hi=hi))
    print(f"\n  negative = removing it HURTS = the component contributes")
    print("\n" + "=" * 86)
    print("PER POSITION  (component flattened for that position only)")
    print("=" * 86)
    print(f"{'component':<16}" + "".join(f"{q:>16}" for q in C.POSITIONS))
    for comp in ABLATE:
        line = f"{comp:<16}"
        for pos in C.POSITIONS:
            a = f"{comp}@{pos}"
            if a not in scores:
                line += f"{'-':>16}"; continue
            mm = scores[a].merge(scores["full"], on=ON, suffixes=("_a", "_f"))
            dd = (mm.value_a - mm.value_f) * 38
            mu2, t2, p2, lo2, hi2 = cluster(dd, mm.season)
            line += f"{mu2:>13.1f}{'*' if abs(t2) > 2.36 else ' ':<3}"
            rows.append(dict(arm=a, component=comp, position=pos,
                             mean=mu2, t=t2, p=p2, lo=lo2, hi=hi2))
        print(line)
    print("  * = |t| > 2.36 (significant at G=8).  negative = it matters there.")
    pickle.dump({"scores": scores, "summary": pd.DataFrame(rows)},
                open(C.cache("fpl2_ablation.pkl"), "wb"))
    print("\ndone")


if __name__ == "__main__":
    main()
