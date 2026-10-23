#!/usr/bin/env python3
"""
O55 -- make BONUS conditional on the model's own predicted returns.

The user's observation: B.Fernandes was forecast 5.7-7.1 every week while
returning 2,23,2,2,2,1,1. The over-prediction is not his attacking rates --
2.66 xG and 1.50 xA for 3 goals and 1 assist is performing TO xG -- it is
BONUS: predicted 1.055/90 against an actual ~0.43/90.

And bonus is overwhelmingly CONDITIONAL. Over 105,659 played player-gameweeks
across ten seasons:

| condition | P(bonus>0) | E[bonus] |
|---|---|---|
| points (ex-bonus) <= 5 | 0.024 | 0.038 |
| points (ex-bonus) > 5  | 0.523 | 1.099 |
| no goal or assist      | 0.041 | 0.071 |
| 1+ goal or assist      | 0.509 | 1.089 |

A 29x ratio. `rates` fits `bonus_per90` as a FLAT MARGINAL RATE, so the model
simultaneously predicts "probably blanks" and "collects bonus anyway" for the
same player in the same fixture. That is internally inconsistent, and unlike
the ceiling (S54) it is not a level effect -- it reweights players BY their
predicted return probability, so it reorders and can move a rank decision
(R41).

ARMS. One fit per season, xp recomputed per arm, so only the bonus term moves.

  bon_flat    shipped. MUST reproduce `m_full_us` bit-identically.
  bon_cond    E[bonus] = P(ret)*B_hit + (1-P(ret))*B_miss, with B_hit and
              B_miss per-position per-90 rates from TRAINING only.
  bon_scale   the fitted bonus_per90 REWEIGHTED by P(ret), renormalised so the
              population mean is unchanged. This is the clean test of the
              MECHANISM: bon_cond also lowers the overall level, and R41 says
              a level shift should do nothing, so if bon_cond works and
              bon_scale does not, the gain was not the reweighting.
  bon_hard    bonus only where a return is predicted, else zero -- the
              degenerate limit, included as a dose-response anchor.

`ret` includes a clean sheet for DEF and GK, whose bonus comes from shut-outs
and saves rather than from goals and assists.
"""
from __future__ import annotations
import os, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, features, model, rates, scoring, team  # noqa: E402

SCRATCH = Path(os.environ.get("BAKE_DIR", "/tmp/bakeoff"))
OUT = SCRATCH / "bake"; OUT.mkdir(parents=True, exist_ok=True)
P = rates.RateModel.PREFIX
KEEP = ["season", "gameweek", "player_code", "points", "cost",
        "position", "team", "minutes", "fixtures"]
ARMS = ["bon_flat", "bon_cond", "bon_scale", "bon_hard"]


class _WP:
    def __init__(self, est):
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.est = est
    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight); return self
    def predict(self, X): return self.est.predict(self.pre.transform(X))


def conditional_bonus(train: pd.DataFrame) -> dict:
    """Per-position per-90 bonus rates, split by whether a RETURN happened."""
    d = train[train["minutes"] > 0].copy()
    g = pd.to_numeric(d.get("goals_scored"), errors="coerce").fillna(0)
    a = pd.to_numeric(d.get("assists"), errors="coerce").fillna(0)
    cs = pd.to_numeric(d.get("clean_sheets"), errors="coerce").fillna(0)
    ret = (g > 0) | (a > 0) | (d["position"].isin(["DEF", "GK"]) & (cs > 0))
    out = {}
    for pos in C.POSITIONS:
        s = d[d["position"].eq(pos)]; r = ret[d["position"].eq(pos)]
        def rate(sub):
            m = float(sub["minutes"].sum())
            return 90.0 * float(pd.to_numeric(sub["bonus"], errors="coerce")
                                .fillna(0).sum()) / max(m, 1.0)
        out[pos] = {"hit": rate(s[r]), "miss": rate(s[~r])}
    return out


def p_return(o: pd.DataFrame) -> np.ndarray:
    """P(at least one return this fixture), from the model's OWN components."""
    mins = pd.to_numeric(o["expected_minutes"], errors="coerce").fillna(0).to_numpy(float)
    fx = pd.to_numeric(o["fixtures"], errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
    lam = ((o[P + "goals_per90"].to_numpy(float) + o[P + "assists_per90"].to_numpy(float))
           * mins / 90.0 * np.clip(fx, 0, None))
    p_ga = 1.0 - np.exp(-np.clip(lam, 0, None))
    cs = np.clip(o["clean_sheet_prob"].to_numpy(float), 0, 1)
    back = o["position"].isin(["DEF", "GK"]).to_numpy()
    return np.where(back, 1.0 - (1.0 - p_ga) * (1.0 - cs), p_ga)


def recompute(o: pd.DataFrame, bonus: np.ndarray) -> np.ndarray:
    fx = pd.to_numeric(o["fixtures"], errors="coerce").fillna(0).clip(lower=1)
    v = scoring.expected_points(
        p_appear=o["p_appear"], p_sixty=o["p_sixty"],
        exp_minutes=o["expected_minutes"], position=o["position"],
        goals_per90=o[P + "goals_per90"], assists_per90=o[P + "assists_per90"],
        saves_per90=o[P + "saves_per90"], clean_sheet_prob=o["clean_sheet_prob"],
        conceded_per90=o["expected_conceded"].to_numpy(float) / fx.to_numpy(float),
        bonus_per90=bonus, yellow_per90=o[P + "yellow_per90"],
        defensive_return_prob=o[P + "defensive_return_prob"])
    return np.clip(np.nan_to_num(np.asarray(v, float), nan=0.0), 0, None)


def run(holdout: str):
    from sklearn.linear_model import PoissonRegressor
    t0 = time.time()
    features.USE_UNDERSTAT_FEATURES = True
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    feats = pd.read_pickle(SCRATCH / "feats_us.pkl")
    names = pickle.load(open(SCRATCH / "names_us.pkl", "rb"))
    tf = pd.read_pickle(SCRATCH / "tf.pkl")
    feats = feats[feats["season"] <= holdout]
    for c in feats.columns:
        if feats[c].dtype == "float64":
            feats[c] = feats[c].astype("float32")
    tr = feats[feats["season"] < holdout]
    o = model.walk_forward(feats, tf, names, holdout)
    print(f"[{holdout}] fitted+predicted in {time.time()-t0:.0f}s", flush=True)

    cb = conditional_bonus(tr)
    print("   per-90 bonus, TRAINING only:  " + "  ".join(
        f"{p}: hit {cb[p]['hit']:.3f} / miss {cb[p]['miss']:.3f}" for p in C.POSITIONS),
        flush=True)

    pr = p_return(o)
    flat = o[P + "bonus_per90"].to_numpy(float)
    hit = o["position"].map({p: cb[p]["hit"] for p in cb}).astype(float).to_numpy()
    miss = o["position"].map({p: cb[p]["miss"] for p in cb}).astype(float).to_numpy()

    out = o[KEEP].copy()
    out["bon_flat"] = recompute(o, flat)
    out["bon_cond"] = recompute(o, pr * hit + (1.0 - pr) * miss)
    # renormalise so the POPULATION mean bonus is unchanged -- isolates the
    # reweighting from the level drop (R41)
    w = pr / max(float(np.nanmean(pr)), 1e-9)
    out["bon_scale"] = recompute(o, flat * w)
    out["bon_hard"] = recompute(o, np.where(pr >= 0.5, flat, 0.0))
    for a in ARMS:
        print(f"   {a:<10} mean xp {out[a].mean():.4f}", flush=True)

    ref = OUT / f"us_{holdout}.pkl"
    if ref.exists():
        prev = pd.read_pickle(ref)[["gameweek", "player_code", "m_full_us"]]
        j = out[["gameweek", "player_code", "bon_flat"]].merge(
            prev, on=["gameweek", "player_code"])
        d = float(np.abs(j["bon_flat"] - j["m_full_us"]).max())
        print(f"   control: max|bon_flat - m_full_us| = {d:.10f} "
              f"({'IDENTICAL' if d < 1e-9 else '*** DIFFERS ***'})", flush=True)
    out.to_pickle(OUT / f"bon_{holdout}.pkl")
    print(f"[{holdout}] WROTE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
