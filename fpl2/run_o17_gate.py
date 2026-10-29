#!/usr/bin/env python3
"""
O17 completed gate: evaluate the clause the first run declared but never tested.

run_o17.py declared "wins 3/3 on both top-20 and captain-added" but computed
those only as SEASON MEANS, so the per-season clause went unevaluated -- the
same class of error as O20's mismatched-fold comparison. `all` passed what was
implemented (mean gain > 148, better in 3/3 seasons on the season total); this
evaluates what was declared.

Also added, because S12 got it and `all` did not: a BASELINE SEED SPREAD. The
baseline was tight in the S12 replication (-251..-274 over 4 seeds) but that is
an assumption here until measured, and `all` must beat the BEST seed, not the
mean -- comparing a chosen arm against an average is how four-way searches
manufacture winners.

GATE, as originally declared plus the seed control:
  1. per-season top-20 better than baseline in 3/3
  2. per-season captain-added better than baseline in 3/3
  3. per-season season-total better than baseline in 3/3
  4. mean gain > 148
  5. mean gain beats the BEST of 3 baseline seeds
Anything less is recorded as a partial pass with the failing clause named.
"""
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, legacy, model, rates, simulate, team  # noqa: E402
from fpl2 import minutes as minutes_mod   # noqa: E402

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

B_RATES, B_CLF, B_REG = rates._reg, minutes_mod._clf, minutes_mod._reg
B_TEAM = team.HistGradientBoostingRegressor


def _lin(est):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)


class _WeightedPipe:
    def __init__(self, est):
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.est = est

    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.est.predict(self.pre.transform(X))


def restore():
    rates._reg, minutes_mod._clf, minutes_mod._reg = B_RATES, B_CLF, B_REG
    team.HistGradientBoostingRegressor = B_TEAM


def set_all_glm():
    rates._reg = lambda s=0: _WeightedPipe(PoissonRegressor(alpha=1.0, max_iter=500))
    minutes_mod._clf = lambda s=0: _lin(LogisticRegression(max_iter=1000, C=1.0))
    minutes_mod._reg = lambda s=0: _lin(PoissonRegressor(alpha=1.0, max_iter=500))
    team.HistGradientBoostingRegressor = (
        lambda **kw: _lin(PoissonRegressor(alpha=1.0, max_iter=500)))


def set_baseline_seed(k):
    restore()
    rates._reg = lambda s=0: HistGradientBoostingRegressor(
        loss="poisson", max_iter=300, learning_rate=0.06, max_depth=6,
        min_samples_leaf=40, l2_regularization=1.0, random_state=k)
    minutes_mod._clf = lambda s=0: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=6, min_samples_leaf=40,
        l2_regularization=1.0, random_state=k + s)
    minutes_mod._reg = lambda s=0: HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_depth=6, min_samples_leaf=40,
        l2_regularization=1.0, random_state=k + s)


def per_season(out):
    t = out[out.fixtures > 0]
    t20, cap = [], []
    for gw, g in t.groupby("gameweek"):
        if len(g) < 50:
            continue
        t20.append(float(g.nlargest(20, "xp").points.sum()))
        cap.append(float(g.loc[g.xp.idxmax(), "points"]))
    return float(np.mean(t20)), float(np.mean(cap))


def run(label, leg, feats, names, tf):
    t20, cap, marg = {}, {}, {}
    for s in C.HOLDOUT_SEASONS:
        out = model.walk_forward(feats, tf, names, s)
        a, b = per_season(out)
        g = leg[s][0].merge(out[["season", "gameweek", C.PLAYER_KEY, "xp"]],
                            on=["season", "gameweek", C.PLAYER_KEY], how="left")
        g["xp"] = g["xp"].fillna(0.0)
        g.loc[g["fixtures"] <= 0, "xp"] = 0.0
        t20[s], cap[s] = a, b
        marg[s] = float(simulate.simulate_season(g, "xp").total) - leg[s][1]
    print(f"  {label:18s} " + "  ".join(
        f"{s}: t20 {t20[s]:5.1f} cap {cap[s]:4.1f} m {marg[s]:+5.0f}"
        for s in C.HOLDOUT_SEASONS), flush=True)
    return t20, cap, marg


def main():
    t0 = time.time()
    raw = data.load_raw(); panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)
    leg = {}
    for s in C.HOLDOUT_SEASONS:
        g = legacy.attach(data.with_blanks(panel, s)).rename(columns={"xp": "xp_legacy"})
        g.loc[g["fixtures"] <= 0, "xp_legacy"] = 0.0
        leg[s] = (g, float(simulate.simulate_season(g, "xp_legacy").total))
    print(f"[{time.time()-t0:4.0f}s] setup\n", flush=True)

    seeds = {}
    for k in (0, 1, 2):
        set_baseline_seed(k)
        seeds[k] = run(f"baseline_seed{k}", leg, feats, names, tf)
    restore()
    set_all_glm()
    g_t20, g_cap, g_m = run("all_glm", leg, feats, names, tf)
    restore()

    b0 = seeds[0]
    w_t20 = sum(1 for s in C.HOLDOUT_SEASONS if g_t20[s] > b0[0][s])
    w_cap = sum(1 for s in C.HOLDOUT_SEASONS if g_cap[s] > b0[1][s])
    w_m = sum(1 for s in C.HOLDOUT_SEASONS if g_m[s] > b0[2][s])
    gain = np.mean([g_m[s] for s in C.HOLDOUT_SEASONS]) - np.mean(
        [b0[2][s] for s in C.HOLDOUT_SEASONS])
    seed_means = {k: np.mean([v[2][s] for s in C.HOLDOUT_SEASONS]) for k, v in seeds.items()}
    best_seed = max(seed_means.values())
    glm_mean = float(np.mean([g_m[s] for s in C.HOLDOUT_SEASONS]))

    print(f"\nCOMPLETED GATE", flush=True)
    print(f"  baseline seed means : "
          f"{ {k: round(v) for k, v in seed_means.items()} }  best {best_seed:+.0f}", flush=True)
    print(f"  all_glm mean        : {glm_mean:+.0f}", flush=True)
    for n, (lbl, w) in enumerate([("per-season top-20", w_t20),
                                  ("per-season captain", w_cap),
                                  ("per-season total", w_m)], 1):
        print(f"  {n}. {lbl:22s} {w}/3   {'PASS' if w == 3 else 'FAIL'}", flush=True)
    print(f"  4. mean gain > 148       {gain:+.0f}   "
          f"{'PASS' if gain > 148 else 'FAIL'}", flush=True)
    print(f"  5. beats BEST seed       {glm_mean:+.0f} vs {best_seed:+.0f}   "
          f"{'PASS' if glm_mean > best_seed else 'FAIL'}", flush=True)
    ok = w_t20 == 3 and w_cap == 3 and w_m == 3 and gain > 148 and glm_mean > best_seed
    print(f"\n  VERDICT: {'PASS' if ok else 'PARTIAL/FAIL -- failing clauses named above'}",
          flush=True)
    pickle.dump({"seeds": seeds, "all_glm": (g_t20, g_cap, g_m)},
                open(C.cache("fpl2_o17_gate.pkl"), "wb"))
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
