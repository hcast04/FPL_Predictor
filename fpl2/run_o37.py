#!/usr/bin/env python3
"""
O37: XI-aware squad optimisation, against the gate pre-registered 2026-10-22.

Gate (verbatim from the ledger, written before this file existed):
  1. mean season total across 3 holdouts must improve >= +100 pts/season
  2. 3/3 seasons positive -> SHIP;  2/3 -> REAL BUT NOT ESTABLISHED;  <=1/3 -> REFUTED
  3. w swept over {1.0, 0.5, 0.25, 0.1, 0.0}, whole curve reported, nothing selected on the run
  4. negative control: with xp shuffled within (season, gameweek, position),
     the w=0.1 minus w=1.0 delta must sit inside the noise floor

Also produced, NOT gated: five arms per holdout season --
legacy, legacy+fix, model, model+fix, FPL average.
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, features, legacy, model, rates, simulate, team  # noqa: E402
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor

GRID = (1.0, 0.5, 0.25, 0.1, 0.0)
AVG_PER_GW = 59.4                      # 2026-27 global average; x38 = 2257 (S22/S23/S25)
FPL_AVG_SEASON = AVG_PER_GW * 38


class _WP:
    """The shipped GLM rate component (S12)."""
    def __init__(self, est):
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.est = est
    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight); return self
    def predict(self, X):
        return self.est.predict(self.pre.transform(X))


def run(grid, signal, w):
    r = simulate.simulate_season(grid, signal, bench_weight=w)
    return r


def main():
    t0 = time.time()
    cur = pd.read_pickle(C.cache("fpl2_live2627.pkl"))
    raw = pd.concat([data.load_raw(), cur], ignore_index=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    seasons = list(C.HOLDOUT_SEASONS) + ["2026-27"]
    print(f"[{time.time()-t0:5.0f}s] setup done", flush=True)

    grids = {}
    for s in seasons:
        out = model.walk_forward(feats, tf, names, s)
        g = data.with_blanks(panel, s).merge(
            out[["season", "gameweek", C.PLAYER_KEY, "xp"]],
            on=["season", "gameweek", C.PLAYER_KEY], how="left")
        g["xp"] = g["xp"].fillna(0.0)
        g.loc[g["fixtures"] <= 0, "xp"] = 0.0
        if s != "2026-27":
            lg = legacy.attach(data.with_blanks(panel, s))[
                ["season", "gameweek", C.PLAYER_KEY, "xp"]].rename(columns={"xp": "xp_leg"})
            g = g.merge(lg, on=["season", "gameweek", C.PLAYER_KEY], how="left")
            g["xp_leg"] = g["xp_leg"].fillna(0.0)
            g.loc[g["fixtures"] <= 0, "xp_leg"] = 0.0
        grids[s] = g
        print(f"[{time.time()-t0:5.0f}s] forecast {s}", flush=True)

    # ---------------- dose-response sweep, both signals ---------------------
    rows = []
    for s in seasons:
        g = grids[s]
        for w in GRID:
            rows.append(dict(season=s, arm="model", w=w, total=run(g, "xp", w).total))
            if "xp_leg" in g.columns:
                rows.append(dict(season=s, arm="legacy", w=w,
                                 total=run(g, "xp_leg", w).total))
        print(f"[{time.time()-t0:5.0f}s] swept {s}", flush=True)
    R = pd.DataFrame(rows)

    print("\n" + "="*78)
    print("DOSE-RESPONSE: season total by bench_weight (w=1.0 is the control)")
    print("="*78)
    for arm in ("model", "legacy"):
        sub = R[R.arm.eq(arm)]
        if sub.empty: continue
        piv = sub.pivot_table(index="season", columns="w", values="total")
        print(f"\n  {arm.upper()}")
        print(piv.round(0).to_string())
        base = piv[1.0]
        print("  delta vs w=1.0:")
        print((piv.sub(base, axis=0)).round(0).to_string())

    # ---------------- the gate ----------------------------------------------
    print("\n" + "="*78)
    print("GATE (pre-registered 2026-10-22)")
    print("="*78)
    hold = R[R.season.isin(C.HOLDOUT_SEASONS) & R.arm.eq("model")]
    piv = hold.pivot_table(index="season", columns="w", values="total")
    base = piv[1.0]
    for w in GRID:
        if w == 1.0: continue
        d = piv[w] - base
        npos = int((d > 0).sum())
        mean = float(d.mean())
        verdict = ("SHIP" if mean >= 100 and npos == 3 else
                   "REAL BUT NOT ESTABLISHED" if mean >= 100 and npos == 2 else
                   "FAIL")
        print(f"  w={w:<5} mean {mean:+7.1f}  seasons positive {npos}/3  "
              f"[gate: >=+100 and 3/3]  -> {verdict}")
        print(f"          per season: " +
              "  ".join(f"{s} {d[s]:+.0f}" for s in C.HOLDOUT_SEASONS))

    # ---------------- negative control (R26) --------------------------------
    print("\n" + "="*78)
    print("NEGATIVE CONTROL: xp shuffled within (season, gameweek, position)")
    print("="*78)
    rng = np.random.default_rng(0)
    ctrl = []
    for s in C.HOLDOUT_SEASONS:
        g = grids[s].copy()
        g["xp"] = (g.groupby(["gameweek", "position"])["xp"]
                     .transform(lambda v: rng.permutation(v.to_numpy())))
        for w in (1.0, 0.1):
            ctrl.append(dict(season=s, w=w, total=run(g, "xp", w).total))
    Cc = pd.DataFrame(ctrl).pivot_table(index="season", columns="w", values="total")
    dctrl = Cc[0.1] - Cc[1.0]
    print(Cc.round(0).to_string())
    print(f"\n  shuffled delta (w=0.1 - w=1.0): " +
          "  ".join(f"{s} {dctrl[s]:+.0f}" for s in C.HOLDOUT_SEASONS))
    print(f"  mean {dctrl.mean():+.1f}   [noise floor ~150/season, R1]")
    print(f"  -> control {'PASSES (no spurious gain)' if abs(dctrl.mean()) < 150 else 'FAILS'}")

    # ---------------- five arms, as requested -------------------------------
    BEST = 0.0     # the strongest dose; reported, NOT selected as a ship value
    print("\n" + "="*78)
    print(f"FIVE ARMS (fix shown at w={BEST}; FPL average = {AVG_PER_GW}/gw x 38 = "
          f"{FPL_AVG_SEASON:.0f}, a 2026-27-derived PROXY)")
    print("="*78)
    print(f"{'season':<10}{'legacy':>9}{'legacy+fix':>12}{'model':>9}"
          f"{'model+fix':>11}{'FPL avg':>9}")
    acc = {k: [] for k in ("legacy", "legacy+fix", "model", "model+fix")}
    for s in seasons:
        g = R[R.season.eq(s)]
        def get(arm, w):
            v = g[(g.arm == arm) & (g.w == w)]["total"]
            return float(v.iloc[0]) if len(v) else float("nan")
        lg, lgf = get("legacy", 1.0), get("legacy", BEST)
        md, mdf = get("model", 1.0), get("model", BEST)
        n_gw = grids[s]["gameweek"].nunique()
        avg = AVG_PER_GW * n_gw
        print(f"{s:<10}{lg:>9.0f}{lgf:>12.0f}{md:>9.0f}{mdf:>11.0f}{avg:>9.0f}"
              + ("   (7 gw)" if s == "2026-27" else ""))
        if s in C.HOLDOUT_SEASONS:
            for k, v in zip(acc, (lg, lgf, md, mdf)):
                acc[k].append(v)
    print(f"{'MEAN':<10}" + "".join(f"{np.mean(acc[k]):>{w_}.0f}" for k, w_ in
          zip(("legacy","legacy+fix","model","model+fix"), (9,12,9,11)))
          + f"{FPL_AVG_SEASON:>9.0f}")

    pickle.dump({"sweep": R, "control": Cc}, open(C.cache("fpl2_o37.pkl"),"wb"))
    print(f"\n[{time.time()-t0:.0f}s] cached -> ~/fpl2_o37.pkl")


if __name__ == "__main__":
    main()
