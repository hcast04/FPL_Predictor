#!/usr/bin/env python3
"""
O51 / S52 -- rebuild the forecast layer and race the candidates.

WHY NOW. Ten seasons are loaded (2016-17..2025-26) and `fpl2/run_ablation.py`
(S48) concluded the model is "a minutes predictor wearing expensive
decoration". M22 shows that conclusion was measured on a broken instrument of
a different kind: FPL publishes `expected_goals`/`expected_assists` for only
3 of the 10 seasons, so on six of the eight holdout folds `RateModel.fit`
skipped the xG and xA targets entirely and `predict` returned the 0.0 it was
initialised with. **The model forecast zero goals and zero assists for every
player in the league on 6/8 folds.** Ablating a component that is already zero
measures nothing. Understat's `us_xg`/`us_xa` cover 94.8-99.0% of played rows
in ALL ten seasons.

So this is a genuine restart of the forecast layer, not another tuning pass.

PHASE A (this file) fits every arm per holdout season and writes the signals.
PHASE B (`run_bakeoff_score.py`) scores them. The split is deliberate: fitting
costs ~40 minutes, scoring costs seconds, and arms can be re-scored or new
views added without refitting. Every (family, season) is pickled the moment it
finishes -- a crash after 30 minutes of compute with nothing on disk has
happened twice on this project.

PRE-REGISTERED GATE (declared before any arm was scored):
  primary   metric 1, neutral fixed squads, XI+captain, K=10, 8 holdouts,
            paired on (season, squad, gameweek), SEASON-CLUSTERED SE (R32),
            G=8 -> bar |t| > 2.36.
  reference `m_full_fpl`, the model shipped today.
  BETTER    clustered 95% CI excludes 0 AND point estimate > +13.6/season
            (the instrument's MDE at this n -- a gate below it cannot be
            passed except by luck, R24).
  EQUIV     |mean| < 13.6 and CI contains 0.
  Directions are NOT part of the gate (R27).
"""
from __future__ import annotations
import os, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))

SCRATCH = Path(os.environ.get("BAKE_DIR", "/tmp/bakeoff"))
OUT = SCRATCH / "bake"; OUT.mkdir(parents=True, exist_ok=True)
HOLDOUTS = ["2018-19", "2019-20", "2020-21", "2021-22",
            "2022-23", "2023-24", "2024-25", "2025-26"]
KEEP = ["season", "gameweek", "player_code", "points", "cost",
        "position", "team", "minutes", "fixtures"]

# Pre-registered hyperparameters. NOT tuned on any holdout -- they mirror the
# settings already in `rates.py`/`team.py` so the comparison is architecture
# and data, not a tuning budget. Round 2 may tune, on an inner split.
XGB = dict(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
           colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0,
           tree_method="hist", n_jobs=int(os.environ.get("NJOBS", "3")),
           random_state=0)
HGB = dict(loss="poisson", max_iter=300, learning_rate=0.06, max_depth=6,
           min_samples_leaf=40, l2_regularization=1.0, random_state=0)


class _WP:
    """median-impute + standardise around a linear estimator (S12's config)."""
    def __init__(self, est):
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.est = est

    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.est.predict(self.pre.transform(X))


def _glm():
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import PoissonRegressor
    # median imputation + standardisation is ~55 of S12's 194 points
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         PoissonRegressor(alpha=1.0, max_iter=500))


def run(family: str, holdout: str):
    from fpl2 import config as C, features, model, rates   # noqa: E402
    import xgboost as xgb
    from sklearn.ensemble import HistGradientBoostingRegressor

    t0 = time.time()
    use_us = family in ("us", "pos", "s47", "gbdtus")
    features.USE_UNDERSTAT_FEATURES = use_us
    tag = "feats_us" if use_us else "feats"
    feats = pd.read_pickle(SCRATCH / f"{tag}.pkl")
    names = pickle.load(open(SCRATCH / f"{tag.replace('feats','names')}.pkl", "rb"))
    tf = pd.read_pickle(SCRATCH / "tf.pkl")
    feats = feats[feats["season"] <= holdout]
    for c in feats.columns:                       # halve the resident set
        if feats[c].dtype == "float64":
            feats[c] = feats[c].astype("float32")

    tr = feats[feats["season"] < holdout]
    use = features.usable(tr, list(names))        # per-fold, as every model does
    Xtr = tr[use].to_numpy("float32")
    ytr = np.clip(pd.to_numeric(tr["points"], errors="coerce").fillna(0).to_numpy(float), 0, None)
    print(f"[{family}/{holdout}] {len(use)} feats, train {len(tr):,}", flush=True)

    def done(name):
        print(f"   [{time.time()-t0:5.0f}s] {name}", flush=True)

    # `walk_forward` returns the holdout CONCATENATED BY GAMEWEEK, which is not
    # the panel's row order. Every arm is therefore read off `o` and `o` alone:
    # assigning a prediction positionally from a differently-sorted frame
    # scrambles the join and reports a working model at spearman -0.0145.
    # THE RATE ESTIMATOR. `rates._reg` defaults to HistGradientBoosting, but
    # every runner on this project -- run_ablation, run_o50, run_chips,
    # build_wildcard_* -- overrides it to a Poisson GLM on median-imputed,
    # standardised inputs (S12, magnitude corrected by S39). The GLM is
    # therefore the DE FACTO shipped configuration and the honest reference;
    # measuring against the module default would be measuring against a
    # straw model nobody runs. `m_gbdt_fpl` keeps the module default as its
    # own arm so the choice is measured rather than inherited.
    from sklearn.linear_model import PoissonRegressor
    gbdt_reg = rates._reg
    rates._reg = lambda seed=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))

    o = model.walk_forward(feats, tf, names, holdout)
    te = o
    Xte = te[use].to_numpy("float32")
    out = te[KEEP].copy()

    if family == "fpl":
        out["b_price"] = te["price"].to_numpy(float)
        out["b_form5"] = te["points_r5"].fillna(0).to_numpy(float)
        out["b_form10"] = te["points_r10"].fillna(0).to_numpy(float)
        out["m_full_fpl"] = te["xp"].to_numpy(float); done("baselines + m_full_fpl")
        out["d_xgb_fpl"] = xgb.XGBRegressor(objective="count:poisson", **XGB) \
            .fit(Xtr, ytr).predict(Xte); done("d_xgb_fpl")
        rates._reg = gbdt_reg                     # the module default
        g = model.walk_forward(feats, tf, names, holdout)
        assert g[["gameweek", "player_code"]].reset_index(drop=True).equals(
            te[["gameweek", "player_code"]].reset_index(drop=True)), "row order"
        out["m_gbdt_fpl"] = g["xp"].to_numpy(float)
        # M26 negative control, stated correctly. bonus/saves/yellow ARE fitted
        # on every fold, so the GLM and GBDT arms legitimately differ there --
        # an earlier version of this check asserted they must be identical and
        # was simply wrong. What M26 actually claims is narrower and exact:
        # before 2023-24 there is no xG or xA target, so the GOALS and ASSISTS
        # predictions are identically zero for every player.
        gz = float(np.abs(te["pred_goals_per90"]).max())
        az = float(np.abs(te["pred_assists_per90"]).max())
        print(f"   M26 control: max pred_goals/90={gz:.4f} "
              f"max pred_assists/90={az:.4f} "
              f"(expect 0.0000 before 2023-24)", flush=True)
        done("m_gbdt_fpl")
    elif family == "gbdtus":
        # S12 measured the GLM rate layer at +194 over the GBDT; S39 corrected
        # that to ~21; `m_gbdt_fpl` in this bake-off measured -0.9 with an MDE
        # of 11.1 -- a tight null. But all of those compared the two estimators
        # on a rate layer whose xG and xA heads DID NOT EXIST on five of eight
        # folds, so most of what they compared was bonus and cards. With the
        # data repaired the question is live again and is asked here.
        rates._reg = gbdt_reg
        g2 = model.walk_forward(feats, tf, names, holdout)
        assert g2[["gameweek", "player_code"]].reset_index(drop=True).equals(
            te[["gameweek", "player_code"]].reset_index(drop=True)), "row order"
        out["m_gbdt_us"] = g2["xp"].to_numpy(float); done("m_gbdt_us")
    elif family == "s47":
        # REPLICATION TEST of ledger S47, which measured Understat features at
        # +1.6 [-3.7, +6.8] on this exact instrument (K=10, same 8 holdouts,
        # seed=1 squads, season-clustered) while this bake-off measures the
        # same swap at +55.5. One of the two is wrong and the difference must
        # be located, not averaged over.
        #
        # S47 predates `_xa_source` (M26), so its "understat" arm swapped the
        # xG target only and still had NO assist model on five folds. This arm
        # reproduces that exactly: Understat features and us_xg, but xA pinned
        # back to FPL's column. If it lands near +1.6 the two results are
        # consistent and the xA half carries the effect; if it lands near +55
        # then S47 has a defect.
        orig_xa = rates._xa_source
        rates._xa_source = lambda: "expected_assists"
        try:
            s47 = model.walk_forward(feats, tf, names, holdout)
        finally:
            rates._xa_source = orig_xa
        assert s47[["gameweek", "player_code"]].reset_index(drop=True).equals(
            te[["gameweek", "player_code"]].reset_index(drop=True)), "row order"
        out["m_s47_xgonly"] = s47["xp"].to_numpy(float)
        done("m_s47_xgonly (S47 replication)")
    elif family == "pos":
        # THE PER-POSITION AXIS, isolated.
        #
        # `rates.RateModel._groups` fits the rate layer SEPARATELY for GK, DEF,
        # MID and FWD, with per-position conversion factors and ceilings. The
        # direct arms in the `us` family are POOLED, carrying position only as
        # the numeric feature `pos_code`. So "component vs direct" as first
        # built confounds architecture with the position split, and these two
        # arms break the confound from both sides:
        #
        #   d_xgb_pos   the direct model, split four ways like the rate layer
        #   m_pool_us   the component model, rate layer POOLED
        #
        # Ledger F7 rejected per-position models -- on the 3-season instrument
        # whose noise floor was later shown unfit (S37), by "spearman 0/3".
        # This is a re-test, not a new idea.
        out["z_m_full_us_ref"] = te["xp"].to_numpy(float)
        pos_tr = tr["position"].to_numpy(); pos_te = te["position"].to_numpy()
        pred = np.zeros(len(te), float)
        for q in C.POSITIONS:
            mtr, mte = pos_tr == q, pos_te == q
            if mtr.sum() < 300 or mte.sum() == 0:
                continue
            pred[mte] = xgb.XGBRegressor(objective="count:poisson", **XGB) \
                .fit(Xtr[mtr], ytr[mtr]).predict(Xte[mte])
        out["d_xgb_pos"] = pred; done("d_xgb_pos")

        # A SUBCLASS, not a lambda. `model.predict` reads
        # `rates_mod.RateModel.PREFIX` as a CLASS attribute, so swapping the
        # name for a function gives "'function' object has no attribute
        # 'PREFIX'" eight jobs into the run. A subclass keeps every class
        # attribute and only changes the default.
        orig = rates.RateModel

        class _Pooled(orig):
            def __init__(self, *a, **kw):
                kw.setdefault("per_position", False)
                super().__init__(*a, **kw)

        rates.RateModel = _Pooled
        try:
            pooled = model.walk_forward(feats, tf, names, holdout)
        finally:
            rates.RateModel = orig
        assert pooled[["gameweek", "player_code"]].reset_index(drop=True).equals(
            te[["gameweek", "player_code"]].reset_index(drop=True)), "row order"
        out["m_pool_us"] = pooled["xp"].to_numpy(float); done("m_pool_us")
        # Carried for the OPTIMISER bake-off, which races decision rules on a
        # fixed signal. A captain rule that wants to penalise minutes risk
        # needs p_sixty; one that wants upside needs the dispersion of recent
        # returns. Neither survives into the signal columns, so save them here
        # rather than refit the whole component model again later.
        for c in ("p_sixty", "p_appear", "expected_minutes"):
            out["z_" + c] = te[c].to_numpy(float)
        for c in ("points_r5", "points_r10", "minutes_sd_r5", "minutes_r5"):
            if c in te.columns:
                out["z_" + c] = pd.to_numeric(te[c], errors="coerce").to_numpy(float)
        done("optimiser inputs")
    else:
        out["m_full_us"] = te["xp"].to_numpy(float); done("m_full_us")
        # harvest the fitted minutes hierarchy -- d_per90 reuses it rather than
        # refitting, so the two arms differ ONLY in the rate layer
        exp_min = o["expected_minutes"].to_numpy(float)

        out["d_glm"] = np.clip(_glm().fit(tr[use], ytr).predict(te[use]), 0, None)
        done("d_glm")
        out["d_hgb"] = np.clip(HistGradientBoostingRegressor(**HGB)
                               .fit(Xtr, ytr).predict(Xte), 0, None); done("d_hgb")
        out["d_xgb"] = xgb.XGBRegressor(objective="count:poisson", **XGB) \
            .fit(Xtr, ytr).predict(Xte); done("d_xgb")

        # hurdle: P(played at all) x E[points | played]. FPL points are ~57%
        # exact zeros, which a single Poisson head has to spend capacity on.
        played = tr["minutes"].to_numpy(float) > 0
        clf = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                **XGB).fit(Xtr, played.astype(int))
        reg = xgb.XGBRegressor(objective="count:poisson", **XGB) \
            .fit(Xtr[played], ytr[played])
        out["d_hurdle"] = clf.predict_proba(Xte)[:, 1] * reg.predict(Xte)
        done("d_hurdle")

        # points per 90 among players who were on the pitch, x expected minutes.
        # Same decomposition as the component model, one head instead of seven.
        mins = tr["minutes"].to_numpy(float)
        ok = mins >= 20.0
        p90 = xgb.XGBRegressor(objective="count:poisson", **XGB).fit(
            Xtr[ok], ytr[ok] * 90.0 / mins[ok], sample_weight=mins[ok] / 90.0)
        out["d_per90"] = p90.predict(Xte) * exp_min / 90.0
        done("d_per90")

        # the captain pick is argmax, which samples the UPPER tail -- a
        # conditional mean is the wrong functional for it (S20).
        out["d_xgb_q80"] = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=0.80, **XGB) \
            .fit(Xtr, ytr).predict(Xte)
        done("d_xgb_q80")

    # GUARD. A scrambled join produces a signal that ranks at chance while
    # looking entirely healthy in mean/sd -- that is how `m_full_us` was first
    # reported at spearman -0.0145. No real arm here, not even price alone,
    # ranks below 0.10 within a gameweek. Fail loudly rather than write it.
    arms = [c for c in out.columns if c.split("_")[0] in ("b", "m", "d", "s")]
    pl = out[out["fixtures"] > 0]
    rho = {a: float(pl.groupby("gameweek")
                      .apply(lambda g: g[a].corr(g["points"], method="spearman"),
                             include_groups=False).mean()) for a in arms}
    print("   rho: " + "  ".join(f"{a}={v:+.3f}" for a, v in rho.items()), flush=True)
    bad = {a: v for a, v in rho.items() if not np.isfinite(v) or v < 0.10}
    if bad:
        raise SystemExit(f"ABORT {family}/{holdout}: ranks at chance -> {bad}")

    p = OUT / f"{family}_{holdout}.pkl"
    out.to_pickle(p)                              # persist IMMEDIATELY
    print(f"[{family}/{holdout}] WROTE {p.name} {out.shape} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
