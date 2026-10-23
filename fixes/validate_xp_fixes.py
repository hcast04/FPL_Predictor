"""
Chronological out-of-sample validation of the v2.6 mechanism fixes.

Methodology follows the project audit log (sections 7 and 17):
  * chronological train/test split, no test-season data in any fit;
  * acceptance criteria declared BEFORE the result is inspected (below);
  * decision metrics (ranking / top-K), not only calibration.

Primary test season is 2024-25 so that defensive contribution -- which did not
exist before 2025/26 and has no training support in the earlier seasons -- is
correctly zero for both predicted and actual points.

ACCEPTANCE CRITERIA (declared in advance)
-----------------------------------------
A fix is accepted only if, on the held-out season:
  G1  mean signed bias |predicted - actual| moves toward zero;
  G2  overall MAE does not degrade by more than 0.02;
  G3  mean per-gameweek Spearman does not degrade by more than 0.003;
  G4  NDCG@10 does not degrade by more than 0.002;
  G5  mean top-10 realised points per gameweek does not fall by more than 0.25.
These mirror the gates the audit log applied to the rejected v2.4 layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fpl_xp_fixes as fx  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
VAASTAV_ROOT = (
    "https://raw.githubusercontent.com/vaastav/"
    "Fantasy-Premier-League/master/data"
)


def ensure_data() -> None:
    """Download the archived gameweek files once if they are not present."""
    import urllib.request

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        target = DATA_DIR / f"{season}.csv"
        if target.exists() and target.stat().st_size > 100_000:
            continue
        url = f"{VAASTAV_ROOT}/{season}/gws/merged_gw.csv"
        print(f"[validate] downloading {season} ...")
        urllib.request.urlretrieve(url, target)
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
LEAGUE_GOALS = 1.42

GOAL_POINTS_BASELINE = {"GK": 10.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
CLEAN_SHEET_POINTS = {"GK": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_seasons(mask_coverage: bool) -> pd.DataFrame:
    ensure_data()
    frames = []
    for season in SEASONS:
        frame = pd.read_csv(DATA_DIR / f"{season}.csv", low_memory=False)
        frame["season"] = season
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["position"] = data["position"].replace({"GKP": "GK"})
    data = data[data["position"].isin(fx.PLAYER_SCORING_POSITIONS)].copy()
    data["model_position"] = data["position"]
    data["gameweek"] = pd.to_numeric(data["round"], errors="coerce")
    for column in [
        "minutes", "goals_scored", "assists", "expected_goals",
        "expected_assists", "bonus", "bps", "saves", "yellow_cards",
        "red_cards", "clean_sheets", "goals_conceded", "total_points",
        "starts", "threat", "creativity", "value", "was_home",
        "defensive_contribution",
    ]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        else:
            data[column] = np.nan
    data["kickoff_time"] = pd.to_datetime(
        data["kickoff_time"], errors="coerce", utc=True
    )
    if mask_coverage:
        data = fx.mask_fabricated_event_coverage(data)
    return data.sort_values(["season", "element", "kickoff_time"]).reset_index(drop=True)


def add_team_strength(data: pd.DataFrame) -> pd.DataFrame:
    team = (
        data.groupby(["season", "team", "fixture"], as_index=False)
        .agg(gc=("goals_conceded", "max"), ko=("kickoff_time", "min"))
    )
    scored = (
        data.groupby(["season", "team", "fixture"], as_index=False)
        .agg(gf=("goals_scored", "sum"))
    )
    team = team.merge(scored, on=["season", "team", "fixture"])
    team = team.sort_values(["season", "team", "ko"])
    for raw, out in [("gc", "team_gc_prior"), ("gf", "team_gf_prior")]:
        team[out] = (
            team.groupby(["season", "team"])[raw]
            .transform(lambda s: s.shift().expanding().mean())
        )
    team["n_prior"] = team.groupby(["season", "team"]).cumcount()
    shrink = 6.0
    for out in ["team_gc_prior", "team_gf_prior"]:
        team[out] = (
            team["n_prior"] * team[out].fillna(LEAGUE_GOALS) + shrink * LEAGUE_GOALS
        ) / (team["n_prior"] + shrink)
    return data.merge(
        team[["season", "team", "fixture", "team_gc_prior", "team_gf_prior", "n_prior"]],
        on=["season", "team", "fixture"], how="left",
    )


RATE_SOURCES = {
    "xg": "expected_goals", "xa": "expected_assists", "bonus": "bonus",
    "bps": "bps", "saves": "saves", "yellow": "yellow_cards",
    "points": "total_points", "threat": "threat", "creativity": "creativity",
}


def add_history_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    grp = out.groupby(["season", "element"], sort=False)
    out["prev_minutes"] = grp["minutes"].shift()
    out["prev_started"] = grp["starts"].shift()
    mins = grp["minutes"].transform(lambda s: s.shift())

    for window in (3, 5, 10):
        out[f"minutes_l{window}"] = grp["minutes"].transform(
            lambda s, w=window: s.shift().rolling(w, min_periods=1).mean()
        )
        out[f"starts_l{window}"] = grp["starts"].transform(
            lambda s, w=window: s.shift().rolling(w, min_periods=1).mean()
        )
    out["career_minutes"] = grp["minutes"].transform(
        lambda s: s.shift().expanding().mean()
    )
    out["career_starts"] = grp["starts"].transform(
        lambda s: s.shift().expanding().mean()
    )
    out["appearances_so_far"] = grp["minutes"].transform(
        lambda s: s.shift().gt(0).cumsum()
    )

    for label, source in RATE_SOURCES.items():
        shifted_event = grp[source].transform(lambda s: s.shift())
        for window in (5, 10):
            num = shifted_event.groupby(
                [out["season"], out["element"]], sort=False
            ).transform(lambda s, w=window: s.rolling(w, min_periods=1).sum())
            den = mins.groupby(
                [out["season"], out["element"]], sort=False
            ).transform(lambda s, w=window: s.rolling(w, min_periods=1).sum())
            out[f"{label}90_l{window}"] = (
                90.0 * num / den.where(den >= 45.0)
            )
        num = shifted_event.groupby(
            [out["season"], out["element"]], sort=False
        ).transform(lambda s: s.expanding().sum())
        den = mins.groupby(
            [out["season"], out["element"]], sort=False
        ).transform(lambda s: s.expanding().sum())
        out[f"career_{label}90"] = 90.0 * num / den.where(den >= 90.0)

    out["target_minutes"] = out["minutes"]
    out["target_appeared"] = (out["minutes"] > 0).astype(int)
    out["target_started"] = (out["starts"].fillna(0) > 0).astype(int)
    out["target_sixty"] = (out["minutes"] >= 60).astype(int)
    for label, source in [
        ("xg", "expected_goals"), ("xa", "expected_assists"),
        ("saves", "saves"), ("bonus", "bonus"),
        ("yellow_cards", "yellow_cards"), ("red_cards", "red_cards"),
    ]:
        out[f"target_{label}_per90"] = (
            90.0 * out[source] / out["minutes"]
        ).where(out["minutes"] > 0)
    out["target_fpl_points"] = out["total_points"]

    # Defensive contribution did not exist before 2025/26 and has no training
    # support in the earlier seasons. Remove its points from the realised
    # target so predicted and actual are on one scoring basis.
    threshold = out["model_position"].map(
        {"GK": np.inf, "DEF": 10.0, "MID": 12.0, "FWD": 12.0}
    )
    actions = pd.to_numeric(out.get("defensive_contribution"), errors="coerce")
    dc_points = 2.0 * (actions.ge(threshold) & threshold.notna()).astype(float)
    out["target_fpl_points"] = out["target_fpl_points"] - dc_points.fillna(0.0)
    return out


FEATURES = (
    ["minutes_l3", "minutes_l5", "minutes_l10", "starts_l3", "starts_l5",
     "starts_l10", "career_minutes", "career_starts", "appearances_so_far",
     "prev_minutes", "prev_started", "team_gc_prior", "team_gf_prior",
     "was_home", "gameweek"]
    + [f"{l}90_l{w}" for l in RATE_SOURCES for w in (5, 10)]
    + [f"career_{l}90" for l in RATE_SOURCES]
)


FEATURE_CLIPS: dict[str, tuple[float, float]] = {}


def design(frame: pd.DataFrame) -> pd.DataFrame:
    X = frame.reindex(columns=FEATURES).astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    for column in X.columns:
        if column not in FEATURE_CLIPS:
            continue
        low, high = FEATURE_CLIPS[column]
        X[column] = X[column].clip(low, high)
    for position in fx.PLAYER_SCORING_POSITIONS:
        X[f"pos_{position}"] = frame["model_position"].eq(position).astype(float)
    return X.replace([np.inf, -np.inf], np.nan)


def clf(C=1.0):
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("s", StandardScaler()),
                     ("m", LogisticRegression(C=C, max_iter=3000))])


def reg(alpha=10.0):
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("s", StandardScaler()), ("m", Ridge(alpha=alpha))])


def pois(alpha):
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("s", StandardScaler()),
                     ("m", PoissonRegressor(alpha=alpha, max_iter=3000))])


def build_forecast(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    Xtr, Xte = design(train), design(test)
    out = pd.DataFrame(index=test.index)
    out["model_position"] = test["model_position"].to_numpy()

    appear = clf().fit(Xtr, train["target_appeared"])
    p_app = appear.predict_proba(Xte)[:, 1]
    started = train["target_appeared"].eq(1)
    start = clf().fit(Xtr[started.to_numpy()], train.loc[started, "target_started"])
    p_start = p_app * start.predict_proba(Xte)[:, 1]
    starters = train["target_started"].eq(1)
    sixty = clf().fit(Xtr[starters.to_numpy()], train.loc[starters, "target_sixty"])
    p_sixty = p_start * sixty.predict_proba(Xte)[:, 1]

    m_start = reg().fit(Xtr[starters.to_numpy()], train.loc[starters, "target_minutes"])
    subs = train["target_appeared"].eq(1) & train["target_started"].eq(0)
    m_sub = reg().fit(Xtr[subs.to_numpy()], train.loc[subs, "target_minutes"])
    mins_start = np.clip(m_start.predict(Xte), 1, 90)
    mins_sub = np.clip(m_sub.predict(Xte), 1, 90)

    out["appearance_probability"] = p_app
    out["start_probability"] = p_start
    out["sixty_plus_probability"] = p_sixty
    out["minutes_if_start"] = mins_start
    out["expected_minutes"] = np.clip(
        p_start * mins_start + (p_app - p_start) * mins_sub, 0, 90
    )

    played = train["target_minutes"].gt(0)
    weight = (train.loc[played, "target_minutes"] / 90.0).clip(0.05, 1.0)
    specs = {"xg": (0.4, 2.0, ["DEF", "MID", "FWD"]),
             "xa": (0.4, 2.0, ["DEF", "MID", "FWD"]),
             "saves": (2.0, 12.0, ["GK"]),
             "bonus": (2.0, 3.0, fx.PLAYER_SCORING_POSITIONS),
             "yellow_cards": (4.0, 1.5, fx.PLAYER_SCORING_POSITIONS),
             "red_cards": (10.0, 0.5, fx.PLAYER_SCORING_POSITIONS)}
    for name, (alpha, cap, positions) in specs.items():
        col = f"target_{name}_per90"
        ok = played & train[col].notna() & train["model_position"].isin(positions)
        pred = np.zeros(len(test))
        if ok.sum() > 50:
            model = pois(alpha).fit(
                Xtr[ok.to_numpy()], train.loc[ok, col].clip(0, cap),
                m__sample_weight=weight[ok[played].to_numpy()],
            )
            pred = np.clip(model.predict(Xte), 0, cap)
        pred = np.where(test["model_position"].isin(positions), pred, 0.0)
        out[f"predicted_{name}_per90"] = pred

    out["expected_team_goals_conceded"] = test["team_gc_prior"].to_numpy()
    out["team_clean_sheet_probability"] = np.exp(-out["expected_team_goals_conceded"])
    out["prior_season_fixture_count"] = test["appearances_so_far"].fillna(0).to_numpy()
    out["prior_career_fixture_count"] = out["prior_season_fixture_count"]
    out["price"] = (test["value"] / 10.0).to_numpy()
    out["actual_points"] = test["target_fpl_points"].to_numpy()
    out["gameweek"] = test["gameweek"].to_numpy()
    out["element"] = test["element"].to_numpy()
    return out


# --------------------------------------------------------------------------
# xP assembly (mirrors calculate_component_expected_points, DC excluded)
# --------------------------------------------------------------------------

def expected_floor_groups(rate, group, max_count=40):
    rate = np.clip(np.asarray(rate, float), 0, 20)
    counts = np.arange(max_count + 1)
    probs = poisson.pmf(counts[:, None], rate[None, :])
    return (np.floor_divide(counts, group).astype(float)[:, None] * probs).sum(0)


def components(frame: pd.DataFrame, goal_points: dict) -> pd.DataFrame:
    pos = frame["model_position"]
    mins = frame["expected_minutes"].clip(0, 90)
    frac = mins / 90.0
    r = pd.DataFrame(index=frame.index)
    r["xp_appearance"] = frame["appearance_probability"] + frame["sixty_plus_probability"]
    r["projected_goals"] = frame["predicted_xg_per90"] * frac
    r["projected_assists"] = frame["predicted_xa_per90"] * frac
    r["xp_goals"] = r["projected_goals"] * pos.map(goal_points).astype(float)
    r["xp_assists"] = 3.0 * r["projected_assists"]
    lam = frame["expected_team_goals_conceded"].clip(lower=0)
    exposure = frame["minutes_if_start"].clip(60, 90)
    r["xp_clean_sheet"] = (
        frame["sixty_plus_probability"]
        * np.exp(-lam * exposure / 90.0)
        * pos.map(CLEAN_SHEET_POINTS).astype(float)
    )
    saves = (frame["predicted_saves_per90"] * frac).where(pos.eq("GK"), 0.0)
    r["xp_saves"] = expected_floor_groups(saves, 3)
    conceded = lam * frac
    r["xp_goals_conceded"] = np.where(
        pos.isin(["GK", "DEF"]), -expected_floor_groups(conceded, 2), 0.0
    )
    r["xp_bonus"] = (frame["predicted_bonus_per90"] * frac).clip(0, 3)
    r["xp_yellow_cards"] = -(frame["predicted_yellow_cards_per90"] * frac)
    r["xp_red_cards"] = -3.0 * (frame["predicted_red_cards_per90"] * frac)
    r["expected_points"] = r[[c for c in r.columns if c.startswith("xp_")]].sum(axis=1)
    return r


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def ndcg_at_k(pred, actual, k=10):
    order = np.argsort(-pred)
    gain = np.clip(actual[order][:k], 0, None)
    disc = 1.0 / np.log2(np.arange(2, len(gain) + 2))
    ideal = np.clip(np.sort(actual)[::-1][:k], 0, None)
    denom = (ideal * disc[: len(ideal)]).sum()
    return float((gain * disc).sum() / denom) if denom > 0 else np.nan


def evaluate(frame: pd.DataFrame, xp: pd.Series, label: str) -> dict:
    actual = frame["actual_points"].to_numpy(float)
    pred = xp.to_numpy(float)
    sp, nd, top = [], [], []
    for _, g in frame.assign(_p=pred).groupby("gameweek"):
        if len(g) < 30:
            continue
        a = g["actual_points"].to_numpy(float)
        p = g["_p"].to_numpy(float)
        sp.append(spearmanr(p, a).statistic)
        nd.append(ndcg_at_k(p, a))
        top.append(a[np.argsort(-p)][:10].sum())
    return {
        "variant": label,
        "MAE": float(np.mean(np.abs(pred - actual))),
        "bias": float(np.mean(pred - actual)),
        "spearman": float(np.nanmean(sp)),
        "ndcg@10": float(np.nanmean(nd)),
        "top10_pts": float(np.mean(top)),
    }


def bias_table(frame: pd.DataFrame, xp: pd.Series) -> pd.DataFrame:
    bucket = pd.cut(xp, [-99, 1, 2, 3, 4, 5, 6, 99],
                    labels=["<1", "1-2", "2-3", "3-4", "4-5", "5-6", "6+"])
    return (
        pd.DataFrame({"b": bucket, "xp": xp, "a": frame["actual_points"]})
        .groupby("b", observed=True)
        .agg(n=("xp", "size"), mean_xp=("xp", "mean"),
             mean_actual=("a", "mean"))
        .assign(bias=lambda f: f.mean_xp - f.mean_actual)
    )


def main(test_season="2024-25"):
    train_seasons = [s for s in SEASONS if s < test_season]
    print("=" * 78)
    print(f"Train: {train_seasons}   Test (held out): {test_season}")
    print("=" * 78)

    raw = add_history_features(add_team_strength(load_seasons(mask_coverage=True)))
    train = raw[raw["season"].isin(train_seasons)].copy()
    test = raw[raw["season"].eq(test_season)].copy()
    test = test[test["gameweek"] >= 6].copy()
    print(f"train rows {len(train):,}   test rows {len(test):,}\n")

    FEATURE_CLIPS.clear()
    train_design = train.reindex(columns=FEATURES).astype(float).replace(
        [np.inf, -np.inf], np.nan
    )
    for column in FEATURES:
        series = train_design[column].dropna()
        if len(series) > 100:
            FEATURE_CLIPS[column] = (
                float(series.quantile(0.001)), float(series.quantile(0.999))
            )

    fc = build_forecast(train, test).reset_index(drop=True)

    assist_factors = fx.fit_event_conversion_factors(
        train, realised_col="assists", expected_col="expected_assists",
        bounds=fx.ASSIST_CONVERSION_BOUNDS,
    )
    goal_factors = fx.fit_event_conversion_factors(
        train, realised_col="goals_scored", expected_col="expected_goals",
        bounds=fx.GOAL_CONVERSION_BOUNDS,
    )
    print("Conversion factors fitted on TRAIN seasons only:")
    print("  assists/xA :", {k: round(v, 3) for k, v in assist_factors.items()})
    print("  goals/xG   :", {k: round(v, 3) for k, v in goal_factors.items()}, "\n")

    base = components(fc, GOAL_POINTS_BASELINE)
    fixed = fx.apply_event_conversion(
        base.copy(), fc, assist_factors=assist_factors,
        goal_factors=goal_factors, goal_points=fx.GOAL_POINTS_BY_POSITION,
    )

    rows = [evaluate(fc, base["expected_points"], "baseline (current notebook)"),
            evaluate(fc, fixed["expected_points"], "v2.6 fixed")]
    res = pd.DataFrame(rows).set_index("variant")
    print("HELD-OUT SEASON RESULTS")
    print(res.round(4).to_string(), "\n")

    d = res.loc["v2.6 fixed"] - res.loc["baseline (current notebook)"]
    gates = {
        "G1 |bias| -> 0": abs(res.loc["v2.6 fixed", "bias"]) < abs(res.loc["baseline (current notebook)", "bias"]),
        "G2 MAE not worse by >0.02": d["MAE"] <= 0.02,
        "G3 Spearman not worse by >0.003": d["spearman"] >= -0.003,
        "G4 NDCG@10 not worse by >0.002": d["ndcg@10"] >= -0.002,
        "G5 top-10 not worse by >0.25": d["top10_pts"] >= -0.25,
    }
    print("ACCEPTANCE GATES (declared before inspecting results)")
    for name, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  deltas: MAE {d['MAE']:+.4f}  bias {d['bias']:+.4f}  "
          f"spearman {d['spearman']:+.4f}  ndcg {d['ndcg@10']:+.4f}  "
          f"top10 {d['top10_pts']:+.3f}")
    print(f"\n  OVERALL: {'ACCEPT' if all(gates.values()) else 'REJECT'}\n")

    print("BIAS BY PREDICTED-xP BUCKET (reproduces audit section 7.5)")
    print("\n-- baseline --"); print(bias_table(fc, base["expected_points"]).round(3).to_string())
    print("\n-- v2.6 fixed --"); print(bias_table(fc, fixed["expected_points"]).round(3).to_string())
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2024-25")
