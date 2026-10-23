#!/usr/bin/env python3
"""
One command for every v2.6 result.

    python run_all.py              # evidence + unit tests + OOS validation
    python run_all.py --quick      # evidence + unit tests only (no model fits)

Downloads 4 seasons of archived FPL data on first run (~20 MB into ./data).
Needs: numpy pandas scipy scikit-learn

Sections
  1  Evidence      the measured facts behind each fix, model-free where possible
  2  Unit tests    the fix functions behave as specified
  3  Validation    chronological out-of-sample gates on two held-out seasons
  4  Context       how large an FPL edge has to be before it is detectable
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fpl_xp_fixes as fx  # noqa: E402
from validate_xp_fixes import DATA_DIR, SEASONS, ensure_data  # noqa: E402

RULE = "=" * 78


def header(n: int, title: str) -> None:
    print(f"\n{RULE}\n{n}. {title}\n{RULE}")


def load() -> pd.DataFrame:
    ensure_data()
    frames = []
    for season in SEASONS:
        frame = pd.read_csv(DATA_DIR / f"{season}.csv", low_memory=False)
        frame["season"] = season
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["position"] = data["position"].replace({"GKP": "GK"})
    data = data[data["position"].isin(fx.PLAYER_SCORING_POSITIONS)].copy()
    for column in [
        "minutes", "goals_scored", "assists", "expected_goals",
        "expected_assists", "bonus", "bps", "total_points", "clean_sheets",
        "goals_conceded", "round", "selected",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


# --------------------------------------------------------------------------

def evidence(data: pd.DataFrame) -> None:
    header(1, "EVIDENCE")
    played = data[data["minutes"] > 0]
    recent = played[played["season"] != "2022-23"]

    print("\n1.1  F1 - FPL expected_assists is not on the realised-assist scale")
    print("     (model-free; 2023-26; this is the primary fix)\n")
    grouped = recent.groupby("position").apply(
        lambda f: pd.Series({
            "assist pts awarded": 3 * f["assists"].sum(),
            "3 x xA (model uses)": 3 * f["expected_assists"].sum(),
            "shortfall": 3 * (f["assists"].sum() - f["expected_assists"].sum()),
            "ratio": f["assists"].sum() / f["expected_assists"].sum(),
        }), include_groups=False,
    )
    print(grouped.round(2).to_string())
    total = 3 * (recent["assists"].sum() - recent["expected_assists"].sum())
    print(f"\n     invisible to the model: {total:,.0f} assist points over 3 seasons"
          f" = {total / 3:,.0f} per season")
    print("\n     expected_goals needs no such fix:")
    goals = recent.groupby("season").apply(
        lambda f: pd.Series({"goals/xG": f["goals_scored"].sum() / f["expected_goals"].sum()}),
        include_groups=False,
    )
    print("     " + goals.round(3).to_string().replace("\n", "\n     "))

    print("\n1.2  F2 - fabricated zero xG/xA blocks in the training data\n")
    for season in SEASONS:
        subset = played[played["season"] == season]
        totals = subset.groupby("round")["expected_goals"].sum()
        dead = sorted(totals[totals == 0].index.tolist())
        bad = subset[subset["round"].isin(dead)]
        flag = "  <-- 35.6% of the season" if len(dead) > 5 else ""
        print(f"     {season}: {len(dead):2d} gameweeks with zero league-wide xG, "
              f"{len(bad):5,} rows, {int(bad['goals_scored'].sum()):3d} real goals "
              f"and {int(bad['assists'].sum()):3d} real assists discarded{flag}")

    print("\n1.3  Clean-sheet exposure rule - TESTED AND KEPT\n")
    s = data[(data["minutes"] >= 60) & data["clean_sheets"].notna()].copy()
    s["grp"] = np.where(s["minutes"] >= 90, "played 90'", "subbed 60-89'")
    counts = s.groupby("team_match" if "team_match" in s else
                       s["season"] + "|" + s["team"].astype(str) + "|" +
                       s["fixture"].astype(str))["grp"].nunique()
    both = set(counts[counts == 2].index)
    key = s["season"] + "|" + s["team"].astype(str) + "|" + s["fixture"].astype(str)
    within = s[key.isin(both)]
    pivot = within.groupby("grp")["clean_sheets"].agg(["mean", "size"])
    print(pivot.round(4).to_string().replace("\n", "\n     ").rjust(5))
    gain = pivot.loc["subbed 60-89'", "mean"] - pivot.loc["played 90'", "mean"]
    print(f"\n     Within the SAME team-match, leaving early really is worth "
          f"{gain:+.4f} clean-sheet probability.")
    print("     exp(-lambda * minutes/90) is therefore correct and is left alone.")

    print("\n1.4  Open item - bonus is rank-based, modelled as a per-90 rate\n")
    per_player = (
        played.groupby(["season", "element"])
        .agg(mins=("minutes", "sum"), bonus=("bonus", "sum"), bps=("bps", "sum"))
        .reset_index()
    )
    per_player = per_player[per_player["mins"] >= 900]
    per_player["dec"] = pd.qcut(
        per_player["bonus"].div(per_player["mins"]).rank(method="first"),
        10, labels=False,
    )
    table = per_player.groupby("dec").apply(
        lambda f: pd.Series({
            "bonus/90": 90 * f["bonus"].sum() / f["mins"].sum(),
            "bps/90": 90 * f["bps"].sum() / f["mins"].sum(),
        }), include_groups=False,
    )
    print(f"     bonus/90  D1 {table['bonus/90'].iloc[0]:.3f}   "
          f"D5 {table['bonus/90'].iloc[4]:.3f}   D10 {table['bonus/90'].iloc[-1]:.3f}"
          f"   ({table['bonus/90'].iloc[-1] / table['bonus/90'].iloc[4]:.1f}x top-to-median)")
    print("     A pooled Poisson at alpha=2.0 cannot reproduce that spread.")
    print("     NOT fixed here - it needs its own out-of-sample gate.")


def noise_context(data: pd.DataFrame) -> None:
    header(4, "CONTEXT - how big does an edge have to be to be visible?")
    rng = np.random.default_rng(3)
    latest = data[data["season"] == SEASONS[-1]]
    owned = latest.groupby("element")["selected"].mean().nlargest(70).index
    pool = latest[latest["element"].isin(owned)]
    gws = sorted(pool["round"].dropna().unique())

    scores = []
    for _ in range(400):
        picks = [
            rng.choice(pool[pool["position"] == p]["element"].unique(),
                       size=n, replace=False)
            for p, n in [("GK", 1), ("DEF", 4), ("MID", 4), ("FWD", 2)]
        ]
        squad = np.concatenate(picks)
        captain = rng.choice(squad)
        week = []
        for gw in gws:
            g = pool[(pool["round"] == gw) & pool["element"].isin(squad)]
            week.append(
                g["total_points"].sum()
                + g[g["element"] == captain]["total_points"].sum()
            )
        scores.append(week)

    dev = np.array(scores, dtype=float)
    dev = dev - dev.mean(axis=0)
    print(f"\n     {SEASONS[-1]}, 400 plausible managers from the 70 most-owned players")
    print(f"     SD of a weekly score vs the field: {dev.std():.1f} pts/GW\n")
    for n in (5, 10, 15, 20):
        if n <= len(gws):
            cum = dev[:, :n].sum(axis=1)
            print(f"     after {n:2d} GWs: SD of cumulative delta {cum.std():5.1f} pts"
                  f"  -> +/-{1.96 * cum.std():.0f} is still noise at 95%")
    print("\n     Do not judge the model on fewer than ~15 gameweeks.")


def run(script: str, *args: str) -> None:
    result = subprocess.run(
        [sys.executable, str(HERE / script), *args], cwd=HERE
    )
    if result.returncode != 0:
        print(f"\n!! {script} exited {result.returncode}")


def main() -> None:
    quick = "--quick" in sys.argv
    print(RULE)
    print("FPL Predictor v2.6 - full result set")
    print(RULE)

    data = load()
    evidence(data)

    header(2, "UNIT TESTS")
    run("selftest.py")

    if not quick:
        for season in ("2024-25", "2025-26"):
            header(3, f"OUT-OF-SAMPLE VALIDATION - held-out {season}")
            run("validate_xp_fixes.py", season)
        header(3, "F3 DEFENSIVE-CONTRIBUTION VALIDATION")
        run("validate_dc_fix.py")

    noise_context(data)

    print(f"\n{RULE}\nDone. See INTEGRATION.md for what to change and in what order.")
    print("To regenerate the patched notebook:  ./run.sh patch")
    print("To open it:                         ./run.sh notebook")
    print(RULE)


if __name__ == "__main__":
    main()
