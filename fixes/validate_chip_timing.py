#!/usr/bin/env python3
"""
Chip-timing validation — the experiment the project never ran.

The audit log validates the transfer policy but never the chip policy, and the
policy scorecard that selected greedy_1gw_no_hits ran with
WALK_FORWARD_INCLUDE_CHIPS = False. The notebook's chip thresholds
(TC 7.0, BB 9.0, FH 8.0, WC 10.0) are hand-chosen and have never been scored
against any alternative.

This measures how much chip TIMING is worth, by simulating a season-long squad
and then comparing timing policies on realised points.

Method
------
1.  Forecast every player-gameweek with the v2.6 component model.
2.  Simulate a squad: 15 players, GBP 100.0m, max 3 per club, 2/5/5/3, then one
    greedy no-hit transfer per gameweek on next-gameweek xP. This mirrors the
    validated greedy_1gw_no_hits policy.
3.  For every gameweek, record both the PREDICTED and the REALISED value of
    each chip given that week's squad.
4.  Decide with predicted value (what a policy could actually know), score with
    realised value.

Chips covered: Triple Captain, Bench Boost, Free Hit.
Wildcard is EXCLUDED: its value is multi-week and inseparable from the transfer
optimiser, so it cannot be scored this way without a full squad re-planner.

Chip rules modelled: two sets (first expires GW19), one chip per gameweek,
Free Hit not in consecutive gameweeks.

Policies compared
-----------------
  oracle            perfect hindsight (upper bound)
  notebook rule     threshold + structural trigger + expiry pressure
  greedy threshold  play as soon as predicted value >= threshold
  best predicted    play in the week with the highest predicted value in the half
  forced at expiry  what the notebook rule degenerates to with no trigger
  random legal      mean over random legal gameweeks (floor)
  never             chips unused

Run:  python validate_chip_timing.py [2024-25|2025-26]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fpl_xp_fixes as fx  # noqa: E402
import validate_xp_fixes as V  # noqa: E402
import squad_optimizer as SO  # noqa: E402

BUDGET = 1000               # tenths of a million
SQUAD = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
FORMATIONS = [(3, 5, 2), (3, 4, 3), (4, 5, 1), (4, 4, 2), (4, 3, 3),
              (5, 4, 1), (5, 3, 2), (5, 2, 3)]
MAX_PER_CLUB = 3
FIRST_HALF_END = 19
THRESHOLD = {"triple_captain": 7.0, "bench_boost": 9.0, "free_hit": 8.0}
EXCEPTIONAL = {"triple_captain": 12.0, "bench_boost": 14.0, "free_hit": 13.0}


# ---------------------------------------------------------------- forecasting

# The chip and wildcard studies were originally run on forecasts WITH the v2.6
# assist conversion applied. That correction later failed its gate on the real
# pipeline, so the chip conclusions need re-checking without it.
APPLY_ASSIST_CONVERSION = True


def build_panel(test_season: str) -> pd.DataFrame:
    train_seasons = [s for s in V.SEASONS if s < test_season]
    raw = V.add_history_features(V.add_team_strength(V.load_seasons(mask_coverage=True)))
    train = raw[raw["season"].isin(train_seasons)].copy()
    test = raw[raw["season"].eq(test_season)].copy()
    test = test[test["gameweek"] >= 6].copy()

    V.FEATURE_CLIPS.clear()
    td = train.reindex(columns=V.FEATURES).astype(float).replace([np.inf, -np.inf], np.nan)
    for c in V.FEATURES:
        col = td[c].dropna()
        if len(col) > 100:
            V.FEATURE_CLIPS[c] = (float(col.quantile(0.001)), float(col.quantile(0.999)))

    fc = V.build_forecast(train, test).reset_index(drop=True)
    af = fx.fit_event_conversion_factors(
        train, realised_col="assists", expected_col="expected_assists",
        bounds=fx.ASSIST_CONVERSION_BOUNDS,
    )
    comp = fx.apply_event_conversion(
        V.components(fc, V.GOAL_POINTS_BASELINE), fc,
        assist_factors=af if APPLY_ASSIST_CONVERSION else None,
    )
    fc["xp"] = comp["expected_points"].to_numpy(float)
    fc["team"] = test["team"].to_numpy()
    fc["cost"] = (pd.to_numeric(test["value"], errors="coerce").fillna(40)).to_numpy()
    fc["fixture"] = test["fixture"].to_numpy()

    # Double/blank gameweeks: fixtures per club per gameweek.
    shape = (fc.groupby(["gameweek", "team"])["fixture"].nunique()
             .rename("fixtures").reset_index())
    fc = fc.merge(shape, on=["gameweek", "team"], how="left")

    # One row per player-gameweek (a double gameweek sums both fixtures).
    panel = (fc.groupby(["gameweek", "element"], as_index=False)
             .agg(xp=("xp", "sum"), actual_points=("actual_points", "sum"),
                  model_position=("model_position", "first"),
                  team=("team", "first"), cost=("cost", "first"),
                  fixtures=("fixtures", "first")))
    return panel


# ----------------------------------------------------------- squad simulation

def pick_initial_squad(panel: pd.DataFrame, gw: int) -> list[int]:
    """Exact MILP squad over the opening horizon (was a greedy value pick)."""
    gws = sorted(panel["gameweek"].unique())[:6]
    squad, _ = SO.optimise_squad(panel, gws, budget=BUDGET)
    if len(squad) == sum(SQUAD.values()):
        return squad
    return _greedy_initial_squad(panel, gw)


def _greedy_initial_squad(panel: pd.DataFrame, gw: int) -> list[int]:
    """Fallback if the solver is infeasible on a degenerate pool."""
    strength = (panel.groupby("element")
                .agg(xp=("xp", "mean"), cost=("cost", "first"),
                     pos=("model_position", "first"), team=("team", "first"))
                .reset_index())
    strength = strength[strength["cost"] > 0]
    strength["eff"] = strength["xp"] / strength["cost"]

    squad: list[int] = []
    spend, per_club, per_pos = 0, {}, {k: 0 for k in SQUAD}
    # Two passes: efficiency first to bank funds, then raw xP for the rest.
    for key in ("eff", "xp"):
        for _, r in strength.sort_values(key, ascending=False).iterrows():
            pos, team, cost = r["pos"], r["team"], int(r["cost"])
            if r["element"] in squad or per_pos[pos] >= SQUAD[pos]:
                continue
            if per_club.get(team, 0) >= MAX_PER_CLUB:
                continue
            remaining = sum(SQUAD.values()) - len(squad) - 1
            if spend + cost + 40 * remaining > BUDGET:
                continue
            squad.append(int(r["element"]))
            spend += cost
            per_club[team] = per_club.get(team, 0) + 1
            per_pos[pos] += 1
            if len(squad) == sum(SQUAD.values()):
                return squad
    return squad


def best_xi(rows: pd.DataFrame, col: str):
    """Exact best legal XI; delegates to the shared optimiser."""
    return SO.best_xi_from_squad(rows, col)


def _best_xi_legacy(rows: pd.DataFrame, col: str):
    """Previous enumeration, kept for reference."""
    by = {p: rows[rows["model_position"].eq(p)].sort_values(col, ascending=False)
          for p in SQUAD}
    if by["GK"].empty:
        return rows.head(0), rows
    gk = by["GK"].iloc[[0]]
    best, best_val = None, -np.inf
    for d, m, f in FORMATIONS:
        if len(by["DEF"]) < d or len(by["MID"]) < m or len(by["FWD"]) < f:
            continue
        xi = pd.concat([by["DEF"].head(d), by["MID"].head(m), by["FWD"].head(f)])
        val = float(xi[col].sum()) + float(gk[col].iloc[0])
        if val > best_val:
            best_val, best = val, pd.concat([gk, xi])
    if best is None:
        return rows.head(0), rows
    bench = rows[~rows["element"].isin(best["element"])]
    return best, bench


def simulate_season(panel: pd.DataFrame) -> pd.DataFrame:
    """Greedy 1-transfer-per-week, no hits. Returns per-gameweek chip values."""
    gws = sorted(panel["gameweek"].unique())
    squad = pick_initial_squad(panel[panel["gameweek"] <= gws[0] + 4], gws[0])
    bank = BUDGET - int(panel[panel["element"].isin(squad)]
                        .groupby("element")["cost"].first().sum())
    out = []

    for gw in gws:
        week = panel[panel["gameweek"].eq(gw)]
        held = week[week["element"].isin(squad)]
        if len(held) < 11:
            out.append({"gameweek": gw}); continue

        xi_p, bench_p = best_xi(held, "xp")            # decided on forecast
        if xi_p.empty:
            # A blank gameweek can leave the squad unable to field a legal XI
            # (fewer than 1 GK / 3 DEF / 2 MID / 1 FWD with a fixture), even
            # when 11+ players nominally remain. Record the week as unusable.
            out.append({"gameweek": gw})
            continue
        cap_p = xi_p.loc[xi_p["xp"].idxmax()]

        # realised value of each chip, given this week's squad
        tc_real = float(cap_p["actual_points"])        # one extra captain copy
        bb_real = float(bench_p["actual_points"].sum())
        pool_xi_pred, _ = best_xi(week, "xp")
        if pool_xi_pred.empty:
            out.append({"gameweek": gw})
            continue
        fh_real = max(0.0, float(pool_xi_pred["actual_points"].sum())
                      + float(pool_xi_pred.loc[pool_xi_pred["xp"].idxmax(), "actual_points"])
                      - (float(xi_p["actual_points"].sum()) + tc_real))

        # predicted value, which is all a policy can act on
        tc_pred = float(cap_p["xp"])
        bb_pred = float(bench_p["xp"].sum())
        fh_pred = max(0.0, float(pool_xi_pred["xp"].sum())
                      + float(pool_xi_pred["xp"].max())
                      - (float(xi_p["xp"].sum()) + tc_pred))

        out.append({
            "gameweek": gw,
            "doubles": int((held["fixtures"] > 1).sum()),
            "blanks": int((held["fixtures"] < 1).sum()),
            "triple_captain_pred": tc_pred, "triple_captain_real": tc_real,
            "bench_boost_pred": bb_pred, "bench_boost_real": bb_real,
            "free_hit_pred": fh_pred, "free_hit_real": fh_real,
        })

        # one greedy no-hit transfer on next gameweek's forecast
        nxt = panel[panel["gameweek"].eq(gw + 1)]
        if not nxt.empty:
            owned = nxt[nxt["element"].isin(squad)]
            costs = panel.groupby("element")["cost"].first()
            clubs = panel.groupby("element")["team"].first()
            best_gain, best_swap = 0.30, None
            for _, outp in owned.iterrows():
                cand = nxt[nxt["model_position"].eq(outp["model_position"])
                           & ~nxt["element"].isin(squad)]
                budget = bank + int(costs.get(outp["element"], 40))
                cand = cand[cand["cost"].le(budget)]
                if cand.empty:
                    continue
                top = cand.loc[cand["xp"].idxmax()]
                counts = clubs.reindex(squad).value_counts()
                c_out, c_in = clubs.get(outp["element"]), top["team"]
                if c_in != c_out and counts.get(c_in, 0) >= MAX_PER_CLUB:
                    continue
                gain = float(top["xp"]) - float(outp["xp"])
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (int(outp["element"]), int(top["element"]),
                                 int(costs.get(outp["element"], 40)), int(top["cost"]))
            if best_swap:
                o, i, co, ci = best_swap
                squad = [i if e == o else e for e in squad]
                bank += co - ci
    return pd.DataFrame(out).dropna(subset=["gameweek"]).reset_index(drop=True)


# ------------------------------------------------------------------- policies

def legal_weeks(weeks: pd.DataFrame, half: str) -> pd.DataFrame:
    return (weeks[weeks["gameweek"] <= FIRST_HALF_END] if half == "first"
            else weeks[weeks["gameweek"] > FIRST_HALF_END])


def notebook_rule(w: pd.DataFrame, chip: str, n_chips: int, half_end: int) -> int | None:
    """Threshold + structural trigger + expiry pressure, as the notebook does."""
    for _, r in w.sort_values("gameweek").iterrows():
        gw = int(r["gameweek"])
        pred = float(r[f"{chip}_pred"])
        remaining_weeks = max(1, half_end - gw + 1)
        expiry = n_chips >= remaining_weeks
        exceptional = pred >= EXCEPTIONAL[chip]
        if chip in ("triple_captain", "bench_boost"):
            structural = r["doubles"] > 0 or expiry or exceptional
        else:
            structural = r["blanks"] > 0 or r["doubles"] > 0 or expiry or exceptional
        if (pred >= THRESHOLD[chip] or expiry) and structural:
            return gw
    return None


def evaluate(weeks: pd.DataFrame, chip: str, rng) -> dict:
    res = {}
    for half, half_end in (("first", FIRST_HALF_END), ("second", 38)):
        w = legal_weeks(weeks, half)
        if w.empty:
            continue
        real, pred = f"{chip}_real", f"{chip}_pred"
        scores = {
            "oracle": float(w[real].max()),
            "notebook rule": None,
            "greedy threshold": None,
            "best predicted": float(w.loc[w[pred].idxmax(), real]),
            "forced at expiry": float(w.loc[w["gameweek"].idxmax(), real]),
            "random legal": float(w[real].mean()),
            "never": 0.0,
        }
        gw_rule = notebook_rule(w, chip, 4, half_end)
        scores["notebook rule"] = (float(w.loc[w["gameweek"].eq(gw_rule), real].iloc[0])
                                   if gw_rule is not None
                                   else float(w.loc[w["gameweek"].idxmax(), real]))
        hit = w[w[pred] >= THRESHOLD[chip]].sort_values("gameweek")
        scores["greedy threshold"] = (float(hit.iloc[0][real]) if not hit.empty
                                      else float(w.loc[w["gameweek"].idxmax(), real]))
        res[half] = scores
    return res


def run_season(test_season: str) -> tuple[dict, dict]:
    rng = np.random.default_rng(0)
    print("=" * 80)
    print(f"CHIP-TIMING VALIDATION — {test_season}")
    print("=" * 80)
    panel = build_panel(test_season)
    weeks = simulate_season(panel)
    print(f"\nsimulated gameweeks: {len(weeks)}   "
          f"double-gameweek squad-players observed: {int(weeks['doubles'].sum())}")

    chips = ["triple_captain", "bench_boost", "free_hit"]
    totals: dict[str, float] = {}
    halves: dict[str, dict[str, float]] = {"first": {}, "second": {}}
    for chip in chips:
        res = evaluate(weeks, chip, rng)
        print(f"\n--- {chip.replace('_',' ').title()} "
              f"(realised points from playing it in the chosen week) ---")
        table = pd.DataFrame(res).round(1)
        print(table.to_string())
        for policy in table.index:
            totals[policy] = totals.get(policy, 0.0) + float(table.loc[policy].sum())
            for half in table.columns:
                halves[half][policy] = halves[half].get(policy, 0.0) + float(table.loc[policy, half])

    print("\n" + "=" * 80)
    print("SEASON TOTAL across Triple Captain + Bench Boost + Free Hit, both halves")
    print("(Wildcard excluded — its value is multi-week and not separable here)")
    print("=" * 80)
    summary = (pd.Series(totals, name="realised points")
               .sort_values(ascending=False).to_frame())
    summary["vs notebook rule"] = summary["realised points"] - totals["notebook rule"]
    print(summary.round(1).to_string())
    gap = totals["oracle"] - totals["notebook rule"]
    print(f"\n  points left on the table by the current rule: {gap:.0f}")
    print(f"  (6 chip decisions; oracle is an upper bound that needs hindsight)")
    print("\n  By half — the first chip set has no double or blank gameweek to")
    print("  aim at in either season, so its structural trigger cannot fire:")
    print(pd.DataFrame(halves).round(1).to_string())
    return totals, halves


def main() -> None:
    seasons = sys.argv[1:] or ["2024-25", "2025-26"]
    collected, half_collected = {}, {}
    for season in seasons:
        totals, halves = run_season(season)
        collected[season] = totals
        half_collected[season] = halves
        print()

    if len(collected) < 2:
        return
    print("=" * 80)
    print("COMBINED — mean realised chip points per season")
    print("=" * 80)
    summary = pd.DataFrame(collected)
    summary["mean"] = summary.mean(axis=1)
    summary = summary.sort_values("mean", ascending=False)
    summary["vs never"] = summary["mean"] - summary.loc["never", "mean"]
    summary["vs random"] = summary["mean"] - summary.loc["random legal", "mean"]
    print(summary.round(1).to_string())

    rule = summary.loc["notebook rule", "mean"]
    print(f"\n  Chips are worth {rule - summary.loc['never','mean']:.0f} points a season "
          f"under the current rule (vs not playing them at all).")
    print(f"  The rule beats random legal timing by "
          f"{rule - summary.loc['random legal','mean']:+.0f} on average, but it was "
          f"WORSE than random in one of the two seasons —")
    print(f"  so that edge is not reliable.")
    print(f"  Hindsight oracle leaves {summary.loc['oracle','mean'] - rule:.0f} "
          f"points a season unclaimed, most of which is not forecastable.")
    print("\n  First half vs second half, mean over seasons:")
    fh = pd.DataFrame({s: h["first"] for s, h in half_collected.items()}).mean(axis=1)
    sh = pd.DataFrame({s: h["second"] for s, h in half_collected.items()}).mean(axis=1)
    print(pd.DataFrame({"first half": fh, "second half": sh}).round(1).to_string())


if __name__ == "__main__":
    main()
