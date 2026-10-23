#!/usr/bin/env python3
"""
Wildcard timing, and whether first-half chip value is forecastable at all.

Two experiments the chip-timing study could not do.

PART A — is first-half chip value predictable by ANY signal?
    validate_chip_timing.py compared three timing policies and found none beat
    random before GW19. Three policies is not proof that no signal exists. This
    measures the correlation between realised chip value and every candidate
    signal available at decision time, split by half.

PART B — Wildcard
    Wildcard was excluded from the chip-timing study because its value is
    multi-week: it rebuilds the whole squad, and the benefit accrues over the
    following weeks. Here it is valued by simulating BOTH paths forward from
    each candidate gameweek:

        value(g) = realised points of (wildcard at g, then greedy)
                 - realised points of (greedy throughout)

    over a horizon of HORIZON gameweeks, capped at the end of the chip half.
    Timing policies are then compared on realised value, as before.

Run:  python validate_wildcard.py [2024-25 2025-26]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_chip_timing as C  # noqa: E402
import squad_optimizer as SO  # noqa: E402

HORIZON = 8
FIRST_HALF_END = C.FIRST_HALF_END


# ------------------------------------------------------------------- Part A

def forecastability(weeks: pd.DataFrame) -> pd.DataFrame:
    """Spearman between each decision-time signal and realised chip value."""
    rows = []
    for chip in ("triple_captain", "bench_boost", "free_hit"):
        for half, mask in (("first", weeks["gameweek"] <= FIRST_HALF_END),
                           ("second", weeks["gameweek"] > FIRST_HALF_END)):
            # Gameweeks where the squad had fewer than 11 available players
            # (blanks) carry NaN chip values and would poison the correlation.
            w = weeks[mask].dropna(subset=[f"{chip}_real", f"{chip}_pred"])
            if len(w) < 6:
                continue
            real = w[f"{chip}_real"]
            signals = {
                "predicted chip value": w[f"{chip}_pred"],
                "squad doubles this GW": w["doubles"],
                "gameweek index": w["gameweek"],
            }
            for name, sig in signals.items():
                if sig.nunique() < 2:
                    rho = np.nan
                else:
                    rho = spearmanr(sig, real).statistic
                rows.append({"chip": chip, "half": half, "signal": name,
                             "spearman_vs_realised": rho, "n": len(w)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- Part B

def squad_cost(panel: pd.DataFrame, squad: list[int]) -> int:
    return int(panel[panel["element"].isin(squad)]
               .groupby("element")["cost"].first().sum())


def rebuild_squad(panel: pd.DataFrame, gw: int, horizon: int, budget: int,
                  must_include: set[int] | None = None) -> list[int]:
    """Exact MILP rebuild over the wildcard horizon.

    A wildcard's entire value is the quality of the squad it builds, so this
    must be solved properly rather than approximated. Falls back to the old
    greedy only if the solver cannot find a feasible squad.
    """
    gws = [int(g) for g in sorted(panel["gameweek"].unique())
           if gw <= g < gw + horizon]
    if gws:
        squad, _ = SO.optimise_squad(panel, gws, budget=int(budget),
                                     must_include=must_include)
        if len(squad) == sum(C.SQUAD.values()):
            return squad
    return _greedy_rebuild(panel, gw, horizon, budget)


def _greedy_rebuild(panel: pd.DataFrame, gw: int, horizon: int, budget: int) -> list[int]:
    window = panel[(panel["gameweek"] >= gw) & (panel["gameweek"] < gw + horizon)]
    if window.empty:
        return []
    strength = (window.groupby("element")
                .agg(xp=("xp", "mean"), cost=("cost", "first"),
                     pos=("model_position", "first"), team=("team", "first"))
                .reset_index())
    strength = strength[strength["cost"] > 0]
    strength["eff"] = strength["xp"] / strength["cost"]

    squad, spend = [], 0
    per_club, per_pos = {}, {k: 0 for k in C.SQUAD}
    for key in ("eff", "xp"):
        for _, r in strength.sort_values(key, ascending=False).iterrows():
            pos, team, cost = r["pos"], r["team"], int(r["cost"])
            if r["element"] in squad or per_pos[pos] >= C.SQUAD[pos]:
                continue
            if per_club.get(team, 0) >= C.MAX_PER_CLUB:
                continue
            remaining = sum(C.SQUAD.values()) - len(squad) - 1
            if spend + cost + 40 * remaining > budget:
                continue
            squad.append(int(r["element"]))
            spend += cost
            per_club[team] = per_club.get(team, 0) + 1
            per_pos[pos] += 1
            if len(squad) == sum(C.SQUAD.values()):
                return squad
    return squad


def play_forward(panel: pd.DataFrame, squad: list[int], bank: int,
                 start_gw: int, end_gw: int, score_col: str = "actual_points") -> float:
    """Greedy 1-transfer-per-week from start_gw to end_gw.

    score_col="actual_points" scores the realised outcome; score_col="xp"
    scores what the forecast PROJECTED at decision time, which is what a
    wildcard policy can actually act on.
    """
    squad = list(squad)
    total = 0.0
    costs = panel.groupby("element")["cost"].first()
    clubs = panel.groupby("element")["team"].first()
    for gw in range(int(start_gw), int(end_gw) + 1):
        week = panel[panel["gameweek"].eq(gw)]
        if week.empty:
            continue
        held = week[week["element"].isin(squad)]
        if len(held) < 11:
            continue
        xi, _ = C.best_xi(held, "xp")
        if xi.empty:
            continue
        cap = xi.loc[xi["xp"].idxmax()]
        total += float(xi[score_col].sum()) + float(cap[score_col])

        nxt = panel[panel["gameweek"].eq(gw + 1)]
        if nxt.empty:
            continue
        owned = nxt[nxt["element"].isin(squad)]
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
            if c_in != c_out and counts.get(c_in, 0) >= C.MAX_PER_CLUB:
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
    return total


def squad_path(panel: pd.DataFrame) -> dict[int, tuple[list[int], int]]:
    """Baseline greedy path: squad and bank held at the start of each gameweek."""
    gws = sorted(panel["gameweek"].unique())
    squad = C.pick_initial_squad(panel[panel["gameweek"] <= gws[0] + 4], gws[0])
    bank = C.BUDGET - squad_cost(panel, squad)
    costs = panel.groupby("element")["cost"].first()
    clubs = panel.groupby("element")["team"].first()
    path = {}
    for gw in gws:
        path[gw] = (list(squad), bank)
        nxt = panel[panel["gameweek"].eq(gw + 1)]
        if nxt.empty:
            continue
        owned = nxt[nxt["element"].isin(squad)]
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
            if c_in != c_out and counts.get(c_in, 0) >= C.MAX_PER_CLUB:
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
    return path


def wildcard_values(panel: pd.DataFrame) -> pd.DataFrame:
    path = squad_path(panel)
    gws = sorted(panel["gameweek"].unique())
    rows = []
    for gw in gws:
        half_end = FIRST_HALF_END if gw <= FIRST_HALF_END else max(gws)
        end = min(gw + HORIZON - 1, half_end, max(gws))
        if end <= gw:
            continue
        squad, bank = path[gw]
        budget = squad_cost(panel, squad) + bank
        new_squad = rebuild_squad(panel, gw, end - gw + 1, budget,
                                  must_include=set(squad))
        if len(new_squad) < sum(C.SQUAD.values()):
            continue
        new_bank = budget - squad_cost(panel, new_squad)

        # Realised: what actually happened down each path.
        baseline_real = play_forward(panel, squad, bank, gw, end, "actual_points")
        wc_real = play_forward(panel, new_squad, new_bank, gw, end, "actual_points")

        # Predicted: what the forecast projected at the deadline. This is the
        # only quantity a real wildcard policy can decide on.
        baseline_pred = play_forward(panel, squad, bank, gw, end, "xp")
        wc_pred = play_forward(panel, new_squad, new_bank, gw, end, "xp")

        rows.append({"gameweek": gw, "horizon_end": end,
                     "wildcard_real": wc_real - baseline_real,
                     "wildcard_pred": wc_pred - baseline_pred})
    return pd.DataFrame(rows)


WILDCARD_THRESHOLD = 10.0      # the notebook's CHIP_MINIMUM_UPLIFT["wildcard"]
WILDCARD_EXCEPTIONAL = 18.0    # WEEKLY_CHIP_EXCEPTIONAL_UPLIFT["wildcard"]


def evaluate_wildcard(wc: pd.DataFrame) -> dict:
    out = {}
    for half, mask in (("first", wc["gameweek"] <= FIRST_HALF_END),
                       ("second", wc["gameweek"] > FIRST_HALF_END)):
        w = wc[mask].sort_values("gameweek")
        if w.empty:
            continue
        # prediction-driven policies -- these are what a real rule would do
        above = w[w["wildcard_pred"] >= WILDCARD_THRESHOLD]
        threshold_pick = (float(above.iloc[0]["wildcard_real"]) if not above.empty
                          else float(w.iloc[-1]["wildcard_real"]))

        # The notebook requires BOTH a cleared threshold AND a structural
        # trigger. chip_structural_readiness() for the wildcard is:
        #   expiry_pressure or exceptional or transfers >= 2 or hit_cost > 0
        # This simulation makes at most one no-hit transfer a week, so the last
        # two can never fire and the gate reduces to expiry or exceptional.
        half_end = FIRST_HALF_END if half == "first" else int(w["gameweek"].max())
        notebook_pick = None
        for _, r in w.iterrows():
            gw = int(r["gameweek"])
            pred = float(r["wildcard_pred"])
            expiry = 4 >= max(1, half_end - gw + 1)
            exceptional = pred >= WILDCARD_EXCEPTIONAL
            if (pred >= WILDCARD_THRESHOLD or expiry) and (expiry or exceptional):
                notebook_pick = float(r["wildcard_real"])
                break
        if notebook_pick is None:
            notebook_pick = float(w.iloc[-1]["wildcard_real"])
        best_pred_pick = float(w.loc[w["wildcard_pred"].idxmax(), "wildcard_real"])
        out[half] = {
            "oracle": float(w["wildcard_real"].max()),
            "best predicted": best_pred_pick,
            f"threshold only (>= {WILDCARD_THRESHOLD:.0f} xP)": threshold_pick,
            "notebook rule (threshold AND structural)": notebook_pick,
            "earliest legal": float(w.iloc[0]["wildcard_real"]),
            "mid-half": float(w.iloc[len(w) // 2]["wildcard_real"]),
            "forced at expiry": float(w.iloc[-1]["wildcard_real"]),
            "random legal": float(w["wildcard_real"].mean()),
            "never": 0.0,
        }
    return out


def wildcard_forecastability(wc: pd.DataFrame) -> pd.DataFrame:
    """Does PREDICTED wildcard value predict REALISED wildcard value?"""
    rows = []
    for half, mask in (("first", wc["gameweek"] <= FIRST_HALF_END),
                       ("second", wc["gameweek"] > FIRST_HALF_END)):
        w = wc[mask].dropna(subset=["wildcard_pred", "wildcard_real"])
        if len(w) < 6:
            continue
        rho = spearmanr(w["wildcard_pred"], w["wildcard_real"]).statistic
        rows.append({"half": half, "n": len(w), "spearman": rho,
                     "mean_pred": float(w["wildcard_pred"].mean()),
                     "mean_real": float(w["wildcard_real"].mean())})
    return pd.DataFrame(rows)


def main() -> None:
    seasons = sys.argv[1:] or ["2024-25", "2025-26"]
    fa, wcs = [], {}
    for season in seasons:
        print("=" * 80)
        print(f"{season}")
        print("=" * 80)
        panel = C.build_panel(season)
        weeks = C.simulate_season(panel)

        print("\nPART A — is chip value forecastable? (Spearman vs realised value)")
        t = forecastability(weeks)
        t["season"] = season
        fa.append(t)
        print(t.pivot_table(index=["chip", "signal"], columns="half",
                            values="spearman_vs_realised").round(3).to_string())

        print("\nPART B — Wildcard value by gameweek "
              f"(realised points gained over a {HORIZON}-week horizon)")
        wc = wildcard_values(panel)
        print("  forecastability of wildcard value "
              "(predicted vs realised, Spearman):")
        print("  " + wildcard_forecastability(wc).round(3)
              .to_string(index=False).replace("\n", "\n  "))
        print()
        res = evaluate_wildcard(wc)
        wcs[season] = res
        print(pd.DataFrame(res).round(1).to_string())
        print()

    print("=" * 80)
    print("COMBINED")
    print("=" * 80)
    allfa = pd.concat(fa)
    print("\nMean |Spearman| between decision-time signals and realised chip value:")
    print(allfa.groupby(["half", "signal"])["spearman_vs_realised"]
          .agg(["mean", "min", "max"]).round(3).to_string())

    print("\nWildcard, mean realised points over both seasons:")
    combined = {}
    for season, res in wcs.items():
        for half, scores in res.items():
            for policy, v in scores.items():
                combined.setdefault(policy, {}).setdefault(half, []).append(v)
    table = pd.DataFrame({
        policy: {half: float(np.mean(v)) for half, v in halves.items()}
        for policy, halves in combined.items()
    }).T
    table["total"] = table.sum(axis=1)
    print(table.sort_values("total", ascending=False).round(1).to_string())


if __name__ == "__main__":
    main()
