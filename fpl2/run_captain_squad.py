#!/usr/bin/env python3
"""
O12 step 6: does the captain ceiling rule survive the DECISION metric?

On the league-wide best XI it is worth +1.52 pts/gw (t=2.94, 4/4 seasons,
z=3.83 vs a shuffled control). But that XI is unconstrained. A real manager
captains from inside a 15-man, 100.0-budget squad, where the candidate pool is
far narrower and the top xp player is often the only plausible pick.

Arms, all sharing ONE squad/XI path (selection stays on `xp`; only the armband
differs, so squad effects cannot contaminate the comparison):
  A  captain = argmax(xp)                        <- must reproduce G9's baseline
  B  captain = argmax(xp + 0.5*up_q75)
  C  captain = argmax(xp + 0.5*shuffle(up_q75))  <- negative control, 20 seeds

REGRESSION CHECK: arm A must return 1717 / 2059 / 1869 / 361, the G9 baseline.
If it does not, the captain_score patch changed existing behaviour.
"""
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, simulate   # noqa: E402

SEASONS = list(C.HOLDOUT_SEASONS) + ["2026-27"]
G9_BASELINE = {"2023-24": 1717, "2024-25": 2059, "2025-26": 1869, "2026-27": 361}
K = 0.5
SEEDS = 20


def main():
    t0 = time.time()
    fc = pickle.load(open("/tmp/fpl2_captain_forecasts.pkl", "rb"))
    raw = pd.concat([data.load_raw(), pd.read_pickle("/tmp/fpl2_live2627.pkl")],
                    ignore_index=True)
    panel = data.build_panel(raw)
    d = panel.sort_values(["season", "gameweek"]).copy()
    g = d.groupby(C.PLAYER_KEY)["points"]
    d["up_q75"] = g.transform(lambda s: s.shift(1).expanding(min_periods=3).quantile(0.75))
    up = d[["season", "gameweek", C.PLAYER_KEY, "up_q75"]]
    print(f"[{time.time()-t0:4.0f}s] setup done\n", flush=True)

    print(f"{'season':10s}{'A argmax':>10}{'B ceiling':>11}{'C shuf':>9}"
          f"{'B-A':>7}{'C-A':>7}{'z':>7}{'regress':>10}", flush=True)
    rows = {}
    for s in SEASONS:
        grid = data.with_blanks(panel, s)
        grid = grid.merge(fc[s][["season", "gameweek", C.PLAYER_KEY, "xp"]],
                          on=["season", "gameweek", C.PLAYER_KEY], how="left")
        grid = grid.merge(up, on=["season", "gameweek", C.PLAYER_KEY], how="left")
        grid["xp"] = grid["xp"].fillna(0.0)
        grid.loc[grid["fixtures"] <= 0, "xp"] = 0.0
        grid["up_q75"] = pd.to_numeric(grid["up_q75"], errors="coerce").fillna(0.0)
        grid["cap_real"] = grid["xp"] + K * grid["up_q75"]
        grid.loc[grid["fixtures"] <= 0, "cap_real"] = 0.0

        a = simulate.simulate_season(grid, "xp")
        b = simulate.simulate_season(grid, "xp", captain_score="cap_real")
        shuf = []
        for seed in range(SEEDS):
            rng = np.random.default_rng(4421 * seed + 7)
            gg = grid.copy()
            gg["_u"] = gg.groupby("gameweek")["up_q75"].transform(
                lambda v: rng.permutation(v.to_numpy()))
            gg["cap_shuf"] = gg["xp"] + K * gg["_u"]
            gg.loc[gg["fixtures"] <= 0, "cap_shuf"] = 0.0
            r = simulate.simulate_season(gg, "xp", captain_score="cap_shuf")
            if r is not None:
                shuf.append(float(r.total))
        at, bt = float(a.total), float(b.total)
        sm, ssd = float(np.mean(shuf)), float(np.std(shuf, ddof=1))
        z = (bt - sm) / ssd if ssd > 0 else np.nan
        ok = "OK" if abs(at - G9_BASELINE[s]) < 0.5 else f"BROKE({at:.0f})"
        rows[s] = (at, bt, sm, z)
        print(f"{s:10s}{at:>10.0f}{bt:>11.0f}{sm:>9.0f}{bt-at:>+7.0f}"
              f"{sm-at:>+7.0f}{z:>7.2f}{ok:>10}", flush=True)

    ba = [b - a for a, b, _, _ in rows.values()]
    ca = [c - a for a, _, c, _ in rows.values()]
    print(f"\n  B beats A in {sum(x>0 for x in ba)}/4   aggregate {sum(ba):+.0f}", flush=True)
    print(f"  shuffled control aggregate {sum(ca):+.0f}", flush=True)
    print(f"  per-season z vs control: "
          f"{', '.join(f'{z:.2f}' for _,_,_,z in rows.values())}", flush=True)
    pickle.dump(rows, open("/tmp/fpl2_captain_squad.pkl", "wb"))
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
