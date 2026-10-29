#!/usr/bin/env python3
"""
The simple model against the SAME G1 gate, unmodified.

PASS requires BOTH (declared 2026-10-20, not edited):
  1. beats legacy in all three holdout seasons, and
  2. mean margin exceeds +148 points (the measured noise floor).

Reported alongside, because they are what the diagnosis turned on:
  * spearman and overall calibration vs legacy AND vs the complex rebuild
  * TOP-END calibration, where the complex model over-predicts (1.07-1.14)
    and legacy under-predicts (0.83-0.93). The optimiser only buys from the
    top, so this is the cell that explains the 290-point gap.
"""
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, legacy, simple, simulate, team  # noqa: E402

NOISE_FLOOR = 148.0
# /tmp is swept by systemd-tmpfiles mid-session; caches written there have been
# lost twice today. The session scratchpad persists.
CACHE = Path("/tmp/claude-1000/-home-mdmsystems-acer-Escritorio-pers-fpl/"
             "a909bbf0-dfbe-4cc0-a5f8-534424e1d2b9/scratchpad")


def main():
    t0 = time.time()
    raw = data.load_raw()
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    cx_path = CACHE / "fpl2_xp.pkl"
    complex_xp = pickle.load(open(cx_path, "rb")) if cx_path.exists() else None
    if complex_xp is None:
        print("  (complex-model cache absent -- simple vs legacy only)", flush=True)
    print(f"[{time.time()-t0:4.0f}s] setup done\n", flush=True)

    fc = {s: simple.build(panel, tf, s) for s in C.HOLDOUT_SEASONS}
    print(f"[{time.time()-t0:4.0f}s] simple forecasts built (no training)\n", flush=True)

    leg = legacy.load_fold_forecasts().rename(columns={"xp": "xp_leg"})
    print("FORECAST QUALITY", flush=True)
    print(f"{'season':10s}{'sp_leg':>9}{'sp_cplx':>9}{'sp_simple':>11}"
          f"{'rat_leg':>9}{'rat_cplx':>10}{'rat_simple':>12}", flush=True)
    tops = []
    for s in C.HOLDOUT_SEASONS:
        sm = fc[s][fc[s].fixtures > 0][["season","gameweek",C.PLAYER_KEY,"points","xp"]]
        m = sm.rename(columns={"xp": "xp_s"}).merge(
            leg, on=["season", "gameweek", C.PLAYER_KEY])
        if complex_xp is not None:
            cx = complex_xp[s]; cx = cx[cx.fixtures > 0][["season","gameweek",C.PLAYER_KEY,"xp"]]
            m = m.merge(cx.rename(columns={"xp":"xp_c"}),
                        on=["season","gameweek",C.PLAYER_KEY])
        else:
            m["xp_c"] = np.nan
        tops.append((s, m))
        print(f"{s:10s}"
              f"{m.xp_leg.corr(m.points,method='spearman'):>9.4f}"
              f"{m.xp_c.corr(m.points,method='spearman'):>9.4f}"
              f"{m.xp_s.corr(m.points,method='spearman'):>11.4f}"
              f"{m.xp_leg.sum()/m.points.sum():>9.4f}"
              f"{m.xp_c.sum()/m.points.sum():>10.4f}"
              f"{m.xp_s.sum()/m.points.sum():>12.4f}", flush=True)

    print("\nTOP-END CALIBRATION  (ratio at xp >= cut; 1.0 is perfect)", flush=True)
    print(f"{'season':10s}{'cut':>5}{'leg':>9}{'cplx':>9}{'simple':>9}", flush=True)
    for s, m in tops:
        for cut in (4.0, 5.0, 6.0):
            a = m[m.xp_leg >= cut]; b = m[m.xp_c >= cut]; c = m[m.xp_s >= cut]
            if len(c) < 30 or len(b) < 30:
                continue
            print(f"{s:10s}{cut:>5.1f}"
                  f"{a.xp_leg.sum()/max(a.points.sum(),1):>9.4f}"
                  f"{b.xp_c.sum()/max(b.points.sum(),1):>9.4f}"
                  f"{c.xp_s.sum()/max(c.points.sum(),1):>9.4f}", flush=True)
        print(flush=True)

    print("DECISION METRIC — G1 gate, unmodified", flush=True)
    print(f"{'season':10s}{'simple':>9}{'legacy':>9}{'margin':>9}{'cplx':>9}", flush=True)
    margins, rows = [], {}
    for s in C.HOLDOUT_SEASONS:
        grid = legacy.attach(data.with_blanks(panel, s)).rename(columns={"xp":"xp_legacy"})
        g = grid.merge(fc[s][["season","gameweek",C.PLAYER_KEY,"xp"]],
                       on=["season","gameweek",C.PLAYER_KEY], how="left")
        g["xp"] = g["xp"].fillna(0.0)
        blank = g["fixtures"] <= 0
        g.loc[blank, ["xp", "xp_legacy"]] = 0.0
        r_s = simulate.simulate_season(g, "xp")
        r_l = simulate.simulate_season(g, "xp_legacy")
        if complex_xp is not None:
            gc = g.merge(complex_xp[s][["season","gameweek",C.PLAYER_KEY,"xp"]]
                         .rename(columns={"xp":"xp_cplx"}),
                         on=["season","gameweek",C.PLAYER_KEY], how="left")
            gc["xp_cplx"] = gc["xp_cplx"].fillna(0.0); gc.loc[blank, "xp_cplx"] = 0.0
            r_c = simulate.simulate_season(gc, "xp_cplx")
        else:
            r_c = type("R", (), {"total": float("nan")})()
        ms = float(r_s.total) - float(r_l.total)
        margins.append(ms); rows[s] = (r_s.total, r_l.total, r_c.total)
        print(f"{s:10s}{r_s.total:>9.0f}{r_l.total:>9.0f}{ms:>+9.0f}"
              f"{r_c.total:>9.0f}", flush=True)

    wins = sum(1 for m in margins if m > 0)
    mean_m = float(np.mean(margins))
    print(f"\nGATE G1 (simple vs legacy)", flush=True)
    print(f"  1. wins all three seasons      : "
          f"{'YES' if wins==3 else 'NO'}  ({wins}/3)", flush=True)
    print(f"  2. mean margin exceeds 148 pts : "
          f"{'YES' if mean_m>NOISE_FLOOR else 'NO'}  (mean {mean_m:+.0f})", flush=True)
    print(f"  => {'PASS' if wins==3 and mean_m>NOISE_FLOOR else 'FAIL'}", flush=True)
    if complex_xp is not None:
        cplx_m = [rows[s][2]-rows[s][1] for s in C.HOLDOUT_SEASONS]
        print(f"\n  simple vs complex: "
              f"{sum(1 for s in C.HOLDOUT_SEASONS if rows[s][0]>rows[s][2])}/3 seasons, "
              f"mean {np.mean([rows[s][0]-rows[s][2] for s in C.HOLDOUT_SEASONS]):+.0f}",
              flush=True)
        print(f"  complex vs legacy: mean {np.mean(cplx_m):+.0f}", flush=True)
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
