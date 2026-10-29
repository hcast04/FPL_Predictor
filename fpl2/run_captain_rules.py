#!/usr/bin/env python3
"""
O12 steps 2-4: captain rules, done properly.

What F9 got wrong (M12): no standard error, no test, no noise floor, no
disagreement rate, and a thin upside proxy borrowed from an unrelated feature.
It compared season means and demanded 3/3 seasons.

Fixes here:
  1. DISAGREEMENT RATE. If a rule picks the same player as argmax(xp) 85% of
     the time, a season mean averages one real decision over six null ones.
     Effects are reported on all gameweeks AND on disagreements only.
  2. REAL UPSIDE PROXIES, strictly lagged: per-player expanding sd, 75th
     percentile and haul rate of realised points, shift(1) before every
     window so no gameweek sees itself.
  3. PAIRED INFERENCE: mean of paired differences with a standard error, a
     t-statistic and a sign test over ~121 gameweeks pooled, with the
     per-season split reported alongside rather than instead.

Rule families:
  A  upside: xp + k*upside, swept over k (dose-response, as F5/F10 did right)
  B  calibration: argmax(xp) maximises expected points BY CONSTRUCTION, so a
     rule can only win if xp is biased. S6 says the model under-predicts high-xP
     players (-0.690 at xp>=3.5, 3/3 seasons). Fit E[points|xp] on PRIOR seasons
     only and re-rank by the corrected value.
"""
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, optimise   # noqa: E402

SEASONS = list(C.HOLDOUT_SEASONS) + ["2026-27"]
FC = Path("/tmp/fpl2_captain_forecasts.pkl")


def upside_features(panel):
    """Strictly-lagged dispersion of realised points, per player."""
    d = panel.sort_values(["season", "gameweek"]).copy()
    g = d.groupby(C.PLAYER_KEY)["points"]
    d["up_sd"] = g.transform(lambda s: s.shift(1).expanding(min_periods=3).std())
    d["up_q75"] = g.transform(lambda s: s.shift(1).expanding(min_periods=3).quantile(0.75))
    d["up_haul"] = g.transform(
        lambda s: s.shift(1).ge(10).expanding(min_periods=3).mean())
    return d[["season", "gameweek", C.PLAYER_KEY, "up_sd", "up_q75", "up_haul"]]


def calibration_curve(fc, season, bins=12):
    """E[points | xp] fitted on PRIOR seasons only. Returns a callable."""
    prior = [fc[s] for s in SEASONS if s < season and s in fc]
    if not prior:
        return None
    t = pd.concat(prior, ignore_index=True)
    t = t[(t.fixtures > 0) & t.xp.notna() & t.points.notna()]
    t = t[t.xp >= 2.0]                      # captain candidates only
    if len(t) < 500:
        return None
    # qcut names its result "xp", which collides with an agg column of the same
    # name and makes the index level ambiguous -- use distinct labels.
    q = pd.qcut(t.xp, bins, duplicates="drop").rename("bin")
    tab = (t.groupby(q, observed=True)
             .agg(xp_mid=("xp", "mean"), y_mean=("points", "mean"))
             .reset_index(drop=True)
             .sort_values("xp_mid"))
    return lambda v: np.interp(v, tab.xp_mid.to_numpy(), tab.y_mean.to_numpy())


def main():
    t0 = time.time()
    fc = pickle.load(open(FC, "rb"))
    raw = pd.concat([data.load_raw(), pd.read_pickle("/tmp/fpl2_live2627.pkl")],
                    ignore_index=True)
    panel = data.build_panel(raw)
    up = upside_features(panel)
    print(f"[{time.time()-t0:4.0f}s] setup done", flush=True)

    recs = []
    for s in SEASONS:
        out = fc[s].merge(up, on=["season", "gameweek", C.PLAYER_KEY], how="left")
        cal = calibration_curve(fc, s)
        for gw, g in out[out.fixtures > 0].groupby("gameweek"):
            if len(g) < 50:
                continue
            xi = optimise.best_xi(g, "xp")
            if xi is None or len(xi) < 11:
                continue
            xi = xi.copy()
            for c in ("up_sd", "up_q75", "up_haul"):
                xi[c] = pd.to_numeric(xi[c], errors="coerce").fillna(0.0)
            xp = xi.xp.to_numpy(float); y = xi.points.to_numpy(float)
            base_i = int(np.argmax(xp))
            row = {"season": s, "gameweek": gw,
                   "argmax_xp": y[base_i], "_base_i": base_i}
            # The floor is sigma-dependent, so record the RELATIVE size of the
            # perturbation each rule applies. That picks which noise-floor row
            # the rule must clear -- the step F9 never took.
            for nm, vec in (("sd", xi.up_sd.to_numpy(float)),
                            ("q75", xi.up_q75.to_numpy(float)),
                            ("haul", xi.up_haul.to_numpy(float))):
                for k in (0.1, 0.25, 0.5, 1.0):
                    i = int(np.argmax(xp + k * vec))
                    row[f"up_{nm}_k{k}"] = y[i]
                    row[f"i_up_{nm}_k{k}"] = i
                    row[f"p_up_{nm}_k{k}"] = float(
                        np.mean(np.abs(k * vec) / np.maximum(xp, 1e-9)))
            if cal is not None:
                adj = cal(xp)
                i = int(np.argmax(adj))
                row["calibrated"] = y[i]; row["i_calibrated"] = i
                row["p_calibrated"] = float(
                    np.mean(np.abs(adj - xp) / np.maximum(xp, 1e-9)))
            recs.append(row)
        print(f"[{time.time()-t0:4.0f}s] {s} done", flush=True)

    d = pd.DataFrame(recs)
    rules = [c for c in d.columns
             if c not in ("season", "gameweek", "argmax_xp", "_base_i")
             and not c.startswith("i_") and not c.startswith("p_")]

    print(f"\n{'rule':18s}{'disagree':>10}{'all_gw_mean':>13}{'se':>8}{'t':>7}"
          f"{'disagree_mean':>15}{'se':>8}{'sign':>10}{'seasons':>9}", flush=True)
    print("-" * 100, flush=True)
    res = {}
    for r in rules:
        sub = d[d[r].notna()]
        if sub.empty:
            continue
        diff = sub[r].to_numpy(float) - sub["argmax_xp"].to_numpy(float)
        ic = f"i_{r}"
        dis = (sub[ic].to_numpy(int) != sub["_base_i"].to_numpy(int)) if ic in sub else diff != 0
        n, nd = len(diff), int(dis.sum())
        se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        mean = diff.mean()
        dd = diff[dis]
        dse = dd.std(ddof=1) / np.sqrt(len(dd)) if len(dd) > 1 else np.nan
        wins = int((dd > 0).sum()); losses = int((dd < 0).sum())
        swin = sum(1 for _, gsub in sub.groupby("season")
                   if (gsub[r] - gsub["argmax_xp"]).mean() > 0)
        res[r] = (mean, se, nd, n)
        pert = sub[f"p_{r}"].mean() if f"p_{r}" in sub else np.nan
        floor = 0.324 if pert < 0.005 else (0.551 if pert < 0.03 else 1.620)
        verdict = "detectable" if abs(mean) > floor else "INSIDE FLOOR"
        print(f"  [{r:16s} perturbs xp by {pert:6.1%} -> floor {floor:.3f} "
              f"pts/gw -> {verdict}]", flush=True)
        print(f"{r:18s}{nd/n:>9.1%}{mean:>13.4f}{se:>8.4f}"
              f"{mean/se if se else np.nan:>7.2f}"
              f"{dd.mean() if len(dd) else np.nan:>15.3f}{dse:>8.3f}"
              f"{wins}W/{losses}L".rjust(10) + f"{swin}/{sub.season.nunique()}".rjust(9),
              flush=True)

    print("\nNOTE: argmax(xp) maximises EXPECTED captain points by construction.", flush=True)
    print("A rule can only beat it where xp is biased -- which is what the", flush=True)
    print("calibrated arm tests directly. Upside arms are the F9 family, rerun", flush=True)
    print("with lagged proxies and a dose-response sweep.", flush=True)
    pickle.dump({"per_gw": d, "res": res}, open("/tmp/fpl2_captain_rules.pkl", "wb"))
    print(f"[{time.time()-t0:4.0f}s] DONE", flush=True)


if __name__ == "__main__":
    main()
