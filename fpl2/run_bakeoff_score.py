#!/usr/bin/env python3
"""
O51 / S52 phase B -- score every arm phase A fitted.

Reads `bake/{family}_{season}.pkl`, merges the two families per season, and
measures on the PRE-REGISTERED instrument:

  metric 1, neutral fixed squads (K=10), XI + captain, 8 holdout seasons,
  paired on (season, squad, gameweek), SEASON-CLUSTERED standard errors
  (R32, G=8 -> bar |t| > 2.36), reference arm `m_full_fpl`.

Reported alongside, NOT gating: within-gameweek Spearman, the realised points
of the best XI pickable from the WHOLE pool (signal quality with no budget),
and captain-added. Those are diagnostics -- the gate is metric 1, because that
is the decision.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy import stats                                          # noqa: E402
from fpl2 import config as C, data, measure, optimise            # noqa: E402

SCRATCH = Path(os.environ.get("BAKE_DIR", "/tmp/bakeoff"))
BAKE = SCRATCH / "bake"
HOLDOUTS = os.environ.get("HOLDOUTS", "").split(",") if os.environ.get("HOLDOUTS") \
    else ["2018-19", "2019-20", "2020-21", "2021-22",
          "2022-23", "2023-24", "2024-25", "2025-26"]
# The reference arm. Was `m_full_fpl` while that was the shipped model; after
# M27 retracted S47 and `USE_UNDERSTAT_FEATURES` was turned on, the shipped
# model is `m_full_us` and every delta must be measured against THAT -- a gate
# against a reference known to be broken flatters everything equally.
REF = os.environ.get("REF_ARM", "m_full_us")
K = int(os.environ.get("KSCORE", "10"))
SQUADS: dict = {}
# Which reporting sections to run. The gate is metric 1 and it costs ~10
# minutes; the diagnostics (full-pool XI searches), the captain/XI
# decomposition and the per-position blend cost ~2 hours between them. When
# the question is "which arm wins", running all four first is just latency.
SECTIONS = set(os.environ.get("SECTIONS", "diag,metric1,decomp,perpos").split(","))
MDE_GATE = 13.6          # pre-registered; the instrument's resolution (R24)

# Specified BEFORE any arm was scored: the incumbent plus the new architecture,
# and the two data regimes of the same architecture. Averaging PERCENTILE RANKS
# within a gameweek, not the raw values -- the arms are on different scales
# (a quantile head sits ~60% above a conditional mean) and a raw average would
# just be the larger-scaled arm.
ENSEMBLES = {
    "e_rank_mdl_xgb": ("m_full_fpl", "d_xgb"),
    "e_rank_fpl_us":  ("m_full_fpl", "m_full_us"),
    "e_rank_mean_q":  ("d_xgb", "d_xgb_q80"),
}


def cluster(d, g):
    """Season-clustered mean, t, p and CI. G-1 df. See ledger R32."""
    d = np.asarray(d, float); g = np.asarray(g)
    n = len(d); db = d.mean(); G = len(np.unique(g))
    ss = sum((d[g == k] - db).sum() ** 2 for k in np.unique(g))
    se = np.sqrt((G / (G - 1)) * ss / n ** 2)
    t = db / se if se > 0 else np.nan
    crit = stats.t.ppf(0.975, G - 1)
    return db, se, t, 2 * (1 - stats.t.cdf(abs(t), G - 1)), db - crit * se, db + crit * se


def load():
    panel = pd.read_pickle(C.cache("fpl2_panel_us.pkl"))
    grids, arms = {}, None
    for s in HOLDOUTS:
        parts = []
        for fam in ("fpl", "us", "pos", "s47", "gbdtus", "ceil", "bon"):
            p = BAKE / f"{fam}_{s}.pkl"
            if p.exists():
                parts.append(pd.read_pickle(p))
        if not parts:
            continue
        def _split(fr):
            """arms get scored; `z_` columns are optimiser INPUTS, carried only."""
            a = [c for c in fr.columns if c.split("_")[0] in ("b", "m", "d", "s", "ceil", "bon")]
            z = [c for c in fr.columns if c.startswith("z_")]
            return a, z
        sig = parts[0]
        cols, zcols = _split(sig)
        for extra in parts[1:]:
            ec, ez = _split(extra)
            ec = [c for c in ec if c not in cols]
            ez = [c for c in ez if c not in zcols]
            sig = sig.merge(extra[["gameweek", "player_code"] + ec + ez],
                            on=["gameweek", "player_code"], how="outer")
            cols += ec; zcols += ez
        g = data.with_blanks(panel, s).merge(
            sig[["gameweek", "player_code"] + cols + zcols],
            on=["gameweek", "player_code"], how="left")
        # a blank gameweek has no forecast and must never be selected
        for c in cols + zcols:
            g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0)
            g.loc[g["fixtures"] <= 0, c] = 0.0
        # Combination arms, built at SCORING time from the saved per-row
        # signals -- no refit, and specified in advance rather than picked
        # after seeing which individual arms won (that would be selection on
        # the same data the gate uses).
        for name, members in ENSEMBLES.items():
            if all(m in g.columns for m in members):
                pl = g["fixtures"] > 0
                r = sum(g.loc[pl, m].groupby([g.loc[pl, "gameweek"]])
                        .rank(pct=True) for m in members) / len(members)
                g[name] = 0.0
                g.loc[pl, name] = r
                if name not in cols:
                    cols.append(name)
        grids[s] = g
        arms = cols if arms is None else [a for a in arms if a in cols]
    return panel, grids, arms


def _show_diag(D):
    print("=" * 78)
    print("DIAGNOSTICS -- mean over seasons (not the gate)")
    print("=" * 78)
    t = D.groupby("arm").agg(spearman=("rho", "mean"),
                             pool_xi=("pool_xi", "mean"),
                             captain=("captain", "mean")).round(2)
    print(t.sort_values("pool_xi", ascending=False).to_string(), flush=True)
    D.to_pickle(SCRATCH / f"diag_{REF}.pkl")


def _guarded(name, fn):
    """
    Run a reporting section; a failure there must not discard what ran before.

    The metric-1 table costs ~8 minutes of XI searches and is pickled before
    the decomposition and per-position sections run. Those two are another
    ~12 minutes and are the newest code here, so they are the likely place for
    a crash -- and a crash that threw away the gate's own table would be a bad
    trade for a reporting bug.
    """
    try:
        return fn()
    except Exception as e:                      # noqa: BLE001
        import traceback
        print(f"\n!! section {name!r} FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("!! earlier sections are already written; continuing\n", flush=True)
        return None


def main():
    t0 = time.time()
    panel, grids, arms = load()
    print(f"[{time.time()-t0:4.0f}s] {len(grids)} seasons, {len(arms)} arms: "
          f"{', '.join(arms)}\n", flush=True)

    # ---- diagnostics (cheap, reported, not gating) -------------------------
    diag = []
    if "diag" not in SECTIONS:
        print("[skipping diagnostics]", flush=True)
        D = None
    for s, g in ({} if "diag" not in SECTIONS else grids).items():
        pl = g[g["fixtures"] > 0]
        for a in arms:
            rho = pl.groupby("gameweek").apply(
                lambda d: d[a].corr(d["points"], method="spearman"),
                include_groups=False).mean()
            pool, cap = [], []
            for _, wk in pl.groupby("gameweek"):
                xi = optimise.best_xi(wk, a)
                if xi is not None:
                    pool.append(float(xi["points"].sum()))
                    cap.append(float(optimise.pick_captain(xi, a)["points"]))
            diag.append({"season": s, "arm": a, "rho": rho,
                         "pool_xi": float(np.sum(pool)),
                         "captain": float(np.sum(cap))})
    if diag:
        D = pd.DataFrame(diag)
        _show_diag(D)

    # ---- metric 1, the pre-registered gate ---------------------------------
    rows = []
    for s, g in grids.items():
        squads = measure.neutral_squads(g, k=K, seed=1)
        SQUADS[s] = squads                 # reused by the decomposition below
        for a in arms:
            sc = measure.fixed_squad_score(g, squads, a, actual="points")
            sc["season"] = s; sc["arm"] = a
            rows.append(sc)
        print(f"   scored {s} ({len(squads)} squads)", flush=True)
    M = pd.concat(rows, ignore_index=True)
    M.to_pickle(SCRATCH / f"metric1_{REF}.pkl")

    ref = M[M["arm"].eq(REF)].set_index(["season", "squad", "gameweek"])["value"]
    print("\n" + "=" * 78)
    print(f"METRIC 1 vs {REF} -- per SEASON (x38 gameweeks), season-clustered")
    print(f"bar |t| > 2.36 (G=8);  BETTER needs CI excluding 0 AND mean > +{MDE_GATE}")
    print("=" * 78)
    # MDE is reported per comparison rather than assumed. The 13.6 gate comes
    # from S48's run at the same K and holdouts; if this run's resolution has
    # drifted, the number that decided the gate should be visible next to it
    # rather than inherited silently (R24).
    print(f"{'arm':<14}{'mean':>9}{'t':>8}{'p':>8}{'95% CI':>20}"
          f"{'MDE':>7}  {'seasons':>8}  verdict")
    res = []
    for a in arms:
        if a == REF:
            continue
        cur = M[M["arm"].eq(a)].set_index(["season", "squad", "gameweek"])["value"]
        j = pd.concat({"a": cur, "r": ref}, axis=1).dropna()
        d = (j["a"] - j["r"]).to_numpy(float) * 38.0      # per-season equivalent
        g = j.index.get_level_values("season").to_numpy()
        mean, se, tt, p, lo, hi = cluster(d, g)
        wins = sum(1 for _, dd in pd.Series(d).groupby(g) if dd.mean() > 0)
        if lo > 0 and mean > MDE_GATE:   v = "BETTER"
        elif hi < 0 and mean < -MDE_GATE: v = "WORSE"
        elif abs(mean) < MDE_GATE and lo < 0 < hi: v = "equivalent"
        else: v = "inconclusive"
        res.append({"arm": a, "mean": mean, "se": se, "mde": 2.8 * se, "t": tt,
                    "p": p, "lo": lo, "hi": hi, "wins": wins, "verdict": v})
        print(f"{a:<14}{mean:>+9.1f}{tt:>8.2f}{p:>8.3f}"
              f"{f'[{lo:+.1f}, {hi:+.1f}]':>20}{2.8*se:>7.1f}  {wins:>5}/8   {v}",
              flush=True)
    pd.DataFrame(res).to_pickle(SCRATCH / f"metric1_res_{REF}.pkl")

    if "decomp" not in SECTIONS:
        print("\n[skipping decomposition]", flush=True)
        dec = []
    # ---- captaincy, isolated ----------------------------------------------
    # The user's standing objection is that the CAPTAIN PICK and the squad are
    # what miss, and metric 1 confounds the two: a better signal can win by
    # choosing a different XI while making the same armband error. So hold the
    # XI on the reference signal and let only the armband move, and then the
    # reverse. `optimise.xi_value(captain_score=...)` exists for exactly this.
    print("\n" + "=" * 78)
    print(f"DECOMPOSITION vs {REF} -- same neutral squads, per season (x38)")
    print("  captain_only : XI picked by the reference, ARMBAND by the arm")
    print("  xi_only      : XI picked by the arm, ARMBAND by the reference")
    print("=" * 78)
    print(f"{'arm':<16}{'captain_only':>14}{'t':>7}{'xi_only':>12}{'t':>7}")
    dec = []
    for a in (arms if "decomp" in SECTIONS else []):
        if a == REF:
            continue
        for mode in ("captain_only", "xi_only"):
            sel, cap = (REF, a) if mode == "captain_only" else (a, REF)
            rows_m = []
            for sname, g in grids.items():
                for si, squad in enumerate(SQUADS[sname]):
                    held = g[g["player_code"].isin(squad)]
                    for gw, wk in held.groupby("gameweek", sort=True):
                        v = optimise.xi_value(wk, sel, actual="points",
                                              captain_score=cap)
                        # the reference value is already in `ref` from the
                        # metric-1 pass -- recomputing it per arm per mode
                        # would be ~190k redundant MILP-free XI searches
                        b = ref.get((sname, si, int(gw)), np.nan)
                        if np.isfinite(v) and np.isfinite(b):
                            rows_m.append((sname, (v - b) * 38.0))
            dd = pd.DataFrame(rows_m, columns=["season", "d"])
            m, se, tt, pp, lo, hi = cluster(dd["d"], dd["season"])
            dec.append({"arm": a, "mode": mode, "mean": m, "t": tt, "p": pp,
                        "lo": lo, "hi": hi})
        r = {d["mode"]: d for d in dec if d["arm"] == a}
        print(f"{a:<16}{r['captain_only']['mean']:>+14.1f}{r['captain_only']['t']:>7.2f}"
              f"{r['xi_only']['mean']:>+12.1f}{r['xi_only']['t']:>7.2f}", flush=True)
    pd.DataFrame(dec).to_pickle(SCRATCH / f"decomp_{REF}.pkl")

    # ---- per position ------------------------------------------------------
    # S48's lesson: a pooled number can read "no effect" while two positions
    # move hard in opposite directions. `no_minutes` was -37.8 for MID and
    # +12.1 for GK and pooled to nothing. So: give the arm's signal to ONE
    # position and the reference's to the other three, and read where an
    # advantage actually lives.
    #
    # Restricted to arms on the POINTS scale. A quantile head sits ~60% above
    # a conditional mean and a rank ensemble is a percentile; blending those
    # across positions inside one column would compare scales, not models.
    scaled = [a for a in arms if a != REF and not a.startswith("e_")
              and not a.endswith("q80")]
    print("\n" + "=" * 78)
    print(f"PER POSITION vs {REF} -- arm's signal for that position only, per season")
    print("  (points-scale arms only; quantile and rank-ensemble arms excluded)")
    print("=" * 78)
    print(f"{'arm':<16}" + "".join(f"{q:>12}" for q in C.POSITIONS))
    pos_rows = []
    for a in (scaled if "perpos" in SECTIONS else []):
        cells = []
        for q in C.POSITIONS:
            rows_q = []
            for sname, g in grids.items():
                blend = np.where(g["position"].to_numpy() == q,
                                 g[a].to_numpy(float), g[REF].to_numpy(float))
                gg = g.assign(_blend=blend)
                for si, squad in enumerate(SQUADS[sname]):
                    held = gg[gg["player_code"].isin(squad)]
                    for gw, wk in held.groupby("gameweek", sort=True):
                        v = optimise.xi_value(wk, "_blend", actual="points")
                        b = ref.get((sname, si, int(gw)), np.nan)
                        if np.isfinite(v) and np.isfinite(b):
                            rows_q.append((sname, (v - b) * 38.0))
            dd = pd.DataFrame(rows_q, columns=["season", "d"])
            m, se, tt, pp, lo, hi = cluster(dd["d"], dd["season"])
            star = "*" if (lo > 0 or hi < 0) else " "
            cells.append(f"{m:>+10.1f}{star} ")
            pos_rows.append({"arm": a, "position": q, "mean": m, "t": tt,
                             "p": pp, "lo": lo, "hi": hi})
        print(f"{a:<16}" + "".join(cells), flush=True)
    pd.DataFrame(pos_rows).to_pickle(SCRATCH / f"per_position_{REF}.pkl")
    npos = len(scaled) * len(C.POSITIONS)
    print(f"\n  * = clustered CI excludes 0. {npos} cells at alpha=0.05 expect "
          f"~{0.05*npos:.1f} false positives -- read patterns, not single stars.")

    print("\n" + "=" * 78)
    print(f"METRIC 1 -- absolute season equivalent per arm")
    print("=" * 78)
    tot = (M.groupby(["arm", "season"])["value"].mean() * 38).unstack()
    tot["MEAN"] = tot.mean(axis=1)
    print(tot.round(0).sort_values("MEAN", ascending=False).to_string(), flush=True)
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
