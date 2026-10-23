#!/usr/bin/env python3
"""
Combine the per-season outputs of `run_opt_bakeoff2.py` and report at G = 8.

The runner is fanned out one process per holdout because each arm is a full
season simulation. A single process therefore holds one season, and a
season-clustered standard error over G = 1 is undefined. Each process writes
its pickle and defers; this reads all of them and clusters properly.
"""
from __future__ import annotations
import os, sys, glob
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bakeoff_score import cluster                                # noqa: E402

SCRATCH = Path(os.environ.get("BAKE_DIR", "/tmp/bakeoff"))


def load(prefix):
    fs = sorted(glob.glob(str(SCRATCH / f"{prefix}_*.pkl")))
    fs = [f for f in fs if "_res" not in f]
    if not fs:
        return None
    return pd.concat([pd.read_pickle(f) for f in fs], ignore_index=True)


def report(M, ref_name, title, key):
    if M is None or ref_name not in set(M["arm"]):
        print(f"\n{title}\n  (no data)"); return
    ref = M[M["arm"].eq(ref_name)].set_index(key)["value"]
    G = M["season"].nunique()
    print("\n" + "=" * 78); print(title)
    print(f"G = {G} seasons, season-clustered, bar |t| > 2.36 at G=8")
    print("=" * 78)
    print(f"{'arm':<14}{'mean':>9}{'t':>8}{'p':>8}{'95% CI':>20}{'MDE':>8}  seasons")
    out = []
    for a in M["arm"].unique():
        if a == ref_name:
            continue
        cur = M[M["arm"].eq(a)].set_index(key)["value"]
        j = pd.concat({"a": cur, "r": ref}, axis=1).dropna()
        if j.empty:
            continue
        d = (j["a"] - j["r"]).to_numpy(float)
        gg = j.index.get_level_values("season").to_numpy()
        m, se, tt, pp, lo, hi = cluster(d, gg)
        w = sum(1 for _, dd in pd.Series(d).groupby(gg) if dd.mean() > 0)
        out.append({"arm": a, "mean": m, "t": tt, "p": pp, "mde": 2.8 * se,
                    "wins": w})
        print(f"{a:<14}{m:>+9.1f}{tt:>8.2f}{pp:>8.3f}"
              f"{f'[{lo:+.1f}, {hi:+.1f}]':>20}{2.8*se:>8.1f}  {w}/{G}")
    return pd.DataFrame(out)


def main():
    A = load("opt2_transfer")
    r = report(A, "base", "A. TRANSFER RULES vs base -- opening squad held "
                          "fixed, only the transfer path differs",
               ["season", "squad"])
    if r is not None:
        r.to_pickle(SCRATCH / "opt2_transfer_res.pkl")
    B = load("opt2_build")
    if B is not None:
        B = B.copy(); B["value"] = B["value"] * 38.0
    r = report(B, "build_base", "B. SQUAD-BUILDING vs build_base -- path-free, "
                                "each arm builds its own 15 then holds it",
               ["season", "gameweek"])
    if r is not None:
        r.to_pickle(SCRATCH / "opt2_build_res.pkl")
    for M, n in ((A, "transfer"), (B, "build")):
        if M is None:
            continue
        print(f"\nabsolute per arm ({n}):")
        t = M.groupby(["arm", "season"])["value"].mean().unstack()
        t["MEAN"] = t.mean(axis=1)
        print(t.round(0).sort_values("MEAN", ascending=False).to_string())


if __name__ == "__main__":
    main()
