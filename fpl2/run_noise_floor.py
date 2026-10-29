#!/usr/bin/env python3
"""
Establish the noise floor. Run this FIRST, before trusting any comparison.

Perturbs the signal with information-free multiplicative noise and re-runs the
full season. The spread of the answer is the smallest difference that can be
believed.

Usage:  .venv/bin/python fpl2/run_noise_floor.py
"""
from __future__ import annotations
import pickle, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, legacy, noise   # noqa: E402

CACHE = Path("/tmp/fpl2_panels.pkl")


def panels():
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    p = data.build_panel()
    out = {s: legacy.attach(data.with_blanks(p, s)) for s in C.HOLDOUT_SEASONS}
    pickle.dump(out, open(CACHE, "wb"))
    return out


def main():
    t0 = time.time()
    ps = panels()
    print("NOISE FLOOR — how much does an information-free perturbation move a season?\n")
    nf = noise.noise_floor(ps, "xp", sigmas=(0.001, 0.01, 0.05), seeds=8)
    print(nf.round(2).to_string(index=False))
    print()
    for sig in (0.001, 0.01, 0.05):
        d = noise.detectable(nf, sigma=sig, k=3.0)
        print(f"  jitter {sig:<6}  smallest believable season-total difference "
              f"(3 sd of the cross-season mean): {d:6.0f} points")
    print(f"\n  elapsed {time.time()-t0:.0f}s")
    nf.to_csv("/tmp/fpl2_noise_floor.csv", index=False)
    print("  written /tmp/fpl2_noise_floor.csv")


if __name__ == "__main__":
    main()
