#!/usr/bin/env python3
"""
O60 -- the VICE-CAPTAIN rule.

Prompted by GW8 of 2026-27: the captain (B.Fernandes) played ZERO minutes with
`status=a`, no news, and a forecast of xp 5.79 / p_appear 0.95. The armband
passed to the vice, Palmer, who scored -1 -- so the insurance COST a point.

`plan_gw` picked that vice as the second-highest xp in the XI, which ignores
the only job a vice has: PLAYING when the captain does not.

PRICED BEFORE BUILDING (R48, `verify.price_ceiling`):

    captain plays 0 minutes in  4.16%  of (squad, gameweek) decisions
    perfect-hindsight headroom  12.84  points per blank
    -> ceiling                   20.3  pts/season   against an MDE of ~14

**So the prediction, stated before the run: this measures NULL.** Even a rule
capturing the entire headroom is barely resolvable, and a realistic one
capturing a quarter of it is ~5 points. It is run anyway because it is cheap
-- no refitting, only re-scoring -- and because the mechanism is real.

ARMS (the FPL rule: the vice's points double iff the captain played 0 minutes)
  vice_xp       second-highest xp.                       REFERENCE, shipped
  vice_avail    argmax(xp * p_appear)
  vice_sixty    argmax(xp * p_sixty)
  vice_mins     argmax(expected_minutes)
  vice_perfect  argmax(REALISED points) -- hindsight, not a strategy. Included
                to show how much of the 20.3 any rule could capture.
"""
from __future__ import annotations
import glob, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, measure, optimise, verify        # noqa: E402

SCRATCH = Path(os.environ.get("BAKE_DIR", "/tmp/bakeoff"))
SIGNAL = "m_full_us"
RULES = ["vice_xp", "vice_avail", "vice_sixty", "vice_mins", "vice_perfect"]


def pick_vice(rest: pd.DataFrame, rule: str):
    if rest.empty:
        return None
    if rule == "vice_xp":
        key = rest[SIGNAL]
    elif rule == "vice_avail":
        key = rest[SIGNAL] * rest["z_p_appear"].fillna(0)
    elif rule == "vice_sixty":
        key = rest[SIGNAL] * rest["z_p_sixty"].fillna(0)
    elif rule == "vice_mins":
        key = rest["z_expected_minutes"].fillna(0)
    else:
        key = rest["points"]                      # hindsight
    return rest.loc[key.idxmax()]


def main():
    t0 = time.time()
    panel = pd.read_pickle(C.cache("fpl2_panel_us.pkl"))
    rows = []
    for f in sorted(glob.glob(str(SCRATCH / "bake" / "us_*.pkl"))):
        season = pd.read_pickle(f)["season"].iloc[0]
        us = pd.read_pickle(f)[["gameweek", "player_code", SIGNAL]]
        pos = SCRATCH / "bake" / f"pos_{season}.pkl"
        z = pd.read_pickle(pos)[["gameweek", "player_code", "z_p_appear",
                                 "z_p_sixty", "z_expected_minutes"]]
        g = (data.with_blanks(panel, season)
             .merge(us, on=["gameweek", "player_code"], how="left")
             .merge(z, on=["gameweek", "player_code"], how="left"))
        g[SIGNAL] = pd.to_numeric(g[SIGNAL], errors="coerce").fillna(0.0)
        g.loc[g["fixtures"] <= 0, SIGNAL] = 0.0
        squads = measure.neutral_squads(g, k=10, seed=1)
        for si, s15 in enumerate(squads):
            held = g[g[C.PLAYER_KEY].isin(s15)]
            for gw, wk in held.groupby("gameweek", sort=True):
                xi = optimise.best_xi(wk, SIGNAL)
                if xi is None:
                    continue
                xi = xi.reset_index(drop=True)
                cap = optimise.pick_captain(xi, SIGNAL)
                base = float(xi["points"].sum())
                played = float(cap["minutes"]) > 0
                rest = xi[xi[C.PLAYER_KEY].ne(cap[C.PLAYER_KEY])]
                for rule in RULES:
                    v = pick_vice(rest, rule)
                    # the FPL rule: the armband moves ONLY if the captain
                    # played no minutes at all
                    extra = float(cap["points"]) if played else (
                        float(v["points"]) if v is not None else 0.0)
                    rows.append({"season": season, "squad": si,
                                 "gameweek": int(gw), "rule": rule,
                                 "value": base + extra})
        print(f"   [{time.time()-t0:5.0f}s] {season}", flush=True)

    M = pd.DataFrame(rows)
    M.to_pickle(SCRATCH / "vice.pkl")
    ref = M[M.rule.eq("vice_xp")].set_index(["season", "squad", "gameweek"])["value"]
    print("\n" + "=" * 84)
    print("O60 VICE-CAPTAIN RULE — metric 1 with the FPL armband rule, 8 holdouts")
    print("pre-registered prediction: NULL. Ceiling 20.3 pts/season vs MDE ~14.")
    print("=" * 84)
    print(f"{'rule':<15}{'mean':>9}{'t':>8}{'p':>8}{'95% CI':>20}{'MDE':>7}  seasons")
    for rule in RULES:
        if rule == "vice_xp":
            continue
        cur = M[M.rule.eq(rule)].set_index(["season", "squad", "gameweek"])["value"]
        j = pd.concat({"a": cur, "r": ref}, axis=1).dropna()
        r = verify.clustered((j["a"] - j["r"]).to_numpy(float),
                             j.index.get_level_values("season").to_numpy(),
                             scale=38.0)
        ci = "[%+.1f, %+.1f]" % (r["ci_lo"], r["ci_hi"])
        print(f"{rule:<15}{r['mean']:>+9.1f}{r['t']:>8.2f}{r['p']:>8.3f}"
              f"{ci:>20}{r['mde']:>7.1f}  {r['wins']}/8", flush=True)
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
