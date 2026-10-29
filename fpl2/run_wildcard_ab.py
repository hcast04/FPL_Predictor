#!/usr/bin/env python3
"""
Adversarial head-to-head: does a MODEL wildcard beat a LEGACY wildcard?

Legacy exists only as cached holdout forecasts (2023-24/24-25/25-26), so the
live GW8 squad cannot be tested against it. What CAN be tested is the
PROCEDURE: at gameweek 8 of each holdout season, build a squad from each
signal under identical constraints and play out the rest of the season.

Controls, because a two-arm comparison cannot tell "better signal" from
"any signal":
  template  -- squad built to maximise OWNERSHIP (what the crowd holds)
  random    -- a random legal squad, 20 draws, mean and sd reported
Both arms and both controls run through the same simulator with the same
budget, so only the signal differs.
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, legacy, optimise, simulate, team  # noqa: E402

START = 8
BUDGET = 100.0


def seasons_grid():
    grids = pickle.load(open(C.cache("fpl2_o37_grids.pkl"),"rb"))
    raw = data.load_raw(); panel = data.build_panel(raw)
    out = {}
    for s in C.HOLDOUT_SEASONS:
        g = grids[s].copy()
        lg = legacy.attach(data.with_blanks(panel, s))[
            ["season","gameweek",C.PLAYER_KEY,"xp"]].rename(columns={"xp":"xp_leg"})
        g = g.merge(lg, on=["season","gameweek",C.PLAYER_KEY], how="left")
        g["xp_leg"] = g["xp_leg"].fillna(0.0)
        g.loc[g["fixtures"]<=0, "xp_leg"] = 0.0
        # contemporaneous ownership, for the template control
        own = (panel[panel.season.eq(s)][["gameweek",C.PLAYER_KEY,"selected"]]
               if "selected" in panel.columns else None)
        if own is not None:
            g = g.merge(own, on=["gameweek",C.PLAYER_KEY], how="left")
            g["selected"] = g["selected"].fillna(0.0)
        out[s] = g
    return out


def build_at(g, signal, gw=START, budget=BUDGET, horizon=5):
    gws = [x for x in sorted(g.gameweek.unique()) if gw <= x < gw+horizon]
    w = g[g.gameweek.isin(gws)]
    pool = (w.groupby(C.PLAYER_KEY).agg(**{signal:(signal,"mean"),
            "cost":("cost","first"), "position":("position","first"),
            "team":("team","first")}).reset_index())
    pool = pool[pool.cost.gt(0)]
    sq = optimise.optimise_squad(pool, signal, budget=budget)
    return None if sq is None else sq[C.PLAYER_KEY].tolist()


def play(g, squad, signal="xp", gw=START):
    rest = g[g.gameweek.ge(gw)].copy()
    r = simulate.simulate_season(rest, signal, opening_squad=squad)
    return None if r is None else r.total


def random_squad(g, rng, gw=START, budget=BUDGET):
    w = g[g.gameweek.eq(gw)]
    pool = (w.groupby(C.PLAYER_KEY).agg(cost=("cost","first"),
            position=("position","first"), team=("team","first")).reset_index())
    pool = pool[pool.cost.gt(0)].copy()
    pool["r"] = rng.random(len(pool))
    sq = optimise.optimise_squad(pool, "r", budget=budget)
    return None if sq is None else sq[C.PLAYER_KEY].tolist()


def main():
    t0=time.time(); G = seasons_grid()
    print(f"[{time.time()-t0:4.0f}s] grids ready\n", flush=True)
    rows=[]
    for s in C.HOLDOUT_SEASONS:
        g = G[s]
        sq_m = build_at(g, "xp"); sq_l = build_at(g, "xp_leg")
        # BOTH squads are then played on the MODEL signal for transfers, so the
        # comparison isolates the OPENING SQUAD and not the transfer policy
        tm = play(g, sq_m); tl = play(g, sq_l)
        # and again on each signal's own transfers, the end-to-end version
        tm2 = play(g, sq_m, "xp"); tl2 = play(g, sq_l, "xp_leg")
        tmpl = None
        if "selected" in g.columns:
            sq_t = build_at(g, "selected")
            tmpl = play(g, sq_t) if sq_t else None
        rng = np.random.default_rng(0)
        rnd = [play(g, random_squad(g, rng)) for _ in range(20)]
        rnd = [x for x in rnd if x is not None]
        rows.append(dict(season=s, model=tm, legacy=tl, model_own=tm2,
                         legacy_own=tl2, template=tmpl,
                         rand_mean=float(np.mean(rnd)), rand_sd=float(np.std(rnd,ddof=1)),
                         overlap=len(set(sq_m)&set(sq_l))))
        print(f"[{time.time()-t0:4.0f}s] {s}: model {tm:.0f}  legacy {tl:.0f}  "
              f"template {tmpl if tmpl is None else round(tmpl)}  "
              f"random {np.mean(rnd):.0f}+-{np.std(rnd,ddof=1):.0f}  "
              f"squad overlap {len(set(sq_m)&set(sq_l))}/15", flush=True)
    R=pd.DataFrame(rows)
    print("\n"+"="*78)
    print(f"WILDCARD AT GW{START}, PLAYED TO GW38 (same simulator, same budget)")
    print("="*78)
    print(R[["season","model","legacy","template","rand_mean","rand_sd","overlap"]]
          .round(0).to_string(index=False))
    d=(R.model-R.legacy).to_numpy()
    print(f"\n  model - legacy (model transfers both):  {['%+.0f'%x for x in d]}"
          f"   mean {d.mean():+.1f}  positive {int((d>0).sum())}/3")
    d2=(R.model_own-R.legacy_own).to_numpy()
    print(f"  end-to-end, each on its own signal:     {['%+.0f'%x for x in d2]}"
          f"   mean {d2.mean():+.1f}  positive {int((d2>0).sum())}/3")
    print(f"\n  both vs RANDOM legal squads:")
    print(f"    model  - random: {['%+.0f'%x for x in (R.model-R.rand_mean)]}"
          f"  mean {(R.model-R.rand_mean).mean():+.1f}")
    print(f"    legacy - random: {['%+.0f'%x for x in (R.legacy-R.rand_mean)]}"
          f"  mean {(R.legacy-R.rand_mean).mean():+.1f}")
    print(f"    random sd across draws: {R.rand_sd.mean():.0f} pts "
          f"-- anything inside that is not a result")
    pickle.dump(R, open(C.cache("fpl2_wc_ab.pkl"),"wb"))
    print(f"\n[{time.time()-t0:.0f}s] cached -> ~/fpl2_wc_ab.pkl")


if __name__ == "__main__":
    main()
