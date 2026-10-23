#!/usr/bin/env python3
"""
What does the CHIP POLICY say about the wildcard, right now, for the real squad?

The user's point, and it is correct: this is a data-driven project, so the
decision to play a chip should come from `chips.rolling_policy` -- which was
measured at +119.8 pts/season over 8 holdout seasons (S50) -- and not from
anybody's reading of the fixtures.

THREE DEFECTS IN THE EXISTING LIVE PATH, fixed here:

  1. **The live cache is never invalidated.** `run_live_glm.py` does
     `if LIVE.exists(): cur = pd.read_pickle(LIVE)` with no freshness check.
     It currently holds GW1-7 and GW7 is the last finished, so it happens to
     be correct today -- but the moment GW8 finishes, every run would keep
     using GW1-7 for ever. This file asks the API which gameweeks are finished
     and REBUILDS whenever the cache does not match.

  2. **It trains on four seasons, not ten.** `data.load_raw()` without
     `extended=True` omits 2016-17..2021-22, which S46 added.

  3. **It predates the Understat switch.** With `USE_UNDERSTAT_FEATURES` now
     True (M27), the training panel carries `us_*` columns and a live panel
     built without them would hand the model all-NaN features. Understat is
     pulled for 2026-27 and attached.
"""
from __future__ import annotations
import json, os, sys, time, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import (config as C, chips, data, features, live, model,  # noqa: E402
                  optimise, rates, team, understat)

API = "https://fantasy.premierleague.com/api"
HDR = {"User-Agent": "Mozilla/5.0"}
SEASON = "2026-27"
LIVE = Path.home() / "fpl2_live2627.pkl"
PANEL = Path.home() / "fpl2_panel_us.pkl"
# your FPL team id; public, but read from the environment so the repo
# does not hardcode one person's team
ENTRY = int(os.environ.get("FPL_ENTRY", "3642415"))


def api(p):
    return json.load(urllib.request.urlopen(
        urllib.request.Request(f"{API}/{p}", headers=HDR), timeout=60))


class _WP:
    def __init__(self, est):
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        self.est = est
    def fit(self, X, y, sample_weight=None):
        self.est.fit(self.pre.fit_transform(X), y, sample_weight=sample_weight); return self
    def predict(self, X): return self.est.predict(self.pre.transform(X))


def main():
    from sklearn.linear_model import PoissonRegressor
    t0 = time.time()
    features.USE_UNDERSTAT_FEATURES = True
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))

    boot = api("bootstrap-static/")
    finished = sorted(e["id"] for e in boot["events"] if e.get("finished"))
    nxt = next((e for e in boot["events"] if e.get("is_next")), None)
    gw = int(nxt["id"]) if nxt else max(finished) + 1
    print(f"finished GW {finished} | next GW{gw} deadline {nxt['deadline_time']}",
          flush=True)

    # --- 1. live panel, with an ACTUAL freshness check -----------------------
    cur = pd.read_pickle(LIVE) if LIVE.exists() else None
    have = sorted(int(g) for g in cur["gameweek"].unique()) if cur is not None else []
    if have != finished:
        print(f"cache has {have}, API says {finished} -> REBUILDING", flush=True)
        cur = live.build(SEASON)
        cur.to_pickle(LIVE)
    else:
        print(f"cache matches the API ({have}) -> reusing", flush=True)

    # --- 2. ten seasons of history, not four --------------------------------
    raw = pd.concat([data.load_raw(extended=True), cur], ignore_index=True)
    # `load_raw` derives `kickoff` from `kickoff_time`; the live frame comes
    # straight from the API and never passes through it, so after the concat
    # every live row has kickoff = NaT. `build_panel` aggregates that into
    # `kickoff_first`, and `understat.join_season` matches teams by FIXTURE
    # DATE -- so the live season silently joined ZERO Understat rows and
    # `attach` then overwrote the historical us_* columns with NaN.
    raw["kickoff"] = pd.to_datetime(raw["kickoff_time"], errors="coerce", utc=True)
    panel = data.build_panel(raw)
    tf = team.team_fixtures(raw)
    print(f"[{time.time()-t0:4.0f}s] panel {panel.shape}, "
          f"seasons {panel['season'].nunique()}", flush=True)

    # --- 3. Understat for the live season too --------------------------------
    # attach ALL seasons in one call. Calling it per-season is unsafe: when a
    # season joins nothing it returns `panel.assign(us_*=NaN)`, which wipes
    # whatever was already there -- that is how a pre-merge of the historical
    # columns followed by `attach(seasons=[SEASON])` produced 0.0% coverage
    # everywhere rather than just in the live season.
    try:
        panel = understat.attach(panel)
        print(f"[{time.time()-t0:4.0f}s] understat attached for {SEASON}", flush=True)
    except Exception as e:                                       # noqa: BLE001
        print(f"[{time.time()-t0:4.0f}s] understat {SEASON} unavailable "
              f"({type(e).__name__}); `_avail` falls back to FPL columns", flush=True)
    cov = panel[panel["season"].eq(SEASON) & (panel["minutes"] > 0)]
    if "us_xg" in panel.columns and len(cov):
        print(f"      us_xg coverage in {SEASON}: {cov['us_xg'].notna().mean()*100:.1f}%",
              flush=True)

    feats, names = features.build(panel, tf)
    out = model.walk_forward(feats, tf, names, SEASON)
    print(f"[{time.time()-t0:4.0f}s] forecast done, {len(names)} features", flush=True)
    out.to_pickle(Path.home() / "fpl2_live_chip_out.pkl")

    # --- the real squad ------------------------------------------------------
    el = {e["id"]: e for e in boot["elements"]}
    picks = api(f"entry/{ENTRY}/event/{max(finished)}/picks/")
    ids = [p["element"] for p in picks["picks"]]
    squad = [el[i]["code"] for i in ids if i in el]
    used = {c["name"] for c in (picks.get("active_chip") and [{"name": picks["active_chip"]}] or [])}
    hist_api = api(f"entry/{ENTRY}/history/")
    played = {c["name"] for c in hist_api.get("chips", [])}
    print(f"\nsquad: {len(squad)} players | chips already played: "
          f"{sorted(played) if played else 'NONE'}", flush=True)

    # --- the policy ----------------------------------------------------------
    last = int(out["gameweek"].max())
    grid = data.with_blanks(panel[panel["season"].eq(SEASON)], SEASON).merge(
        out[["season", "gameweek", C.PLAYER_KEY, "xp"]],
        on=["season", "gameweek", C.PLAYER_KEY], how="left")
    grid["xp"] = grid["xp"].fillna(0.0)
    grid.loc[grid["fixtures"] <= 0, "xp"] = 0.0

    name = {e["code"]: e["web_name"] for e in boot["elements"]}
    week = grid[grid["gameweek"].eq(last)]
    held = week[week[C.PLAYER_KEY].isin(squad)]
    avail = [c for c in ("wildcard", "freehit", "bboost", "3xc")
             if c not in played]
    policy = chips.rolling_policy(grid, "xp")
    decision = policy(last, squad, week, avail)

    print("\n" + "=" * 70)
    print(f"CHIP POLICY (S50: +119.8 pts/season, 7/8 seasons) -- GW{last} state")
    print("=" * 70)
    print(f"  chips available : {avail}")
    print(f"  policy says     : {decision if decision else 'PLAY NOTHING'}")
    xi = optimise.best_xi(held, "xp")
    if xi is not None:
        bench = held[~held[C.PLAYER_KEY].isin(xi[C.PLAYER_KEY])]
        cap = optimise.pick_captain(xi, "xp")
        print(f"  current XI xp   : {xi['xp'].sum():.2f}")
        print(f"  bench xp        : {bench['xp'].sum():.2f}  "
              f"(bench boost is worth about this)")
        print(f"  captain (argmax): {name.get(cap[C.PLAYER_KEY], '?')} "
              f"xp {cap['xp']:.2f}")
        pool = week[week["fixtures"] > 0]
        best = optimise.optimise_squad(
            pool.groupby(C.PLAYER_KEY, as_index=False).agg(
                xp=("xp", "mean"), cost=("cost", "first"),
                position=("position", "first"), team=("team", "first")),
            "xp", budget=C.BUDGET)
        if best is not None:
            bx = optimise.best_xi(best.merge(
                pool[[C.PLAYER_KEY, "xp"]].drop_duplicates(C.PLAYER_KEY),
                on=C.PLAYER_KEY, suffixes=("", "_w")), "xp")
            if bx is not None:
                print(f"  unconstrained-rebuild XI xp: {bx['xp'].sum():.2f}  "
                      f"(wildcard ceiling, ignores your actual budget)")
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
