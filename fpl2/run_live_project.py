#!/usr/bin/env python3
"""
Project GW8-11 from GW7 information, and re-ask the chip policy with the
four-gameweek visibility it was actually measured with.

`chips.rolling_policy` reads `visible = [g for g in present if gw <= g <= gw +
KNOWN]` with KNOWN = 4. The live grid only contains FINISHED gameweeks, so the
policy saw ONE week instead of four and every chip's benefit was estimated
from a single fixture. The observation-phase answer is calendar-based and
unaffected, but from GW9 the benefit estimate is what decides.

`project.project` is the honest forward forecast: player form features are
FROZEN at week t (they are strictly lagged, so week t's value is legitimately
known at week t), the opponent, venue and fixture count come from the
PUBLISHED fixture list, and opponent STRENGTH is rewound to week t by
`_strength_as_of` -- which is the one place the leak would hide (M17).
"""
from __future__ import annotations
import json, os, sys, time, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import (config as C, chips, data, features, live, model,  # noqa: E402
                  optimise, project, rates, team, understat)

API = "https://fantasy.premierleague.com/api"
HDR = {"User-Agent": "Mozilla/5.0"}
SEASON = "2026-27"
ENTRY = int(os.environ.get("FPL_ENTRY") or 0)   # export FPL_ENTRY=<your team id>
LIVE = C.cache("fpl2_live2627.pkl")
HORIZON = 4


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
    finished = sorted(e["id"] for e in boot["events"]
                      if e.get("finished") or (e.get("is_current") and all(
                          x.get("finished_provisional")
                          for x in api(f"fixtures/?event={e['id']}"))))
    gw = max(finished)
    cur = pd.read_pickle(LIVE) if LIVE.exists() else None
    if cur is None or sorted(int(g) for g in cur["gameweek"].unique()) != finished:
        cur = live.build(SEASON); cur.to_pickle(LIVE)

    raw = pd.concat([data.load_raw(extended=True), cur], ignore_index=True)
    raw["kickoff"] = pd.to_datetime(raw["kickoff_time"], errors="coerce", utc=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    panel = understat.attach(panel)
    feats, names = features.build(panel, tf)

    tr_p = feats[feats["season"] < SEASON]; tr_t = tf[tf["season"] < SEASON]
    m = model.ComponentModel(feats=list(names)).fit(tr_p, tr_t)
    print(f"[{time.time()-t0:4.0f}s] fitted on {tr_p['season'].nunique()} seasons",
          flush=True)

    # --- the published fixture list for the weeks ahead ---------------------
    fx = api("fixtures/")
    teams = {t["id"]: t["short_name"] for t in boot["teams"]}
    fut = [f for f in fx if f.get("event") and gw < f["event"] <= gw + HORIZON]
    print(f"[{time.time()-t0:4.0f}s] fixture list: {len(fut)} fixtures over "
          f"GW{gw+1}-{gw+HORIZON}", flush=True)
    rows = []
    for f in fut:
        for tid, opp, home in ((f["team_h"], f["team_a"], True),
                               (f["team_a"], f["team_h"], False)):
            rows.append({"gameweek": f["event"], "team": teams[tid],
                         "opponent": teams[opp], "is_home": float(home)})
    FX = pd.DataFrame(rows)
    cnt = FX.groupby(["gameweek", "team"]).size().rename("n").reset_index()
    print("   fixtures per team per gameweek: "
          f"min {cnt['n'].min()}, max {cnt['n'].max()}  "
          f"(blanks: {int((cnt['n']==0).sum())}, doubles: {int((cnt['n']>1).sum())})",
          flush=True)
    FX.to_pickle(C.cache("fpl2_live_fixtures.pkl"))

    # `project.frames` iterates gameweeks PRESENT IN THE FEATURE FRAME, which
    # for a live season means finished weeks only -- so it returned exactly one
    # frame (the current week) and the chip policy ran with 1 week of
    # visibility instead of the KNOWN=4 it was measured with. It was built for
    # walk-forward backtests, where future weeks already exist in the data.
    #
    # The fix is to give `project.project` the team-fixture rows it needs for
    # the unplayed weeks, built from the PUBLISHED fixture list. Strength
    # columns are left out deliberately: `_strength_as_of` rewinds them to
    # week `gw`, which is the whole point.
    fut_tf = FX.assign(season=SEASON,
                       fixture_id=[f"fut_{i}" for i in range(len(FX))])
    tf_ext = pd.concat([tf, fut_tf], ignore_index=True, sort=False)
    frames = []
    for target in range(gw, gw + HORIZON + 1):
        fr = project.project(m, feats, tf_ext, SEASON, gw, target)
        if fr.empty:
            print(f"   GW{target}: empty", flush=True); continue
        fr = fr.copy(); fr["xp_plan"] = pd.to_numeric(fr["xp"], errors="coerce").fillna(0.0)
        frames.append(fr)
    print(f"[{time.time()-t0:4.0f}s] projected {len(frames)} frames "
          f"(GW{gw}-{gw+HORIZON})", flush=True)
    proj = pd.concat(frames, ignore_index=True)
    proj.to_pickle(C.cache("fpl2_live_proj.pkl"))
    nm = {e["code"]: e["web_name"] for e in boot["elements"]}
    for g, fr in proj.groupby("gameweek"):
        t = fr.groupby(C.PLAYER_KEY)["xp"].mean().nlargest(5)
        print(f"   GW{int(g)} top: " +
              ", ".join(f"{nm.get(k,'?')} {v:.2f}" for k, v in t.items()), flush=True)
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
