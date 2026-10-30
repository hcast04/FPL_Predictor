#!/usr/bin/env python3
"""
Project the next four gameweeks from everything played so far, and re-ask the
chip policy with the four-gameweek visibility it was actually measured with.

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
    last = max(finished)
    nxt = next(e for e in boot["events"] if e.get("is_next"))
    # THE WEEK BEING DECIDED, not the last one played.
    #
    # `project(from_gw=g, to_gw=g)` early-returns `model.predict(feats[gw==g])`,
    # which is byte-identical to what `walk_forward` does for week g -- so
    # `from_gw` means THE WEEK BEING DECIDED. This read `gw = max(finished)`,
    # the week already PLAYED, and passed it as `from_gw`. Every form feature
    # is `shift(1).rolling(w)`, so the week-`last` row averages weeks
    # last-3..last-1 (GW5-7 at GW8): the gameweek just finished was invisible
    # to the forecast deciding the next one. B.Fernandes read 4.89 on form
    # through GW7 and 3.33 through GW8; Guehi went from rank 23 to 180.
    #
    # S63 priced it on 8 holdout seasons through the neutral-squad instrument:
    # +36.1 pts/season, se 9.0, t +4.02, p 0.005, MDE 25.1, 7/8 seasons. The
    # negative control (arm B must reproduce `walk_forward` exactly) passed at
    # 0.00e+00 on all 7 seasons it could run on.
    #
    # Same defect class as M30, which fixed `decide_gw` in plan_gw and
    # run_live_chip. This one survived because `gw = max(finished)` reads as
    # obviously right: it IS the last week you have data for.
    gw = int(nxt["id"])
    cur = pd.read_pickle(LIVE) if LIVE.exists() else None
    if cur is None or sorted(int(g) for g in cur["gameweek"].unique()) != finished:
        cur = live.build(SEASON); cur.to_pickle(LIVE)

    raw = pd.concat([data.load_raw(extended=True), cur], ignore_index=True)
    raw["kickoff"] = pd.to_datetime(raw["kickoff_time"], errors="coerce", utc=True)
    panel = data.build_panel(raw); tf = team.team_fixtures(raw)
    panel = understat.attach(panel)

    # --- the published fixture list for the weeks ahead ---------------------
    # Built BEFORE the features, because the decide-week row below needs each
    # team's fixture count for that week.
    fx = api("fixtures/")
    teams = {t["id"]: t["short_name"] for t in boot["teams"]}
    fut = [f for f in fx if f.get("event") and gw <= f["event"] < gw + HORIZON]
    print(f"[{time.time()-t0:4.0f}s] fixture list: {len(fut)} fixtures over "
          f"GW{gw}-{gw+HORIZON-1}", flush=True)
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

    # --- the decide-week feature row, which does not exist yet --------------
    # `project` needs `feats[gameweek == gw]`, and week `gw` is unplayed, so
    # the panel has no row for it. Build one: the week-`last` rows with the
    # gameweek advanced and every STAT column blanked. The form features are
    # `shift(1).rolling(w)`, so this row averages the PLAYED weeks ending at
    # `last` -- it reads nothing from the future, and the assertion below
    # proves that rather than asserting it in a comment.
    src = panel[panel["season"].eq(SEASON) & panel["gameweek"].eq(last)]
    syn = src.copy()
    syn["gameweek"] = gw
    syn["t"] = syn["t"] + (gw - last)
    syn["fixtures"] = (syn["team"]
                       .map(FX[FX["gameweek"].eq(gw)].groupby("team").size())
                       .fillna(0.0))
    for c in ("points", "minutes", "goals_scored", "assists", "clean_sheets",
              "goals_conceded", "saves", "bonus", "bps", "expected_goals",
              "expected_assists", "starts", "defensive_contribution",
              "yellow_cards", "red_cards", "us_xg", "us_shots",
              "us_key_passes", "us_xg_chain", "us_xg_buildup"):
        if c in syn.columns:
            syn[c] = np.nan
    panel = pd.concat([panel, syn], ignore_index=True, sort=False)
    feats, names = features.build(panel, tf_ext)

    # The row must average the PLAYED weeks ending at `last`. If this ever
    # fails the projection is reading something it should not, and a silent
    # wrong answer here is exactly what cost +36.1 pts/season before.
    _p = panel[panel["season"].eq(SEASON) & panel["minutes"].gt(0)]
    _n = _p.groupby(C.PLAYER_KEY)["gameweek"].nunique()
    _ok = _tried = 0
    for _pc in _n[_n.eq(last)].index[:5]:
        _pts = (_p[_p[C.PLAYER_KEY].eq(_pc)]
                .set_index("gameweek")["points"].sort_index())
        _r = feats[feats[C.PLAYER_KEY].eq(_pc) & feats["season"].eq(SEASON)
                   & feats["gameweek"].eq(gw)]
        if _r.empty:
            continue
        _tried += 1
        _ok += abs(_r.iloc[0]["points_r3"] - _pts.loc[last - 2:last].mean()) < 1e-9
    if not _tried or _ok != _tried:
        raise RuntimeError(
            f"lag check FAILED ({_ok}/{_tried}): the GW{gw} feature row is not "
            f"mean(GW{last-2}..GW{last}). Do not trust this projection.")
    print(f"[{time.time()-t0:4.0f}s] lag check {_ok}/{_tried}: GW{gw} row "
          f"= mean(GW{last-2}..GW{last})", flush=True)

    tr_p = feats[feats["season"] < SEASON]; tr_t = tf_ext[tf_ext["season"] < SEASON]
    m = model.ComponentModel(feats=list(names)).fit(tr_p, tr_t)
    print(f"[{time.time()-t0:4.0f}s] fitted on {tr_p['season'].nunique()} seasons",
          flush=True)

    frames = []
    for target in range(gw, gw + HORIZON):
        fr = project.project(m, feats, tf_ext, SEASON, gw, target)
        if fr.empty:
            print(f"   GW{target}: empty", flush=True); continue
        fr = fr.copy(); fr["xp_plan"] = pd.to_numeric(fr["xp"], errors="coerce").fillna(0.0)
        frames.append(fr)
    print(f"[{time.time()-t0:4.0f}s] projected {len(frames)} frames "
          f"(GW{gw}-{gw+HORIZON-1})", flush=True)
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
