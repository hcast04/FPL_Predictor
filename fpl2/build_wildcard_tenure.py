#!/usr/bin/env python3
"""
The wildcard squad as the model gives it, and as it would be with the
club-tenure correction applied.

CONTEXT, stated because it bounds what this is worth. S58 found a real bias:
a top-10-ranked player who changed club is over-predicted by 0.85 pts/gameweek,
decaying 7x with tenure (-0.952 at 0-9 fixtures to -0.140 at 50+). S59 built
the correction -- shrink the 21 `*_career` features toward the training
position mean by `w = n/(n+K)` -- and it WORKS on its target, closing the gap
36% monotonically in K. It is worth **+1.0 pts/season against an MDE of 6.9**.
A null. It is NOT shipped.

So the right reading of the second squad is "which names does the known bias
touch", not "this one is better". The fix is below the instrument's
resolution; the diagnosis is what has value, for a human reading the output.
"""
from __future__ import annotations
import json, os, sys, time, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import (config as C, data, features, live, model,          # noqa: E402
                  optimise, project, rates, team, understat)

API = "https://fantasy.premierleague.com/api"; HDR = {"User-Agent": "Mozilla/5.0"}
SEASON = "2026-27"; LIVE = Path.home() / "fpl2_live2627.pkl"
BUDGET = float(os.environ.get("WC_BUDGET", "99.1"))
K_TEN = int(os.environ.get("TENURE_K", "10"))
MIN_CHANCE = 75


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


def shrink(feats, names, K):
    if K <= 0:
        return feats
    d = feats.copy()
    n = pd.to_numeric(d["fixtures_at_current_club"], errors="coerce").fillna(0.0)
    w = (n / (n + K)).to_numpy(float)
    tr = d[d["season"] < SEASON]
    for c in [x for x in names if x.endswith("_career")]:
        pm = tr.groupby("position")[c].mean()
        prior = d["position"].map(pm).astype(float).to_numpy()
        v = pd.to_numeric(d[c], errors="coerce").to_numpy(float)
        d[c] = np.where(np.isnan(v), v, w * v + (1.0 - w) * np.nan_to_num(prior))
    return d


def main():
    from sklearn.linear_model import PoissonRegressor
    t0 = time.time()
    features.USE_UNDERSTAT_FEATURES = True
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    boot = api("bootstrap-static/")
    fin = sorted(e["id"] for e in boot["events"] if e.get("finished"))
    nxt = next(e for e in boot["events"] if e.get("is_next")); target = int(nxt["id"])
    cur = pd.read_pickle(LIVE) if LIVE.exists() else None
    if cur is None or sorted(int(g) for g in cur.gameweek.unique()) != fin:
        cur = live.build(SEASON); cur.to_pickle(LIVE)
    raw = pd.concat([data.load_raw(extended=True), cur], ignore_index=True)
    raw["kickoff"] = pd.to_datetime(raw["kickoff_time"], errors="coerce", utc=True)
    panel = understat.attach(data.build_panel(raw)); tf = team.team_fixtures(raw)
    feats, names = features.build(panel, tf)
    gw = max(fin)

    fxs = api("fixtures/"); tm = {t["id"]: t["short_name"] for t in boot["teams"]}
    rows = []
    for f in fxs:
        if f.get("event") and gw < f["event"] <= target:
            for tid, opp, home in ((f["team_h"], f["team_a"], 1.0),
                                   (f["team_a"], f["team_h"], 0.0)):
                rows.append({"gameweek": f["event"], "team": tm[tid],
                             "opponent": tm[opp], "is_home": home})
    FX = pd.DataFrame(rows).assign(season=SEASON)
    FX["fixture_id"] = [f"fut_{i}" for i in range(len(FX))]
    tf_ext = pd.concat([tf, FX], ignore_index=True, sort=False)

    el = pd.DataFrame([{"code": e["code"], "name": e["web_name"],
                        "cost": e["now_cost"] / 10.0, "owned": float(e["selected_by_percent"]),
                        "status": e["status"], "chance": e["chance_of_playing_next_round"],
                        "team_id": e["team"], "etype": e["element_type"]}
                       for e in boot["elements"]])
    el["team"] = el.team_id.map(tm)
    el["position"] = el.etype.map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})
    ten = (feats[feats.season.eq(SEASON)].groupby(C.PLAYER_KEY)
           ["fixtures_at_current_club"].max().rename("tenure"))

    squads = {}
    for label, K in (("MODEL AS SHIPPED", 0), (f"WITH TENURE FIX (K={K_TEN})", K_TEN)):
        f2 = shrink(feats, names, K)
        tr_p = f2[f2.season < SEASON]
        m = model.ComponentModel(feats=list(names)).fit(tr_p, tf[tf.season < SEASON])
        fr = project.project(m, f2, tf_ext, SEASON, gw, target)
        xp = fr.groupby(C.PLAYER_KEY, as_index=False).agg(xp=("xp", "mean"))
        pool = el.merge(xp, left_on="code", right_on=C.PLAYER_KEY, how="inner")
        pool = pool.drop(columns=["code"]).reset_index(drop=True)
        av = pool[pool.status.eq("a") & (pool.chance.isna() | (pool.chance >= MIN_CHANCE))]
        orig = C.MAX_PER_CLUB; C.MAX_PER_CLUB = 2       # your max-2/club preference
        try:
            sq = optimise.optimise_squad(av.reset_index(drop=True), "xp", budget=BUDGET)
        finally:
            C.MAX_PER_CLUB = orig
        sq = sq.reset_index(drop=True)
        xi = optimise.best_xi(sq, "xp").reset_index(drop=True)
        cap = optimise.pick_captain(xi, "xp")
        sq["role"] = np.where(sq[C.PLAYER_KEY].isin(xi[C.PLAYER_KEY]), "XI", "bench")
        sq.loc[sq[C.PLAYER_KEY].eq(cap[C.PLAYER_KEY]), "role"] = "CAPT"
        sq = sq.merge(ten, left_on=C.PLAYER_KEY, right_index=True, how="left")
        squads[label] = (sq, float(xi.xp.sum()) + float(cap.xp))
        print(f"[{time.time()-t0:4.0f}s] {label}: "
              f"{float(xi.xp.sum())+float(cap.xp):.2f} xp", flush=True)

    for label, (sq, tot) in squads.items():
        print("\n" + "=" * 72); print(f"{label}   GW{target}   £{BUDGET}m, max 2/club")
        print("=" * 72)
        print(sq.sort_values(["role", "xp"], ascending=[True, False])
              [["name", "position", "team", "cost", "xp", "owned", "tenure", "role"]]
              .to_string(index=False, formatters={
                  "cost": "£{:.1f}m".format, "xp": "{:.2f}".format,
                  "owned": "{:.1f}%".format,
                  "tenure": lambda v: "new" if pd.isna(v) or v < 10 else f"{int(v)}"}))
        print(f"  spend £{sq.cost.sum():.1f}m | XI + captain {tot:.2f} xp")
    a, b = list(squads.values())
    sa, sb = set(a[0].name), set(b[0].name)
    print("\n" + "=" * 72); print("DIFFERENCE")
    print("=" * 72)
    print(f"  dropped by the fix : {', '.join(sorted(sa - sb)) or '(none)'}")
    print(f"  added by the fix   : {', '.join(sorted(sb - sa)) or '(none)'}")
    print(f"  common             : {len(sa & sb)}/15")


if __name__ == "__main__":
    main()
