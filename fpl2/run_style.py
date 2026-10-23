#!/usr/bin/env python3
"""
O58 -- do TEAM PLAYING STYLE features add anything?

The user's observation: every team feature the model carries is an OUTCOME
(`team_scored_r5`, `team_conceded_r10` and the opponent twins). None describes
HOW a side plays. A possession team working the ball into the box and a
counter-attacking side crossing from deep can score at identical rates and
offer their forwards completely different chances.

S29 priced OPPONENT identity at zero for ranking. That is who you play, not
how your own team plays, so it does not cover this.

FEATURES (`fpl2/style.py`): possession, shots, shots on target, SoT%, goals
per shot (quality, not volume), crosses, interceptions, tackles won and fouls
-- rolled at r5 and r10, for the team AND the opponent. 36 features.
Match-level FBref data, so nothing is a season aggregate applied to a
mid-season fixture (that is how M28 produced a fictitious +235/season).

LEAK TESTED: every value verified to equal the mean of STRICTLY EARLIER
matches (1,060 rows, 0 mismatches), with a deliberately unshifted control
failing 1,034 -- so the test can fail.

ARMS:
  sty_off   shipped, 128 features. REFERENCE.
  sty_on    + the 36 style features.
  sty_shuf  + the same 36, SHUFFLED between teams within each gameweek.

**The shuffle is the point** (R26). Adding any feature block perturbs a fit;
only the shuffled control says whether a gain came from the INFORMATION. It
is what killed F16's Championship priors when the raw numbers looked like a
pass. `sty_on` must beat `sty_shuf`, not merely beat `sty_off`.

GATE: metric 1, 8 holdouts, K=10, season-clustered, bar |t| > 2.36. BETTER
needs the CI to exclude 0 AND the mean above +13.6 AND a win over the shuffle.
"""
from __future__ import annotations
import os, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, features, model, rates, style          # noqa: E402

SCRATCH = Path(os.environ.get("BAKE_DIR", "/tmp/bakeoff"))
OUT = SCRATCH / "bake"; OUT.mkdir(parents=True, exist_ok=True)
KEEP = ["season", "gameweek", "player_code", "points", "cost",
        "position", "team", "minutes", "fixtures"]


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


def run(holdout: str):
    from sklearn.linear_model import PoissonRegressor
    t0 = time.time()
    features.USE_UNDERSTAT_FEATURES = True
    rates._reg = lambda s=0: _WP(PoissonRegressor(alpha=1.0, max_iter=500))
    feats = pd.read_pickle(SCRATCH / "feats_us.pkl")
    names = pickle.load(open(SCRATCH / "names_us.pkl", "rb"))
    tf = pd.read_pickle(SCRATCH / "tf.pkl")
    panel = pd.read_pickle(Path.home() / "fpl2_panel_us.pkl")

    sty, made = style.build(panel)
    key = ["season", "gameweek", C.PLAYER_KEY]
    feats = feats.merge(sty[key + made].drop_duplicates(key), on=key, how="left")
    feats = feats[feats["season"] <= holdout]
    for c in feats.columns:
        if feats[c].dtype == "float64":
            feats[c] = feats[c].astype("float32")

    # the shuffle: permute each style column BETWEEN TEAMS within a gameweek,
    # so the distribution and the within-week structure survive and only the
    # team->style assignment is destroyed
    rng = np.random.default_rng(7)
    shuf = feats.copy()
    tkey = feats.groupby(["season", "gameweek", "team"], sort=False).ngroup()
    for c in made:
        byteam = feats.groupby(["season", "gameweek", "team"])[c].first()
        perm = (byteam.groupby(level=[0, 1], group_keys=False)
                .apply(lambda s: pd.Series(rng.permutation(s.values), index=s.index)))
        shuf[c] = feats.set_index(["season", "gameweek", "team"]).index.map(perm).to_numpy()

    out = None
    for arm, frame, nm in (("sty_off", feats, list(names)),
                           ("sty_on", feats, list(names) + made),
                           ("sty_shuf", shuf, list(names) + made)):
        o = model.walk_forward(frame, tf, nm, holdout)
        if out is None:
            out = o[KEEP].copy()
        else:
            assert o[["gameweek", "player_code"]].reset_index(drop=True).equals(
                out[["gameweek", "player_code"]].reset_index(drop=True)), "row order"
        out[arm] = o["xp"].to_numpy(float)
        print(f"   [{time.time()-t0:5.0f}s] {arm} ({len(nm)} feats) "
              f"mean xp {out[arm].mean():.4f}", flush=True)

    ref = OUT / f"us_{holdout}.pkl"
    if ref.exists():
        prev = pd.read_pickle(ref)[["gameweek", "player_code", "m_full_us"]]
        j = out[["gameweek", "player_code", "sty_off"]].merge(
            prev, on=["gameweek", "player_code"])
        d = float(np.abs(j["sty_off"] - j["m_full_us"]).max())
        print(f"   control: max|sty_off - m_full_us| = {d:.10f} "
              f"({'IDENTICAL' if d < 1e-9 else '*** DIFFERS ***'})", flush=True)
    out.to_pickle(OUT / f"sty_{holdout}.pkl")
    print(f"[{holdout}] WROTE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
