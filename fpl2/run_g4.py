#!/usr/bin/env python3
"""G4+G5: xG/xA attacking targets and an xGC team model. Logs per season."""
import pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, data, model, rates, team   # noqa: E402
from sklearn.metrics import roc_auc_score                # noqa: E402

def main():
    t0 = time.time()
    feats = pd.read_pickle("/tmp/fpl2_features.pkl")
    names = pickle.load(open("/tmp/fpl2_featnames.pkl", "rb"))
    tf = team.team_fixtures(data.load_raw())
    print(f"[{time.time()-t0:5.0f}s] team fixtures built", flush=True)

    print("\nMECHANISM 1 — clean-sheet model on xGC (was 0.6036/0.6177/0.6177)", flush=True)
    for s in C.HOLDOUT_SEASONS:
        p = team.walk_forward(tf, s).dropna(subset=["conceded"])
        m = team.TeamModel().fit(tf[tf.season < s])
        print(f"  {s}: AUC {roc_auc_score((p.conceded==0).astype(int), p.clean_sheet_prob):.4f}"
              f"  CS pred {p.clean_sheet_prob.mean():.3f} actual {(p.conceded==0).mean():.3f}"
              f"  scale {m.scale:.3f}   [{time.time()-t0:5.0f}s]", flush=True)

    print("\nMECHANISM 2 — rate tails and xP tail (legacy: a90max 0.289, xp max 5.04)", flush=True)
    P = rates.RateModel.PREFIX
    res = {}
    for s in C.HOLDOUT_SEASONS:
        out = model.walk_forward(feats, tf, names, s)
        res[s] = out[["season","gameweek","player_code","position","team",
                      "cost","fixtures","points","xp","xp_raw"]]
        t = out[out.fixtures > 0]; hi = t[t.expected_minutes >= 60]
        print(f"  {s}: ratio {t.xp.sum()/t.points.sum():5.3f}"
              f"  spearman {t.xp.corr(t.points, method='spearman'):.4f}"
              f"  g90max {hi[P+'goals_per90'].max():.3f}"
              f"  a90max {hi[P+'assists_per90'].max():.3f}"
              f"  xp q99 {hi.xp.quantile(.99):5.2f} max {hi.xp.max():5.2f}"
              f"   [{time.time()-t0:5.0f}s]", flush=True)
    pickle.dump(res, open("/tmp/fpl2_xp.pkl", "wb"))
    print(f"\n[{time.time()-t0:5.0f}s] forecasts cached -> /tmp/fpl2_xp.pkl", flush=True)

if __name__ == "__main__":
    main()
