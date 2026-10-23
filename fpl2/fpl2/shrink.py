"""
Hierarchical shrinkage toward a position x price prior.  ** REJECTED **

Tested 2026-10-20 and NOT in the pipeline. Kept so the idea is not re-proposed.

Swept k over {0, 0.5, 1, 2, 3, 5, 8, 15} on out-of-sample forecasts, judged on
top-20 realised points (the metric the optimiser consumes, per ledger R16):

      k      2023-24  2024-25  2025-26    mean   wins
      0 (none) 4.229    4.630    4.312    4.390    --
      3        4.193    4.713    4.120    4.342   1/3
      5        4.139    4.736    4.143    4.339   1/3

**No k wins in all three seasons, and the mean is worse at every k than at
k=0.** Shrinking toward a crude positional average flattens genuine differences
faster than it removes over-confidence: spearman fell from ~0.68 to ~0.59 and
calibration flipped to over-prediction (ratios 1.12 / 1.06 / 1.21).

The diagnosis it was built for is still real -- the model rates Salah 6.33 and
buys him at 14.5m for 123 points while legacy caps out at 3.71 -- but this is
not the fix. A per-player reliability estimate, or an explicit decline/role
model, would target it without flattening everyone.

The failure this fixes: at gameweek 1 there is no within-season evidence, so
every feature is carried from the previous season. The model then extrapolates
a declining premium's career rates with full confidence -- it rated Salah at
6.33 xP and spent GBP 14.5m on a player who returned 123 points, while the
legacy signal rated nobody above 3.71 and bought better. Flatness at the top is
the CORRECT posture when there is no current-season data.

    xp_final = w * xp_model + (1 - w) * prior(position, price)
    w        = n / (n + k)        n = appearances THIS season, strictly before

`prior` is E[points | position, price band], fitted on training seasons. Price
is used because it is the market's expectation and is the one informative
signal available before a ball is kicked.

`k` is selected on TRAINING seasons only (`select_k`), never against the gate.
Tuning it on the holdout would be gate-shopping wearing a different hat.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C

PRICE_BANDS = [0, 4.5, 5.5, 6.5, 8.0, 10.0, 99.0]
K_CANDIDATES = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)


@dataclass
class ShrinkagePrior:
    k: float = 0.0        # 0 = no shrinkage. See the REJECTED note above.
    table: dict = field(default_factory=dict)
    fallback: float = 0.0

    def fit(self, train: pd.DataFrame, target: str = "points") -> "ShrinkagePrior":
        d = train[train["fixtures"] > 0].copy()
        d["band"] = pd.cut(pd.to_numeric(d["cost"], errors="coerce"),
                           PRICE_BANDS, labels=False)
        self.fallback = float(d[target].mean())
        grp = d.groupby(["position", "band"], observed=True)[target].agg(["mean", "size"])
        # a band with few rows is itself unreliable: pull it toward the global
        # mean rather than trusting a noisy cell
        n = grp["size"].to_numpy(float)
        shrunk = (n * grp["mean"].to_numpy(float) + 50.0 * self.fallback) / (n + 50.0)
        self.table = {key: float(v) for key, v in zip(grp.index, shrunk)}
        return self

    def prior_for(self, frame: pd.DataFrame) -> np.ndarray:
        band = pd.cut(pd.to_numeric(frame["cost"], errors="coerce"),
                      PRICE_BANDS, labels=False)
        keys = list(zip(frame["position"], band))
        return np.array([self.table.get(k, self.fallback) for k in keys], float)

    def weight(self, frame: pd.DataFrame) -> np.ndarray:
        n = pd.to_numeric(frame.get("season_appearances_so_far"),
                          errors="coerce").fillna(0.0).to_numpy(float)
        if self.k <= 0:
            return np.ones_like(n)
        return n / (n + self.k)

    def apply(self, frame: pd.DataFrame, xp: np.ndarray) -> np.ndarray:
        w = self.weight(frame)
        return w * np.asarray(xp, float) + (1.0 - w) * self.prior_for(frame)


def select_k(train: pd.DataFrame, xp_col: str = "xp",
             target: str = "points") -> tuple[float, pd.DataFrame]:
    """
    Choose k on TRAINING data only, by held-out top-20 realised points.

    Top-20 rather than MAE because that is what the optimiser consumes
    (ledger R16), and because MAE on a bimodal outcome rewards predicting the
    mode (R13). The last training season is held out as the selection fold.
    """
    seasons = sorted(train["season"].unique())
    if len(seasons) < 2:
        return 3.0, pd.DataFrame()
    fit_on = train[train["season"] < seasons[-1]]
    pick_on = train[train["season"].eq(seasons[-1])]
    pick_on = pick_on[pick_on["fixtures"] > 0]
    rows = []
    for k in K_CANDIDATES:
        pri = ShrinkagePrior(k=k).fit(fit_on, target=target)
        adj = pri.apply(pick_on, pick_on[xp_col].to_numpy(float))
        tmp = pick_on.assign(_adj=adj)
        per = [g.nlargest(20, "_adj")[target].mean()
               for _, g in tmp.groupby("gameweek") if len(g) >= 50]
        rows.append({"k": k, "top20": float(np.mean(per))})
    tab = pd.DataFrame(rows)
    best = float(tab.loc[tab["top20"].idxmax(), "k"])
    return best, tab
