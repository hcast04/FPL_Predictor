"""
Per-player reliability: has the model been wrong about THIS player?

Shrinkage toward a positional prior was rejected (see `shrink.py`) because it
flattened everyone to fix a few. The failure it was aimed at is specific: the
model rated Salah 6.33 and bought him at GBP 14.5m for 123 points, while being
fine about most players. So the correction should be per player, estimated from
the model's own track record on him.

    ratio_p = (sum actual + alpha) / (sum predicted + alpha)
    xp_adj  = xp * clip(ratio_p, lo, hi)

`alpha` shrinks the ratio toward 1 when the sample is small, so a player with
two appearances is left alone. That makes the correction self-limiting: it can
only fire where there is enough evidence to justify it.

Two sources of evidence, pooled with the current season weighted more:
  * LAST season, from an inner walk-forward (honest, out-of-sample). This is
    what makes a gameweek-1 correction possible at all.
  * THIS season so far, strictly lagged.

In-sample residuals must never be used. A model's errors on its own training
data are biased small and would make every player look reliable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

RATIO_LO, RATIO_HI = 0.60, 1.40     # a reliability estimate is a nudge, not a veto


@dataclass
class Reliability:
    alpha: float = 20.0
    prior_weight: float = 0.5        # weight on last season vs this season
    table: dict = None

    def fit_prior_season(self, scored: pd.DataFrame,
                         pred_col: str = "xp", actual_col: str = "points"):
        """
        Build per-player ratios from honest out-of-sample predictions.

        `scored` must be predictions on a season the model did NOT train on.
        """
        d = scored[scored["fixtures"] > 0]
        g = d.groupby(C.PLAYER_KEY).agg(pred=(pred_col, "sum"),
                                        act=(actual_col, "sum"),
                                        n=(pred_col, "size"))
        ratio = ((g["act"] + self.alpha) / (g["pred"] + self.alpha)).clip(RATIO_LO, RATIO_HI)
        self.table = {"prior": ratio.to_dict(), "prior_n": g["n"].to_dict()}
        return self

    def within_season(self, history: pd.DataFrame, pred_col: str = "xp",
                      actual_col: str = "points") -> tuple[dict, dict]:
        """Ratios from earlier gameweeks of the season being forecast."""
        if history is None or history.empty:
            return {}, {}
        d = history[history["fixtures"] > 0]
        if d.empty:
            return {}, {}
        g = d.groupby(C.PLAYER_KEY).agg(pred=(pred_col, "sum"),
                                        act=(actual_col, "sum"),
                                        n=(pred_col, "size"))
        ratio = ((g["act"] + self.alpha) / (g["pred"] + self.alpha)).clip(RATIO_LO, RATIO_HI)
        return ratio.to_dict(), g["n"].to_dict()

    def apply(self, frame: pd.DataFrame, xp: np.ndarray,
              history: pd.DataFrame | None = None) -> np.ndarray:
        """Scale xp by the pooled reliability ratio. Unknown players untouched."""
        prior = (self.table or {}).get("prior", {})
        cur, cur_n = self.within_season(history)
        codes = frame[C.PLAYER_KEY].to_numpy()
        out = np.asarray(xp, float).copy()
        w_prior = self.prior_weight
        for i, code in enumerate(codes):
            p = prior.get(code)
            c = cur.get(code)
            if p is None and c is None:
                continue                      # no evidence: leave it alone
            if c is None:
                r = p
            elif p is None:
                r = c
            else:
                # the current season is more relevant, and more so as it grows
                n = cur_n.get(code, 0)
                wc = n / (n + 6.0)
                r = wc * c + (1 - wc) * (w_prior * p + (1 - w_prior) * 1.0)
            out[i] *= float(np.clip(r, RATIO_LO, RATIO_HI))
        return out
