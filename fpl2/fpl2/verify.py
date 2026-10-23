"""
The checks that actually caught things, as code instead of prose.

Every rule in `fixes/EXPERIMENT_LEDGER.md` is a sentence I have to remember to
read. On 2026-10-23 four of them were broken on the same day BY THE PERSON WHO
WROTE THEM -- R29 (a grouping whose membership changes with the treatment),
R26 (no negative control), R38 (a monotone dose-response is a leak until
proven otherwise) and R43 (a correction monotone in the score cannot reorder
it). Each cost between twenty minutes and two hours.

A rule you have to remember is a rule you will break. These are the same
rules as functions, so an experiment can call them and fail loudly.

Nothing here knows about FPL. It is all "did this comparison measure what it
claims to measure".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class VerificationError(AssertionError):
    """A check that must pass before a result may be believed."""


# --------------------------------------------------------------------------
# R?? -- the reference arm must reproduce the shipped model EXACTLY
# --------------------------------------------------------------------------
def reference_identical(new, old, name: str = "reference", tol: float = 1e-9):
    """
    An arm meant to reproduce the shipped model must do so bit-identically.

    This caught two separate things on 2026-10-23: a real state leak (arms
    inheriting each other's refitted defensive model) and a non-bug
    (HistGradientBoosting sums histograms in parallel, so a different
    OMP_NUM_THREADS gives a different, equally valid answer -- 0.0016 at 8
    threads, 0.0000 at 4). Pin threads before trusting an exact control.
    """
    a = np.asarray(new, float); b = np.asarray(old, float)
    if a.shape != b.shape:
        raise VerificationError(f"{name}: shape {a.shape} vs {b.shape}")
    d = float(np.nanmax(np.abs(a - b)))
    if d > tol:
        raise VerificationError(
            f"{name}: max|new - old| = {d:.10f} > {tol}. The arm that should "
            f"reproduce the shipped model does not, so NOTHING else in this "
            f"experiment is interpretable. Check for state carried between "
            f"arms, and pin OMP_NUM_THREADS.")
    return d


# --------------------------------------------------------------------------
# R43 / R41 -- a change monotone in the score cannot reorder it
# --------------------------------------------------------------------------
def reorders(new, old, by=None, min_rho_drop: float = 1e-4) -> dict:
    """
    Does the treatment actually change the ORDERING, or only the level?

    `argmax` is blind to any monotone transform of the score. S54's ceiling
    clipped Salah in 32 of 32 gameweeks and moved the ranking by 2.4 places
    out of ~600; S56's conditional bonus had a 122x mechanism and a rank
    correlation of 0.9998. Both measured null. One call to this would have
    priced them before the experiment was built.

    Returns the within-group rank correlation and mean rank shift. A rho
    above ~0.999 means the decision cannot move much, whatever the mechanism.
    """
    s_new = pd.Series(np.asarray(new, float))
    s_old = pd.Series(np.asarray(old, float))
    g = (pd.Series(np.zeros(len(s_new))) if by is None
         else pd.Series(np.asarray(by)).reset_index(drop=True))
    rho, shift, top1 = [], [], []
    for _, idx in g.groupby(g).groups.items():
        a, b = s_new.loc[idx], s_old.loc[idx]
        if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
            continue
        rho.append(stats.spearmanr(a, b).statistic)
        shift.append(float((a.rank(ascending=False) - b.rank(ascending=False))
                           .abs().mean()))
        top1.append(float(a.idxmax() != b.idxmax()))
    out = {"rank_rho": float(np.nanmean(rho)) if rho else np.nan,
           "mean_rank_shift": float(np.mean(shift)) if shift else np.nan,
           "top1_changed": float(np.mean(top1)) if top1 else np.nan,
           "groups": len(rho)}
    out["reorders"] = bool(out["rank_rho"] < 1 - min_rho_drop)
    return out


# --------------------------------------------------------------------------
# R29 -- a grouping whose membership changes with the treatment is not
#        comparable across arms
# --------------------------------------------------------------------------
def stable_population(masks: dict, name: str = "population") -> dict:
    """
    Every arm must be scored on the SAME rows.

    On 2026-10-23 the tenure experiment ranked "top 10 by xp" using EACH ARM'S
    OWN xp. The shrink demoted movers out of the band -- 195 rows at K=0, 138
    at K=40 -- so the arms were scored on different populations and the fix
    appeared to make the bias WORSE. Fixing the population reversed it.
    """
    sizes = {k: int(np.asarray(v).sum()) for k, v in masks.items()}
    uniq = set(sizes.values())
    ok = len(uniq) == 1
    if not ok:
        raise VerificationError(
            f"{name}: arms are scored on DIFFERENT populations {sizes}. "
            f"Fix the population on one arm (usually the reference) and vary "
            f"only the prediction (R29).")
    return sizes


# --------------------------------------------------------------------------
# R38 -- a dose-response that never saturates is a leak until proven otherwise
# --------------------------------------------------------------------------
def dose_response(doses, effects, name: str = "dose-response") -> dict:
    """
    Real effects have an optimum. Monotone-to-the-boundary means the treatment
    tracks something that keeps increasing, and "how much of the future can I
    see" is the usual candidate.

    The opening-horizon sweep ran monotone to +312.8 at h=19 (t 4.52, 8/8
    seasons) and was F10's leak, already in the ledger. Measured honestly the
    sign flips.
    """
    d = np.asarray(doses, float); e = np.asarray(effects, float)
    o = np.argsort(d); d, e = d[o], e[o]
    diffs = np.diff(e)
    mono = bool(np.all(diffs >= -1e-12) or np.all(diffs <= 1e-12))
    peak_at_edge = bool(np.argmax(np.abs(e)) in (0, len(e) - 1))
    out = {"monotone": mono, "peak_at_boundary": peak_at_edge,
           "argmax_dose": float(d[int(np.argmax(np.abs(e)))])}
    out["suspect_leak"] = bool(mono and peak_at_edge)
    if out["suspect_leak"]:
        out["warning"] = (f"{name}: monotone across all {len(d)} doses with the "
                          f"peak at the boundary (dose {out['argmax_dose']:g}). "
                          f"Treat as a LEAK until a leak-free construction "
                          f"reproduces it (R38).")
    return out


# --------------------------------------------------------------------------
# R48 -- size the affected population before building the fix
# --------------------------------------------------------------------------
def price_ceiling(per_unit_effect: float, affected_rows: int,
                  total_rows: int, units_per_season: float = 38.0) -> dict:
    """
    What is the most this fix could possibly be worth?

    S58 measured a -0.85 pts/gameweek bias, which sounds material. The
    population carrying it was 0.6 players per gameweek league-wide, so the
    product was always ~1 point. The gate agreed: +1.0 against an MDE of 6.9.
    """
    share = affected_rows / max(total_rows, 1)
    per_gw = abs(per_unit_effect) * share * 11.0      # an XI, not the league
    return {"affected_share": share, "ceiling_per_season": per_gw * units_per_season,
            "note": "upper bound: assumes the fix removes the bias entirely"}


# --------------------------------------------------------------------------
# leak test with its own negative control
# --------------------------------------------------------------------------
def strictly_lagged(frame: pd.DataFrame, group: str, source: str,
                    feature: str, window: int, order: str = None,
                    sample: int = 2000) -> dict:
    """
    Verify `feature` is the mean of STRICTLY EARLIER `source` values, and
    verify the check can fail by running it against an unshifted copy.

    A leak test that has never failed is not a test (R26's sibling). The
    style features passed this at 0 mismatches while a deliberately unshifted
    version failed 1,034 times.
    """
    d = frame.sort_values([group] + ([order] if order else [])).reset_index(drop=True)
    bad = checked = 0
    leaky_bad = 0
    for _, g in d.groupby(group, sort=False):
        g = g.reset_index(drop=True)
        src = g[source].to_numpy(float); feat = g[feature].to_numpy(float)
        leaky = pd.Series(src).rolling(window, min_periods=2).mean().to_numpy()
        for n in range(2, len(g)):
            want = np.nanmean(src[max(0, n - window):n])
            if np.isfinite(feat[n]):
                checked += 1
                if abs(feat[n] - want) > 1e-9:
                    bad += 1
            if np.isfinite(leaky[n]) and abs(leaky[n] - want) > 1e-9:
                leaky_bad += 1
            if checked >= sample:
                break
        if checked >= sample:
            break
    if bad:
        raise VerificationError(
            f"{feature}: {bad}/{checked} rows READ THE FUTURE (R?? / M14).")
    if not leaky_bad:
        raise VerificationError(
            f"{feature}: the leak test could not be made to fail, so it is "
            f"vacuous. An unshifted control must mismatch.")
    return {"checked": checked, "mismatches": bad, "control_failures": leaky_bad}


# --------------------------------------------------------------------------
# R32 -- cluster by season, and always report the MDE
# --------------------------------------------------------------------------
def clustered(diff, groups, scale: float = 1.0) -> dict:
    """Season-clustered mean, t, p, CI and MDE. G-1 degrees of freedom."""
    d = np.asarray(diff, float) * scale
    g = np.asarray(groups)
    ok = np.isfinite(d)
    d, g = d[ok], g[ok]
    n, G = len(d), len(np.unique(g))
    if G < 2:
        raise VerificationError(
            f"clustered: G = {G}. A season-clustered SE needs at least two "
            f"clusters; a single-holdout process must defer reporting (R32).")
    mean = d.mean()
    ss = sum((d[g == k] - mean).sum() ** 2 for k in np.unique(g))
    se = np.sqrt((G / (G - 1)) * ss / n ** 2)
    t = mean / se if se > 0 else np.nan
    crit = stats.t.ppf(0.975, G - 1)
    wins = sum(1 for k in np.unique(g) if d[g == k].mean() > 0)
    return {"n": n, "G": G, "mean": mean, "se": se, "t": t,
            "p": 2 * (1 - stats.t.cdf(abs(t), G - 1)),
            "ci_lo": mean - crit * se, "ci_hi": mean + crit * se,
            "mde": 2.8 * se, "wins": wins,
            "resolvable": bool(abs(mean) > 2.8 * se)}
