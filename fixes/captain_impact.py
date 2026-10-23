"""
How often does the v2.6 fix change the captain pick, and does it help?

Uses the notebook's PRODUCTION captain rule, not a naive argmax. Production
maximises the joint captain/vice objective

    uplift = captain_xP + P(captain does not appear) * vice_xP

over the starting XI (PRODUCTION_CAPTAIN_HIGHEST_XP is False). An earlier
revision of this script picked the single highest-xP player, which is the
*fallback* branch, so its result did not describe production behaviour.

This harness has no squad optimiser, so the XI is proxied by the top 11 by xP
within each gameweek. That is a proxy, not a reproduction -- treat the output
as indicative of direction, not as a production validation.
"""
import sys; sys.path.insert(0,".")
import numpy as np, pandas as pd
import validate_xp_fixes as V, fpl_xp_fixes as fx

def run(test_season):
    train_seasons=[s for s in V.SEASONS if s<test_season]
    raw=V.add_history_features(V.add_team_strength(V.load_seasons(True)))
    tr=raw[raw.season.isin(train_seasons)].copy(); te=raw[raw.season.eq(test_season)].copy()
    te=te[te.gameweek>=6].copy()
    V.FEATURE_CLIPS.clear()
    td=tr.reindex(columns=V.FEATURES).astype(float).replace([np.inf,-np.inf],np.nan)
    for c in V.FEATURES:
        s=td[c].dropna()
        if len(s)>100: V.FEATURE_CLIPS[c]=(float(s.quantile(.001)),float(s.quantile(.999)))
    fc=V.build_forecast(tr,te).reset_index(drop=True)
    af=fx.fit_event_conversion_factors(tr,realised_col="assists",expected_col="expected_assists",
                                       bounds=fx.ASSIST_CONVERSION_BOUNDS)
    gf=fx.fit_event_conversion_factors(tr,realised_col="goals_scored",expected_col="expected_goals",
                                       bounds=fx.GOAL_CONVERSION_BOUNDS)
    base=V.components(fc,V.GOAL_POINTS_BASELINE)
    fixed=fx.apply_event_conversion(base.copy(),fc,assist_factors=af,goal_factors=gf,
                                    goal_points=fx.GOAL_POINTS_BY_POSITION)
    d=fc.assign(base=base.expected_points.to_numpy(), fixed=fixed.expected_points.to_numpy())
    # captain = highest xP among credible starters, as the notebook does
    d=d[d.expected_minutes>=45]
    def joint_captain(xi, col):
        """Production rule: argmax over (captain, vice) of the joint uplift."""
        p = xi[col].to_numpy(float)
        miss = 1.0 - xi["appearance_probability"].to_numpy(float).clip(0.0, 1.0)
        n = len(p)
        best, pair = -np.inf, (0, 0)
        for c in range(n):
            for v in range(n):
                if c == v:
                    continue
                u = p[c] + miss[c] * p[v]
                if u > best:
                    best, pair = u, (c, v)
        return xi.iloc[pair[0]], xi.iloc[pair[1]]

    changed=same=0; bpts=[]; fpts=[]
    for gw,g in d.groupby("gameweek"):
        if len(g)<30: continue
        xi_b = g.nlargest(11, "base")      # proxy starting XI
        xi_f = g.nlargest(11, "fixed")
        cb,_ = joint_captain(xi_b, "base")
        cf,_ = joint_captain(xi_f, "fixed")
        bpts.append(cb.actual_points); fpts.append(cf.actual_points)
        if int(cb.element)==int(cf.element): same+=1
        else: changed+=1
    n=same+changed
    print(f"  {test_season}: captain pick changed in {changed}/{n} gameweeks ({100*changed/n:.0f}%)")
    print(f"     captain actual points per GW -- baseline {np.mean(bpts):5.2f}   "
          f"v2.6 {np.mean(fpts):5.2f}   delta {np.mean(fpts)-np.mean(bpts):+5.2f}")
    print(f"     doubled (captain counts twice):                              "
          f"delta {2*(np.mean(fpts)-np.mean(bpts)):+5.2f} pts/GW")
    return np.mean(fpts)-np.mean(bpts)

print("CAPTAIN IMPACT OF THE v2.6 FIX (held-out seasons)")
print("Production joint captain/vice objective, XI proxied by top 11 xP\n")
ds=[run(s) for s in ("2024-25","2025-26")]
print(f"\n  mean captain delta across both seasons: {np.mean(ds):+.2f} pts/GW "
      f"({2*np.mean(ds):+.2f} doubled)")
print("\n  NOTE: ~33 gameweeks per season is a tiny sample for a single pick")
print("  per week. Treat the direction as weakly informative and the magnitude")
print("  as very uncertain. This is NOT a production captain validation --")
print("  the starting XI is proxied, not optimised.")
