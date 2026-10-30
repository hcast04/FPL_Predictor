#!/usr/bin/env python3
"""
S64 / O61 — what does holding BOTH SIDES of a fixture actually cost?

Raised when the GW9 wildcard squad came back with ten of fifteen players in six
matches, five of them containing players from both teams: Haaland (MCI)
captained against Verbruggen and De Cuyper (BHA) in the same fixture.

The claim is about the COVARIANCE of realised points between two players. That
is an outcome, not a forecast, so eleven seasons of results answer it directly
and no model is fitted anywhere in this file.

  part 1  correlation of realised points by pair type, with the control that
          makes it readable: pairs from DIFFERENT matches in the same gameweek
          must come out at zero, because they cannot influence each other
  part 2  what the measured covariances do to a given XI's variance
  part 3  what breaking the captain's own conflict costs in expected points

Verdict recorded in the ledger: CONFIRMED MECHANISM, MARGINAL EFFECT. 11/11
seasons on every within-match class, worth 0.62 points of standard deviation.
"""
from __future__ import annotations
import itertools, json, sys, time, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fpl2 import config as C, optimise                            # noqa: E402

API = "https://fantasy.premierleague.com/api"
HDR = {"User-Agent": "Mozilla/5.0"}
MAXPAIR = 400_000
ATT = {"MID", "FWD"}


def _get(u):
    return json.load(urllib.request.urlopen(
        urllib.request.Request(u, headers=HDR), timeout=60))


def _cls(p1, p2):
    a1, a2 = p1 in ATT, p2 in ATT
    return ("att_vs_att" if a1 and a2
            else "def_vs_def" if not a1 and not a2 else "att_vs_def")


def pair_correlations(panel: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Part 1. Correlation of realised points for three pair populations."""
    rng = np.random.default_rng(seed)
    d = panel[(panel["fixtures"] == 1) & panel["opponents"].notna()].copy()
    d["opp"] = d["opponents"].map(
        lambda t: t[0] if isinstance(t, (tuple, list)) and len(t) else None)
    d = d[d["opp"].notna() & d["team"].notna() & d["points"].notna()]
    d["match"] = [f"{s}|{g}|{'-'.join(sorted([a, b]))}"
                  for s, g, a, b in zip(d.season, d.gameweek, d.team, d["opp"])]
    # a 0-minute row is not an outcome
    d = d[d["minutes"] > 0][["season", "gameweek", "match", "team", "opp",
                             "position", "points"]]
    print(f"  {len(d):,} player-fixtures, {d.match.nunique():,} matches, "
          f"{d.season.nunique()} seasons", flush=True)

    # Matches are sampled at RANDOM across all seasons. Iterating in panel
    # order and breaking at MAXPAIR drew every within-match pair from the
    # earliest four seasons while the control used all eleven -- the two
    # populations would then differ by era as well as by the thing under test.
    groups = list(d.groupby("match", sort=False))
    rows = []
    for i in rng.permutation(len(groups)):
        _m, g = groups[i]
        if len(g) < 4:
            continue
        a = g[g.team.eq(g.team.iloc[0])]
        b = g[g.team.ne(g.team.iloc[0])]
        if a.empty or b.empty:
            continue
        for x in range(len(a)):
            for y in range(len(b)):
                rows.append(("opposite",
                             _cls(a.position.iloc[x], b.position.iloc[y]),
                             a.points.iloc[x], b.points.iloc[y],
                             a.season.iloc[x], a.gameweek.iloc[x]))
        for side in (a, b):
            for x, y in itertools.combinations(range(len(side)), 2):
                rows.append(("same_team",
                             _cls(side.position.iloc[x], side.position.iloc[y]),
                             side.points.iloc[x], side.points.iloc[y],
                             side.season.iloc[x], side.gameweek.iloc[x]))
        if len(rows) > MAXPAIR:
            break

    ctrl = []
    for (s, gw), g in d.groupby(["season", "gameweek"], sort=False):
        if g.match.nunique() < 2 or len(g) < 20:
            continue
        g = g.iloc[rng.permutation(len(g))]
        half = len(g) // 2
        L = g.iloc[:half].reset_index(drop=True)
        R = g.iloc[half:2 * half].reset_index(drop=True)
        keep = L.match.values != R.match.values
        for xi, yi, p1, p2 in zip(L.points[keep], R.points[keep],
                                  L.position[keep], R.position[keep]):
            ctrl.append(("unrelated", _cls(p1, p2), xi, yi, s, gw))

    ALL = pd.concat([pd.DataFrame(rows, columns=["grp", "cls", "x", "y",
                                                 "season", "gameweek"]),
                     pd.DataFrame(ctrl, columns=["grp", "cls", "x", "y",
                                                 "season", "gameweek"])],
                    ignore_index=True)

    def demean(fr):
        """Remove each gameweek's own mean: a high-scoring week lifts everyone
        and would read as positive correlation in every group alike."""
        o = fr.copy()
        for c in ("x", "y"):
            o[c] = o[c] - o.groupby(["season", "gameweek"])[c].transform("mean")
        return o

    out = []
    for grp in ("opposite", "same_team", "unrelated"):
        for cls in ("att_vs_def", "att_vs_att", "def_vs_def"):
            sub = ALL[(ALL.grp == grp) & (ALL.cls == cls)]
            if len(sub) < 500:
                continue
            sd = demean(sub)
            per = [np.corrcoef(demean(q).x, demean(q).y)[0, 1]
                   for _, q in sd.groupby("season") if len(q) > 200]
            out.append({"grp": grp, "cls": cls, "n": len(sub),
                        "corr": float(np.corrcoef(sd.x, sd.y)[0, 1]),
                        "cov": float(np.cov(sd.x, sd.y)[0, 1]),
                        "neg": sum(1 for v in per if v < 0), "G": len(per)})
    return pd.DataFrame(out)


def squad_variance(xi: list, cov: dict, var: dict, captain: str) -> dict:
    """
    Part 2. Var(total) = sum Var_i + 2 sum Cov_ij.

    `xi` is [(name, team, position, match_id)]. The captain's points are
    DOUBLED, so his variance enters x4 and his covariances x2.
    """
    def w(n):
        return 2.0 if n == captain else 1.0
    var_ind = sum(w(n) ** 2 * var[p] for n, _t, p, _m in xi)
    pairs = []
    for (n1, t1, p1, m1), (n2, t2, p2, m2) in itertools.combinations(xi, 2):
        if m1 != m2:
            continue                  # different matches: the control says ~0
        grp = "same_team" if t1 == t2 else "opposite"
        c = cov[(grp, _cls(p1, p2))]
        pairs.append({"a": n1, "b": n2, "match": m1, "grp": grp,
                      "cls": _cls(p1, p2), "cov": c,
                      "contrib": 2 * w(n1) * w(n2) * c})
    P = pd.DataFrame(pairs)
    tot = float(P.contrib.sum()) if len(P) else 0.0
    return {"pairs": P, "var_independent": var_ind, "cov_contribution": tot,
            "var_total": var_ind + tot, "sd": float(np.sqrt(var_ind + tot)),
            "sd_if_unrelated": float(np.sqrt(var_ind))}


def main():
    t0 = time.time()
    panel = pd.read_pickle(C.cache("fpl2_panel_us.pkl"))

    print("S64.1  CORRELATION OF REALISED POINTS, BY PAIR TYPE", flush=True)
    R = pair_correlations(panel)
    print(f'\n{"group":<11}{"class":<13}{"n":>9}{"corr":>9}{"cov":>9}  seasons negative')
    for _, r in R.iterrows():
        print(f'{r.grp:<11}{r.cls:<13}{r.n:>9,}{r["corr"]:>9.4f}{r["cov"]:>9.4f}'
              f'  {r.neg}/{r.G}')
    print("\n  `unrelated` is the CONTROL: different matches cannot influence\n"
          "  each other, so it must read zero. If it does not, nothing above\n"
          "  is a statement about fixtures.", flush=True)
    R.to_pickle(C.cache("fpl2_s64_pairs.pkl"))

    played = panel[(panel["fixtures"] == 1) & (panel["minutes"] > 0)
                   & panel["points"].notna()]
    VAR = played.groupby("position")["points"].var().to_dict()
    COV = R.set_index(["grp", "cls"])["cov"].to_dict()

    # --- part 2/3 need the live squad; skip cleanly if it is not built ------
    proj = C.cache("fpl2_live_proj.pkl")
    if not proj.exists():
        print("\n(no live projection on disk; parts 2 and 3 need one)")
        return
    boot = _get(f"{API}/bootstrap-static/")
    nxt = next(e for e in boot["events"] if e.get("is_next"))
    target = int(nxt["id"])
    tm = {t["id"]: t["short_name"] for t in boot["teams"]}
    opp = {}
    for f in _get(f"{API}/fixtures/"):
        if f.get("event") != target:
            continue
        h, a = tm[f["team_h"]], tm[f["team_a"]]
        mid = "-".join(sorted([h, a]))
        opp[h] = (a, mid)
        opp[a] = (h, mid)
    el = pd.DataFrame([{"player_code": e["code"], "name": e["web_name"],
                        "cost": e["now_cost"] / 10, "team": tm[e["team"]],
                        "position": {1: "GK", 2: "DEF", 3: "MID",
                                     4: "FWD"}[e["element_type"]],
                        "status": e["status"],
                        "chance": e["chance_of_playing_next_round"]}
                       for e in boot["elements"]])
    pr = pd.read_pickle(proj)
    xp = (pr[pr.gameweek.eq(target)]
          .groupby("player_code", as_index=False).agg(xp=("xp", "mean")))
    pool = el.merge(xp, on="player_code", how="inner")
    pool = pool[pool.status.eq("a")
                & (pool.chance.isna() | (pool.chance >= 75))].reset_index(drop=True)

    sq = optimise.optimise_squad(pool, "xp", budget=99.7)
    s = sq.merge(pool.drop_duplicates("player_code"), on="player_code",
                 suffixes=("", "_p"))
    best = optimise.best_xi(s, "xp").reset_index(drop=True)
    cap = best.loc[best.xp.idxmax(), "name"]
    xi = [(r["name"], r.team, r.position, opp[r.team][1]) for _, r in best.iterrows()]

    print(f"\nS64.2  VARIANCE OF THE GW{target} XI (captain {cap})")
    V = squad_variance(xi, COV, VAR, cap)
    if len(V["pairs"]):
        print(V["pairs"].to_string(index=False,
                                   float_format=lambda v: f"{v:7.3f}"))
    print(f'\n  sum of individual variances  {V["var_independent"]:8.2f}')
    print(f'  covariance contribution      {V["cov_contribution"]:+8.2f}')
    print(f'  sd as built                  {V["sd"]:8.2f}')
    print(f'  sd if every pair unrelated   {V["sd_if_unrelated"]:8.2f}')
    print(f'  -> variance reduced by       '
          f'{100*(1-V["var_total"]/V["var_independent"]):7.1f}%')
    print("\n  E[sum] = sum E[.] whatever the covariance, so the expected\n"
          "  total is UNCHANGED. Only the spread moves.")

    foe, _mid = opp[best.loc[best.xp.idxmax(), "team"]]
    base = float(best.xp.sum()) + float(best.xp.max())
    ban = set(pool[pool.team.eq(foe)].player_code)
    alt = optimise.optimise_squad(pool, "xp", budget=99.7, banned=ban)
    print(f"\nS64.3  PRICE OF BREAKING THE CAPTAIN'S OWN CONFLICT (vs {foe})")
    if alt is None:
        print("  no legal squad without them")
        return
    a = alt.merge(pool.drop_duplicates("player_code"), on="player_code",
                  suffixes=("", "_p"))
    axi = optimise.best_xi(a, "xp")
    v1 = float(axi.xp.sum()) + float(axi.xp.max())
    print(f"  unconstrained      {base:.2f} xp")
    print(f"  {foe} banned        {v1:.2f} xp   cost {v1-base:+.2f} xp")
    print(f"  out: {sorted(set(best.name)-set(axi.name))}")
    print(f"  in : {sorted(set(axi.name)-set(best.name))}")
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
