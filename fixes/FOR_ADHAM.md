---
to: Adham
from: Héctor (via a modelling session on my FPL pipeline)
date: 2026-10-19
---

# Things we tested that bear on your model

We spent a session properly testing several of the ideas you and I talked
about. Most of them **didn't help**, which is more useful to you than another
feature suggestion. Everything below was tested walk-forward on three held-out
seasons (2023-24, 2024-25, 2025-26), with the rule that a result only counts if
it wins in **all three**.

Big caveat up front: these were run on *my* feature set and *my* harness. A
negative here is not proof it won't work in yours — your features and targets
differ. Treat them as "this is harder than it looks", not "don't bother".

---

## 1. Read this one first: measure your noise floor

Before believing any result, perturb your predictions by a tiny amount and
re-run the whole thing.

I jittered my xP by **0.1%** — far smaller than any real modelling change — and
a single season's total moved by **up to 112 points**. At 1% jitter, the
standard deviation of the 3-season mean was 39.

**So any single-season improvement under ~150 points is noise.** I had several
"findings" smaller than that which evaporated. If you're comparing model
versions on one season, you are almost certainly reading randomness.

Corollary: with 3 seasons, a null signal wins all three **12.5%** of the time.
Across ~50 comparisons we ran, roughly **6 spurious "wins in all 3 seasons"**
are expected by chance alone. Count your comparisons.

---

## 2. Per-position models — didn't beat pooled

Your idea, the one I said I'd steal. Separate model per GK/DEF/MID/FWD vs one
pooled model with position as a feature, everything else identical:

| metric | 2023-24 | 2024-25 | 2025-26 | wins |
|---|---:|---:|---:|---:|
| spearman | −0.0090 | −0.0078 | −0.0045 | **0/3** |
| best legal XI pts | +3.42 | −1.97 | −0.88 | 1/3 |

Ranking got **worse in all three seasons**. The likely reason: splitting the
data four ways costs more than the position-specific structure gains, and a
GBDT already learns position interactions natively from a position feature.

Might still work for you if your per-position models use genuinely *different
features* per position (e.g. saves/clean-sheet features only for GK/DEF) rather
than the same feature matrix split four ways. That's a different experiment and
I didn't run it.

---

## 3. Understat team context (PPDA, npxG, deep completions) — added nothing

This was the gap I was most confident about, because my feature set has **zero**
team-level pressing or possession signal. I pulled Understat team match stats
for four seasons, built 5-game trailing form for PPDA, xG, npxG, deep
completions and expected points — for both the player's own team **and** the
opponent — 24 new features, **99.3% join coverage**, so the data was clean.

| metric | 2023-24 | 2024-25 | 2025-26 | wins |
|---|---:|---:|---:|---:|
| spearman | −0.0014 | −0.0003 | −0.0010 | **0/3** |
| best XI pts | −0.39 | +1.61 | −2.36 | 1/3 |

Nothing. My read: rolling team goals-for / goals-against already capture most of
what PPDA proxies for, and the extra columns are mostly noise on top. PPDA is a
*style* metric; FPL points care about *outcomes*.

---

## 4. "Use windows — avg xG over last 3/5/10" — as features yes, as a blend no

Two different things here, and they come out differently:

- **As features** (3/5/10-game trailing means as model inputs): fine, both our
  models already do this. No reason to change.
- **As a post-hoc blend** (mix model output with recent form): **failed**.
  Blending xP with the trailing 3-gameweek mean was worse at *every* weight in
  *every* season on the decision metric. Best weight = **0.0**.

---

## 5. Training on only the last X gameweeks — strictly worse

Héctor's instinct was "football is form, so just train on recent games". Swept
X ∈ {3, 5, 8, 12, 20, 38, 76, all} crossed with feature-window sets:

| training window | mean season pts |
|---|---:|
| 3 | 2050 |
| 8 | 2043 |
| 20 | 2103 |
| 38 | 2146 |
| **all history** | **2200** |

Roughly monotonic — **more data wins**. Short windows lose badly (up to −12.6
points per gameweek in one season). Form belongs in the features, not in
throwing away training data.

---

## 6. Before you build the 4–5 gameweek horizon: we found no benefit

You said you want to generalise from next-gameweek to the next 4–5. I tested
exactly that — sweeping the horizon used for squad and transfer decisions while
always *scoring* on the current gameweek only.

First result said **+196 points/season** for an 8-week horizon. It was
**leakage**: my cached forecast for gameweek g+8 had been generated walk-forward
using data up to g+7, so looking ahead was reading the future.

Rebuilt it so the only forward-looking information is the **published fixture
schedule** plus team strength known at decision time — which is all a real
manager has. Result:

| horizon | mean Δ vs 1-gameweek |
|---|---:|
| 2 | −169 |
| 3 | −78 |
| 4 | −84 |
| 6 | −90 |
| 8 | −104 |

**Every horizon from 2 to 10 was worse than just using next gameweek.** One
horizon (12) appeared to win all three seasons, but that turned out to be
*further* leakage in the "fixed" script — once cleaned, nothing beats H=1.

I'd still build it, because my fixture-ease proxy was crude (a goals-conceded
ratio) and a better one might genuinely help. But **budget time to validate it**
rather than assuming it's an upgrade, and watch hard for leakage — mine survived
two rounds of "I fixed it".

---

## 7. The one thing I'd prioritise instead

We found a real, replicated defect that nobody's hypothesis predicted:

**The model over-predicts players whose role shrank and under-predicts those
whose role grew.**

- signed points bias on role-downgraded players: **+0.124 pts/GW**
- significant in **all three seasons**, p < 0.0001, permutation p = 0.021
- it does **not** decay over a season, so it isn't a stale-prior problem and
  recency weighting won't fix it

Measure role change as (minutes-per-game in the first 4 gameweeks of the new
season) minus (last season's minutes-per-game), evaluated from GW5 onward so
there's no leakage. If your model has the same bias, that's a cheap, real fix.

---

## 8. On xP calibration — your numbers vs mine

Your GW8 output had Haaland at **8.17** and Rogers at 6.87. My model's single
highest xP in the *entire game* that week was **4.75** — zero players above 5.0.

That's a big divergence and I can't tell you who's right. What I measured on my
side:

- Historically (86k out-of-sample player-gameweeks), my model **under-predicts**
  at the top: bias **−0.69** for players at xP ≥ 3.5.
- But in the live 2026-27 season so far, it **over**-predicts there (+0.46 at
  xP ≥ 3.5, n=72, both gameweeks agreeing in sign).

Opposite signs, unresolved. Worth checking your own: bucket your predictions and
compare bucket means to realised points. If your 8.17 players are realising ~6,
you're over-confident at the top in a way that hits captaincy hardest — the
captain is by definition your maximum prediction, so it's where calibration
error costs double.

---

## 9. The optimizer gap is bigger than it looks

You said you don't have an optimizer, just xPoints. Worth quantifying what that
costs, because it surprised me.

I ran a season simulation under real constraints (£100m, 2/5/5/3, max 3 per
club, one free transfer, no hits), swapping **only** the ranking signal:

| signal | mean season total |
|---|---:|
| my model xP | **2145** |
| most-owned (template) | 1923 |
| 5-game form | 1894 |
| most expensive | 1602 |
| random | 644 |

My model beats the template by 222/season. **But when I held the XI and captain
choice fixed and swapped only the squad/transfer signal, the gap dropped to
79–152.** So roughly **half to two-thirds of a good model's advantage comes from
XI and captain selection**, not from picking a better squad.

That's the part a pure-xPoints setup leaves on the table: budget-constrained
squad selection, bench ordering, and captaincy under the 1 GK / 3-5 DEF /
2-5 MID / 1-3 FWD constraint. An exact MILP over 15 players is maybe 60 lines
with `scipy.optimize.milp` and it's where the points are.

One caution against my own earlier instinct: I *assumed* concentrating starters
in one fixture was a mistake, and advised it twice. When I finally tested
capping it, it **lost on both mean and downside in all three seasons**.
Concentration in good fixtures is where the points are. Don't add that rule.

---

## 10. Things I tested and rejected, so you don't repeat them

| idea | verdict |
|---|---|
| Rank-based bonus model (predict BPS, simulate the 3/2/1 ranking) | recovers dispersion but **ranks worse** than a simple per-90 rate |
| Value-aware transfer tiebreak (break xP ties on predicted price rises) | +7 one season, −0.7 the next. Price prediction itself is excellent though — AUC **0.95** for rises |
| Variance-adjusted captaincy (prefer upside over mean) | failed in 2 of 3 seasons |
| Direct-GBM xP replacing a component model (appearance × goals × assists × CS …) | lost to the component approach |

That last one may interest you most, since XGBoost-direct-to-points is your
architecture. In my setup the component model won — but my direct-GBM was a
fair-effort baseline, not a tuned one, so treat it as "not automatically
better", not "worse".

---

## Summary

- Measure your **noise floor** before trusting any comparison.
- **Per-position models** and **Understat team context** didn't help here.
- **Don't shorten your training window**; keep windows as features.
- **Validate the multi-gameweek horizon** before trusting it, and hunt for
  leakage twice.
- **Role-change miscalibration** is the real defect we found — worth checking.
- **Check your top-end calibration** — your numbers are much higher than mine.
- **An optimizer is worth more than it looks**, and it's mostly XI + captain.
