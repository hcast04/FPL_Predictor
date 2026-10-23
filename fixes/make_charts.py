#!/usr/bin/env python3
"""Diagnostic charts for the FPL model audit. Writes PNGs to outputs/."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
INK_MUTED = "#8a8880"
GRID      = "#e6e5e1"
BLUE      = "#2a78d6"
RED       = "#d03b3b"
GRAY_MID  = "#c9c7c1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.family": "DejaVu Sans", "figure.dpi": 110,
})


def style(ax, title, sub=None):
    # title and subtitle are stacked ABOVE the axes; pad must clear both or
    # they overprint each other (they did on the first render)
    ax.set_title(title, fontsize=12.5, fontweight="bold", color=INK,
                 loc="left", pad=30)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9,
                color=INK_2, va="bottom", ha="left")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ---------------------------------------------------------------- figure 1
GW = np.arange(1, 8)
YOU = np.array([54, 67, 47, 63, 39, 30, 37])
AVG = np.array([50, 81, 51, 69, 48, 58, 55])
DEF_ = YOU - AVG

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.0))

w = 0.38
a1.bar(GW - w/2, YOU, w, color=BLUE, label="You", zorder=3)
a1.bar(GW + w/2, AVG, w, color=GRAY_MID, label="Global average", zorder=3)
for x, y in zip(GW, YOU):
    a1.text(x - w/2, y + 1.5, str(y), ha="center", fontsize=8.5, color=INK_2)
a1.set_xticks(GW); a1.set_xticklabels([f"GW{g}" for g in GW])
a1.set_ylabel("points"); a1.set_ylim(0, 92)
a1.grid(axis="x", visible=False)
a1.legend(frameon=False, loc="upper right", fontsize=9)
style(a1, "You scored below average in 6 of 7 gameweeks",
      "337 points against an average of 412")

cols = [BLUE if d >= 0 else RED for d in DEF_]
a2.bar(GW, DEF_, 0.62, color=cols, zorder=3)
a2.axhline(0, color=INK_MUTED, lw=1.0, zorder=4)
m = DEF_.mean()
a2.axhline(m, color=INK, lw=1.4, ls="--", zorder=5)
a2.axhspan(-20.3, -1.1, color=RED, alpha=0.08, zorder=1)
# label the mean line inside the axes, left of the bars, so it cannot clip
a2.text(0.62, m - 0.9, f"mean {m:+.1f}", va="top", ha="left", fontsize=9,
        color=INK, fontweight="bold")
for x, d in zip(GW, DEF_):
    # labels sit outside the bar end; nudge any that would land on the mean line
    off = 1.2 if d >= 0 else -1.2
    va = "bottom" if d >= 0 else "top"
    yy = d + off
    if abs(yy - m) < 1.6:
        yy = m - 2.2
    a2.text(x, yy, f"{d:+d}", ha="center", va=va, fontsize=8.5, color=INK_2)
a2.set_xticks(GW); a2.set_xticklabels([f"GW{g}" for g in GW])
a2.set_ylabel("points vs average")
a2.set_xlim(0.4, 7.6); a2.set_ylim(-33, 9)
a2.grid(axis="x", visible=False)
style(a2, "The gap is real, not variance",
      "t(6) = -2.72, p = 0.034 · shaded band = 95% CI [-20.3, -1.1]")
fig.tight_layout(rect=(0, 0.02, 1, 0.94))
fig.savefig("outputs/chart_season_diagnosis.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- figure 2
fig, (b1, b2) = plt.subplots(1, 2, figsize=(13.5, 5.0))

names = ["model xP", "template\n(most owned)", "form (5gw)", "form (3gw)",
         "most expensive", "points per £m", "random"]
vals = [2145, 1923, 1894, 1868, 1602, 1583, 644]
y = np.arange(len(vals))[::-1]
colr = [BLUE] + [GRAY_MID]*5 + [INK_MUTED]
b1.barh(y, vals, 0.62, color=colr, zorder=3)
for yy, v in zip(y, vals):
    b1.text(v + 25, yy, f"{v:,}", va="center", fontsize=9, color=INK_2)
b1.set_yticks(y); b1.set_yticklabels(names, fontsize=9)
b1.set_xlabel("season points"); b1.set_xlim(0, 2520)
b1.grid(axis="y", visible=False)
style(b1, "The model's xP beats every dumb baseline",
      "same optimiser, budget and rules · wins in all 3 holdout seasons")

labels = ["optimal\nstart", "top 5%\nremoved", "top 15%\nremoved", "top 30%\nremoved"]
tot = [2145, 1964, 1914, 1846]
x = np.arange(len(tot))
b2.bar(x, tot, 0.58, color=[BLUE, GRAY_MID, GRAY_MID, GRAY_MID], zorder=3)
for xx, v in zip(x, tot):
    b2.text(xx, v + 18, f"{v:,}", ha="center", fontsize=9, color=INK_2)
YOUR_PACE = 1829
b2.axhline(YOUR_PACE, color=RED, lw=1.6, ls="--", zorder=5)
b2.text(0.5, YOUR_PACE - 14, f"your pace {YOUR_PACE:,}", va="top", ha="center",
        fontsize=9, color=RED, fontweight="bold")
b2.set_xticks(x); b2.set_xticklabels(labels, fontsize=9)
b2.set_ylabel("season points"); b2.set_ylim(1700, 2260)
b2.set_xlim(-0.6, 3.6)
b2.grid(axis="x", visible=False)
b2.yaxis.set_major_locator(MultipleLocator(100))
style(b2, "A weak opening squad costs 181-299 points a season",
      "one free transfer a week does not recover it")
fig.tight_layout(rect=(0, 0.02, 1, 0.94))
fig.savefig("outputs/chart_what_we_tested.png", dpi=170)
plt.close(fig)
print("wrote outputs/chart_season_diagnosis.png")
print("wrote outputs/chart_what_we_tested.png")
