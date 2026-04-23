"""Figures for Section 7.6 Model Selection Rationale.

  fig_rat_rows.png    — nine-row rationale map (row → model → lesson)
  fig_rat_routing.png — primary-row segment routing decision diagram
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report_figures_v3"
OUT.mkdir(exist_ok=True)

MOVIE   = "#3B82F6"
GAME    = "#EF4444"
USER    = "#8B5CF6"
NEUTRAL = "#64748B"
ACCENT  = "#F59E0B"
TEAL    = "#14B8A6"
GREEN   = "#10B981"
PINK    = "#EC4899"
INK     = "#0F172A"
MUTED   = "#94A3B8"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "figure.facecolor": "white",
})


def chip(ax, x, y, w, h, text, color, fc, fontsize=10, text_color=INK,
         fontweight="bold"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fc, edgecolor=color, linewidth=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight=fontweight)


def arrow(ax, x1, y1, x2, y2, color=NEUTRAL, lw=1.3, style="->",
          connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color, lw=lw,
        mutation_scale=12, connectionstyle=connectionstyle))


def setup(ax, xlim, ylim):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


# ── Figure A: nine-row rationale map ────────────────────────────────────

def fig_rat_rows():
    """Each row: name chip — model chip — motivating lesson chip."""
    fig, ax = plt.subplots(figsize=(12, 8.2), dpi=180)
    setup(ax, (0, 12), (0, 9.2))

    rows = [
        ("Top Picks\n(segment-routed)",    "LightGCN / PTUPCDR /\npopularity",  "Lesson 2 + 6",  USER),
        ("Based on Your\nMovie Taste",     "EMCDR cross-domain",                "Lesson 4",      ACCENT),
        ("Because Others\nLiked …",        "Cooc standalone",                   "Lesson 8",      TEAL),
        ("Hidden Gems",                    "SBERT-CDR on niche",                "Lesson 7",      PINK),
        ("Movies You Might\nLike",         "SBERT movie-similarity",            "Lesson 7",      MOVIE),
        ("Popular with\nStrategy Fans",    "LightGCN-Movies",                   "Lesson 3",      MOVIE),
        ("Because You\nPlay Games",        "Reverse cooc (G → M)",              "Lesson 8",      GAME),
        ("Popular Games",                  "Popularity baseline",               "Lesson 6",      NEUTRAL),
        ("Popular Movies",                 "Popularity baseline",               "Lesson 6",      NEUTRAL),
    ]

    # Column headers
    ax.text(1.6,  8.8, "ROW", ha="center", fontsize=11,
            color=INK, fontweight="bold")
    ax.text(5.7,  8.8, "MODEL", ha="center", fontsize=11,
            color=INK, fontweight="bold")
    ax.text(9.8,  8.8, "MOTIVATING LESSON", ha="center", fontsize=11,
            color=INK, fontweight="bold")

    # Header underline
    ax.plot([0.2, 11.8], [8.55, 8.55], color=MUTED, lw=0.8)

    row_h = 0.82
    for i, (row, model, lesson, col) in enumerate(rows):
        y = 7.6 - i * row_h
        # row chip
        chip(ax, 0.2, y, 2.8, 0.66, row, col, "white",
             fontsize=9.5, text_color=INK)
        # model chip
        chip(ax, 4.1, y, 3.2, 0.66, model, NEUTRAL, "#F8FAFC",
             fontsize=9.5, text_color=INK, fontweight="normal")
        # lesson chip
        chip(ax, 8.4, y, 2.8, 0.66, lesson, col, _tint(col),
             fontsize=10, text_color=INK)
        # connectors
        arrow(ax, 3.0, y + 0.33, 4.1, y + 0.33, color=MUTED, lw=1.0)
        arrow(ax, 7.3, y + 0.33, 8.4, y + 0.33, color=MUTED, lw=1.0)

    # Bottom caption
    ax.text(6.0, 0.35,
            "Every row is traceable to a lesson finding — rationale maps one-to-one to empirical evidence",
            ha="center", fontsize=10, color=MUTED, style="italic")

    fig.suptitle("Model selection rationale — nine rows, one lesson each",
                 fontsize=14, fontweight="bold", y=0.98, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig_rat_rows.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def _tint(hex_color):
    """Very light tint of a hex color (95% white)."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hex_color)
    return (0.93 + r * 0.07, 0.93 + g * 0.07, 0.93 + b * 0.07)


# ── Figure B: primary-row segment routing ───────────────────────────────

def fig_rat_routing():
    fig, ax = plt.subplots(figsize=(12, 6.4), dpi=180)
    setup(ax, (0, 12), (0, 7.2))

    # Root decision
    chip(ax, 4.5, 5.9, 3.0, 0.9,
         "user's game-rating count",
         USER, "#F5F3FF", fontsize=11, text_color=INK)

    # Three branches: 0 / 1–2 / 3+
    branch_labels = [
        ("n = 0",       1.0,  "#FEF2F2", GAME),
        ("n = 1–2",     5.0,  "#FFFBEB", ACCENT),
        ("n ≥ 3",       9.0,  "#ECFDF5", GREEN),
    ]
    for text, x, fc, col in branch_labels:
        chip(ax, x, 4.0, 2.0, 0.6, text, col, fc,
             fontsize=10, text_color=INK)

    # arrows root → branch
    arrow(ax, 5.1, 5.9, 2.0, 4.6, color=GAME,
          connectionstyle="arc3,rad=0.15", lw=1.6)
    arrow(ax, 6.0, 5.9, 6.0, 4.6, color=ACCENT, lw=1.6)
    arrow(ax, 6.9, 5.9, 10.0, 4.6, color=GREEN,
          connectionstyle="arc3,rad=-0.15", lw=1.6)

    # Model chips under branches
    models = [
        ("popularity\n+ cold-start SBERT-CDR",  1.0,  GAME,    "#FEF2F2"),
        ("PTUPCDR cross-domain\ntransfer",      5.0,  ACCENT,  "#FFFBEB"),
        ("LightGCN\nin-domain",                 9.0,  GREEN,   "#ECFDF5"),
    ]
    for text, x, col, fc in models:
        chip(ax, x, 2.3, 2.0, 1.0, text, col, fc,
             fontsize=10, text_color=INK)

    for x, col in [(1.0, GAME), (5.0, ACCENT), (9.0, GREEN)]:
        arrow(ax, x + 1.0, 4.0, x + 1.0, 3.3, color=col, lw=1.4)

    # Motivation footnotes
    ax.text(2.0, 1.7, "no target-side signal —\nuse what everyone else liked",
            ha="center", fontsize=9, color=NEUTRAL, style="italic")
    ax.text(6.0, 1.7, "thin warm history —\ntransfer from movies",
            ha="center", fontsize=9, color=NEUTRAL, style="italic")
    ax.text(10.0, 1.7, "enough signal to train\ncollaborative model",
            ha="center", fontsize=9, color=NEUTRAL, style="italic")

    # Bottom pill summarising rule
    chip(ax, 3.0, 0.3, 6.0, 0.7,
         "Top-Picks row switches model per user — never one model for everyone",
         USER, "#F5F3FF", fontsize=10.5, text_color=INK)

    fig.suptitle("Primary row — segment-routed top picks",
                 fontsize=14, fontweight="bold", y=0.98, color=INK)
    fig.text(0.5, 0.92,
             "A single row, three models — picked by how much target-domain signal the user has",
             ha="center", fontsize=10.5, color=MUTED, style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / "fig_rat_routing.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    fig_rat_rows()
    fig_rat_routing()
    print("Saved rationale figures to", OUT)
