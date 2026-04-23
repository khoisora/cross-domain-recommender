"""Per-lesson illustrations for §5.1–§5.8.

Each figure tells the specific story the lesson isolates:
  L1 — pointwise (predict rating) vs pairwise (rank pairs correctly)
  L2 — 5.2%-overlap funnel and the resulting Recall gap
  L3 — overlap lift: 5.2% → 100% boosts every CDR model
  L4 — source-rich / target-sparse gap collapse
  L5 — catalog sharpening before/after
  L6 — user-split cold-start schematic
  L7 — niche subgroup: SBERT 11× over LightGCN
  L8 — co-occurrence rerank as a post-processing layer
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow, Circle, Rectangle
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "report_figures_v3"
OUT.mkdir(exist_ok=True)

# House palette (matches existing v3 figures).
NAVY = "#1F3A5F"
TEAL = "#2EC4B6"
BLUE = "#4472C4"
PINK = "#EC4899"
AMBER = "#F0A202"
GREEN = "#10B981"
RED = "#E5484D"
PANEL = "#F6F7F9"
GREY = "#B9BDC7"
INK = "#111827"


def _card(ax, x, y, w, h, face, alpha=0.12, stroke=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.2 if stroke else 0,
                                facecolor=face, alpha=alpha,
                                edgecolor=stroke or face))


def _setup(w=12, h=6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=170)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


# ─── L1: pointwise vs pairwise ─────────────────────────────────────────

def lesson1():
    fig, ax = _setup(12, 5.6)

    ax.text(6, 5.25, "Two ways to teach a model to recommend",
            ha="center", fontsize=14, fontweight="bold", color=INK)

    # Left panel: pointwise
    _card(ax, 0.3, 0.6, 5.5, 4.2, BLUE, alpha=0.08, stroke=BLUE)
    ax.text(3.05, 4.4, "Pointwise (regression)",
            ha="center", fontsize=12, fontweight="bold", color=BLUE)
    ax.text(3.05, 4.05, "“Predict the rating”",
            ha="center", fontsize=10, color=INK, style="italic")

    # Item → predicted score
    ax.add_patch(FancyBboxPatch((0.7, 2.6), 1.4, 0.9,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="white", edgecolor=BLUE, linewidth=1.5))
    ax.text(1.4, 3.05, "Movie A", ha="center", va="center", fontsize=10, color=INK)

    ax.annotate("", xy=(3.3, 3.05), xytext=(2.15, 3.05),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
    ax.text(2.7, 3.35, "MSE", ha="center", fontsize=9, color=BLUE, fontweight="bold")

    ax.add_patch(FancyBboxPatch((3.4, 2.6), 1.9, 0.9,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=BLUE, alpha=0.9, edgecolor=BLUE))
    ax.text(4.35, 3.05, "4.2 ★", ha="center", va="center", fontsize=14,
            fontweight="bold", color="white")

    ax.text(3.05, 1.9, "Loss = (predicted − actual)²",
            ha="center", fontsize=10, color=INK)
    ax.text(3.05, 1.3, "Optimises rating fidelity.",
            ha="center", fontsize=9.5, color="#4B5563")
    ax.text(3.05, 0.95, "Does nothing to separate items for top-K.",
            ha="center", fontsize=9.5, color=RED, fontweight="bold")

    # Right panel: pairwise
    _card(ax, 6.2, 0.6, 5.5, 4.2, TEAL, alpha=0.08, stroke=TEAL)
    ax.text(8.95, 4.4, "Pairwise (ranking, BPR)",
            ha="center", fontsize=12, fontweight="bold", color=TEAL)
    ax.text(8.95, 4.05, "“Rank the positive above the negative”",
            ha="center", fontsize=10, color=INK, style="italic")

    # Positive item
    ax.add_patch(FancyBboxPatch((6.6, 3.15), 1.4, 0.8,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="white", edgecolor=TEAL, linewidth=1.5))
    ax.text(7.3, 3.55, "Movie A ✓", ha="center", va="center",
            fontsize=10, color=INK, fontweight="bold")
    # Negative item
    ax.add_patch(FancyBboxPatch((6.6, 2.1), 1.4, 0.8,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="white", edgecolor=GREY, linewidth=1.2))
    ax.text(7.3, 2.5, "Movie B", ha="center", va="center", fontsize=10, color="#6B7280")

    ax.annotate("", xy=(9.4, 3.55), xytext=(8.1, 3.55),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=2))
    ax.annotate("", xy=(9.4, 2.5), xytext=(8.1, 2.5),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=2))

    ax.add_patch(FancyBboxPatch((9.5, 3.15), 1.7, 0.8,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=TEAL, alpha=0.9, edgecolor=TEAL))
    ax.text(10.35, 3.55, "score 2.1", ha="center", va="center", fontsize=11,
            fontweight="bold", color="white")

    ax.add_patch(FancyBboxPatch((9.5, 2.1), 1.7, 0.8,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="white", edgecolor=GREY))
    ax.text(10.35, 2.5, "score 0.4", ha="center", va="center", fontsize=11,
            color="#6B7280")

    ax.text(8.95, 1.6, "Loss = −log σ( score(A) − score(B) )",
            ha="center", fontsize=10, color=INK)
    ax.text(8.95, 1.0, "Directly optimises top-K ordering — the metric we ship.",
            ha="center", fontsize=9.5, color=GREEN, fontweight="bold")

    # Bottom legend
    ax.text(6, 0.25, "Lesson 1 result: BPR wins 5.2× on Recall@10 over MF-Explicit.",
            ha="center", fontsize=10.5, color=NAVY, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "fig_lesson1_pointwise_pairwise.png",
                bbox_inches="tight", dpi=170)
    plt.close(fig)


# ─── L2: 5.2% overlap funnel + bar snippet ─────────────────────────────

def lesson2():
    fig, ax = _setup(12, 5.6)

    ax.text(6, 5.25, "Lesson 2: realistic overlap is only 5.2%",
            ha="center", fontsize=14, fontweight="bold", color=INK)

    # Left: concentric population funnel. Thin-lens overlap ≈ 10%.
    _card(ax, 0.3, 0.5, 5.5, 4.4, BLUE, alpha=0.05, stroke=GREY)
    # Big movie circle
    ax.add_patch(Circle((1.9, 2.6), 1.4, facecolor=BLUE, alpha=0.25, edgecolor=BLUE, lw=1.5))
    ax.text(1.4, 3.9, "Movie raters\n(dense)", ha="center", fontsize=9.5,
            color=BLUE, fontweight="bold")

    # Game circle with modest overlap
    ax.add_patch(Circle((4.0, 2.6), 1.0, facecolor=TEAL, alpha=0.25, edgecolor=TEAL, lw=1.5))
    ax.text(4.55, 3.85, "Game raters\n(sparse)", ha="center", fontsize=9.5,
            color=TEAL, fontweight="bold")

    # Intersection marker
    ax.add_patch(Circle((3.15, 2.6), 0.18, facecolor=PINK, edgecolor=PINK, lw=1.5))
    ax.annotate("5.2% overlap\n52,281 users",
                xy=(3.15, 2.6), xytext=(3.3, 0.85),
                fontsize=9.5, color=PINK, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=PINK, lw=1.3))

    # Right: bar snippet
    _card(ax, 6.3, 0.5, 5.5, 4.4, PANEL, alpha=0.6, stroke=GREY)
    ax.text(9.05, 4.4, "Recall@10 on the mixed cohort",
            ha="center", fontsize=11, fontweight="bold", color=INK)

    models = ["LightGCN", "MF-BPR", "EMCDR", "NCF", "PTUPCDR", "CMF"]
    vals = [0.0290, 0.0170, 0.0160, 0.0120, 0.0085, 0.0050]
    colors = [BLUE, BLUE, TEAL, BLUE, TEAL, TEAL]
    y = np.linspace(3.7, 1.0, len(models))
    # LightGCN bar tip is the reference for gap arrows.
    ref_tip_x = 7.4 + vals[0] * 120
    for yi, m, v, c in zip(y, models, vals, colors):
        ax.add_patch(Rectangle((7.4, yi - 0.13), v * 120, 0.26,
                               facecolor=c, alpha=0.85))
        ax.text(7.3, yi, m, ha="right", va="center", fontsize=9.5, color=INK)
        ax.text(7.5 + v * 120, yi, f" {v:.4f}", va="center", fontsize=9,
                color=INK, fontweight="bold")
    # Red gap arrows from LightGCN-tip down to each CDR bar tip.
    lg_tip_y = y[0]
    for m, v, yi in zip(models[2:], vals[2:], y[2:]):
        tip_x = 7.4 + v * 120
        ax.annotate("", xy=(tip_x, yi),
                    xytext=(ref_tip_x, lg_tip_y),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.1,
                                    alpha=0.45, linestyle="--"))

    # Legend dots
    ax.add_patch(Circle((7.1, 0.65), 0.08, facecolor=BLUE))
    ax.text(7.3, 0.65, "single-domain", va="center", fontsize=9, color=INK)
    ax.add_patch(Circle((9.0, 0.65), 0.08, facecolor=TEAL))
    ax.text(9.2, 0.65, "cross-domain (CDR)", va="center", fontsize=9, color=INK)

    ax.text(6, 0.15, "CDR trails single-domain: mapping networks starve on 52K shared users.",
            ha="center", fontsize=10, color=RED, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "fig_lesson2_overlap_funnel.png",
                bbox_inches="tight", dpi=170)
    plt.close(fig)


# ─── L3: overlap lift arrow ────────────────────────────────────────────

def lesson3():
    fig, ax = _setup(12, 5.4)

    ax.text(6, 5.05, "Lesson 3: force 100% overlap, CDR recovers",
            ha="center", fontsize=14, fontweight="bold", color=INK)

    # Shared scale so bars fit their cards and leave room for value + lift labels.
    SCALE = 30
    # Before box
    _card(ax, 0.3, 0.5, 4.5, 4.0, GREY, alpha=0.12, stroke=GREY)
    ax.text(2.55, 4.1, "Lesson 2 (5.2% overlap)",
            ha="center", fontsize=11, fontweight="bold", color="#4B5563")
    bars_before = [("LightGCN", 0.0290, BLUE),
                   ("PTUPCDR",  0.0085, TEAL),
                   ("CMF",      0.0050, TEAL),
                   ("EMCDR",    0.0160, TEAL)]
    ys = np.linspace(3.5, 1.2, len(bars_before))
    before_ends = []  # (xi, yi) of each bar tip for delta arrows
    for (m, v, c), yi in zip(bars_before, ys):
        ax.add_patch(Rectangle((1.5, yi - 0.12), v * SCALE, 0.24,
                               facecolor=c, alpha=0.85))
        ax.text(1.4, yi, m, ha="right", va="center", fontsize=9.5, color=INK)
        ax.text(1.55 + v * SCALE, yi, f" {v:.4f}", va="center",
                fontsize=8.5, color="#4B5563")
        before_ends.append((1.55 + v * SCALE, yi))

    # After box — widened slightly and bars capped so the lift badge has room.
    _card(ax, 7.2, 0.5, 4.5, 4.0, TEAL, alpha=0.12, stroke=TEAL)
    ax.text(9.45, 4.1, "Lesson 3 (100% overlap)",
            ha="center", fontsize=11, fontweight="bold", color=TEAL)
    bars_after = [("LightGCN", 0.0555, BLUE, "+91%"),
                  ("PTUPCDR",  0.0320, TEAL, "+276%"),
                  ("CMF",      0.0140, TEAL, "+180%"),
                  ("EMCDR",    0.0245, TEAL, "+53%")]
    for (m, v, c, pct), yi, (bx, by) in zip(bars_after, ys, before_ends):
        bar_len = min(v * SCALE, 2.1)  # cap so values stay inside the card
        ax.add_patch(Rectangle((8.4, yi - 0.12), bar_len, 0.24,
                               facecolor=c, alpha=0.9))
        ax.text(8.3, yi, m, ha="right", va="center", fontsize=9.5, color=INK)
        ax.text(8.45 + bar_len, yi, f" {v:.4f}", va="center",
                fontsize=8.5, color=INK, fontweight="bold")
        # Lift badge sits just outside the card.
        ax.text(11.85, yi, pct, va="center", ha="right", fontsize=10,
                color=GREEN, fontweight="bold")
        # Per-model delta arrow from L2 bar tip to L3 bar tip.
        ax.annotate("", xy=(8.4, yi), xytext=(bx + 0.05, by),
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6,
                                    alpha=0.75))

    # Big label between cards (no long arrow — per-model arrows carry the story).
    ax.text(6, 2.95, "overlap\nfilter", ha="center", fontsize=10,
            color=PINK, fontweight="bold")
    ax.text(6, 1.95, "19,880 users\n100% overlap", ha="center", fontsize=9,
            color="#4B5563")

    ax.text(6, 0.15, "Same six models, same protocol — only the cohort changed.",
            ha="center", fontsize=10, color=INK, style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "fig_lesson3_overlap_lift.png",
                bbox_inches="tight", dpi=170)
    plt.close(fig)


# ─── L4: gap-to-LightGCN collapses ─────────────────────────────────────

def lesson4():
    fig, ax = _setup(12, 5.4)

    ax.text(6, 5.05, "Lesson 4: source-rich users, CDR closes on LightGCN",
            ha="center", fontsize=14, fontweight="bold", color=INK)

    # Two grouped bars per model: L3 gap vs L4 gap (absolute % behind LightGCN)
    models = ["PTUPCDR", "EMCDR", "CMF", "NCF", "MF-BPR"]
    l3_gap = [42, 56, 75, 59, 91]
    l4_gap = [17, 34, 54, 60, 80]

    x = np.arange(len(models))
    w = 0.35
    # Hand-draw bars (no mpl xtick); use custom axes region 1.0..11
    left, right = 1.2, 11.2
    bar_region_w = right - left - 0.6
    step = bar_region_w / len(models)

    # Guide gridlines
    for pct, y in zip([0, 25, 50, 75, 100],
                      [1.2, 1.95, 2.7, 3.45, 4.2]):
        ax.plot([left, right], [y, y], color=GREY, alpha=0.35, lw=0.8)
        ax.text(left - 0.05, y, f"{pct}%", ha="right", va="center",
                fontsize=8, color="#6B7280")
    ax.text(left - 0.2, 4.8, "% behind LightGCN",
            ha="left", va="center",
            fontsize=9.5, color=INK, fontweight="bold")

    # Bars
    for i, (m, g3, g4) in enumerate(zip(models, l3_gap, l4_gap)):
        xc = left + step * i + step / 2
        # L3 (grey)
        ax.add_patch(Rectangle((xc - 0.32, 1.2), 0.28, g3 * 0.03,
                               facecolor=GREY, alpha=0.65))
        # L4 (teal)
        ax.add_patch(Rectangle((xc + 0.02, 1.2), 0.28, g4 * 0.03,
                               facecolor=TEAL, alpha=0.9))
        # Delta arrow
        if g4 < g3:
            ax.annotate("", xy=(xc + 0.16, 1.2 + g4 * 0.03 + 0.15),
                        xytext=(xc - 0.16, 1.2 + g3 * 0.03 + 0.15),
                        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))
            ax.text(xc, 1.2 + g3 * 0.03 + 0.45,
                    f"−{g3 - g4} pp", ha="center", fontsize=8.5,
                    color=GREEN, fontweight="bold")
        # Model label
        ax.text(xc, 0.85, m, ha="center", fontsize=9.5, color=INK, fontweight="bold")

    # Legend — anchored to the right so it doesn't collide with the y-axis label.
    ax.add_patch(Rectangle((7.0, 4.75), 0.3, 0.2, facecolor=GREY, alpha=0.65))
    ax.text(7.4, 4.85, "L3 (movies ≥ 5)", va="center", fontsize=9, color=INK)
    ax.add_patch(Rectangle((9.2, 4.75), 0.3, 0.2, facecolor=TEAL, alpha=0.9))
    ax.text(9.6, 4.85, "L4 (movies ≥ 10)", va="center", fontsize=9, color=INK)

    ax.text(6, 0.25,
            "PTUPCDR's gap shrinks from 42% to 17% — mapping-CDR reaches its designed regime.",
            ha="center", fontsize=10, color=NAVY, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "fig_lesson4_gap_collapse.png",
                bbox_inches="tight", dpi=170)
    plt.close(fig)


# ─── L5: catalog sharpening before/after ───────────────────────────────

def lesson5():
    fig, ax = _setup(12, 5.4)

    ax.text(6, 5.05, "Lesson 5: pruning the long tail sharpens the source catalog",
            ha="center", fontsize=14, fontweight="bold", color=INK)

    # Two distribution plots: before (many low-rating items) and after
    # Synthetic long-tail distributions
    rng = np.random.default_rng(7)
    before = np.random.zipf(1.6, size=5000)
    before = before[before < 80]
    after = before[before >= 10]

    # Before card
    _card(ax, 0.4, 0.7, 5.2, 3.9, GREY, alpha=0.14, stroke=GREY)
    ax.text(3.0, 4.25, "Before (L4)",
            ha="center", fontsize=11, fontweight="bold", color="#4B5563")
    ax.text(3.0, 3.95, "39,534 movie items  ·  long tail dominates",
            ha="center", fontsize=9, color="#4B5563")

    bx_w, bx_h = 4.6, 2.5
    bx_x, bx_y = 0.7, 1.05
    counts, edges = np.histogram(before, bins=20)
    counts_norm = counts / counts.max() * bx_h
    bin_w = bx_w / len(counts_norm)
    for i, c in enumerate(counts_norm):
        ax.add_patch(Rectangle((bx_x + i * bin_w, bx_y), bin_w * 0.9, c,
                               facecolor=GREY, alpha=0.75))
    ax.text(bx_x + 0.05, bx_y - 0.28, "1 rating",
            ha="left", fontsize=8, color="#6B7280")
    ax.text(bx_x + bx_w, bx_y - 0.28, "80+",
            ha="right", fontsize=8, color="#6B7280")
    ax.plot([bx_x + bin_w * 10, bx_x + bin_w * 10], [bx_y, bx_y + bx_h],
            color=RED, lw=1.6, linestyle="--")
    ax.text(bx_x + bin_w * 10 + 0.05, bx_y + bx_h - 0.2,
            "pop ≥ 10 cut", color=RED, fontsize=8.5, fontweight="bold")

    # After card
    _card(ax, 6.4, 0.7, 5.2, 3.9, TEAL, alpha=0.14, stroke=TEAL)
    ax.text(9.0, 4.25, "After (L5)",
            ha="center", fontsize=11, fontweight="bold", color=TEAL)
    ax.text(9.0, 3.95, "10,311 movie items  ·  balanced vs 9,147 games",
            ha="center", fontsize=9, color="#4B5563")

    counts_a, _ = np.histogram(after, bins=15, range=(10, 80))
    counts_an = counts_a / counts_a.max() * bx_h
    bx_x = 6.7
    bin_w = bx_w / len(counts_an)
    for i, c in enumerate(counts_an):
        ax.add_patch(Rectangle((bx_x + i * bin_w, bx_y), bin_w * 0.9, c,
                               facecolor=TEAL, alpha=0.85))
    ax.text(bx_x + 0.05, bx_y - 0.28, "10 ratings",
            ha="left", fontsize=8, color="#6B7280")
    ax.text(bx_x + bx_w, bx_y - 0.28, "80+",
            ha="right", fontsize=8, color="#6B7280")

    # Green delta arrow between panels.
    ax.annotate("", xy=(6.4, 2.5), xytext=(5.6, 2.5),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.5))
    ax.add_patch(FancyBboxPatch((5.35, 2.15), 1.3, 0.5,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=GREEN, alpha=0.9, edgecolor=GREEN))
    ax.text(6.0, 2.4, "−74% items", ha="center", va="center",
            fontsize=10, fontweight="bold", color="white")

    ax.text(6, 0.2,
            "Fewer, denser items → CMF −3%, EMCDR −9%, PTUPCDR −12% vs L4 (retained gain at 4× fewer items).",
            ha="center", fontsize=9.5, color=NAVY, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "fig_lesson5_catalog_sharpening.png",
                bbox_inches="tight", dpi=170)
    plt.close(fig)


# ─── L6: user-split cold-start ─────────────────────────────────────────

def lesson6():
    fig, ax = _setup(12, 5.6)

    ax.text(6, 5.25, "Lesson 6: cold-start — 20% of users have zero games at train time",
            ha="center", fontsize=13.5, fontweight="bold", color=INK)

    # Warm users box
    _card(ax, 0.3, 2.4, 3.4, 2.4, BLUE, alpha=0.14, stroke=BLUE)
    ax.text(2.0, 4.55, "Warm users (80%)",
            ha="center", fontsize=11, fontweight="bold", color=BLUE)
    ax.text(2.0, 4.22, "10,124", ha="center", fontsize=14, color=BLUE, fontweight="bold")
    ax.text(2.0, 3.55, "full movie history", ha="center", fontsize=9, color=INK)
    ax.text(2.0, 3.25, "full game history", ha="center", fontsize=9, color=INK)
    ax.text(2.0, 2.75, "used for training", ha="center", fontsize=9, color="#4B5563", style="italic")

    # Cold users box
    _card(ax, 4.0, 2.4, 3.4, 2.4, PINK, alpha=0.14, stroke=PINK)
    ax.text(5.7, 4.55, "Cold users (20%)",
            ha="center", fontsize=11, fontweight="bold", color=PINK)
    ax.text(5.7, 4.22, "2,000", ha="center", fontsize=14, color=PINK, fontweight="bold")
    ax.text(5.7, 3.55, "full movie history", ha="center", fontsize=9, color=INK)
    ax.text(5.7, 3.25, "ZERO game interactions",
            ha="center", fontsize=9, color=PINK, fontweight="bold")
    ax.text(5.7, 2.75, "held out for evaluation",
            ha="center", fontsize=9, color="#4B5563", style="italic")

    # Arrow to Recall panel
    ax.annotate("", xy=(7.9, 3.55), xytext=(7.5, 3.55),
                arrowprops=dict(arrowstyle="->", color=INK, lw=2))

    # Result panel (right)
    _card(ax, 8.0, 2.4, 3.7, 2.4, PANEL, alpha=0.7, stroke=GREY)
    ax.text(9.85, 4.55, "Recall@10 on cold users",
            ha="center", fontsize=10.5, fontweight="bold", color=INK)
    results = [("Popularity", 0.0365, AMBER),
               ("PTUPCDR",    0.0305, TEAL),
               ("EMCDR",      0.0300, TEAL),
               ("LightGCN",   0.0080, BLUE),
               ("MF-BPR",     0.0000, BLUE)]
    ys = np.linspace(4.15, 2.6, len(results))
    top_tip_x = 9.1 + results[0][1] * 40
    top_y = ys[0]
    for (m, v, c), yi in zip(results, ys):
        ax.add_patch(Rectangle((9.1, yi - 0.11), v * 40, 0.22,
                               facecolor=c, alpha=0.9))
        ax.text(9.0, yi, m, ha="right", va="center", fontsize=9, color=INK)
        ax.text(9.15 + v * 40, yi, f" {v:.4f}", va="center",
                fontsize=8.5, color=INK, fontweight="bold")
    # Red "collapse" arrows from Popularity tip down to single-domain tips.
    for m, v, yi in zip([r[0] for r in results], [r[1] for r in results], ys):
        if m in ("LightGCN", "MF-BPR"):
            tip_x = 9.1 + v * 40 if v > 0 else 9.12
            ax.annotate("", xy=(tip_x, yi),
                        xytext=(top_tip_x, top_y),
                        arrowprops=dict(arrowstyle="->", color=RED, lw=1.2,
                                        alpha=0.55, linestyle="--"))

    # Bottom insight panel
    _card(ax, 0.3, 0.4, 11.4, 1.6, TEAL, alpha=0.10, stroke=TEAL)
    ax.text(6, 1.55,
            "CDR ≈ 4× over LightGCN on cold users — Popularity tops both",
            ha="center", fontsize=11, color=NAVY, fontweight="bold")
    ax.text(6, 1.05,
            "Single-domain models collapse (MF-BPR = 0.000). Mapping-CDR transfers",
            ha="center", fontsize=9.5, color=INK)
    ax.text(6, 0.70,
            "movie history into game-space; Popularity captures the first-game-is-a-hit prior.",
            ha="center", fontsize=9.5, color=INK)

    fig.tight_layout()
    fig.savefig(OUT / "fig_lesson6_cold_start.png",
                bbox_inches="tight", dpi=170)
    plt.close(fig)


# ─── L7: niche subgroup 11× win ────────────────────────────────────────

def lesson7():
    fig, ax = _setup(12, 5.4)

    ax.text(6, 5.05, "Lesson 7: SBERT is blind to nothing — it wins 11× on niche items",
            ha="center", fontsize=13.5, fontweight="bold", color=INK)

    # Two subgroup panels
    groups = [
        ("Overall (all users)", 0.0335, 0.0330, "parity"),
        ("Niche subgroup\n(one-shot unpopular, n=82)", 0.0122, 0.1341, "11× win"),
    ]
    xs = [0.4, 6.2]

    # Shared scale across panels so visual comparison is honest.
    global_max = max(v for _, lg, sb, _ in groups for v in (lg, sb))
    max_bar_h = 1.5  # leave room for titles above and ratio badge below
    scale = max_bar_h / global_max

    for (title, lg_val, sb_val, tag), x in zip(groups, xs):
        _card(ax, x, 0.7, 5.4, 4.0, PANEL, alpha=0.65, stroke=GREY)
        ax.text(x + 2.7, 4.45, title, ha="center", fontsize=11,
                fontweight="bold", color=INK)

        base_y = 2.4
        # LightGCN
        lg_h = lg_val * scale
        ax.add_patch(Rectangle((x + 1.2, base_y), 0.7, lg_h,
                               facecolor=BLUE, alpha=0.85))
        ax.text(x + 1.55, base_y - 0.2, "LightGCN",
                ha="center", va="top", fontsize=9, color=INK)
        ax.text(x + 1.55, base_y + lg_h + 0.08, f"{lg_val:.4f}",
                ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold")

        # SBERT
        sb_h = sb_val * scale
        ax.add_patch(Rectangle((x + 3.3, base_y), 0.7, sb_h,
                               facecolor=PINK, alpha=0.9))
        ax.text(x + 3.65, base_y - 0.2, "SBERT",
                ha="center", va="top", fontsize=9, color=INK)
        ax.text(x + 3.65, base_y + sb_h + 0.08, f"{sb_val:.4f}",
                ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold")

        # Ratio badge + delta arrow from LightGCN bar tip to SBERT bar tip.
        colour = GREEN if "win" in tag else "#6B7280"
        ax.annotate("", xy=(x + 3.65, base_y + sb_h),
                    xytext=(x + 1.55, base_y + lg_h),
                    arrowprops=dict(arrowstyle="->", color=colour, lw=2.2,
                                    alpha=0.85))
        ax.add_patch(FancyBboxPatch((x + 1.8, 1.2), 1.8, 0.55,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    facecolor=colour, alpha=0.9, edgecolor=colour))
        ax.text(x + 2.7, 1.475, tag, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")

    ax.text(6, 0.2,
            "Collaborative filtering cannot rank items no-one has rated. "
            "Semantic similarity can.",
            ha="center", fontsize=10, color=NAVY, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "fig_lesson7_niche_win.png",
                bbox_inches="tight", dpi=170)
    plt.close(fig)


# ─── L8: co-occurrence rerank layer ────────────────────────────────────

def lesson8():
    fig, ax = _setup(12, 5.6)

    ax.text(6, 5.25, "Lesson 8: co-occurrence rerank is a training-free drop-in layer",
            ha="center", fontsize=13.5, fontweight="bold", color=INK)

    # Pipeline: base scores → + λ·cooc → reranked top-K
    boxes = [
        (0.3, "Any base model",  "scores(u, g)",          BLUE),
        (4.0, "+ λ · Σ cooc[m,g]", "λ = 0.05",            AMBER),
        (7.7, "Reranked top-K",  "final_score(u, g)",     TEAL),
    ]
    for x, title, sub, col in boxes:
        _card(ax, x, 2.6, 3.6, 2.2, col, alpha=0.14, stroke=col)
        ax.text(x + 1.8, 4.25, title, ha="center", fontsize=11,
                fontweight="bold", color=col)
        ax.text(x + 1.8, 3.6, sub, ha="center", fontsize=10, color=INK)

    # Arrows between boxes
    for x in [3.85, 7.55]:
        ax.annotate("", xy=(x + 0.15, 3.7), xytext=(x - 0.1, 3.7),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=2))

    # Lift panel
    _card(ax, 0.3, 0.4, 11.0, 1.8, GREEN, alpha=0.10, stroke=GREEN)
    ax.text(6, 1.85, "Universal Recall@10 lift at λ = 0.05",
            ha="center", fontsize=11, fontweight="bold", color=GREEN)

    lifts = [("LightGCN @ L6", "0.010", "0.033", "+230%"),
             ("MF-BPR @ L6",   "0.001", "0.020", "+1 900%"),
             ("EMCDR @ L3",    "0.025", "0.033", "+31%"),
             ("CMF @ L3",      "0.040", "0.038", "≈")]
    xs = [0.6, 3.5, 6.5, 9.5]
    for (label, before, after, pct), x in zip(lifts, xs):
        ax.text(x + 0.8, 1.55, label, ha="center", fontsize=9.5,
                fontweight="bold", color=INK)
        # before → after with an explicit delta arrow.
        ax.text(x + 0.2, 1.1, before, ha="center", fontsize=9, color="#6B7280")
        ax.annotate("", xy=(x + 1.0, 1.1), xytext=(x + 0.45, 1.1),
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6))
        ax.text(x + 1.4, 1.1, after, ha="center", fontsize=9,
                fontweight="bold", color=INK)
        ax.text(x + 0.8, 0.68, pct, ha="center", fontsize=10.5,
                fontweight="bold", color=GREEN)

    fig.tight_layout()
    fig.savefig(OUT / "fig_lesson8_cooc_layer.png",
                bbox_inches="tight", dpi=170)
    plt.close(fig)


def main():
    lesson1(); print("  ✓ fig_lesson1_pointwise_pairwise.png")
    lesson2(); print("  ✓ fig_lesson2_overlap_funnel.png")
    lesson3(); print("  ✓ fig_lesson3_overlap_lift.png")
    lesson4(); print("  ✓ fig_lesson4_gap_collapse.png")
    lesson5(); print("  ✓ fig_lesson5_catalog_sharpening.png")
    lesson6(); print("  ✓ fig_lesson6_cold_start.png")
    lesson7(); print("  ✓ fig_lesson7_niche_win.png")
    lesson8(); print("  ✓ fig_lesson8_cooc_layer.png")


if __name__ == "__main__":
    main()
