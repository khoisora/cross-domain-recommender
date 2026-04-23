"""Redraw model architecture figures with modern visual arrays and matrices.

Replaces the text-heavy "User embedding" / "Movie embedding" boxes with
actual colored-cell vectors, matrices, and small neural-net layer stacks —
in the distill.pub style.

Outputs PNGs to report_figures_v3/:
  fig06_mf_bpr_training.png
  fig07_mf_bpr_inference.png
  fig08_ncf_gmf.png
  fig09_ncf_mlp.png
  fig11_lightgcn_prop.png
  fig12_lightgcn_training.png
  fig13_cmf.png
  fig14_emcdr.png
  fig15_ptupcdr.png
  fig16_sbert.png
  fig18_cooc.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report_figures_v3"
OUT.mkdir(exist_ok=True)

# Colors — distill.pub / modern palette
MOVIE   = "#3B82F6"   # blue
GAME    = "#EF4444"   # red
USER    = "#8B5CF6"   # purple
NEUTRAL = "#64748B"   # slate
ACCENT  = "#F59E0B"   # amber
TEAL    = "#14B8A6"
GREEN   = "#10B981"
PINK    = "#EC4899"

BG      = "#F8FAFC"
INK     = "#0F172A"
MUTED   = "#94A3B8"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": INK,
    "axes.edgecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
})


# ── primitives ──────────────────────────────────────────────────────────

def vector(ax, x, y, values, color, cell_w=0.36, cell_h=0.48, label=None,
           label_above=True, vmin=-1, vmax=1, fontsize=9):
    """Draw a 1-D vector as a row of colored cells. Returns (x_end, y)."""
    import matplotlib.colors as mc
    base = np.array(mc.to_rgb(color))
    white = np.array([1, 1, 1])
    for i, v in enumerate(values):
        t = (v - vmin) / (vmax - vmin)
        t = np.clip(t, 0, 1)
        c = white * (1 - t) + base * t
        ax.add_patch(Rectangle((x + i * cell_w, y), cell_w, cell_h,
                               facecolor=c, edgecolor=color, linewidth=0.9))
    w = len(values) * cell_w
    if label:
        ly = y + cell_h + 0.12 if label_above else y - 0.22
        va = "bottom" if label_above else "top"
        ax.text(x + w / 2, ly, label, ha="center", va=va,
                fontsize=fontsize, color=INK, fontweight="bold")
    return x + w, y


def matrix(ax, x, y, rows, cols, color, cell=0.22, label=None, seed=None,
           fontsize=9, label_above=True):
    rng = np.random.default_rng(seed)
    vals = rng.uniform(-1, 1, (rows, cols))
    import matplotlib.colors as mc
    base = np.array(mc.to_rgb(color))
    white = np.array([1, 1, 1])
    for i in range(rows):
        for j in range(cols):
            t = (vals[i, j] + 1) / 2
            c = white * (1 - t) + base * t
            ax.add_patch(Rectangle((x + j * cell, y + (rows - 1 - i) * cell),
                                   cell, cell,
                                   facecolor=c, edgecolor=color, linewidth=0.5))
    w = cols * cell
    h = rows * cell
    if label:
        if label_above:
            ax.text(x + w / 2, y + h + 0.12, label, ha="center", va="bottom",
                    fontsize=fontsize, color=INK, fontweight="bold")
        else:
            ax.text(x + w / 2, y - 0.18, label, ha="center", va="top",
                    fontsize=fontsize, color=INK, fontweight="bold")
    return x + w, y + h


def mlp_stack(ax, x, y, widths, color=NEUTRAL, layer_w=0.32, gap=0.18,
              label=None, fontsize=9, label_above=True):
    """Stack of layer rectangles — heights encode widths."""
    cur = x
    max_h = max(widths) * 0.16
    for w in widths:
        h = w * 0.16
        ax.add_patch(FancyBboxPatch(
            (cur, y + (max_h - h) / 2), layer_w, h,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor=color, edgecolor=color, alpha=0.85, linewidth=0))
        cur += layer_w + gap
    total = cur - x - gap
    if label:
        ly = y + max_h + 0.12 if label_above else y - 0.18
        va = "bottom" if label_above else "top"
        ax.text(x + total / 2, ly, label, ha="center", va=va,
                fontsize=fontsize, color=INK, fontweight="bold")
    return x + total, y + max_h / 2


def arrow(ax, x1, y1, x2, y2, color=NEUTRAL, style="->", lw=1.5,
          connectionstyle=None, alpha=0.9, label=None, label_pos=0.5,
          label_offset=(0, 0.15), dashed=False):
    ls = "--" if dashed else "-"
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle=style, color=color, lw=lw,
                          mutation_scale=14, alpha=alpha, linestyle=ls,
                          connectionstyle=connectionstyle or "arc3,rad=0")
    ax.add_patch(arr)
    if label:
        mx = x1 + (x2 - x1) * label_pos + label_offset[0]
        my = y1 + (y2 - y1) * label_pos + label_offset[1]
        ax.text(mx, my, label, fontsize=9, color=NEUTRAL, ha="center",
                va="center", style="italic")


def chip(ax, x, y, w, h, text, color=NEUTRAL, facecolor=None, fontsize=10,
         fontweight="bold", text_color=None):
    fc = facecolor if facecolor else "white"
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fc, edgecolor=color, linewidth=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=text_color if text_color else color)


def icon_movie(ax, x, y, size=0.28):
    ax.add_patch(Rectangle((x, y), size, size * 1.25,
                           facecolor=MOVIE, edgecolor=MOVIE, alpha=0.85))
    for i in range(4):
        ax.add_patch(Rectangle((x + 0.02, y + 0.03 + i * 0.08),
                               0.035, 0.035,
                               facecolor="white", edgecolor="white"))
        ax.add_patch(Rectangle((x + size - 0.055, y + 0.03 + i * 0.08),
                               0.035, 0.035,
                               facecolor="white", edgecolor="white"))


def icon_game(ax, x, y, size=0.28):
    ax.add_patch(FancyBboxPatch((x, y + 0.05), size, size * 0.72,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor=GAME, edgecolor=GAME, alpha=0.85))
    ax.add_patch(plt.Circle((x + 0.08, y + 0.17), 0.025,
                            facecolor="white", edgecolor="white"))
    ax.add_patch(plt.Circle((x + size - 0.08, y + 0.17), 0.025,
                            facecolor="white", edgecolor="white"))


def icon_user(ax, x, y, size=0.3):
    ax.add_patch(plt.Circle((x + size/2, y + size*0.78),
                            size * 0.22, facecolor=USER, edgecolor=USER))
    ax.add_patch(FancyBboxPatch((x + size*0.15, y + size*0.05),
                                size * 0.7, size * 0.55,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor=USER, edgecolor=USER, alpha=0.95))


def title(fig, text, sub=None):
    fig.suptitle(text, fontsize=14, fontweight="bold", y=0.98, color=INK)
    if sub:
        fig.text(0.5, 0.925, sub, ha="center", va="top",
                 fontsize=10.5, color=MUTED, style="italic")


def setup_ax(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


# ── Figure 6: MF-BPR Training — two stacked panels ──────────────────────

def fig_mf_bpr_training():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.2), dpi=180,
                                   gridspec_kw={"height_ratios": [1, 1]})
    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)

    rng = np.random.default_rng(0)
    user_vec = rng.uniform(-1, 1, 8)
    pos_vec  = rng.uniform(-1, 1, 8)
    neg_vec  = rng.uniform(-1, 1, 8)

    # ── Panel A: compute two scores ────────────────────────────────────
    ax1.set_xlim(0, 11); ax1.set_ylim(0, 3.6)
    ax1.text(0.1, 3.3, "(a) compute two scores", fontsize=11,
             color=INK, fontweight="bold")

    icon_user(ax1, 0.5, 2.5)
    vector(ax1, 1.3, 2.55, user_vec, USER, label="user vector")

    icon_game(ax1, 0.5, 1.0)
    vector(ax1, 1.3, 1.05, pos_vec, GREEN, label="liked game (+)")

    icon_game(ax1, 5.2, 0.05)
    vector(ax1, 6.0, 0.15, neg_vec, GAME, label="random game (−)", label_above=False)

    # positive score chip
    chip(ax1, 8.3, 2.3, 2.2, 0.8, "user · (+)", color=GREEN,
         facecolor="#ECFDF5", text_color=INK)
    ax1.text(9.4, 2.15, "positive score", ha="center", fontsize=9,
             color=NEUTRAL, style="italic")
    arrow(ax1, 4.5, 2.8, 8.3, 2.7, color=USER, lw=1.4)
    arrow(ax1, 4.5, 1.3, 8.3, 2.6, color=GREEN, connectionstyle="arc3,rad=-0.1", lw=1.4)

    # negative score chip
    chip(ax1, 8.3, 0.8, 2.2, 0.8, "user · (−)", color=GAME,
         facecolor="#FEF2F2", text_color=INK)
    ax1.text(9.4, 0.65, "negative score", ha="center", fontsize=9,
             color=NEUTRAL, style="italic")
    arrow(ax1, 4.5, 2.8, 8.3, 1.3, color=USER, connectionstyle="arc3,rad=0.18",
          lw=1.2, alpha=0.7)
    arrow(ax1, 7.5, 0.4, 8.3, 1.0, color=GAME, lw=1.4)

    # ── Panel B: BPR loss pushes scores apart ──────────────────────────
    ax2.set_xlim(0, 11); ax2.set_ylim(0, 3.6)
    ax2.text(0.1, 3.3, "(b) BPR loss pushes scores apart",
             fontsize=11, color=INK, fontweight="bold")

    chip(ax2, 0.8, 2.3, 2.2, 0.8, "positive score", color=GREEN,
         facecolor="#ECFDF5", text_color=INK)
    chip(ax2, 0.8, 0.8, 2.2, 0.8, "negative score", color=GAME,
         facecolor="#FEF2F2", text_color=INK)

    chip(ax2, 4.5, 1.6, 2.0, 1.0, "BPR loss", color=ACCENT,
         facecolor="#FFFBEB", fontsize=12, text_color=INK)

    arrow(ax2, 3.0, 2.7, 4.5, 2.3, color=GREEN, lw=1.4)
    arrow(ax2, 3.0, 1.2, 4.5, 1.8, color=GAME, lw=1.4)

    # Gradient arrow
    arrow(ax2, 6.5, 2.1, 8.5, 2.75, color=NEUTRAL, lw=1.6)
    ax2.text(8.7, 2.75, "gradient update", fontsize=10, color=INK,
             fontweight="bold", va="center")

    # Score-line diagram at right
    ax2.plot([7.8, 10.5], [1.2, 1.2], color=NEUTRAL, lw=1.2)
    ax2.plot([8.4, 8.4], [1.1, 1.3], color=GAME, lw=2.5)
    ax2.plot([10.0, 10.0], [1.1, 1.3], color=GREEN, lw=2.5)
    ax2.text(8.4, 0.85, "(−)", ha="center", color=GAME, fontsize=10, fontweight="bold")
    ax2.text(10.0, 0.85, "(+)", ha="center", color=GREEN, fontsize=10, fontweight="bold")
    ax2.annotate("", xy=(10.0, 1.5), xytext=(8.4, 1.5),
                 arrowprops=dict(arrowstyle="->", color=NEUTRAL, lw=1.5))
    ax2.text(9.2, 1.65, "push apart", ha="center", fontsize=9,
             color=NEUTRAL, style="italic")

    fig.suptitle("MF-BPR — training", fontsize=14, fontweight="bold",
                 y=0.98, color=INK)
    fig.text(0.5, 0.94, "For every (user, liked, random) triplet, pull the liked score above the random one",
             ha="center", fontsize=10.5, color=MUTED, style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig06_mf_bpr_training.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ── Figure 7: MF-BPR Inference ──────────────────────────────────────────

def fig_mf_bpr_inference():
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=180)
    setup_ax(ax, (0, 11), (0, 5.2))

    rng = np.random.default_rng(1)
    user_vec = rng.uniform(-1, 1, 8)

    # user vector far left
    icon_user(ax, 0.2, 2.25)
    vector(ax, 0.9, 2.3, user_vec, USER, label="user vector")

    # column headers
    ax.text(4.7, 4.7, "candidate games", fontsize=10,
            color=INK, fontweight="bold", ha="center")
    ax.text(8.3, 4.7, "score = dot product", fontsize=10,
            color=INK, fontweight="bold", ha="center")

    names = ["Stellaris", "Civ VI", "Cyberpunk", "Hades", "Factorio"]
    scores = [0.87, 0.74, 0.61, 0.49, 0.33]

    for i, (n, s) in enumerate(zip(names, scores)):
        y = 4.1 - i * 0.7
        # game label + icon + vector
        ax.text(3.7, y + 0.16, n, ha="right", va="center",
                fontsize=9.5, color=INK)
        icon_game(ax, 3.8, y - 0.05, size=0.24)
        v = rng.uniform(-1, 1, 8)
        vector(ax, 4.2, y, v, GAME, cell_w=0.22, cell_h=0.32, fontsize=8)
        # score bar
        ax.add_patch(Rectangle((7.2, y + 0.03), s * 2.2, 0.26,
                               facecolor=GAME, edgecolor=GAME, alpha=0.8))
        ax.text(7.1, y + 0.16, f"{s:.2f}", ha="right", va="center",
                fontsize=9.5, color=INK, fontweight="bold")

    # single fan of arrows from END of user vector → candidate column header
    ax.text(3.15, 2.8, "dot(·)", fontsize=9, color=NEUTRAL, style="italic",
            ha="left")
    for i in range(5):
        y = 4.1 - i * 0.7 + 0.16
        arrow(ax, 3.0, 2.56, 3.55, y, color=NEUTRAL, alpha=0.35, lw=1.0,
              connectionstyle="arc3,rad=0")

    # small inline helper
    ax.text(5.5, 0.3, "sort by score, show top-K",
            fontsize=9.5, color=NEUTRAL, style="italic", ha="center")

    title(fig, "MF-BPR — inference",
          "Score each game by dot product; sort; recommend top-K")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig07_mf_bpr_inference.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ── Figure 8: NCF GMF branch ────────────────────────────────────────────

def fig_ncf_gmf():
    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=180)
    setup_ax(ax, (0, 12), (0, 4.6))

    rng = np.random.default_rng(2)
    u = rng.uniform(-1, 1, 8); g = rng.uniform(-1, 1, 8)
    prod = u * g

    icon_user(ax, 0.3, 3.2)
    vector(ax, 1.1, 3.3, u, USER, label="user (GMF)")
    icon_game(ax, 0.3, 1.3)
    vector(ax, 1.1, 1.4, g, GAME, label="game (GMF)")

    # hadamard op centred between the two rows
    ax.add_patch(plt.Circle((5.0, 2.4), 0.3, facecolor="white",
                            edgecolor=INK, linewidth=1.8))
    ax.text(5.0, 2.4, "×", fontsize=22, ha="center", va="center",
            color=INK, fontweight="bold")
    ax.text(5.0, 1.8, "element-wise", fontsize=9, ha="center",
            color=NEUTRAL, style="italic")

    arrow(ax, 4.2, 3.45, 4.75, 2.7, color=USER, lw=1.4)
    arrow(ax, 4.2, 1.65, 4.75, 2.1, color=GAME, lw=1.4)

    # product vector with gap
    vector(ax, 5.8, 2.25, prod, ACCENT, label="u × g", fontsize=10)
    arrow(ax, 5.35, 2.4, 5.8, 2.5, color=NEUTRAL, lw=1.4)

    # linear layer chip — generous gap from product vector
    chip(ax, 9.0, 2.15, 1.0, 0.65, "linear", color=NEUTRAL,
         facecolor="white", fontsize=10, text_color=INK)
    arrow(ax, 8.7, 2.5, 9.0, 2.48, color=NEUTRAL, lw=1.4)

    # score chip
    chip(ax, 10.5, 2.05, 1.3, 0.85, "score", color=ACCENT,
         facecolor="#FFFBEB", fontsize=12, text_color=INK)
    arrow(ax, 10.0, 2.48, 10.5, 2.48, color=ACCENT, lw=1.4)

    title(fig, "NCF — GMF branch",
          "Element-wise multiply, then a linear readout → a score")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / "fig08_ncf_gmf.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ── Figure 9: NCF MLP branch ────────────────────────────────────────────

def fig_ncf_mlp():
    fig, ax = plt.subplots(figsize=(13, 4.6), dpi=180)
    setup_ax(ax, (0, 13), (0, 4.6))

    rng = np.random.default_rng(3)
    u = rng.uniform(-1, 1, 8); g = rng.uniform(-1, 1, 8)
    cat = np.concatenate([u, g])

    icon_user(ax, 0.3, 3.2)
    vector(ax, 1.1, 3.3, u, USER, label="user (MLP)")
    icon_game(ax, 0.3, 1.3)
    vector(ax, 1.1, 1.4, g, GAME, label="game (MLP)")

    # concat symbol in a small white circle
    ax.add_patch(plt.Circle((4.7, 2.4), 0.3, facecolor="white",
                            edgecolor=INK, linewidth=1.8))
    ax.text(4.7, 2.4, "‖", fontsize=20, ha="center", va="center",
            color=INK, fontweight="bold")
    ax.text(4.7, 1.75, "concat", fontsize=9, ha="center",
            color=NEUTRAL, style="italic")
    arrow(ax, 4.2, 3.45, 4.45, 2.7, color=USER, lw=1.4)
    arrow(ax, 4.2, 1.65, 4.45, 2.1, color=GAME, lw=1.4)

    # concatenated vector — fits between x=5.5 and x=9.02 (16 cells × 0.22)
    vector(ax, 5.5, 2.25, cat, NEUTRAL, cell_w=0.22, cell_h=0.4,
           label="[u ‖ g]", fontsize=9)
    arrow(ax, 5.0, 2.4, 5.5, 2.45, color=NEUTRAL, lw=1.4)

    # MLP stack — well after the concat vector ends
    mlp_stack(ax, 9.7, 1.5, [6, 4, 2], color=PINK, label="MLP")
    arrow(ax, 9.02, 2.45, 9.7, 2.45, color=NEUTRAL, lw=1.4)

    chip(ax, 11.6, 2.15, 1.0, 0.7, "score", color=ACCENT,
         facecolor="#FFFBEB", fontsize=11, text_color=INK)
    arrow(ax, 11.2, 2.45, 11.6, 2.5, color=ACCENT, lw=1.4)

    title(fig, "NCF — MLP branch",
          "Concatenate the two embeddings, then a stack of non-linear layers → a score")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / "fig09_ncf_mlp.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ── Figure 11: LightGCN propagation ─────────────────────────────────────

def fig_lightgcn_prop():
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=180)
    setup_ax(ax, (0, 11), (0, 5.6))

    rng = np.random.default_rng(5)

    # bipartite graph: 3 users left, 4 games right
    user_pos = [(1.2, 4.2), (1.2, 2.9), (1.2, 1.6)]
    game_pos = [(3.6, 4.6), (3.6, 3.4), (3.6, 2.2), (3.6, 1.0)]
    edges = [(0,0),(0,1),(1,0),(1,2),(2,1),(2,2),(2,3)]

    for e in edges:
        u, g = user_pos[e[0]], game_pos[e[1]]
        ax.plot([u[0]+0.15, g[0]-0.15], [u[1], g[1]],
                color=NEUTRAL, alpha=0.35, lw=1.2)
    for x, y in user_pos:
        ax.add_patch(plt.Circle((x, y), 0.18, facecolor=USER, edgecolor=USER))
    for x, y in game_pos:
        ax.add_patch(plt.Circle((x, y), 0.18, facecolor=GAME, edgecolor=GAME))

    ax.text(1.2, 5.05, "users", ha="center", fontsize=10,
            color=INK, fontweight="bold")
    ax.text(3.6, 5.3, "games", ha="center", fontsize=10,
            color=INK, fontweight="bold")

    # 3 layers of embedding evolution
    titles = ["layer 0", "layer 1", "layer 2"]
    colors = [USER, "#7C3AED", "#5B21B6"]
    for i in range(3):
        x0 = 5.3 + i * 1.7
        for j, (xu, yu) in enumerate(user_pos):
            v = rng.uniform(-1, 1, 6)
            vector(ax, x0, yu - 0.2, v, colors[i],
                   cell_w=0.2, cell_h=0.3, fontsize=7)
        ax.text(x0 + 0.6, 5.15, titles[i], ha="center",
                fontsize=10, color=INK, fontweight="bold")

    # propagation arrows between layers
    for i in range(2):
        xs = 6.5 + i * 1.7
        xe = 7.0 + i * 1.7
        for _, yu in user_pos:
            arrow(ax, xs, yu - 0.05, xe, yu - 0.05, color=MUTED, lw=1.0)

    # final aggregation
    chip(ax, 10.2, 2.5, 0.7, 1.0, "mean", color=TEAL,
         facecolor="#F0FDFA", fontsize=11, text_color=INK)
    for _, yu in user_pos:
        arrow(ax, 9.7, yu - 0.05, 10.2, 3.0, color=MUTED, lw=1.0,
              alpha=0.6)
    ax.text(10.55, 2.25, "final user\nembedding", ha="center", fontsize=9,
            color=INK, style="italic")

    title(fig, "LightGCN — graph propagation",
          "Each layer averages neighbour embeddings; final = mean of layers")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fig11_lightgcn_prop.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ── Figure 12: LightGCN training loop ───────────────────────────────────

def fig_lightgcn_training():
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=180)
    setup_ax(ax, (0, 11), (0, 4.8))

    # chips in a cycle
    steps = [
        ("graph\npropagation", USER, "#F5F3FF"),
        ("sample\ntriplet", NEUTRAL, "#F8FAFC"),
        ("BPR\nloss", ACCENT, "#FFFBEB"),
        ("update\nembeddings", GREEN, "#ECFDF5"),
    ]
    positions = [(1.2, 2.4), (3.8, 2.4), (6.4, 2.4), (9.0, 2.4)]
    for (lbl, col, fc), (x, y) in zip(steps, positions):
        chip(ax, x, y, 1.7, 1.1, lbl, color=col, facecolor=fc,
             text_color=INK, fontsize=11)

    for i in range(3):
        x1 = positions[i][0] + 1.7
        x2 = positions[i+1][0]
        arrow(ax, x1, 2.95, x2, 2.95, color=NEUTRAL, lw=1.8)

    # loop back — routed BELOW the chip row so it never crosses a chip
    arrow(ax, 9.85, 2.4, 2.05, 2.4, color=NEUTRAL,
          connectionstyle="arc3,rad=-0.4", lw=1.6, dashed=True)
    ax.text(5.5, 0.9, "repeat each epoch", fontsize=10,
            color=NEUTRAL, style="italic", ha="center")

    # vector cluster decoration — pushed higher to avoid chip row
    rng = np.random.default_rng(7)
    for i in range(3):
        vector(ax, 0.4 + i*0.1, 4.1 - i*0.08, rng.uniform(-1, 1, 6),
               USER, cell_w=0.18, cell_h=0.24)
    ax.text(1.3, 4.55, "user embeddings", fontsize=9, color=NEUTRAL,
            style="italic")

    for i in range(3):
        vector(ax, 9.2 + i*0.1, 4.1 - i*0.08, rng.uniform(-1, 1, 6),
               GAME, cell_w=0.18, cell_h=0.24)
    ax.text(9.85, 4.55, "updated", fontsize=9, color=NEUTRAL,
            style="italic")

    title(fig, "LightGCN — training loop",
          "Propagate → sample → BPR → update; repeat until the rank stabilises")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / "fig12_lightgcn_training.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ── Figure 13: CMF shared user embedding ────────────────────────────────

def fig_cmf():
    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=180)
    setup_ax(ax, (0, 12), (0, 5.6))

    rng = np.random.default_rng(8)
    shared_u = rng.uniform(-1, 1, 10)

    # shared user embedding at top-centre
    icon_user(ax, 5.15, 4.4)
    vector(ax, 5.7, 4.5, shared_u, USER, cell_w=0.32, cell_h=0.45,
           label="shared user embedding (one per user)")

    # movie matrix (left) and game matrix (right) — bottom row
    matrix(ax, 0.6, 1.2, 10, 8, MOVIE, cell=0.22,
           label="movie item matrix", seed=9)
    matrix(ax, 9.2, 1.2, 10, 8, GAME, cell=0.22,
           label="game item matrix", seed=10)

    # Score arrows: user → movie matrix  and  user → game matrix
    arrow(ax, 5.9, 4.45, 2.0, 3.3, color=MOVIE, lw=1.6,
          connectionstyle="arc3,rad=0.2")
    arrow(ax, 8.8, 4.45, 10.2, 3.3, color=GAME, lw=1.6,
          connectionstyle="arc3,rad=-0.2")

    ax.text(3.3, 4.15, "score movies", fontsize=9, color=MOVIE,
            style="italic", ha="center")
    ax.text(9.5, 4.15, "score games", fontsize=9, color=GAME,
            style="italic", ha="center")

    # Loss chips DIRECTLY below each matrix (connected with short vertical arrows)
    chip(ax, 1.1, 0.2, 2.2, 0.7, "movie BPR loss", color=MOVIE,
         facecolor="#EFF6FF", fontsize=10, text_color=INK)
    arrow(ax, 2.2, 1.15, 2.2, 0.9, color=MOVIE, lw=1.4)

    chip(ax, 9.6, 0.2, 2.2, 0.7, "game BPR loss", color=GAME,
         facecolor="#FEF2F2", fontsize=10, text_color=INK)
    arrow(ax, 10.7, 1.15, 10.7, 0.9, color=GAME, lw=1.4)

    # Central caption
    ax.text(6.0, 2.8, "same user vector\nscores both domains",
            ha="center", va="center", fontsize=10.5,
            color=NEUTRAL, style="italic")

    title(fig, "CMF — shared user embedding",
          "One user vector is learned jointly from both domains' interactions")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(OUT / "fig13_cmf.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ── Figure 14: EMCDR three-phase ────────────────────────────────────────

def fig_emcdr():
    fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=180)
    setup_ax(ax, (0, 11.5), (0, 5.8))

    rng = np.random.default_rng(11)

    # Phase 1: movie MF — user_m and item_m matrices
    ax.text(1.9, 5.35, "Phase 1 — train on movies",
            fontsize=10.5, fontweight="bold", color=MOVIE, ha="center")
    matrix(ax, 0.3, 3.9, 5, 6, MOVIE, cell=0.2,
           label="user_M", seed=12, fontsize=9)
    matrix(ax, 2.4, 3.9, 5, 6, MOVIE, cell=0.2,
           label="item_M", seed=13, fontsize=9)

    # Phase 2: game MF
    ax.text(1.9, 2.55, "Phase 2 — train on games",
            fontsize=10.5, fontweight="bold", color=GAME, ha="center")
    matrix(ax, 0.3, 1.1, 5, 6, GAME, cell=0.2,
           label="user_G", seed=14, fontsize=9)
    matrix(ax, 2.4, 1.1, 5, 6, GAME, cell=0.2,
           label="item_G", seed=15, fontsize=9)

    # Phase 3 translator (MLP)
    ax.text(6.75, 5.35, "Phase 3 — learn translator",
            fontsize=10.5, fontweight="bold", color=USER, ha="center")

    # movie vec (overlap user)
    v_m = rng.uniform(-1, 1, 6)
    vector(ax, 5.0, 4.6, v_m, MOVIE, cell_w=0.22, cell_h=0.34,
           label="user's movie vec", fontsize=9)
    # mlp
    mlp_stack(ax, 6.9, 4.1, [5, 4, 5], color=USER, label="MLP")
    # translated
    v_gt = rng.uniform(-1, 1, 6)
    vector(ax, 8.9, 4.6, v_gt, GREEN, cell_w=0.22, cell_h=0.34,
           label="translated vec", fontsize=9)

    arrow(ax, 6.35, 4.77, 6.9, 4.55, color=MOVIE, lw=1.6)
    arrow(ax, 7.95, 4.55, 8.9, 4.77, color=GREEN, lw=1.6)

    # ground-truth game vec + loss
    v_g = rng.uniform(-1, 1, 6)
    vector(ax, 8.9, 3.3, v_g, GAME, cell_w=0.22, cell_h=0.34,
           label="actual game vec", fontsize=9, label_above=False)
    arrow(ax, 10.22, 3.75, 10.22, 4.55, color=NEUTRAL, style="<->", lw=1.4)
    ax.text(10.6, 4.15, "MSE", fontsize=9.5, color=NEUTRAL, style="italic")

    # Serve arrow
    ax.text(6.75, 1.95, "Serve — for any user",
            fontsize=10.5, fontweight="bold", color=INK, ha="center")

    v_mu = rng.uniform(-1, 1, 6)
    vector(ax, 5.0, 0.8, v_mu, MOVIE, cell_w=0.22, cell_h=0.34,
           label="any movie vec", fontsize=9, label_above=False)
    mlp_stack(ax, 6.9, 0.35, [5, 4, 5], color=USER)
    v_tr = rng.uniform(-1, 1, 6)
    vector(ax, 8.9, 0.8, v_tr, GREEN, cell_w=0.22, cell_h=0.34,
           label="→ rank all games", fontsize=9, label_above=False)
    arrow(ax, 6.35, 0.97, 6.9, 0.78, color=MOVIE, lw=1.4)
    arrow(ax, 7.95, 0.78, 8.9, 0.97, color=GREEN, lw=1.4)

    # Divider line
    ax.plot([4.4, 4.4], [0.3, 5.3], color=MUTED, lw=0.8, linestyle="--")

    title(fig, "EMCDR — three phases",
          "Train each side → learn an MLP that translates movie-space → game-space on overlap users")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fig14_emcdr.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ── Figure 15: PTUPCDR ──────────────────────────────────────────────────

def fig_ptupcdr():
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=180)
    setup_ax(ax, (0, 11.5), (0, 5.2))

    rng = np.random.default_rng(17)

    # user movie vec on left
    v_m = rng.uniform(-1, 1, 8)
    icon_user(ax, 0.2, 2.7)
    vector(ax, 0.9, 2.8, v_m, MOVIE, cell_w=0.26, cell_h=0.38,
           label="user's movie vec", fontsize=9)

    # meta-network
    ax.text(4.3, 4.55, "meta-network (MLP)",
            fontsize=10, fontweight="bold", color=USER, ha="center")
    mlp_stack(ax, 3.3, 3.5, [5, 6, 7], color=USER, layer_w=0.3)

    # user-specific bridge matrix (output of meta-net)
    matrix(ax, 6.2, 3.4, 6, 6, TEAL, cell=0.22,
           label="personalised bridge\n(weights output by meta-net)", seed=18,
           fontsize=9)

    arrow(ax, 3.15, 3.1, 3.3, 3.9, color=MOVIE, lw=1.6)  # movie vec → meta
    arrow(ax, 5.3, 3.9, 6.2, 4.05, color=USER, lw=1.6)   # meta → bridge

    # movie vec also feeds directly into bridge (dashed)
    arrow(ax, 3.15, 3.0, 6.2, 3.4, color=MOVIE, lw=1.3,
          dashed=True, connectionstyle="arc3,rad=-0.2")
    ax.text(4.7, 2.0, "movie vec also applied to the bridge",
            fontsize=9, color=NEUTRAL, style="italic", ha="center")

    # translated game-space vec
    v_t = rng.uniform(-1, 1, 8)
    vector(ax, 9.6, 2.8, v_t, GREEN, cell_w=0.26, cell_h=0.38,
           label="translated game-space vec", fontsize=9)

    arrow(ax, 7.5, 4.05, 9.6, 3.0, color=GREEN, lw=1.6,
          connectionstyle="arc3,rad=-0.2")

    # contrast arrow — a second user would get different bridge
    icon_user(ax, 0.2, 0.25)
    vector(ax, 0.9, 0.35, rng.uniform(-1, 1, 8), MOVIE, cell_w=0.26, cell_h=0.38,
           fontsize=9)
    matrix(ax, 6.2, 0.25, 6, 6, ACCENT, cell=0.22,
           seed=19, fontsize=9, label_above=False)
    arrow(ax, 3.15, 0.55, 6.2, 0.9, color=ACCENT, lw=1.2,
          connectionstyle="arc3,rad=-0.15")
    ax.text(9.0, 1.4, "a different user → a different bridge",
            fontsize=9.5, color=NEUTRAL, style="italic", ha="center")

    title(fig, "PTUPCDR — personalised transfer",
          "A meta-network reads the user's movie vec and outputs a user-specific bridge matrix")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fig15_ptupcdr.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ── Figure 16: SBERT encoding ───────────────────────────────────────────

def fig_sbert():
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=180)
    setup_ax(ax, (0, 11), (0, 5.6))

    # 3 movies + 3 games
    titles_m = ["Blade Runner", "Oppenheimer", "Dune"]
    titles_g = ["Stellaris", "Civilization VI", "Cyberpunk 2077"]

    rng = np.random.default_rng(20)

    # movies block
    ax.text(1.0, 5.2, "user's liked movies", fontsize=10,
            color=MOVIE, fontweight="bold")
    for i, t in enumerate(titles_m):
        y = 4.5 - i * 0.9
        chip(ax, 0.3, y, 1.7, 0.55, t, color=MOVIE, facecolor="#EFF6FF",
             fontsize=9.5, text_color=INK)
        v = rng.uniform(-1, 1, 12)
        vector(ax, 2.3, y, v, MOVIE, cell_w=0.14, cell_h=0.5, fontsize=8)
        arrow(ax, 2.0, y + 0.27, 2.3, y + 0.27, color=MOVIE, lw=1.2)

    ax.text(3.0, 5.2, "SBERT vectors", fontsize=10,
            color=NEUTRAL, fontweight="bold")

    # Mean → user profile vector
    chip(ax, 5.0, 2.6, 1.0, 1.1, "mean", color=USER, facecolor="#F5F3FF",
         fontsize=11, text_color=INK)
    for i in range(3):
        y = 4.5 - i * 0.9 + 0.27
        arrow(ax, 3.95, y, 5.0, 3.15, color=MUTED, alpha=0.6, lw=1.0)

    v_u = rng.uniform(-1, 1, 12)
    vector(ax, 6.5, 3.0, v_u, USER, cell_w=0.14, cell_h=0.5,
           label="user profile", fontsize=9)
    arrow(ax, 6.0, 3.15, 6.5, 3.25, color=USER, lw=1.4)

    # Game candidates + cosine
    ax.text(9.5, 5.2, "game SBERT vectors", fontsize=10,
            color=GAME, fontweight="bold")
    for i, t in enumerate(titles_g):
        y = 4.5 - i * 0.9
        chip(ax, 9.4, y, 1.5, 0.55, t, color=GAME, facecolor="#FEF2F2",
             fontsize=9.5, text_color=INK)
        v = rng.uniform(-1, 1, 12)
        vector(ax, 7.8, y, v, GAME, cell_w=0.14, cell_h=0.5, fontsize=8)

    # cosine similarity chip
    ax.text(7.1, 1.6, "score = cos(user profile, game)",
            fontsize=10, color=NEUTRAL, fontweight="bold", ha="center")
    for i in range(3):
        y = 4.5 - i * 0.9 + 0.27
        arrow(ax, 8.48, 3.25, 7.8, y, color=MUTED, alpha=0.6, lw=1.0)

    title(fig, "SBERT-CDR — content-based scoring",
          "Average the liked-movie title vectors, then cosine-score every game")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fig16_sbert.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ── Figure 18: Co-occurrence construction ───────────────────────────────

def fig_cooc():
    fig, ax = plt.subplots(figsize=(11.5, 5.0), dpi=180)
    setup_ax(ax, (0, 11.5), (0, 5.0))

    # a tiny user-item matrix (movies then games)
    rng = np.random.default_rng(21)
    n_users, n_movies, n_games = 6, 4, 4
    M = (rng.uniform(0, 1, (n_users, n_movies)) > 0.55).astype(int)
    G = (rng.uniform(0, 1, (n_users, n_games))  > 0.55).astype(int)

    # Movie likes
    ax.text(1.5, 4.6, "users × movies", fontsize=10,
            color=MOVIE, fontweight="bold", ha="center")
    for i in range(n_users):
        for j in range(n_movies):
            v = M[i, j]
            ax.add_patch(Rectangle((0.5 + j*0.32, 3.8 - i*0.32),
                                   0.32, 0.32,
                                   facecolor=MOVIE if v else "white",
                                   edgecolor=MOVIE, linewidth=0.6))

    # Game likes
    ax.text(4.6, 4.6, "users × games", fontsize=10,
            color=GAME, fontweight="bold", ha="center")
    for i in range(n_users):
        for j in range(n_games):
            v = G[i, j]
            ax.add_patch(Rectangle((3.6 + j*0.32, 3.8 - i*0.32),
                                   0.32, 0.32,
                                   facecolor=GAME if v else "white",
                                   edgecolor=GAME, linewidth=0.6))

    # M^T G arrow
    ax.text(6.85, 3.35, r"$M^{\top} G$", fontsize=18, fontweight="bold",
            color=INK, ha="center")
    arrow(ax, 2.05, 3.1, 6.45, 3.35, color=MOVIE, lw=1.4,
          connectionstyle="arc3,rad=0.15")
    arrow(ax, 5.15, 3.1, 6.85, 3.35, color=GAME, lw=1.4,
          connectionstyle="arc3,rad=-0.15")
    ax.text(6.85, 2.85, "count co-likes",
            fontsize=9, color=NEUTRAL, style="italic", ha="center")

    # Co-occurrence count matrix (movies × games)
    counts = M.T @ G
    ax.text(8.5, 4.6, "co-occurrence count", fontsize=10,
            color=INK, fontweight="bold", ha="center")
    mx = counts.max()
    for i in range(n_movies):
        for j in range(n_games):
            t = counts[i, j] / max(mx, 1)
            c = np.array([1,1,1]) * (1-t) + np.array([0.54,0.17,0.89]) * t
            ax.add_patch(Rectangle((7.8 + j*0.38, 3.9 - i*0.38),
                                   0.38, 0.38,
                                   facecolor=c, edgecolor=USER, linewidth=0.7))
            ax.text(7.8 + j*0.38 + 0.19, 3.9 - i*0.38 + 0.19,
                    str(counts[i, j]), ha="center", va="center",
                    fontsize=9, color=INK,
                    fontweight="bold" if counts[i,j] else "normal")

    arrow(ax, 7.3, 3.25, 7.75, 3.5, color=NEUTRAL, lw=1.4)

    # PPMI output
    ax.text(10.8, 4.6, "PPMI", fontsize=10,
            color=ACCENT, fontweight="bold", ha="center")
    rng2 = np.random.default_rng(22)
    for i in range(n_movies):
        for j in range(n_games):
            v = rng2.uniform(0, 1) * (counts[i,j] > 0)
            c = np.array([1,1,1]) * (1-v) + np.array([0.96,0.62,0.04]) * v
            ax.add_patch(Rectangle((10.0 + j*0.36, 3.9 - i*0.36),
                                   0.36, 0.36,
                                   facecolor=c, edgecolor=ACCENT, linewidth=0.6))
    arrow(ax, 9.95, 3.3, 9.95, 3.3, color=NEUTRAL, lw=1.4)
    arrow(ax, 9.65, 3.4, 10.0, 3.4, color=NEUTRAL, lw=1.4)
    ax.text(11.25, 2.55, "log lift\n(observed / chance)", fontsize=9,
            color=NEUTRAL, style="italic", ha="center")

    # Explanation at bottom
    ax.text(5.5, 0.8,
            "A non-zero cell = at least one user liked both. "
            "PPMI rescales by how often chance would produce the same overlap.",
            ha="center", fontsize=9.5, color=NEUTRAL, style="italic")

    title(fig, "Co-occurrence matrix — construction",
          "Count movie × game co-likes on shared users, then rescale to PPMI")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fig18_cooc.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ── run all ─────────────────────────────────────────────────────────────

def fig_neumf():
    """Figure 10 — NeuMF: combine GMF and MLP branches."""
    fig, ax = plt.subplots(figsize=(13, 5.6), dpi=180)
    setup_ax(ax, (0, 13), (0, 5.6))

    rng = np.random.default_rng(30)

    # GMF branch output (top)
    ax.text(2.5, 5.25, "GMF branch", fontsize=10.5,
            color=ACCENT, fontweight="bold", ha="center")
    v1 = rng.uniform(-1, 1, 8)
    vector(ax, 1.1, 4.3, v1, ACCENT, cell_w=0.3, cell_h=0.45,
           label="u × g  (GMF output)", fontsize=9)

    # MLP branch output (bottom)
    ax.text(2.5, 2.95, "MLP branch", fontsize=10.5,
            color=PINK, fontweight="bold", ha="center")
    v2 = rng.uniform(-1, 1, 8)
    vector(ax, 1.1, 2.1, v2, PINK, cell_w=0.3, cell_h=0.45,
           label="MLP hidden (MLP output)", fontsize=9)

    # Concat symbol inside a circle
    ax.add_patch(plt.Circle((4.5, 3.55), 0.3, facecolor="white",
                            edgecolor=INK, linewidth=1.8))
    ax.text(4.5, 3.55, "‖", fontsize=20, color=INK, ha="center",
            va="center", fontweight="bold")
    ax.text(4.5, 2.95, "concat", fontsize=9, ha="center",
            color=NEUTRAL, style="italic")
    arrow(ax, 3.6, 4.5, 4.25, 3.8, color=ACCENT, lw=1.5)
    arrow(ax, 3.6, 2.35, 4.25, 3.3, color=PINK, lw=1.5)

    # concatenated vector — 16 cells × 0.18 = 2.88 wide, starts at x=5.3
    v3 = np.concatenate([v1, v2])
    vector(ax, 5.3, 3.3, v3, NEUTRAL, cell_w=0.18, cell_h=0.5,
           label="concatenated", fontsize=9)
    arrow(ax, 4.8, 3.55, 5.3, 3.55, color=NEUTRAL, lw=1.4)

    # Linear chip — pushed past concat vector end (x=8.18), plenty of gap
    chip(ax, 9.1, 3.25, 1.1, 0.65, "linear", color=NEUTRAL, facecolor="white",
         fontsize=11, text_color=INK)
    arrow(ax, 8.2, 3.55, 9.1, 3.55, color=NEUTRAL, lw=1.4)
    ax.text(9.65, 3.1, "+ sigmoid", fontsize=9, ha="center",
            color=NEUTRAL, style="italic")

    # Score chip
    chip(ax, 11.1, 3.15, 1.2, 0.85, "score", color=ACCENT,
         facecolor="#FFFBEB", fontsize=12, text_color=INK)
    arrow(ax, 10.2, 3.58, 11.1, 3.58, color=ACCENT, lw=1.4)

    title(fig, "NeuMF — combined",
          "Concatenate the GMF and MLP branch outputs, then one linear layer → final score")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fig10_neumf.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def fig_sbert_cdr():
    """Figure 17 — SBERT-CDR cross-domain flow through shared semantic space."""
    fig, ax = plt.subplots(figsize=(11.5, 5.4), dpi=180)
    setup_ax(ax, (0, 11.5), (0, 5.4))

    rng = np.random.default_rng(31)

    # User side - movies
    ax.text(1.2, 5.0, "movie titles", fontsize=10, color=MOVIE,
            fontweight="bold")
    for i, n in enumerate(["Dune", "Blade Runner", "Arrival"]):
        y = 4.3 - i * 0.6
        chip(ax, 0.3, y, 1.6, 0.42, n, color=MOVIE, facecolor="#EFF6FF",
             fontsize=9, text_color=INK)

    # Game side
    ax.text(10.1, 5.0, "game titles", fontsize=10, color=GAME,
            fontweight="bold")
    for i, n in enumerate(["Stellaris", "Cyberpunk 2077", "Half-Life"]):
        y = 4.3 - i * 0.6
        chip(ax, 9.5, y, 1.7, 0.42, n, color=GAME, facecolor="#FEF2F2",
             fontsize=9, text_color=INK)

    # Central SBERT encoder
    ax.add_patch(FancyBboxPatch((4.3, 1.8), 3.0, 2.8,
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor="#F1F5F9", edgecolor=USER,
                                linewidth=1.8))
    ax.text(5.8, 4.25, "SBERT encoder",
            fontsize=11, color=USER, fontweight="bold", ha="center")
    ax.text(5.8, 3.95, "(frozen)", fontsize=9, color=NEUTRAL,
            ha="center", style="italic")

    # Inside encoder — show mini transformer layers
    for i in range(4):
        ax.add_patch(FancyBboxPatch(
            (4.5 + i*0.65, 2.4), 0.5, 1.3,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=USER, edgecolor=USER, alpha=0.75 - i*0.1))
    ax.text(5.8, 2.0, "L × self-attention", fontsize=8.5,
            color=NEUTRAL, ha="center", style="italic")

    # Arrows into encoder
    for i in range(3):
        y = 4.3 - i * 0.6 + 0.21
        arrow(ax, 1.9, y, 4.3, 3.3, color=MOVIE, lw=1.0, alpha=0.6)
        arrow(ax, 9.5, y, 7.3, 3.3, color=GAME, lw=1.0, alpha=0.6)

    # shared 384-d semantic space + 2 example vectors side by side
    ax.text(5.8, 1.2, "shared 384-d semantic space",
            fontsize=10, color=INK, fontweight="bold", ha="center")

    v_m = rng.uniform(-1, 1, 16)
    v_g = rng.uniform(-1, 1, 16)
    vector(ax, 1.8, 0.4, v_m, MOVIE, cell_w=0.18, cell_h=0.45, fontsize=8,
           label="movie vec", label_above=False)
    vector(ax, 6.5, 0.4, v_g, GAME, cell_w=0.18, cell_h=0.45, fontsize=8,
           label="game vec", label_above=False)

    arrow(ax, 5.8, 1.8, 3.25, 0.8, color=MOVIE, lw=1.2,
          connectionstyle="arc3,rad=-0.15")
    arrow(ax, 5.8, 1.8, 8.0, 0.8, color=GAME, lw=1.2,
          connectionstyle="arc3,rad=0.15")

    # cosine link
    arrow(ax, 3.25, 0.6, 6.5, 0.6, color=NEUTRAL, style="<->", lw=1.4)
    ax.text(4.88, 0.95, "cosine similarity",
            fontsize=9.5, color=INK, fontweight="bold", ha="center")

    title(fig, "SBERT-CDR — cross-domain flow",
          "Movies and games land in the same 384-d space — score by cosine similarity")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "fig17_sbert_cdr.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def fig_cooc_rerank():
    """Figure 19 — Co-occurrence reranking at inference."""
    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=180)
    setup_ax(ax, (0, 12), (0, 5.6))

    games = ["Stellaris", "Civ VI", "Cyberpunk", "Hades", "Factorio"]
    base = [0.72, 0.65, 0.61, 0.55, 0.48]
    lift = [0.05, 0.02, 0.60, 0.11, 0.09]

    # Base model scores (left column)
    ax.text(1.5, 4.55, "base model score", fontsize=10,
            color=NEUTRAL, fontweight="bold", ha="center")
    for i, (n, s) in enumerate(zip(games, base)):
        y = 3.85 - i * 0.55
        ax.add_patch(Rectangle((0.5, y), s * 2.0, 0.38,
                               facecolor=USER, edgecolor=USER, alpha=0.7))
        ax.text(0.45, y + 0.19, n, ha="right", va="center", fontsize=9.5,
                color=INK)
        ax.text(0.55 + s*2.0 + 0.05, y + 0.19, f"{s:.2f}", ha="left",
                va="center", fontsize=9, color=NEUTRAL)

    # + λ ·
    ax.text(3.7, 2.5, "+  λ  ·", fontsize=19, color=INK,
            ha="center", va="center", fontweight="bold")

    # Cooc lookup (middle)
    ax.text(5.6, 4.55, "cooc lift  (from PPMI)", fontsize=10,
            color=ACCENT, fontweight="bold", ha="center")
    mx = max(lift)
    for i, (n, v) in enumerate(zip(games, lift)):
        y = 3.85 - i * 0.55
        t = v / mx
        c = np.array([1,1,1]) * (1-t) + np.array([0.96,0.62,0.04]) * t
        ax.add_patch(Rectangle((4.6, y), 2.0, 0.38,
                               facecolor=c, edgecolor=ACCENT, linewidth=1.0))
        ax.text(5.6, y + 0.19, f"+{v:.2f}", ha="center", va="center",
                fontsize=9.5, color=INK, fontweight="bold")

    # = symbol
    ax.text(7.5, 2.5, "=", fontsize=22, color=INK,
            ha="center", va="center", fontweight="bold")

    # Final reranked scores (right)
    ax.text(9.4, 4.55, "reranked", fontsize=10,
            color=INK, fontweight="bold", ha="center")
    final = [b + 0.3 * l for b, l in zip(base, lift)]
    order = sorted(range(len(games)), key=lambda i: -final[i])
    medal_colors = [GREEN, GREEN, GREEN, NEUTRAL, NEUTRAL]
    for rank, idx in enumerate(order):
        y = 3.85 - rank * 0.55
        ax.add_patch(Rectangle((8.4, y), final[idx] * 2.0, 0.38,
                               facecolor=medal_colors[rank],
                               edgecolor=medal_colors[rank], alpha=0.85))
        ax.text(8.35, y + 0.19, games[idx], ha="right", va="center",
                fontsize=9.5, color=INK,
                fontweight="bold" if games[idx] == "Cyberpunk" else "normal")
        ax.text(8.45 + final[idx]*2.0 + 0.05, y + 0.19, f"{final[idx]:.2f}",
                ha="left", va="center", fontsize=9, color=NEUTRAL)
        if games[idx] == "Cyberpunk":
            ax.annotate("↑ 3rd → 1st", xy=(11.2, y + 0.19),
                        fontsize=10, color=GREEN,
                        fontweight="bold", ha="left", va="center")

    # Bottom note
    ax.text(6.0, 0.4,
            "Cyberpunk had the biggest cooc lift → it jumps to the top",
            fontsize=10, color=NEUTRAL, style="italic", ha="center")

    title(fig, "Co-occurrence reranking at inference",
          "Final score = base score + λ × cooc lift — tiny λ, big ranking effect")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fig19_cooc_rerank.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    fig_mf_bpr_training()
    fig_mf_bpr_inference()
    fig_ncf_gmf()
    fig_ncf_mlp()
    fig_neumf()
    fig_lightgcn_prop()
    fig_lightgcn_training()
    fig_cmf()
    fig_emcdr()
    fig_ptupcdr()
    fig_sbert()
    fig_sbert_cdr()
    fig_cooc()
    fig_cooc_rerank()
    print("All model diagrams saved to", OUT)
