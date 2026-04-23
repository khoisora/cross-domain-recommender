"""Render fig14a_emcdr_architecture.png and fig14b_emcdr_inference.png.

Architecture figure: the three embedding spaces (U^M, V^M on the movie
side; U^G, V^G on the game side) plus the MLP translator f_θ that
bridges the two user spaces. Emphasises *structure*, not training order.

Inference figure: user's movie-space embedding u_u^M → f_θ → translated
vector → dot product with every V^G row → ranked game scores.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "report_figures_v3"
OUT.mkdir(exist_ok=True)

BLUE = "#4C8AE7"
BLUE_LIGHT = "#CAD9F5"
RED = "#E74C5E"
RED_LIGHT = "#F7B4BC"
GREEN = "#2EB872"
GREEN_LIGHT = "#BFEBD3"
PURPLE = "#6C5CE7"
PURPLE_LIGHT = "#C9C2F5"
GREY = "#8892A6"
INK = "#222B45"


def _matrix(ax, x, y, w, h, color_dark, color_light, label,
            sublabel=None, rng=None, rows=6, cols=6):
    rng = rng or np.random.default_rng(0)
    cell_w = w / cols
    cell_h = h / rows
    for r in range(rows):
        for c in range(cols):
            v = rng.uniform()
            col = color_dark if v > 0.66 else color_light if v > 0.33 else "white"
            ax.add_patch(mp.Rectangle(
                (x + c * cell_w, y + r * cell_h),
                cell_w, cell_h,
                facecolor=col, edgecolor="#8892A6", lw=0.4))
    ax.text(x + w / 2, y + h + 0.025, label,
            ha="center", fontsize=11, fontweight="bold", color=INK)
    if sublabel:
        ax.text(x + w / 2, y - 0.035, sublabel,
                ha="center", fontsize=8.5, style="italic", color=GREY)


def _row_heat(ax, x, y, n, cw, ch, color_dark, color_light, rng):
    vals = rng.uniform(0, 1, size=n)
    for i, v in enumerate(vals):
        c = color_dark if v > 0.66 else color_light if v > 0.33 else "white"
        ax.add_patch(mp.Rectangle((x + i * cw, y), cw, ch,
                                  facecolor=c, edgecolor="#8892A6", lw=0.5))


def _mlp(ax, x, y, w=0.10, h=0.18):
    """Draw a stylised MLP: two vertical rounded 'layers' with an arrow."""
    for i, xoff in enumerate((0.0, w / 2)):
        ax.add_patch(mp.FancyBboxPatch(
            (x + xoff, y), w / 4, h,
            boxstyle="round,pad=0.003,rounding_size=0.015",
            facecolor=PURPLE, edgecolor=PURPLE))
    ax.text(x + w / 2 + w / 8, y + h + 0.03, "f_θ  (MLP)",
            ha="center", fontsize=10.5, fontweight="bold", color=INK)


# ---------------------------------------------------------------
#                     Architecture figure
# ---------------------------------------------------------------

def render_architecture():
    fig, ax = plt.subplots(figsize=(11.0, 5.6), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(0.5, 0.94, "EMCDR — architecture",
             ha="center", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.5, 0.885,
             "Two independent embedding spaces bridged by a learned translator f_θ",
             ha="center", fontsize=11, style="italic", color=GREY)

    rng = np.random.default_rng(11)

    # Movie side (left).
    _matrix(ax, 0.04, 0.52, 0.16, 0.22, BLUE, BLUE_LIGHT,
            r"$\mathbf{U}^{M}$", sublabel="movie-domain users", rng=rng)
    _matrix(ax, 0.23, 0.52, 0.16, 0.22, BLUE, BLUE_LIGHT,
            r"$\mathbf{V}^{M}$", sublabel="movie items", rng=rng)
    # Domain tag.
    ax.add_patch(mp.FancyBboxPatch(
        (0.03, 0.79), 0.37, 0.06,
        boxstyle="round,pad=0.004,rounding_size=0.015",
        facecolor="white", edgecolor=BLUE, lw=1.4))
    ax.text(0.215, 0.82, "movie-domain MF-BPR (phase 1)",
            ha="center", fontsize=10.5, color=BLUE, fontweight="bold")

    # Game side (right).
    _matrix(ax, 0.60, 0.52, 0.16, 0.22, RED, RED_LIGHT,
            r"$\mathbf{U}^{G}$", sublabel="game-domain users", rng=rng)
    _matrix(ax, 0.79, 0.52, 0.16, 0.22, RED, RED_LIGHT,
            r"$\mathbf{V}^{G}$", sublabel="game items", rng=rng)
    ax.add_patch(mp.FancyBboxPatch(
        (0.59, 0.79), 0.37, 0.06,
        boxstyle="round,pad=0.004,rounding_size=0.015",
        facecolor="white", edgecolor=RED, lw=1.4))
    ax.text(0.775, 0.82, "game-domain MF-BPR (phase 2)",
            ha="center", fontsize=10.5, color=RED, fontweight="bold")

    # Translator bridge below the two sides.
    # Source: a single u_u^M row on the left; target: a u_u^G row on the right.
    # Centered MLP f_θ in the middle.
    row_n = 10
    row_cw = 0.018
    row_ch = 0.06
    src_x = 0.12
    src_y = 0.24
    _row_heat(ax, src_x, src_y, row_n, row_cw, row_ch, BLUE, BLUE_LIGHT, rng)
    ax.text(src_x + row_n * row_cw / 2, src_y - 0.05,
            r"$\mathbf{u}_{u}^{M}$  (row of $\mathbf{U}^{M}$)",
            ha="center", fontsize=10, color=INK)

    tgt_x = 0.71
    tgt_y = 0.24
    _row_heat(ax, tgt_x, tgt_y, row_n, row_cw, row_ch, GREEN, GREEN_LIGHT, rng)
    ax.text(tgt_x + row_n * row_cw / 2, tgt_y - 0.05,
            r"$f_{\theta}(\mathbf{u}_{u}^{M})$  — in game space",
            ha="center", fontsize=10, color=INK)

    # MLP block center.
    _mlp(ax, 0.44, 0.19, w=0.12, h=0.16)

    # Arrows.
    ax.annotate("", xy=(0.44, 0.27),
                xytext=(src_x + row_n * row_cw + 0.005, src_y + row_ch / 2),
                arrowprops=dict(arrowstyle="->", lw=1.4, color=BLUE))
    ax.annotate("", xy=(tgt_x - 0.005, tgt_y + row_ch / 2),
                xytext=(0.44 + 0.12, 0.27),
                arrowprops=dict(arrowstyle="->", lw=1.4, color=GREEN))

    # Supervision: bracket connecting f_θ(u^M) to u^G on overlap users.
    sup_y = 0.09
    ax.text(0.5, sup_y,
            r"phase 3 fits $f_{\theta}$ by minimising  "
            r"$\|f_{\theta}(\mathbf{u}_{u}^{M}) - \mathbf{u}_{u}^{G}\|_{2}^{2}$  "
            r"on overlap users  $u \in \mathcal{U}_{\mathrm{overlap}}$",
            ha="center", fontsize=10.5, color=PURPLE, style="italic")

    # Divider between the two domain boxes.
    ax.plot([0.50, 0.50], [0.48, 0.88], color=GREY, lw=0.6, ls=":")

    out = OUT / "fig14a_emcdr_architecture.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.2, facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ---------------------------------------------------------------
#                     Inference figure
# ---------------------------------------------------------------

def render_inference():
    fig, ax = plt.subplots(figsize=(11.0, 5.6), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(0.5, 0.94, "EMCDR — inference",
             ha="center", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.5, 0.885,
             r"Translate the user's movie vector with $f_{\theta}$, then score every game by dot product",
             ha="center", fontsize=11, style="italic", color=GREY)

    rng = np.random.default_rng(23)

    # Stage 1: movie-side user vector (far left, compact).
    n = 8
    cw = 0.018
    ch = 0.06
    x0 = 0.02
    y0 = 0.50
    _row_heat(ax, x0, y0, n, cw, ch, BLUE, BLUE_LIGHT, rng)
    ax.text(x0 + n * cw / 2, y0 + ch + 0.04,
            r"$\mathbf{u}_{u}^{M}$", ha="center",
            fontsize=12, fontweight="bold", color=INK)
    ax.text(x0 + n * cw / 2, y0 - 0.05,
            "user's movie vector",
            ha="center", fontsize=9, style="italic", color=GREY)

    # Stage 2: MLP.
    mlp_x = x0 + n * cw + 0.04
    _mlp(ax, mlp_x, y0 - 0.03, w=0.09, h=ch + 0.08)

    # Stage 3: translated vector (green).
    tgt_x = mlp_x + 0.13
    _row_heat(ax, tgt_x, y0, n, cw, ch, GREEN, GREEN_LIGHT, rng)
    ax.text(tgt_x + n * cw / 2, y0 + ch + 0.04,
            r"$f_{\theta}(\mathbf{u}_{u}^{M})$",
            ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.text(tgt_x + n * cw / 2, y0 - 0.05,
            "in game space",
            ha="center", fontsize=9, style="italic", color=GREY)

    # Arrows stage 1 → MLP → stage 3.
    ax.annotate("", xy=(mlp_x, y0 + ch / 2),
                xytext=(x0 + n * cw + 0.005, y0 + ch / 2),
                arrowprops=dict(arrowstyle="->", lw=1.6, color=BLUE))
    ax.annotate("", xy=(tgt_x - 0.005, y0 + ch / 2),
                xytext=(mlp_x + 0.09, y0 + ch / 2),
                arrowprops=dict(arrowstyle="->", lw=1.6, color=GREEN))

    trans_right_x_override = tgt_x + n * cw

    # Stage 4: candidate games + scores (right column).
    games = [("Stellaris", 0.81), ("Civ VI", 0.69), ("Cyberpunk", 0.55),
             ("Hades", 0.42), ("Factorio", 0.28)]
    top_y = 0.78
    dy = 0.12
    name_x = 0.62
    row_gx = 0.645
    n_g = 9
    gw = 0.020
    gh = 0.060
    score_x = 0.88

    ax.text(row_gx + n_g * gw / 2, top_y + gh + 0.05,
            r"candidate games (rows of $\mathbf{V}^{G}$)",
            ha="center", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(score_x, top_y + gh + 0.05,
            r"score = $f_{\theta}(\mathbf{u}_{u}^{M}) \cdot \mathbf{v}_{g}$",
            ha="left", fontsize=10.5, fontweight="bold", color=INK)

    trans_right_x = trans_right_x_override
    trans_mid_y = y0 + ch / 2

    for i, (name, score) in enumerate(games):
        gy = top_y - i * dy
        ax.text(name_x, gy + gh / 2, name,
                ha="right", va="center", fontsize=10.5, color=INK)
        _row_heat(ax, row_gx, gy, n_g, gw, gh, RED, RED_LIGHT, rng)
        ax.annotate("", xy=(name_x - 0.06, gy + gh / 2),
                    xytext=(trans_right_x + 0.005, trans_mid_y),
                    arrowprops=dict(arrowstyle="->", lw=0.7,
                                    color=GREY, alpha=0.35,
                                    connectionstyle="arc3,rad=-0.05"))
        bar_w = score * 0.09
        ax.add_patch(mp.Rectangle((score_x, gy + 0.004), bar_w, gh - 0.008,
                                  facecolor=RED, alpha=0.65, edgecolor="none"))
        ax.text(score_x - 0.008, gy + gh / 2, f"{score:.2f}",
                ha="right", va="center", fontsize=11,
                fontweight="bold", color=INK)

    fig.text(0.5, 0.045,
             "sort scores, return top-K games — one forward pass through f_θ per user",
             ha="center", fontsize=10, style="italic", color=GREY)

    out = OUT / "fig14b_emcdr_inference.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.2, facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out.name}")


if __name__ == "__main__":
    render_architecture()
    render_inference()
