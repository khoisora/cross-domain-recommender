"""Render fig13b_cmf_inference.png — CMF inference diagram.

Mirrors fig07 (MF-BPR inference) but emphasises that the user vector
used for scoring game candidates is the *shared* vector trained jointly
on movie and game signal. Output lands in report_figures_v3/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "report_figures_v3"
OUT.mkdir(exist_ok=True)

PURPLE = "#6C5CE7"
PURPLE_LIGHT = "#C9C2F5"
RED = "#E74C5E"
RED_LIGHT = "#F7B4BC"
BLUE = "#4C8AE7"
GREY = "#8892A6"
INK = "#222B45"


def _heat_row(ax, x, y, values, cell_w, cell_h, colors):
    for i, v in enumerate(values):
        c = colors[0] if v > 0.66 else colors[1] if v > 0.33 else "white"
        ax.add_patch(mp.Rectangle((x + i * cell_w, y), cell_w, cell_h,
                                  facecolor=c, edgecolor="#8892A6", lw=0.6))


def main():
    # Figure canvas: axes span [0,1] x [0,1].
    fig, ax = plt.subplots(figsize=(11.0, 5.6), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title + subtitle.
    fig.text(0.5, 0.94, "CMF — inference",
             ha="center", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.5, 0.885,
             "The shared user vector U[u] scores every game by dot product against V_game",
             ha="center", fontsize=11, style="italic", color=GREY)

    # ---- Shared user vector on the left ----
    n_u = 10
    uw = 0.028
    uh = 0.09
    ux = 0.06
    uy = 0.50
    rng = np.random.default_rng(7)
    u_vals = rng.uniform(0.2, 1, size=n_u)
    _heat_row(ax, ux, uy, u_vals, uw, uh, (PURPLE, PURPLE_LIGHT))

    ax.text(ux + n_u * uw / 2, uy + uh + 0.035,
            "shared user vector  U[u]",
            ha="center", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(ux + n_u * uw / 2, uy - 0.04,
            "trained jointly on movie + game BPR",
            ha="center", fontsize=9, style="italic", color=GREY)

    # Labels for the two training signals that fed into U[u].
    ax.annotate("", xy=(ux + 0.005, uy + 0.045),
                xytext=(ux - 0.05, uy + 0.11),
                arrowprops=dict(arrowstyle="->", lw=1.1, color=BLUE))
    ax.text(ux - 0.055, uy + 0.115, "movie signal",
            ha="right", fontsize=9, color=BLUE, style="italic")
    ax.annotate("", xy=(ux + 0.005, uy + 0.025),
                xytext=(ux - 0.05, uy - 0.04),
                arrowprops=dict(arrowstyle="->", lw=1.1, color=RED))
    ax.text(ux - 0.055, uy - 0.055, "game signal",
            ha="right", fontsize=9, color=RED, style="italic")

    # ---- Candidate games column (center) ----
    # The user vector occupies x-range [0.06, 0.34]. Leave gap → start at 0.48.
    games = [("Stellaris", 0.87), ("Civ VI", 0.74), ("Cyberpunk", 0.61),
             ("Hades", 0.49), ("Factorio", 0.33)]
    n_g = 10
    gw = 0.024
    gh = 0.07
    gx = 0.50   # left edge of each game's heat row
    name_x = 0.48  # right-align item names to just before gx
    top_y = 0.74
    dy = 0.12

    # Column headers.
    ax.text(gx + n_g * gw / 2, top_y + gh + 0.07,
            "candidate games (rows of V_game)",
            ha="center", fontsize=10.5, fontweight="bold", color=INK)

    score_x = 0.80
    ax.text(score_x, top_y + gh + 0.07,
            "score = U[u] · V_g",
            ha="left", fontsize=10.5, fontweight="bold", color=INK)

    user_vec_right_x = ux + n_u * uw
    user_vec_mid_y = uy + uh / 2

    for i, (name, score) in enumerate(games):
        gy = top_y - i * dy
        # Item name — right-aligned to name_x so it cannot overrun heat row.
        ax.text(name_x, gy + gh / 2, name,
                ha="right", va="center", fontsize=10.5, color=INK)

        # Heat row for game embedding.
        g_vals = rng.uniform(0.15, 1, size=n_g)
        _heat_row(ax, gx, gy, g_vals, gw, gh, (RED, RED_LIGHT))

        # Arrow from user-vector end to item name start.
        ax.annotate("", xy=(name_x - 0.07, gy + gh / 2),
                    xytext=(user_vec_right_x + 0.005, user_vec_mid_y),
                    arrowprops=dict(arrowstyle="->", lw=0.7,
                                    color=GREY, alpha=0.35,
                                    connectionstyle="arc3,rad=-0.05"))

        # Score bar (right column).
        bar_w = score * 0.16
        ax.add_patch(mp.Rectangle((score_x, gy + 0.005), bar_w, gh - 0.01,
                                  facecolor=RED, alpha=0.65,
                                  edgecolor="none"))
        ax.text(score_x - 0.008, gy + gh / 2, f"{score:.2f}",
                ha="right", va="center", fontsize=11,
                fontweight="bold", color=INK)

    # Footer.
    fig.text(0.5, 0.045,
             "sort scores, return top-K games — no movie-side compute at inference",
             ha="center", fontsize=10, style="italic", color=GREY)

    out_path = OUT / "fig13b_cmf_inference.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.2,
                facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


if __name__ == "__main__":
    main()
