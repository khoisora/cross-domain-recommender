"""Modern formula cards for the project report.

Each card is a matplotlib figure with:
  - Coloured title strip (model name)
  - Centred mathtext equation
  - Italic legend / where-clause

Output: ./new_figures/formulas/formula_<name>.png at 220 DPI.
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import FONT, BG, TITLE, MUTED, PALETTE, set_mpl_style

set_mpl_style()

OUT = Path(__file__).resolve().parent.parent / "new_figures" / "formulas"
OUT.mkdir(parents=True, exist_ok=True)


def formula_card(name: str, title: str, formula: str, legend: str, color: str,
                 width: float = 8.5, height: float = 2.6, formula_fs: int = 22) -> None:
    """Render one formula card."""
    c = PALETTE[color]

    fig = plt.figure(figsize=(width, height), dpi=220, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Outer rounded panel
    pad = 0.018
    panel = FancyBboxPatch(
        (pad, pad), 1 - 2 * pad, 1 - 2 * pad,
        boxstyle="round,pad=0.0,rounding_size=0.025",
        linewidth=1.6, edgecolor=c["stroke"], facecolor=c["fill"],
    )
    ax.add_patch(panel)

    # Title strip on top
    strip_h = 0.28
    strip = Rectangle(
        (pad, 1 - pad - strip_h), 1 - 2 * pad, strip_h,
        linewidth=0, facecolor=c["stroke"],
    )
    ax.add_patch(strip)
    ax.text(
        0.5, 1 - pad - strip_h / 2, title,
        color="#FFFFFF", fontsize=15, fontweight="bold",
        ha="center", va="center", fontname=FONT,
    )

    # Formula (centered in middle band)
    ax.text(
        0.5, 0.42, formula,
        color=c["text"], fontsize=formula_fs,
        ha="center", va="center",
    )

    # Legend (italic, muted, bottom)
    ax.text(
        0.5, 0.13, legend,
        color=MUTED, fontsize=10, style="italic",
        ha="center", va="center", fontname=FONT,
        wrap=True,
    )

    out = OUT / f"formula_{name}.png"
    fig.savefig(out, dpi=220, facecolor=BG, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"  rendered formulas/formula_{name}.png")


def main():
    print(f"Rendering formula cards to {OUT}/")

    formula_card(
        "bpr", "BPR loss (MF)",
        r"$\mathbf{L}_{BPR} = -\sum_{(u, i^+, i^-)} \log\, \sigma\!\left(\hat r_{u,i^+} - \hat r_{u,i^-}\right) + \lambda\,\|\Theta\|^2$",
        r"$\hat r_{u,i}=p_u \!\cdot q_i$; sample one positive and one negative per user.",
        "indigo",
    )

    formula_card(
        "ncf", "NeuMF score",
        r"$\hat y_{u,i} = \sigma\!\left(h^{\top}\left[\,p_u^G \odot q_i^G\;;\;\phi_{MLP}(p_u^M, q_i^M)\,\right]\right)$",
        r"Concatenate GMF (element-wise product) with MLP output, then a linear head.",
        "sky",
    )

    formula_card(
        "lightgcn", "LightGCN aggregation",
        r"$e^{(k+1)} = \tilde A\, e^{(k)}\,,\quad e^{*} = \frac{1}{K+1}\sum_{k=0}^{K} e^{(k)}$",
        r"$\tilde A = D^{-1/2} A D^{-1/2}$ is the symmetric normalised adjacency; no self-weights.",
        "teal",
    )

    formula_card(
        "cmf", "Collective MF",
        r"$\min_{P,Q_m,Q_g}\;\alpha \|R_m - P Q_m^{\top}\|_F^2 + (1-\alpha)\|R_g - P Q_g^{\top}\|_F^2 + \lambda\,\Omega(P, Q_m, Q_g)$",
        r"Shared user factors $P$ are learned jointly from both domains; $\alpha$ balances them.",
        "emerald",
        height=2.8,
        formula_fs=19,
    )

    formula_card(
        "emcdr", "EMCDR mapping",
        r"$\hat u_g = f_\phi(u_m)\,,\qquad \mathbf{L}_{map} = \sum_{u \in \mathbf{U}_{\cap}} \|f_\phi(u_m) - u_g\|^2$",
        r"Train MF separately; then learn $f_\phi$ (MLP) on users with history in both domains.",
        "amber",
    )

    formula_card(
        "ptupcdr", "PTUPCDR (personalised bridge)",
        r"$\theta_u = g_{\psi}(u_m)\,,\qquad \hat u_g = f_{\theta_u}(u_m)$",
        r"A meta-network $g_\psi$ outputs per-user bridge parameters $\theta_u$ from the source embedding.",
        "rose",
    )

    formula_card(
        "sbert", "SBERT-CDR scoring",
        r"$\text{score}(u, g) = \cos\!\left(\,\frac{1}{|H_u|}\sum_{m \in H_u} v_m\,,\, v_g\right)$",
        r"Mean-pooled sentence-BERT embedding of the user's movie history vs. a game title embedding.",
        "violet",
    )

    formula_card(
        "cooc", "Co-occurrence rerank (PPMI)",
        r"$\text{PPMI}(m, g) = \max\!\left(0,\; \log\frac{p(m, g)}{p(m)\,p(g)}\right)$",
        r"Positive Pointwise Mutual Information over co-liked (movie, game) pairs; used as a prior.",
        "cyan",
    )

    print(f"Done. 8 formula cards in {OUT}/")


if __name__ == "__main__":
    main()
