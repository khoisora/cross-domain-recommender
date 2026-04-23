"""Render one small, clean PNG per model's loss / scoring formula.

Uses matplotlib mathtext (LaTeX-style) on a white canvas at 220 dpi so the
formulas are crisp in Word. Output: report_figures_v3/fig_loss_<model>.png.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "report_figures_v3"
OUT.mkdir(exist_ok=True)

# Use a serif look close to Computer Modern.
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"


def _render(filename, expression, width_in=5.6, height_in=0.9, fontsize=16):
    fig = plt.figure(figsize=(width_in, height_in), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, expression, ha="center", va="center",
            fontsize=fontsize, color="#111111")
    fig.savefig(OUT / filename, bbox_inches="tight", pad_inches=0.12,
                facecolor="white")
    plt.close(fig)
    print(f"  ✓ {filename}")


def main():
    # 1. MF-BPR: pairwise log-sigmoid of positive/negative score margin.
    _render(
        "fig_loss_mfbpr.png",
        r"$\mathcal{L}_{\mathrm{BPR}} \;=\; -\!\!\sum_{(u,\,i^{+},\,i^{-}) \in \mathcal{D}}"
        r"\ln \sigma\!\left(\mathbf{u}_{u}^{\!\top}\mathbf{v}_{i^{+}} \;-\; "
        r"\mathbf{u}_{u}^{\!\top}\mathbf{v}_{i^{-}}\right) "
        r"\;+\; \lambda\,\|\Theta\|_{2}^{\,2}$",
        width_in=6.4, height_in=1.0, fontsize=17,
    )

    # 2. NCF: binary cross-entropy over observed positives and sampled negatives.
    _render(
        "fig_loss_ncf.png",
        r"$\mathcal{L}_{\mathrm{BCE}} \;=\; -\!\!\sum_{(u,i)\in\mathcal{D}^{+}\cup\mathcal{D}^{-}}"
        r"\left[\,y_{u,i}\log\hat{y}_{u,i} \;+\; (1-y_{u,i})\log\!\left(1-\hat{y}_{u,i}\right)\right]$",
        width_in=6.8, height_in=1.0, fontsize=17,
    )

    # 3. LightGCN: BPR on layer-combined embeddings.
    _render(
        "fig_loss_lightgcn.png",
        r"$\tilde{\mathbf{e}}_{v} \;=\; \frac{1}{K+1}\!\sum_{k=0}^{K}\mathbf{e}_{v}^{(k)}"
        r"\qquad\mathcal{L}_{\mathrm{LGCN}} \;=\; -\!\!\sum_{(u,\,i^{+},\,i^{-})}"
        r"\ln\sigma\!\left(\tilde{\mathbf{e}}_{u}^{\!\top}\tilde{\mathbf{e}}_{i^{+}} - "
        r"\tilde{\mathbf{e}}_{u}^{\!\top}\tilde{\mathbf{e}}_{i^{-}}\right) + "
        r"\lambda\,\|\mathbf{E}^{(0)}\|_{2}^{\,2}$",
        width_in=7.6, height_in=1.1, fontsize=16,
    )

    # 4. CMF: weighted sum of per-domain BPR losses over a shared user matrix U.
    _render(
        "fig_loss_cmf.png",
        r"$\mathcal{L}_{\mathrm{CMF}} \;=\; \alpha\,\mathcal{L}_{\mathrm{BPR}}^{\mathrm{movie}}"
        r"(\mathbf{U},\mathbf{V}_{m}) "
        r"\;+\; (1-\alpha)\,\mathcal{L}_{\mathrm{BPR}}^{\mathrm{game}}(\mathbf{U},\mathbf{V}_{g}) "
        r"\;+\; \lambda\,\|\Theta\|_{2}^{\,2}$",
        width_in=7.0, height_in=1.0, fontsize=16,
    )

    # 5. EMCDR: Phase-3 mapping loss (MSE from mapped movie-user to game-user).
    _render(
        "fig_loss_emcdr.png",
        r"$\mathcal{L}_{\mathrm{EMCDR}} \;=\; \sum_{u\in\mathcal{U}_{\mathrm{overlap}}}"
        r"\left\| f_{\theta}\!\left(\mathbf{u}_{u}^{\mathrm{movie}}\right) - "
        r"\mathbf{u}_{u}^{\mathrm{game}}\right\|_{2}^{\,2}$",
        width_in=6.0, height_in=0.95, fontsize=17,
    )

    # 6. PTUPCDR: mixture-of-experts personalized mapping, same MSE target.
    _render(
        "fig_loss_ptupcdr.png",
        r"$\mathcal{L}_{\mathrm{PTUPCDR}} \;=\; \sum_{u\in\mathcal{U}_{\mathrm{overlap}}}"
        r"\left\| \sum_{k=1}^{K} g_{k}\!\left(\mathbf{u}_{u}^{\mathrm{movie}}\right)\;"
        r"f_{k}\!\left(\mathbf{u}_{u}^{\mathrm{movie}}\right) - \mathbf{u}_{u}^{\mathrm{game}}"
        r"\right\|_{2}^{\,2}$",
        width_in=7.2, height_in=1.1, fontsize=16,
    )

    # 7. SBERT-CDR: cosine scoring (no training loss — pre-trained encoder).
    _render(
        "fig_loss_sbert.png",
        r"$\mathbf{p}_{u} = \frac{1}{|\mathcal{I}_{u}|}\sum_{j\in\mathcal{I}_{u}}\mathbf{e}_{j}"
        r"\qquad s_{u,i} = \cos\!\left(\mathbf{p}_{u},\mathbf{e}_{i}\right)"
        r"= \frac{\mathbf{p}_{u}\cdot\mathbf{e}_{i}}"
        r"{\|\mathbf{p}_{u}\|\,\|\mathbf{e}_{i}\|}$",
        width_in=7.0, height_in=1.0, fontsize=16,
    )

    # 8. Co-occurrence reranking: geometric-mean-normalised count + blend.
    _render(
        "fig_loss_cooc.png",
        r"$c_{m,g} = \frac{|\mathcal{U}_{m}\cap\mathcal{U}_{g}|}"
        r"{\sqrt{|\mathcal{U}_{m}|\,|\mathcal{U}_{g}|}}"
        r"\qquad s^{\mathrm{rerank}}_{u,g} \;=\; s^{\mathrm{base}}_{u,g} \;+\; "
        r"\lambda\!\!\sum_{m\in\mathcal{I}_{u}^{\mathrm{movie}}}\!\!c_{m,g}$",
        width_in=7.2, height_in=1.05, fontsize=16,
    )


if __name__ == "__main__":
    main()
