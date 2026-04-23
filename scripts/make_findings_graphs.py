"""Per-lesson "key-findings" comparison graphs (small, focused).

Each graph visualises the single headline number(s) in the key-findings
table — paired before/after bars with green delta arrows where applicable.

Output: report_figures_v3/fig_findings_lessonN.png
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "report_figures_v3"
OUT.mkdir(exist_ok=True)

NAVY = "#1F3A5F"; TEAL = "#2EC4B6"; BLUE = "#4472C4"; PINK = "#EC4899"
AMBER = "#F0A202"; GREEN = "#10B981"; RED = "#E5484D"; GREY = "#B9BDC7"
INK = "#111827"; PANEL = "#F6F7F9"


def _paired(ax, left_vals, right_vals, labels, left_label, right_label,
            left_color=GREY, right_color=TEAL, annotate=None, fmt="{:.4f}",
            title=None, ymax=None, lower_is_better=False):
    """Grouped bars: left state vs right state per model, with green delta arrows."""
    n = len(labels)
    x = np.arange(n)
    w = 0.36
    ax.bar(x - w / 2, left_vals, w, facecolor=left_color, alpha=0.8, label=left_label)
    ax.bar(x + w / 2, right_vals, w, facecolor=right_color, alpha=0.9, label=right_label)
    if ymax is None:
        ymax = max(max(left_vals), max(right_vals)) * 1.35
    ax.set_ylim(0, ymax)

    for i, (l, r) in enumerate(zip(left_vals, right_vals)):
        ax.text(i - w / 2, l + ymax * 0.015, fmt.format(l),
                ha="center", fontsize=8, color="#4B5563")
        ax.text(i + w / 2, r + ymax * 0.015, fmt.format(r),
                ha="center", fontsize=8, color=INK, fontweight="bold")
        improved = (r < l) if lower_is_better else (r > l)
        # Delta arrow
        if r != l:
            colour = GREEN if improved else RED
            ax.annotate("", xy=(i + w / 2, r),
                        xytext=(i - w / 2, l),
                        arrowprops=dict(arrowstyle="->", color=colour, lw=1.4,
                                        alpha=0.75))
        if annotate is not None:
            tag = annotate[i]
            if tag:
                ax.text(i, ymax * 0.88, tag, ha="center", fontsize=9,
                        fontweight="bold",
                        color=GREEN if improved or r == l else RED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GREY, alpha=0.3, lw=0.6)
    if title:
        ax.set_title(title, fontsize=11, color=INK, fontweight="bold", loc="left")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)


def _single_bars(ax, vals, labels, colors, title, fmt="{:.4f}", ymax=None):
    n = len(labels)
    x = np.arange(n)
    ax.bar(x, vals, 0.55, color=colors, alpha=0.9)
    if ymax is None:
        ymax = max(vals) * 1.25 if max(vals) > 0 else 0.05
    ax.set_ylim(0, ymax)
    for i, v in enumerate(vals):
        ax.text(i, v + ymax * 0.015, fmt.format(v),
                ha="center", fontsize=8.5, fontweight="bold", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GREY, alpha=0.3, lw=0.6)
    ax.set_title(title, fontsize=11, color=INK, fontweight="bold", loc="left")


def lesson1():
    fig, ax = plt.subplots(figsize=(8.5, 3.4), dpi=170)
    labels = ["Recall@10", "NDCG@10"]
    mf_explicit = [0.0025, 0.0021]
    mf_bpr = [0.0130, 0.0140]
    _paired(ax, mf_explicit, mf_bpr, labels,
            left_label="MF-Explicit (pointwise, MSE)",
            right_label="MF-BPR (pairwise)",
            left_color=GREY, right_color=TEAL,
            title="Key finding: BPR wins 5.2× on Recall@10, 6.7× on NDCG",
            ymax=0.018)
    # lift badges
    for i, (l, r) in enumerate(zip(mf_explicit, mf_bpr)):
        ax.text(i, max(l, r) * 1.35, f"×{r / l:.1f}",
                ha="center", fontsize=11, fontweight="bold", color=GREEN)
    fig.tight_layout()
    fig.savefig(OUT / "fig_findings_lesson1.png", bbox_inches="tight", dpi=170)
    plt.close(fig)


def lesson2():
    fig, ax = plt.subplots(figsize=(8.5, 3.4), dpi=170)
    models = ["LightGCN", "MF-BPR", "EMCDR", "NCF", "PTUPCDR", "CMF"]
    vals = [0.0290, 0.0170, 0.0160, 0.0120, 0.0085, 0.0050]
    colors = [BLUE, BLUE, TEAL, BLUE, TEAL, TEAL]
    _single_bars(ax, vals, models, colors,
                 "Key finding: LightGCN leads at 5.2% overlap; CDR trails single-domain",
                 ymax=0.035)
    # LightGCN reference line
    ax.axhline(0.0290, color=BLUE, ls="--", lw=0.9, alpha=0.55)
    ax.text(5.3, 0.030, "LightGCN", color=BLUE, fontsize=8.5, ha="right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_findings_lesson2.png", bbox_inches="tight", dpi=170)
    plt.close(fig)


def lesson3():
    fig, ax = plt.subplots(figsize=(8.5, 3.5), dpi=170)
    models = ["LightGCN", "PTUPCDR", "EMCDR", "NCF", "CMF"]
    l2 = [0.0290, 0.0085, 0.0160, 0.0120, 0.0050]
    l3 = [0.0555, 0.0320, 0.0245, 0.0230, 0.0140]
    annotate = ["+91%", "+276%", "+53%", "+92%", "+180%"]
    _paired(ax, l2, l3, models,
            left_label="L2 (5.2% overlap)",
            right_label="L3 (100% overlap)",
            left_color=GREY, right_color=TEAL, annotate=annotate,
            title="Key finding: forcing 100% overlap lifts every model (CDR more)",
            ymax=0.072)
    fig.tight_layout()
    fig.savefig(OUT / "fig_findings_lesson3.png", bbox_inches="tight", dpi=170)
    plt.close(fig)


def lesson4():
    fig, ax = plt.subplots(figsize=(8.5, 3.5), dpi=170)
    models = ["PTUPCDR", "EMCDR", "CMF", "NCF", "MF-BPR"]
    l3 = [42, 56, 75, 59, 91]
    l4 = [17, 34, 54, 60, 80]
    annotate = [f"−{a-b} pp" if a > b else f"+{b-a} pp"
                for a, b in zip(l3, l4)]
    _paired(ax, l3, l4, models,
            left_label="L3 (movies ≥ 5)",
            right_label="L4 (movies ≥ 10)",
            left_color=GREY, right_color=TEAL, annotate=annotate,
            fmt="{:.0f}%", lower_is_better=True,
            title="Key finding: source richness shrinks every CDR's gap to LightGCN",
            ymax=130)
    ax.set_ylabel("% behind LightGCN", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_findings_lesson4.png", bbox_inches="tight", dpi=170)
    plt.close(fig)


def lesson5():
    fig, ax = plt.subplots(figsize=(8.5, 3.5), dpi=170)
    models = ["LightGCN", "PTUPCDR", "NCF", "EMCDR", "CMF"]
    l4 = [0.0350, 0.0290, 0.0140, 0.0230, 0.0160]
    l5 = [0.0315, 0.0255, 0.0215, 0.0210, 0.0155]
    annotate = ["−10%", "−12%", "+54%", "−9%", "−3%"]
    _paired(ax, l4, l5, models,
            left_label="L4 (all movies)",
            right_label="L5 (movies ≥ 10 ratings)",
            left_color=GREY, right_color=TEAL, annotate=annotate,
            title="Key finding: trimming 74% of movie items holds CDR gains; NCF benefits",
            ymax=0.050)
    fig.tight_layout()
    fig.savefig(OUT / "fig_findings_lesson5.png", bbox_inches="tight", dpi=170)
    plt.close(fig)


def lesson6():
    fig, ax = plt.subplots(figsize=(8.5, 3.4), dpi=170)
    models = ["Popularity", "PTUPCDR", "EMCDR", "LightGCN", "NCF", "MF-BPR"]
    vals = [0.0365, 0.0305, 0.0300, 0.0080, 0.0020, 0.0000]
    colors = [AMBER, TEAL, TEAL, BLUE, BLUE, BLUE]
    _single_bars(ax, vals, models, colors,
                 "Key finding: Popularity tops cold-start; CDR 4× over LightGCN; single-domain collapses",
                 ymax=0.048)
    ax.axhline(0.0080, color=BLUE, ls="--", lw=0.9, alpha=0.5)
    ax.text(5.4, 0.010, "LightGCN (cold)", color=BLUE, fontsize=8, ha="right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_findings_lesson6.png", bbox_inches="tight", dpi=170)
    plt.close(fig)


def lesson7():
    fig, ax = plt.subplots(figsize=(8.5, 3.5), dpi=170)
    models = ["LightGCN", "SBERT-CDR", "SBERT", "PTUPCDR"]
    overall = [0.0335, 0.0330, 0.0310, 0.0220]
    niche = [0.0122, 0.1120, 0.1341, 0.0210]
    annotate = ["parity", "wins", "×11", "≈"]
    _paired(ax, overall, niche, models,
            left_label="Overall",
            right_label="Niche subgroup (n=82)",
            left_color=BLUE, right_color=PINK, annotate=annotate,
            title="Key finding: SBERT matches overall, wins 11× on niche long-tail items",
            ymax=0.170)
    fig.tight_layout()
    fig.savefig(OUT / "fig_findings_lesson7.png", bbox_inches="tight", dpi=170)
    plt.close(fig)


def lesson8():
    fig, ax = plt.subplots(figsize=(8.5, 3.5), dpi=170)
    models = ["LightGCN\n(cold)", "MF-BPR\n(cold)", "EMCDR\n(LLO)", "CMF\n(LLO)"]
    before = [0.010, 0.001, 0.0250, 0.0400]
    after = [0.033, 0.020, 0.0330, 0.0380]
    annotate = ["+230%", "+1 900%", "+31%", "≈"]
    _paired(ax, before, after, models,
            left_label="Base score",
            right_label="+ cooc rerank (λ = 0.05)",
            left_color=GREY, right_color=GREEN, annotate=annotate,
            title="Key finding: cooc rerank lifts every model — cold-start single-domain recovers",
            ymax=0.060)
    fig.tight_layout()
    fig.savefig(OUT / "fig_findings_lesson8.png", bbox_inches="tight", dpi=170)
    plt.close(fig)


def main():
    for fn, n in [(lesson1, 1), (lesson2, 2), (lesson3, 3), (lesson4, 4),
                  (lesson5, 5), (lesson6, 6), (lesson7, 7), (lesson8, 8)]:
        fn()
        print(f"  ✓ fig_findings_lesson{n}.png")


if __name__ == "__main__":
    main()
