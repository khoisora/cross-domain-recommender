"""Modernized result charts for the project report.

Generates 10 charts:
  - lesson_1 .. lesson_7 (per-lesson model comparison)
  - lesson_7_subgroups
  - overlap_impact
  - results_summary

Output: ./new_figures/results/<name>.png at 200 DPI.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import FONT, BG, TITLE, MUTED, BORDER, PALETTE, ORDER, set_mpl_style

set_mpl_style()

PROJECT = Path("/sessions/happy-optimistic-gates/mnt/NewCrossDomainRecommenders")
RESULTS_DIR = PROJECT / "artifacts" / "results"
OUT = Path(__file__).resolve().parent.parent / "new_figures" / "results"
OUT.mkdir(parents=True, exist_ok=True)


# ----- helpers -------------------------------------------------------------

DISPLAY_NAME = {
    "mf_bpr": "MF-BPR",
    "mf_explicit": "MF (explicit)",
    "ncf": "NCF",
    "lightgcn": "LightGCN",
    "bitgcf": "BiTGCF",
    "cmf": "CMF",
    "emcdr": "EMCDR",
    "ptupcdr": "PTUPCDR",
    "sbert": "SBERT",
    "sbert-cdr": "SBERT-CDR",
    "sbert_cdr": "SBERT-CDR",
    "popularity": "Popularity",
}

PRIMARY = PALETTE["indigo"]["stroke"]
SECONDARY = PALETTE["amber"]["stroke"]
SUCCESS = PALETTE["emerald"]["stroke"]
DANGER = PALETTE["rose"]["stroke"]


def display_name(model: str) -> str:
    base = model.replace("_cooc", "")
    return DISPLAY_NAME.get(base, base.upper()) + (" + Cooc" if model.endswith("_cooc") else "")


def load_lesson(lesson: int) -> list[dict]:
    out = []
    for p in sorted(RESULTS_DIR.glob(f"*_lesson{lesson}.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            pass
    return out


def annotate_dataset(fig, info: dict) -> None:
    """Footer with cohort + counts in a clean style."""
    if not info:
        return
    def fmt(v): return f"{v:,}" if isinstance(v, int) else str(v)
    games = info.get("n_game_interactions", info.get("n_game_interactions_warm", "?"))
    txt = (
        f"Cohort: {info.get('cohort_filter', '?')}    "
        f"Users: {fmt(info.get('n_users', '?'))}    "
        f"Movies: {fmt(info.get('n_movie_interactions', '?'))}    "
        f"Games: {fmt(games)}    "
        f"Split: {info.get('split', '?')}"
    )
    fig.text(
        0.5, 0.005, txt,
        ha="center", va="bottom", fontsize=8, color=MUTED, fontname=FONT,
    )


# ----- per-lesson chart ----------------------------------------------------

def lesson_chart(lesson: int, title_override: str | None = None) -> None:
    results = load_lesson(lesson)
    if not results:
        print(f"  no data for lesson {lesson}")
        return

    # Sort by recall@10 desc to give a nice ranking
    results.sort(key=lambda r: r.get("recall@10", 0), reverse=True)
    names = [display_name(r.get("model", "?")) for r in results]
    recall = [r.get("recall@10", 0) for r in results]
    ndcg = [r.get("ndcg@10", 0) for r in results]

    n = len(names)
    fig, ax = plt.subplots(figsize=(max(8.5, 1.0 * n + 1.5), 5.2), dpi=200)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.18)

    x = np.arange(n)
    width = 0.38

    bars_r = ax.bar(x - width / 2, recall, width, label="Recall@10",
                    color=PRIMARY, edgecolor="white", linewidth=0.6)
    bars_n = ax.bar(x + width / 2, ndcg, width, label="NDCG@10",
                    color=SECONDARY, edgecolor="white", linewidth=0.6)

    # value labels above bars
    for v, b in list(zip(recall, bars_r)) + list(zip(ndcg, bars_n)):
        ax.text(b.get_x() + b.get_width() / 2, v + max(recall + ndcg) * 0.02,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=TITLE)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10, color=TITLE)
    ax.set_ylabel("Score", color=TITLE)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(recall + ndcg) * 1.18 + 0.005)

    title = title_override or f"Lesson {lesson} — Model Comparison (full-rank @10)"
    fig.suptitle(title, fontsize=14, fontweight="bold", color=TITLE, y=0.96)

    # legend top-right inside axes
    ax.legend(loc="upper right", frameon=False)

    annotate_dataset(fig, results[0].get("dataset_info", {}))

    out = OUT / f"lesson_{lesson}.png"
    fig.savefig(out, dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  rendered results/lesson_{lesson}.png")


# ----- subgroups chart -----------------------------------------------------

def lesson_subgroups(lesson: int) -> None:
    results = load_lesson(lesson)
    if not results:
        return

    # Pick a focused subgroup set
    interesting = [
        ("super_cold_users",                 "Super-cold"),
        ("one_shot_target_user",             "One-shot target"),
        ("one_shot_unpopular_target_user",   "One-shot · unpop"),
        ("high_source_low_target",           "Hi-src · lo-tgt"),
        ("high_source_unpopular_low_target", "Hi-src · unpop"),
    ]
    keys = [k for k, _ in interesting]
    pretty = [n for _, n in interesting]

    # only keep results with subgroup data
    rs = [r for r in results if r.get("subgroups")]
    if not rs:
        return
    rs.sort(key=lambda r: r.get("recall@10", 0), reverse=True)
    rs = rs[:6]

    fig, ax = plt.subplots(figsize=(max(11, 2 * len(keys)), 5.6), dpi=200)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.18)

    x = np.arange(len(keys))
    width = 0.8 / len(rs)
    palette_keys = ORDER * 4

    for i, r in enumerate(rs):
        sg = r.get("subgroups", {})
        vals = [sg.get(k, {}).get("recall@10", 0) for k in keys]
        col = PALETTE[palette_keys[i]]["stroke"]
        ax.bar(
            x + i * width - 0.4 + width / 2, vals, width,
            label=display_name(r.get("model", "?")),
            color=col, edgecolor="white", linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pretty, fontsize=10, color=TITLE)
    ax.set_ylabel("Recall@10", color=TITLE)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    fig.suptitle(f"Lesson {lesson} — Subgroup Recall@10", fontsize=14, fontweight="bold", color=TITLE, y=0.96)
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=9)
    annotate_dataset(fig, rs[0].get("dataset_info", {}))

    out = OUT / f"lesson_{lesson}_subgroups.png"
    fig.savefig(out, dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  rendered results/lesson_{lesson}_subgroups.png")


# ----- overlap_impact -----------------------------------------------------

def overlap_impact() -> None:
    models = ["LightGCN", "EMCDR", "PTUPCDR", "CMF"]
    l2 = [0.0290, 0.0160, 0.0085, 0.0050]
    l3 = [0.0595, 0.0225, 0.0315, 0.0395]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.85, bottom=0.14)

    x = np.arange(len(models))
    w = 0.38
    ax.bar(x - w / 2, l2, w, label="L2 · 5.8% overlap",
           color=PALETTE["rose"]["stroke"], edgecolor="white", linewidth=0.6)
    ax.bar(x + w / 2, l3, w, label="L3 · 100% overlap",
           color=PALETTE["emerald"]["stroke"], edgecolor="white", linewidth=0.6)

    for i, (a, b) in enumerate(zip(l2, l3)):
        ax.text(i - w / 2, a + 0.0015, f"{a:.3f}", ha="center", va="bottom", fontsize=8, color=TITLE)
        ax.text(i + w / 2, b + 0.0015, f"{b:.3f}", ha="center", va="bottom", fontsize=8, color=TITLE)
        if a > 0:
            pct = (b - a) / a * 100
            ax.annotate(f"+{pct:.0f}%", xy=(i + w / 2, b),
                        xytext=(i + w / 2, b + 0.008), ha="center",
                        fontsize=10, fontweight="bold", color=PALETTE["emerald"]["stroke"])

    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11, color=TITLE)
    ax.set_ylabel("Recall@10", color=TITLE)
    ax.set_ylim(0, max(l3) * 1.4)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    fig.suptitle("Impact of User Overlap on Cross-Domain Performance",
                 fontsize=14, fontweight="bold", color=TITLE, y=0.96)
    ax.legend(loc="upper right", frameon=False)

    out = OUT / "overlap_impact.png"
    fig.savefig(out, dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  rendered results/overlap_impact.png")


# ----- results_summary ----------------------------------------------------

def results_summary() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4), dpi=200)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.18, wspace=0.22)

    models_llo = ["LightGCN", "MF-BPR", "CMF", "PTUPCDR", "NCF", "EMCDR"]
    base_llo   = [0.0595, 0.0445, 0.0395, 0.0315, 0.0255, 0.0225]
    cooc_llo   = [0.0625, 0.0520, 0.0380, 0.0355, 0.0370, 0.0335]

    x1 = np.arange(len(models_llo)); w = 0.38
    ax1.bar(x1 - w / 2, base_llo, w, label="Base",
            color=PALETTE["indigo"]["stroke"], edgecolor="white", linewidth=0.6)
    ax1.bar(x1 + w / 2, cooc_llo, w, label="+ Co-occurrence",
            color=PALETTE["amber"]["stroke"], edgecolor="white", linewidth=0.6)
    ax1.set_xticks(x1); ax1.set_xticklabels(models_llo, rotation=20, ha="right", fontsize=10, color=TITLE)
    ax1.set_ylabel("Recall@10", color=TITLE)
    ax1.set_title("Standard regime · LLO · Lesson 3", fontsize=12, fontweight="bold", color=TITLE)
    ax1.legend(frameon=False)
    ax1.set_ylim(0, 0.085); ax1.yaxis.grid(True); ax1.set_axisbelow(True)
    for i, (a, b) in enumerate(zip(base_llo, cooc_llo)):
        ax1.text(i - w / 2, a + 0.001, f"{a:.3f}", ha="center", va="bottom", fontsize=8, color=TITLE)
        ax1.text(i + w / 2, b + 0.001, f"{b:.3f}", ha="center", va="bottom", fontsize=8, color=TITLE)

    models_cs = ["Popularity", "EMCDR", "PTUPCDR", "LightGCN", "CMF", "MF-BPR"]
    base_cs   = [0.0381, 0.0334, 0.0301, 0.0067, 0.0020, 0.0007]
    cooc_cs   = [0.0387, 0.0341, 0.0301, 0.0261, 0.0321, 0.0007]

    x2 = np.arange(len(models_cs))
    ax2.bar(x2 - w / 2, base_cs, w, label="Base",
            color=PALETTE["indigo"]["stroke"], edgecolor="white", linewidth=0.6)
    ax2.bar(x2 + w / 2, cooc_cs, w, label="+ Co-occurrence",
            color=PALETTE["amber"]["stroke"], edgecolor="white", linewidth=0.6)
    ax2.set_xticks(x2); ax2.set_xticklabels(models_cs, rotation=20, ha="right", fontsize=10, color=TITLE)
    ax2.set_ylabel("Recall@10", color=TITLE)
    ax2.set_title("Cold-start regime · Lesson 6 · zero game history", fontsize=12, fontweight="bold", color=TITLE)
    ax2.legend(frameon=False)
    ax2.set_ylim(0, 0.05); ax2.yaxis.grid(True); ax2.set_axisbelow(True)
    for i, (a, b) in enumerate(zip(base_cs, cooc_cs)):
        ax2.text(i - w / 2, a + 0.0008, f"{a:.3f}", ha="center", va="bottom", fontsize=8, color=TITLE)
        ax2.text(i + w / 2, b + 0.0008, f"{b:.3f}", ha="center", va="bottom", fontsize=8, color=TITLE)

    fig.suptitle("Key Results: Recall@10 Across Evaluation Regimes",
                 fontsize=15, fontweight="bold", color=TITLE, y=0.98)

    out = OUT / "results_summary.png"
    fig.savefig(out, dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  rendered results/results_summary.png")


# ----- main ----------------------------------------------------------------

def main():
    print(f"Rendering result charts to {OUT}/")
    for L in (1, 2, 3, 4, 5, 6, 7):
        lesson_chart(L)
    lesson_subgroups(7)
    overlap_impact()
    results_summary()
    print("Done.")


if __name__ == "__main__":
    main()
