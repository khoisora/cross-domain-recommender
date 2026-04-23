"""Regenerate lesson bar charts with readable proportions.

Outputs PNGs to report_figures_v3/ for:
  fig21_lesson2.png       (paragraph 190)
  fig23_lesson3.png       (paragraph 195)
  fig24_lesson4.png       (paragraph 198)
  fig25_lesson5.png       (paragraph 201)
  fig27_lesson6.png       (paragraph 206)
  fig29_lesson7_subgroup.png (paragraph 211)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "report_figures_v3"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE, TEAL, GREEN, ORANGE, PURPLE, GREY = "#4472C4", "#2EC4B6", "#70AD47", "#ED7D31", "#7B68EE", "#A5A5A5"


def load(path):
    try:
        with open(ROOT / path) as f: return json.load(f)
    except FileNotFoundError:
        return None


def grouped_bars(models, r_vals, n_vals, title, outfile, colors, figsize=(9, 5.2)):
    fig, ax = plt.subplots(figsize=figsize, dpi=180)
    x = np.arange(len(models))
    w = 0.38
    b1 = ax.bar(x - w/2, r_vals, w, label="Recall@10", color=colors)
    b2 = ax.bar(x + w/2, n_vals, w, label="NDCG@10",
                color=[c + "99" for c in colors],
                edgecolor=colors, linewidth=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.3f}", (b.get_x() + b.get_width()/2, h),
                        ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(max(r_vals), max(n_vals)) * 1.18)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight", dpi=180)
    plt.close(fig)


def lesson_chart(lesson, model_keys, display_names, title, outfile, colors):
    r_vals, n_vals, names, cs = [], [], [], []
    for key, name, col in zip(model_keys, display_names, colors):
        d = load(f"artifacts/results/{key}_lesson{lesson}.json")
        if d is None or d.get("recall@10") is None:
            continue
        r_vals.append(d["recall@10"])
        n_vals.append(d["ndcg@10"])
        names.append(name)
        cs.append(col)
    grouped_bars(names, r_vals, n_vals, title, outfile, cs)


# Lesson 2 — all models on mixed population (5.8% overlap)
lesson_chart(
    2,
    ["popularity", "mf_bpr", "ncf", "lightgcn", "cmf", "emcdr", "ptupcdr"],
    ["Popularity", "MF-BPR", "NCF", "LightGCN", "CMF", "EMCDR", "PTUPCDR"],
    "Lesson 2 — mixed population (5.8% overlap): single-domain dominates CDR",
    OUT / "fig21_lesson2.png",
    [GREY, BLUE, BLUE, BLUE, ORANGE, ORANGE, ORANGE],
)

# Lesson 3 — 100% overlap
lesson_chart(
    3,
    ["popularity", "mf_bpr", "ncf", "lightgcn", "cmf", "emcdr", "ptupcdr"],
    ["Popularity", "MF-BPR", "NCF", "LightGCN", "CMF", "EMCDR", "PTUPCDR"],
    "Lesson 3 — 100% overlap cohort: CDR still trails LightGCN",
    OUT / "fig23_lesson3.png",
    [GREY, BLUE, BLUE, BLUE, ORANGE, ORANGE, ORANGE],
)

# Lesson 4 — source-rich / target-sparse
lesson_chart(
    4,
    ["popularity", "mf_bpr", "ncf", "lightgcn", "cmf", "emcdr", "ptupcdr"],
    ["Popularity", "MF-BPR", "NCF", "LightGCN", "CMF", "EMCDR", "PTUPCDR"],
    "Lesson 4 — source-rich / target-sparse: CDR narrows the gap",
    OUT / "fig24_lesson4.png",
    [GREY, BLUE, BLUE, BLUE, ORANGE, ORANGE, ORANGE],
)

# Lesson 5 — catalog sharpening
lesson_chart(
    5,
    ["popularity", "mf_bpr", "ncf", "lightgcn", "cmf", "emcdr", "ptupcdr"],
    ["Popularity", "MF-BPR", "NCF", "LightGCN", "CMF", "EMCDR", "PTUPCDR"],
    "Lesson 5 — catalog sharpening: LightGCN still leads, CDR unchanged",
    OUT / "fig25_lesson5.png",
    [GREY, BLUE, BLUE, BLUE, ORANGE, ORANGE, ORANGE],
)

# Lesson 6 — cold-start
lesson_chart(
    6,
    ["popularity", "mf_bpr", "ncf", "lightgcn", "cmf", "emcdr", "ptupcdr", "sbert_cdr"],
    ["Popularity", "MF-BPR", "NCF", "LightGCN", "CMF", "EMCDR", "PTUPCDR", "SBERT-CDR"],
    "Lesson 6 — cold-start (0 games): CDR mapping + popularity win",
    OUT / "fig27_lesson6.png",
    [GREY, BLUE, BLUE, BLUE, ORANGE, TEAL, TEAL, PURPLE],
)

# Lesson 7 — subgroup analysis: niche vs popular (from SBERT-CDR JSON)
def lesson7_subgroups():
    models = [
        ("lightgcn_lesson7.json",  "LightGCN",  BLUE),
        ("ptupcdr_lesson7.json",   "PTUPCDR",   ORANGE),
        ("sbert_lesson7.json",     "SBERT",     PURPLE),
        ("sbert-cdr_lesson7.json", "SBERT-CDR", TEAL),
    ]
    subgroups = [
        ("one_shot_target_user",            "one-shot\ntarget user"),
        ("one_shot_unpopular_target_user",  "one-shot\nniche target"),
        ("high_source_low_target",          "high-source\nlow-target"),
        ("high_source_unpopular_low_target","high-source\nniche target"),
    ]

    data = {}
    for fname, label, col in models:
        d = load(f"artifacts/results/{fname}")
        if d is None:
            continue
        data[label] = {k: d["subgroups"].get(k, {}).get("recall@10", 0.0) for k, _ in subgroups}

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    x = np.arange(len(subgroups))
    w = 0.2
    for i, (_, label, col) in enumerate(models):
        if label not in data: continue
        vals = [data[label][k] for k, _ in subgroups]
        offset = (i - len(models)/2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=label, color=col)
        for b in bars:
            h = b.get_height()
            if h > 0.005:
                ax.annotate(f"{h:.3f}", (b.get_x()+b.get_width()/2, h),
                            ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in subgroups])
    ax.set_ylabel("Recall@10")
    ax.set_title("Lesson 7 — Recall@10 by user subgroup: SBERT-CDR dominates on niche-target rows",
                 fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig29_lesson7_subgroup.png", bbox_inches="tight", dpi=180)
    plt.close(fig)

lesson7_subgroups()

print("Saved charts to", OUT)
