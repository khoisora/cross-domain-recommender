"""Regenerate Figure 30: base vs cooc comparison across LLO and Cold-start."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "report_figures_v3"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})

BLUE, TEAL = "#4472C4", "#2EC4B6"
MODELS = [
    ("lightgcn",    "LightGCN"),
    ("cmf",         "CMF"),
    ("emcdr",       "EMCDR"),
    ("ptupcdr",     "PTUPCDR"),
    ("sbert_cdr",   "SBERT-CDR"),
    ("popularity",  "Popularity"),
]

def load_pair(model, lesson):
    base_p = ROOT / f"artifacts/results/{model}_lesson{lesson}.json"
    cooc_p = ROOT / f"artifacts/results/{model}_cooc_lesson{lesson}.json"
    with open(base_p) as f: b = json.load(f)
    with open(cooc_p) as f: c = json.load(f)
    return b["recall@10"], c["recall@10"]

fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=180)
for ax, lesson, title in [
    (axes[0], 99, "Leave-Last-Out (LLO)"),
    (axes[1],  6, "Cold-start (0 games)"),
]:
    labels, base_vals, cooc_vals = [], [], []
    for key, name in MODELS:
        try:
            b, c = load_pair(key, lesson)
        except FileNotFoundError:
            continue
        labels.append(name); base_vals.append(b); cooc_vals.append(c)

    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w/2, base_vals, w, label="base",  color=BLUE)
    ax.bar(x + w/2, cooc_vals, w, label="+ cooc", color=TEAL)
    for i, (b, c) in enumerate(zip(base_vals, cooc_vals)):
        ax.annotate(f"{b:.3f}", (i - w/2, b), ha="center", va="bottom", fontsize=8)
        ax.annotate(f"{c:.3f}", (i + w/2, c), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Recall@10")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")

fig.suptitle("Figure 30 — Universal co-occurrence reranking improves Recall@10 across models",
             fontsize=12, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig30_cooc_summary.png", bbox_inches="tight", dpi=180)
plt.close(fig)
print("Saved", OUT / "fig30_cooc_summary.png")
