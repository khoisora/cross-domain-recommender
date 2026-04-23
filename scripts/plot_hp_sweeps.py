"""Turn artifacts/sweeps/*.json into per-model plots.

Outputs PNGs to report_figures_v3/:
  fig_hp_cmf.png         (lr × alpha heatmap + best point)
  fig_hp_lightgcn.png    (K layers line chart)
  fig_hp_ptupcdr.png     (n_experts bar chart)
  fig_hp_sbert_cdr.png   (source_weight line chart)
  fig_hp_emcdr_cooc.png  (cooc λ line chart)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SWEEPS = ROOT / "artifacts" / "sweeps"
OUT = ROOT / "report_figures_v3"
OUT.mkdir(exist_ok=True)

BLUE, TEAL, GREEN, ORANGE, PINK, PURPLE, GREY = (
    "#4472C4", "#2EC4B6", "#70AD47", "#ED7D31", "#EC4899", "#7B68EE", "#A5A5A5"
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load(pattern: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(SWEEPS.glob(pattern))]


# ── CMF: lr × alpha heatmap ─────────────────────────────────────────────

def plot_cmf():
    pts = load("cmf__*.json")
    if not pts: return print("no CMF sweep points")
    lrs = sorted({p["hyperparams"]["lr"] for p in pts})
    als = sorted({p["hyperparams"]["alpha"] for p in pts})
    R = np.zeros((len(lrs), len(als)))
    for p in pts:
        i, j = lrs.index(p["hyperparams"]["lr"]), als.index(p["hyperparams"]["alpha"])
        R[i, j] = p["recall@10"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    im = ax.imshow(R, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(als))); ax.set_xticklabels([f"{a}" for a in als])
    ax.set_yticks(range(len(lrs))); ax.set_yticklabels([f"{l}" for l in lrs])
    ax.set_xlabel(r"$\alpha$  (source-domain weight)")
    ax.set_ylabel("learning rate")
    for i in range(len(lrs)):
        for j in range(len(als)):
            v = R[i, j]
            c = "white" if v < R.max() * 0.55 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=10, color=c, fontweight="bold")
    best_i, best_j = np.unravel_index(R.argmax(), R.shape)
    ax.add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                               fill=False, edgecolor="red", lw=2.5))
    fig.colorbar(im, ax=ax, label="Recall@10")
    ax.set_title("CMF — Recall@10 across learning rate × α grid\n"
                 f"best: lr={lrs[best_i]}, α={als[best_j]}  →  {R.max():.4f}",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_hp_cmf.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ── LightGCN: K layers ──────────────────────────────────────────────────

def plot_lightgcn():
    pts = load("lightgcn__*.json")
    if not pts: return print("no LightGCN sweep points")
    pts.sort(key=lambda p: p["hyperparams"]["K"])
    Ks = [p["hyperparams"]["K"] for p in pts]
    R  = [p["recall@10"] for p in pts]
    N  = [p["ndcg@10"]   for p in pts]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    ax.plot(Ks, R, "-o", color=BLUE, lw=2, markersize=8, label="Recall@10")
    ax.plot(Ks, N, "-s", color=TEAL, lw=2, markersize=8, label="NDCG@10")
    for k, r in zip(Ks, R):
        ax.annotate(f"{r:.3f}", (k, r), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    best_idx = int(np.argmax(R))
    ax.axvline(Ks[best_idx], color="red", lw=1.2, linestyle="--", alpha=0.7)
    ax.set_xlabel("K — number of propagation layers")
    ax.set_ylabel("metric value")
    ax.set_xticks(Ks)
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(f"LightGCN — propagation depth (best K = {Ks[best_idx]})",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_hp_lightgcn.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ── PTUPCDR: n_experts bar chart ────────────────────────────────────────

def plot_ptupcdr():
    pts = load("ptupcdr__*.json")
    if not pts: return print("no PTUPCDR sweep points")
    pts.sort(key=lambda p: p["hyperparams"]["n_experts"])
    ns = [p["hyperparams"]["n_experts"] for p in pts]
    R  = [p["recall@10"] for p in pts]
    N  = [p["ndcg@10"]   for p in pts]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    x = np.arange(len(ns)); w = 0.36
    b1 = ax.bar(x - w/2, R, w, color=ORANGE, label="Recall@10")
    b2 = ax.bar(x + w/2, N, w, color=ORANGE + "99", edgecolor=ORANGE,
                linewidth=1.4, label="NDCG@10")
    for bs, vals in ((b1, R), (b2, N)):
        for b, v in zip(bs, vals):
            ax.annotate(f"{v:.3f}", (b.get_x()+b.get_width()/2, v),
                        textcoords="offset points", xytext=(0,3),
                        ha="center", fontsize=9)
    best = int(np.argmax(R))
    ax.axvline(x[best], color="red", lw=1.2, ls="--", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("n_experts  (per-user expert MLPs in the meta-network)")
    ax.set_ylabel("metric value")
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(f"PTUPCDR — expert count (best = {ns[best]})",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_hp_ptupcdr.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ── SBERT-CDR: source_weight ────────────────────────────────────────────

def plot_sbert_cdr():
    pts = load("sbert_cdr__*.json")
    if not pts: return print("no SBERT-CDR sweep points")
    pts.sort(key=lambda p: p["hyperparams"]["source_weight"])
    ws = [p["hyperparams"]["source_weight"] for p in pts]
    R  = [p["recall@10"] for p in pts]
    N  = [p["ndcg@10"]   for p in pts]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    ax.plot(ws, R, "-o", color=PINK, lw=2, markersize=8, label="Recall@10")
    ax.plot(ws, N, "-s", color=PURPLE, lw=2, markersize=8, label="NDCG@10")
    for w, r in zip(ws, R):
        ax.annotate(f"{r:.3f}", (w, r), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    best = int(np.argmax(R))
    ax.axvline(ws[best], color="red", lw=1.2, ls="--", alpha=0.7)
    ax.set_xlabel(r"source_weight  (movie share in the user profile blend)")
    ax.set_ylabel("metric value")
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(f"SBERT-CDR — source-domain blend weight "
                 f"(best = {ws[best]:.1f})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_hp_sbert_cdr.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


# ── EMCDR + cooc: λ ─────────────────────────────────────────────────────

def plot_emcdr_cooc():
    pts = load("emcdr_cooc__*.json")
    if not pts: return print("no EMCDR-cooc sweep points")
    pts.sort(key=lambda p: p["hyperparams"]["lambda"])
    ls = [p["hyperparams"]["lambda"] for p in pts]
    R  = [p["recall@10"] for p in pts]
    N  = [p["ndcg@10"]   for p in pts]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    ax.plot(ls, R, "-o", color=GREEN, lw=2, markersize=8, label="Recall@10")
    ax.plot(ls, N, "-s", color=TEAL, lw=2, markersize=8, label="NDCG@10")
    for l, r in zip(ls, R):
        ax.annotate(f"{r:.3f}", (l, r), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    best = int(np.argmax(R))
    ax.axvline(ls[best], color="red", lw=1.2, ls="--", alpha=0.7)
    ax.set_xlabel(r"$\lambda$  (co-occurrence blend weight)")
    ax.set_ylabel("metric value")
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(f"EMCDR + co-occurrence rerank — λ sweep "
                 f"(best λ = {ls[best]})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_hp_emcdr_cooc.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot_cmf()
    plot_lightgcn()
    plot_ptupcdr()
    plot_sbert_cdr()
    plot_emcdr_cooc()
    print("Saved hp-sweep plots to", OUT)
