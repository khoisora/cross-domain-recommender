#!/usr/bin/env python3
"""Plot benchmark results from JSON files.

Usage:
    python plot_results.py --domain-pair movie_game --lesson 1

Reads artifacts/<domain-pair>/results/*_lesson<N>.json and creates
comparison bar charts with dataset_info annotation.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_lesson_results(results_dir: Path, lesson: int) -> list[dict]:
    """Load all result JSONs for a given lesson."""
    pattern = f"*_lesson{lesson}.json"
    results = []
    for path in sorted(results_dir.glob(pattern)):
        try:
            data = json.loads(path.read_text())
            results.append(data)
        except Exception as e:
            print(f"Warning: failed to load {path}: {e}")
    return results


def create_bar_chart(results: list[dict], output_path: Path, lesson: int):
    """Create side-by-side bar chart for Recall@10 and NDCG@10 (full-rank + sampled).

    Layout: two panels side by side.
      Left panel: full-rank Recall@10 and NDCG@10 (all items scored).
      Right panel: sampled HR@10 and NDCG@10 (1 pos + 99 neg protocol).
    Bottom annotation box shows dataset context from dataset_info.
    """
    if not results:
        print("No results to plot.")
        return

    models = [r.get("model", "unknown") for r in results]
    n = len(models)
    x = np.arange(n)
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 3 * n), 5))

    # --- Full-rank metrics ---
    ax1 = axes[0]
    recall = [r.get("recall@10", 0) for r in results]
    ndcg = [r.get("ndcg@10", 0) for r in results]
    ax1.bar(x - width / 2, recall, width, label="Recall@10", color="#4ECDC4", alpha=0.85)
    ax1.bar(x + width / 2, ndcg, width, label="NDCG@10", color="#FF6B6B", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=30, ha="right")
    ax1.set_title("Full-Rank Evaluation @10")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    # Value labels
    for i, (r, n_) in enumerate(zip(recall, ndcg)):
        ax1.text(i - width / 2, r + 0.001, f"{r:.4f}", ha="center", va="bottom", fontsize=7)
        ax1.text(i + width / 2, n_ + 0.001, f"{n_:.4f}", ha="center", va="bottom", fontsize=7)

    # --- Sampled metrics ---
    ax2 = axes[1]
    hr10 = [r.get("sampled_hr@10", 0) for r in results]
    sndcg = [r.get("sampled_ndcg@10", 0) for r in results]
    ax2.bar(x - width / 2, hr10, width, label="Sampled HR@10", color="#4ECDC4", alpha=0.85)
    ax2.bar(x + width / 2, sndcg, width, label="Sampled NDCG@10", color="#FF6B6B", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=30, ha="right")
    ax2.set_title("Sampled (1+99) Evaluation @10")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    for i, (h, s) in enumerate(zip(hr10, sndcg)):
        ax2.text(i - width / 2, h + 0.002, f"{h:.4f}", ha="center", va="bottom", fontsize=7)
        ax2.text(i + width / 2, s + 0.002, f"{s:.4f}", ha="center", va="bottom", fontsize=7)

    # --- Dataset info annotation ---
    ds = results[0].get("dataset_info", {})
    def _fmt(v): return f"{v:,}" if isinstance(v, int) else str(v)
    games_val = ds.get('n_game_interactions', ds.get('n_game_interactions_warm', '?'))
    info_text = (
        f"Domain: {ds.get('domain_pair', '?')}\n"
        f"Cohort: {ds.get('cohort_filter', '?')}\n"
        f"Users: {_fmt(ds.get('n_users', '?'))}  |  "
        f"Movies: {_fmt(ds.get('n_movie_interactions', '?'))}  |  "
        f"Games: {_fmt(games_val)}\n"
        f"Split: {ds.get('split', '?')}"
    )
    fig.text(
        0.5, -0.02, info_text,
        ha="center", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
        family="monospace",
    )

    fig.suptitle(f"Lesson {lesson} — Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def create_subgroup_chart(results: list[dict], output_path: Path, lesson: int):
    """Create subgroup bar chart if subgroup data is present."""
    # Collect subgroup names across all results
    all_subgroups = set()
    for r in results:
        sg = r.get("subgroups", {})
        all_subgroups.update(k for k, v in sg.items() if v)

    if not all_subgroups:
        return

    # Filter out subgroups with < 10 users
    subgroups = sorted(all_subgroups)
    models = [r.get("model", "unknown") for r in results]

    fig, ax = plt.subplots(figsize=(max(12, 2 * len(subgroups)), 6))
    x = np.arange(len(subgroups))
    width = 0.8 / len(models)
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for i, r in enumerate(results):
        sg_data = r.get("subgroups", {})
        vals = [sg_data.get(sg, {}).get("recall@10", 0) for sg in subgroups]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=models[i], color=colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(subgroups, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Recall@10")
    ax.set_title(f"Lesson {lesson} — Subgroup Recall@10")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved subgroup plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results.")
    parser.add_argument("--domain-pair", default="movie_game")
    parser.add_argument("--lesson", type=int, required=True)
    args = parser.parse_args()

    # Resolve paths
    from ml.scripts.benchmarks.benchmark_common import _DOMAIN_PAIR_PATHS
    if args.domain_pair not in _DOMAIN_PAIR_PATHS:
        print(f"Unknown domain pair: {args.domain_pair}")
        sys.exit(1)

    _, artifacts_dir = _DOMAIN_PAIR_PATHS[args.domain_pair]
    results_dir = artifacts_dir / "results"
    plots_dir = artifacts_dir / "plots"

    results = load_lesson_results(results_dir, args.lesson)
    if not results:
        print(f"No results found for lesson {args.lesson} in {results_dir}")
        sys.exit(1)

    print(f"Found {len(results)} model results for lesson {args.lesson}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = plots_dir / f"lesson_{args.lesson}_{timestamp}.png"
    create_bar_chart(results, chart_path, args.lesson)

    # Subgroup chart if applicable
    subgroup_path = plots_dir / f"lesson_{args.lesson}_subgroups_{timestamp}.png"
    create_subgroup_chart(results, subgroup_path, args.lesson)


if __name__ == "__main__":
    main()
