"""Recommendation evaluation metrics: Recall@K, NDCG@K, HitRate@K.

All metrics use binary relevance (relevant or not, no graded relevance).
Default k_values=[10] — we only report @10 across the project.
These are per-user metrics; aggregate_metrics() averages across users.
"""

from __future__ import annotations

import numpy as np


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant items in the top-K recommendations."""
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Binary: did any relevant item appear in top-K?"""
    return 1.0 if set(recommended[:k]) & relevant else 0.0


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K (binary relevance).

    DCG rewards relevant items appearing earlier in the ranked list.
    IDCG is the best possible DCG (all relevant items at the top).
    NDCG = DCG / IDCG normalizes to [0, 1].
    """
    if not relevant:
        return 0.0
    # DCG = sum(1/log2(rank+1)) for each relevant item in the top-K.
    # Position index i is 0-based so we use log2(i+2) to get log2(rank) with rank starting at 1.
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )
    # IDCG = best possible DCG if all relevant items were ranked at the top positions
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def compute_all_metrics(
    recommended: list[int],
    relevant: set[int],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute recall, NDCG, and hit rate for a single user at each K."""
    if k_values is None:
        k_values = [10]
    results = {}
    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(recommended, relevant, k)
        results[f"ndcg@{k}"] = ndcg_at_k(recommended, relevant, k)
        results[f"hit_rate@{k}"] = hit_rate_at_k(recommended, relevant, k)
    return results


def aggregate_metrics(per_user_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Mean of per-user metric dicts."""
    if not per_user_metrics:
        return {}
    return {
        key: float(np.mean([m[key] for m in per_user_metrics if key in m]))
        for key in per_user_metrics[0]
    }
