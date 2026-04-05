"""Recommendation evaluation metrics.

Implements standard information retrieval metrics for recommender
system evaluation:
- Recall@K
- HitRate@K  
- NDCG@K
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def recall_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """Compute Recall@K.

    Fraction of relevant items that appear in the top-K recommendations.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of ground-truth relevant item IDs.
        k: Cutoff position.

    Returns:
        Recall value in [0, 1].
    """
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def hit_rate_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """Compute HitRate@K (binary: did any relevant item appear in top-K?).

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of ground-truth relevant item IDs.
        k: Cutoff position.

    Returns:
        1.0 if any hit, 0.0 otherwise.
    """
    top_k = set(recommended[:k])
    return 1.0 if len(top_k & relevant) > 0 else 0.0


def ndcg_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at K.

    Uses binary relevance: relevant items have relevance 1, others 0.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of ground-truth relevant item IDs.
        k: Cutoff position.

    Returns:
        NDCG value in [0, 1].
    """
    if not relevant:
        return 0.0

    dcg = 0.0
    for i, item_id in enumerate(recommended[:k]):
        if item_id in relevant:
            dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed

    # Ideal DCG: all relevant items ranked at the top
    ideal_len = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_all_metrics(
    recommended: list[int],
    relevant: set[int],
    k_values: list[int] = None,
) -> dict[str, float]:
    """Compute all metrics for a single user at multiple K values.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of ground-truth relevant item IDs.
        k_values: List of K cutoffs to evaluate.

    Returns:
        Dict mapping metric names to values.
    """
    if k_values is None:
        k_values = [5, 10, 20, 50]

    results = {}
    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(recommended, relevant, k)
        results[f"ndcg@{k}"] = ndcg_at_k(recommended, relevant, k)
        # HitRate@K == Recall@K when each user has exactly 1 test item (leave-last-out),
        # so we omit it to avoid redundancy.

    return results


def aggregate_metrics(
    per_user_metrics: list[dict[str, float]],
) -> dict[str, float]:
    """Aggregate per-user metrics into mean values.

    Args:
        per_user_metrics: List of per-user metric dicts.

    Returns:
        Dict mapping metric names to mean values.
    """
    if not per_user_metrics:
        return {}

    all_keys = per_user_metrics[0].keys()
    aggregated = {}
    for key in all_keys:
        values = [m[key] for m in per_user_metrics if key in m]
        aggregated[key] = float(np.mean(values))

    return aggregated
