"""Cross-domain full-rank evaluation.

Score ALL items, mask non-target and train-seen items, rank top-K, compute
Recall@K, NDCG@K, and HitRate@K. Reports per-subgroup breakdowns for
fine-grained analysis.
"""

from __future__ import annotations

import logging

import numpy as np

from ml.evaluation.metrics import compute_all_metrics, aggregate_metrics

logger = logging.getLogger(__name__)

# All metrics are @10 only
K = 10


def evaluate_full_rank(model_name: str, predict_fn, data) -> dict:
    """Full-rank evaluation on held-out target-domain test set.

    For each eval user:
      1. Get model scores for ALL items
      2. Mask out non-target-domain items (e.g., movies when evaluating games)
      3. Mask out items the user already interacted with in training
      4. Take top-K items from remaining scores
      5. Compute Recall@K and NDCG@K against ground-truth relevant items

    Args:
        predict_fn: (user_idx) -> np.ndarray of scores for all items.
        data: CrossDomainSplit instance.
    """
    num_items = data.num_items
    # Boolean mask: True for target-domain items only.
    # Non-target items get -inf scores so they never appear in top-K.
    target_mask = np.zeros(num_items, dtype=bool)
    for idx in data.target_item_indices:
        target_mask[idx] = True

    per_user_metrics = []
    per_user_uid = []

    for uid in data.eval_user_indices:
        # Skip users with no ground-truth relevant items in the test set
        relevant = data.target_test_relevant.get(uid, set())
        if not relevant:
            continue
        try:
            scores = predict_fn(uid).copy()
        except Exception:
            continue

        # Two-stage masking ensures fair evaluation:
        # 1. Non-target items → -inf: only rank within the target domain (e.g., games)
        scores[~target_mask] = -np.inf
        # 2. Train-seen items → -inf: don't credit models for re-ranking known items
        for iid in data.target_train_seen.get(uid, set()):
            if 0 <= iid < num_items:
                scores[iid] = -np.inf

        top_indices = np.argsort(scores)[::-1][:K].tolist()
        m = compute_all_metrics(top_indices, relevant, k_values=[K])
        per_user_metrics.append(m)
        per_user_uid.append(uid)

    overall = aggregate_metrics(per_user_metrics)
    logger.info(
        "%s [full-rank] (%d users): Recall@10=%.4f  NDCG@10=%.4f  HitRate@10=%.4f",
        model_name, len(per_user_metrics),
        overall.get("recall@10", 0), overall.get("ndcg@10", 0), overall.get("hit_rate@10", 0),
    )

    # Subgroup aggregation
    subgroup_results = _aggregate_subgroups(
        data.user_subgroups, per_user_uid, per_user_metrics
    )

    return {
        **overall,
        "subgroups": subgroup_results,
        "n_eval_users": len(per_user_metrics),
    }


def _aggregate_subgroups(
    user_subgroups: dict[str, list[int]],
    per_user_uid: list[int],
    per_user_metrics: list[dict],
) -> dict[str, dict[str, float]]:
    """Aggregate metrics per subgroup. Silently skips empty subgroups.

    Maps each subgroup's user indices back to their per-user metrics,
    then averages. This enables comparing model performance across
    user segments (e.g., cold-start vs warm, movie-heavy vs game-heavy).
    """
    uid_to_pos = {uid: i for i, uid in enumerate(per_user_uid)}
    results = {}
    for sg_name, sg_uids in user_subgroups.items():
        sg_m = [per_user_metrics[uid_to_pos[uid]] for uid in sg_uids if uid in uid_to_pos]
        if sg_m:
            results[sg_name] = {
                key: float(np.mean([m[key] for m in sg_m]))
                for key in sg_m[0]
            }
    return results
