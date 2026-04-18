"""Cross-domain evaluation: full-rank and sampled-negative protocols.

Two complementary evaluation protocols:
  1. Full-rank: score ALL items, mask non-target and train-seen items, rank top-K.
     Measures true ranking ability but is sensitive to item catalog size.
  2. Sampled (1+99): score 1 positive + 99 random negatives, check if positive is in top-K.
     More comparable across different catalog sizes, standard in RecSys papers.

Both protocols report per-subgroup breakdowns for fine-grained analysis.
"""

from __future__ import annotations

import logging

import numpy as np

from ml.evaluation.metrics import compute_all_metrics, aggregate_metrics

logger = logging.getLogger(__name__)

# All metrics are @10 only
K = 10


def evaluate_full_rank(
    model_name: str,
    predict_fn,
    data,
    eval_user_override: list[int] | None = None,
) -> dict:
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

    eval_users = eval_user_override or data.eval_user_indices
    for uid in eval_users:
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
        "%s [full-rank] (%d users): Recall@10=%.4f  NDCG@10=%.4f",
        model_name, len(per_user_metrics),
        overall.get("recall@10", 0), overall.get("ndcg@10", 0),
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


def evaluate_sampled(
    model_name: str,
    predict_fn,
    data,
    n_negatives: int = 99,
    seed: int = 42,
) -> dict:
    """Sampled-negative evaluation: 1 positive + n_negatives random target items.

    For each eval user:
      1. Take one ground-truth positive test item
      2. Sample n_negatives random target items (excluding train-seen and test item)
      3. Score all candidates, compute rank of the positive item
      4. HR@K = 1 if positive ranked in top-K, else 0
      5. NDCG@K = 1/log2(rank+1) if ranked in top-K, else 0

    This protocol is widely used in RecSys papers (e.g., NCF, LightGCN) and is
    more comparable across datasets with different catalog sizes.
    """
    rng = np.random.RandomState(seed)
    target_items_arr = np.array(sorted(data.target_item_indices), dtype=np.int64)

    per_user_metrics = []
    per_user_uid = []

    for uid in data.eval_user_indices:
        relevant = data.target_test_relevant.get(uid, set())
        if not relevant:
            continue
        # Standard sampled protocol: pick ONE positive test item as the target
        test_item = next(iter(relevant))

        # Negative pool: all target items except train-seen and the test item.
        # This ensures negatives are plausibly unobserved, not just held-out.
        exclude = data.target_train_seen.get(uid, set()) | {test_item}
        pool = target_items_arr[~np.isin(target_items_arr, list(exclude))]
        if len(pool) < n_negatives:
            continue

        neg_items = rng.choice(pool, size=n_negatives, replace=False)
        # candidates[0] = the positive; candidates[1:] = negatives
        candidates = np.concatenate([[test_item], neg_items])

        try:
            scores = predict_fn(uid)
        except Exception:
            continue

        # Rank the positive among all 100 candidates (1 pos + 99 neg).
        # rank=1 means the model scored the positive highest.
        cand_scores = scores[candidates]
        rank = int((cand_scores > cand_scores[0]).sum()) + 1

        m = {}
        for k in [10]:
            m[f"hr@{k}"] = float(rank <= k)
            m[f"ndcg@{k}"] = (1.0 / np.log2(rank + 2)) if rank <= k else 0.0
        per_user_metrics.append(m)
        per_user_uid.append(uid)

    overall = {}
    if per_user_metrics:
        for key in per_user_metrics[0]:
            overall[f"sampled_{key}"] = float(np.mean([m[key] for m in per_user_metrics]))

    logger.info(
        "%s [sampled@%d] (%d users): HR@10=%.4f  NDCG@10=%.4f",
        model_name, n_negatives, len(per_user_metrics),
        overall.get("sampled_hr@10", 0), overall.get("sampled_ndcg@10", 0),
    )

    subgroup_results = _aggregate_subgroups(
        data.user_subgroups, per_user_uid, per_user_metrics, prefix="sampled_"
    )

    return {
        **overall,
        "sampled_subgroups": subgroup_results,
        "n_sampled_users": len(per_user_metrics),
    }


def _aggregate_subgroups(
    user_subgroups: dict[str, list[int]],
    per_user_uid: list[int],
    per_user_metrics: list[dict],
    prefix: str = "",
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
            agg = {}
            for key in sg_m[0]:
                agg[f"{prefix}{key}"] = float(np.mean([m[key] for m in sg_m]))
            results[sg_name] = agg
    return results
