"""Evaluate saved model embeddings with updated subgroup definitions.

Loads pre-trained embeddings from artifacts/demo/ and runs full-rank evaluation
using the current benchmark_common subgroup logic (no retraining required).

Uses the saved cross_domain_user_to_idx.json / cross_domain_item_to_idx.json
mappings so embedding indices are aligned to the trained model's index space,
while evaluating the full user set (including newly included cold-start users
that were previously filtered by min_target_user_interactions).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_bench_dir = str(Path(__file__).resolve().parent / "benchmarks")
_root = str(Path(__file__).resolve().parent.parent.parent)
for p in (_bench_dir, _root):
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark_common import (
    load_cross_domain_split,
    evaluate_cross_domain,
    save_result,
    setup_logging,
    DEMO_DIR,
)

logger = logging.getLogger(__name__)

# ── Model registry: name → (user_emb_file, item_emb_file, result_key) ───────
MODELS = {
    "MF_BPR": ("mf_bpr_game_user.npy", "mf_bpr_game_item.npy", "MF_BPR_game"),
    "MF_Explicit": ("mf_explicit_game_user.npy", "mf_explicit_game_item.npy", "MF_Explicit_game"),
    "NCF": ("ncf_game_user.npy", "ncf_game_item.npy", "NCF_game"),
    "LightGCN": ("lightgcn_game_user.npy", "lightgcn_game_item.npy", "LightGCN_game"),
    "CMF": ("cmf_user.npy", "cmf_item.npy", "CMF"),
    "DeepAPF": ("deepapf_user.npy", "deepapf_item.npy", "DeepAPF"),
}


def load_saved_mappings() -> tuple[dict[str, int], dict[str, int], dict[int, str]]:
    """Load user/item index mappings saved during training."""
    u2i_path = DEMO_DIR / "cross_domain_user_to_idx.json"
    i2i_path = DEMO_DIR / "cross_domain_item_to_idx.json"
    i2item_path = DEMO_DIR / "cross_domain_idx_to_item.json"

    with open(u2i_path) as f:
        saved_u2i: dict[str, int] = json.load(f)
    with open(i2i_path) as f:
        saved_i2item_str: dict[str, int] = json.load(f)
    with open(i2item_path) as f:
        idx_to_item_str: dict[str, str] = json.load(f)

    saved_i2i: dict[str, int] = saved_i2item_str
    idx_to_item: dict[int, str] = {int(k): v for k, v in idx_to_item_str.items()}
    return saved_u2i, saved_i2i, idx_to_item


def make_predict_fn(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    saved_u2i: dict[str, int],
    new_u2i: dict[str, int],
    new_i2i: dict[str, int],
    saved_i2i: dict[str, int],
):
    """Build a predict_fn bridging new user/item indices to saved embeddings.

    For each (new_uid_idx, new_item_idx):
      1. Resolve new_uid_idx → user_id_str → saved_uid_idx → user_emb row
      2. Build aligned item embedding matrix: new_item_idx → saved_item_idx → item_emb row
    Items not in saved mapping get zero embeddings (invisible to ranking).
    Users not in saved mapping return -inf scores (skipped in evaluation).
    """
    new_idx_to_user = {v: k for k, v in new_u2i.items()}
    n_new_items = len(new_i2i)

    # Build aligned item embedding matrix in new index space
    emb_dim = item_emb.shape[1]
    aligned_item_emb = np.zeros((n_new_items, emb_dim), dtype=np.float32)
    new_item_ids = {iid: new_idx for iid, new_idx in new_i2i.items()}
    for item_id, new_idx in new_item_ids.items():
        saved_idx = saved_i2i.get(item_id)
        if saved_idx is not None and saved_idx < len(item_emb):
            aligned_item_emb[new_idx] = item_emb[saved_idx]

    def predict(new_uid_idx: int) -> np.ndarray:
        uid_str = new_idx_to_user.get(new_uid_idx)
        if uid_str is None:
            return np.full(n_new_items, -np.inf, dtype=np.float32)
        saved_idx = saved_u2i.get(uid_str)
        if saved_idx is None or saved_idx >= len(user_emb):
            return np.full(n_new_items, -np.inf, dtype=np.float32)
        return user_emb[saved_idx] @ aligned_item_emb.T

    return predict


def eval_model(
    model_name: str,
    user_emb_file: str,
    item_emb_file: str,
    result_key: str,
    data,
    saved_u2i: dict[str, int],
    saved_i2i: dict[str, int],
) -> dict:
    user_emb_path = DEMO_DIR / user_emb_file
    item_emb_path = DEMO_DIR / item_emb_file

    if not user_emb_path.exists() or not item_emb_path.exists():
        logger.warning("Skipping %s — embeddings not found: %s", model_name, user_emb_path)
        return {}

    logger.info("Loading %s embeddings...", model_name)
    user_emb = np.load(user_emb_path)
    item_emb = np.load(item_emb_path)
    logger.info("  user_emb: %s, item_emb: %s", user_emb.shape, item_emb.shape)

    predict_fn = make_predict_fn(
        user_emb, item_emb,
        saved_u2i, data.user_to_idx,
        data.item_to_idx, saved_i2i,
    )

    t0 = time.time()
    metrics = evaluate_cross_domain(model_name, predict_fn, data)
    elapsed = time.time() - t0
    logger.info("%s eval completed in %.1fs", model_name, elapsed)

    save_result(result_key, metrics, train_time_s=0.0,
                description=f"eval-only re-run with updated subgroups (full-rank)")
    return metrics


def main():
    setup_logging()
    logger.info("=== Eval-only re-run with updated subgroups (full-rank) ===")

    saved_u2i, saved_i2i, _ = load_saved_mappings()
    logger.info("Saved mappings: %d users, %d items", len(saved_u2i), len(saved_i2i))

    logger.info("Loading data (no min_target_user_interactions filter)...")
    data = load_cross_domain_split(
        max_eval_users=None,
        target_domain="game",
    )
    logger.info(
        "Eval users: %d  |  new subgroups: %s",
        len(data.eval_user_indices),
        {k: len(v) for k, v in data.user_subgroups.items()},
    )

    results = {}
    for model_name, (uf, itf, rkey) in MODELS.items():
        logger.info("\n── %s ──", model_name)
        m = eval_model(model_name, uf, itf, rkey, data, saved_u2i, saved_i2i)
        if m:
            results[model_name] = m

    print("\n" + "=" * 70)
    print("EVAL SUMMARY (full-rank, new subgroups)")
    print("=" * 70)
    for name, m in sorted(results.items(), key=lambda x: x[1].get("recall@10", 0), reverse=True):
        r10 = m.get("recall@10", 0)
        n10 = m.get("ndcg@10", 0)
        n_users = m.get("n_eval_users", 0)
        print(f"  {name:15s}  Recall@10={r10:.4f}  NDCG@10={n10:.4f}  ({n_users} users)")
        sgs = m.get("subgroups", {})
        for sg, sg_m in sgs.items():
            if sg_m:
                sg_r10 = sg_m.get("recall@10", 0)
                print(f"    {sg:35s}  Recall@10={sg_r10:.4f}")


if __name__ == "__main__":
    main()
