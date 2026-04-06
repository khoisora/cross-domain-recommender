"""Lesson 6: User-split cold-start benchmark.

80% warm users: all game interactions in training.
20% cold users: zero game interactions in training (movies only).

Evaluates all models on cold users only — measures cross-domain transfer
under true zero-game-history conditions.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np

from ml.scripts.benchmarks.benchmark_common import (
    add_common_args, evaluate_cross_domain,
    load_user_split_cold_start_split, save_result, setup_logging,
    POSITIVE_THRESHOLD,
)
from ml.models.matrix_factorization_bpr import MatrixFactorizationBPR
from ml.models.ncf import NCF
from ml.models.lightgcn import LightGCN
from ml.models.cmf import CMF
from ml.models.emcdr import EMCDRWrapper
from ml.models.ptupcdr import PTUPCDRWrapper
from ml.models.bitgcf import BiTGCFWrapper

logger = logging.getLogger(__name__)


def run_popularity(data) -> dict:
    """Rank games by warm-user interaction count."""
    from collections import Counter
    pop_counts: Counter = Counter()
    for _, row in data.game_train.iterrows():
        iid = data.item_to_idx.get(str(row["item_id"]))
        if iid is None:
            from ml.models.id_utils import normalize_id
            iid = data.item_to_idx.get(normalize_id(row["item_id"]))
        if iid is not None:
            pop_counts[iid] += 1
    pop_scores = np.array([pop_counts.get(i, 0) for i in range(data.num_items)], dtype=np.float64)

    def predict(uid):
        return pop_scores.copy()

    return predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Cold-start user-split benchmark")
    add_common_args(parser)
    parser.add_argument("--cold-ratio", type=float, default=0.2)
    args = parser.parse_args()

    setup_logging()

    data = load_user_split_cold_start_split(
        cold_start_ratio=args.cold_ratio,
        domain_pair=args.domain_pair,
    )

    # Force CPU — unified cross-domain ID space is too large for MPS
    data.device = "cpu"

    logger.info(
        "Dataset: %d eval cold users, %d warm users, %d game train interactions",
        len(data.eval_user_indices),
        data.dataset_info["n_warm_users"],
        data.dataset_info["n_game_interactions_warm"],
    )

    results = {}

    # --- Popularity baseline ---
    t0 = time.time()
    pop_fn = run_popularity(data)
    metrics = evaluate_cross_domain("Popularity", pop_fn, data)
    results["Popularity"] = metrics
    save_result(
        algo="Popularity", metrics=metrics, dataset_info=data.dataset_info,
        lesson=args.lesson, train_time=time.time() - t0,
        description="Global game popularity from warm-user interactions",
    )

    # --- MF-BPR ---
    t0 = time.time()
    model = MatrixFactorizationBPR(data.num_users, data.num_items, embedding_dim=64, device=data.device)
    model.fit(data.game_train, data.user_to_idx, data.item_to_idx,
              epochs=50, lr=0.001, reg_lambda=0.01, batch_size=4096,
              positive_threshold=POSITIVE_THRESHOLD)
    metrics = evaluate_cross_domain("MF_BPR", lambda uid: model.predict(uid), data)
    save_result(
        algo="MF_BPR", metrics=metrics, dataset_info=data.dataset_info,
        lesson=args.lesson, train_time=time.time() - t0,
        description="MF-BPR cold-start (game-only, warm users only)",
    )

    # --- LightGCN ---
    t0 = time.time()
    model = LightGCN(data.num_users, data.num_items, embedding_dim=64, device=data.device)
    model.fit(data.game_train, data.user_to_idx, data.item_to_idx,
              epochs=50, lr=0.001, reg_lambda=1e-4, batch_size=4096,
              positive_threshold=POSITIVE_THRESHOLD)
    metrics = evaluate_cross_domain("LightGCN", lambda uid: model.predict(uid), data)
    save_result(
        algo="LightGCN", metrics=metrics, dataset_info=data.dataset_info,
        lesson=args.lesson, train_time=time.time() - t0,
        description="LightGCN cold-start (game-only, warm users only)",
    )

    # --- NCF ---
    t0 = time.time()
    model = NCF(data.num_users, data.num_items, embedding_dim=64, device=data.device)
    model.fit(data.game_train, data.user_to_idx, data.item_to_idx,
              epochs=50, lr=0.001, reg_lambda=1e-4, batch_size=4096,
              positive_threshold=POSITIVE_THRESHOLD)
    metrics = evaluate_cross_domain("NCF", lambda uid: model.predict(uid), data)
    save_result(
        algo="NCF", metrics=metrics, dataset_info=data.dataset_info,
        lesson=args.lesson, train_time=time.time() - t0,
        description="NCF cold-start (game-only, warm users only)",
    )

    # --- CMF ---
    t0 = time.time()
    model = CMF(data.num_users, data.num_items, embedding_dim=64, device=data.device)
    model.fit(data.cross_train, data.user_to_idx, data.item_to_idx,
              epochs=50, lr=0.001, reg_lambda=0.01, batch_size=4096,
              positive_threshold=POSITIVE_THRESHOLD)
    metrics = evaluate_cross_domain("CMF", lambda uid: model.predict(uid), data)
    save_result(
        algo="CMF", metrics=metrics, dataset_info=data.dataset_info,
        lesson=args.lesson, train_time=time.time() - t0,
        description="CMF cold-start (joint MF on movies+games)",
    )

    # --- EMCDR ---
    t0 = time.time()
    model = EMCDRWrapper(data.num_users, data.num_items, embedding_dim=64, device=data.device)
    model.fit(data.cross_train, data.user_to_idx, data.item_to_idx,
              epochs=50, lr=0.001, reg_lambda=1e-4, batch_size=4096,
              positive_threshold=POSITIVE_THRESHOLD)
    metrics = evaluate_cross_domain("EMCDR", lambda uid: model.predict(uid), data)
    save_result(
        algo="EMCDR", metrics=metrics, dataset_info=data.dataset_info,
        lesson=args.lesson, train_time=time.time() - t0,
        description="EMCDR cold-start (global mapping from movie → game space)",
    )

    # --- PTUPCDR ---
    t0 = time.time()
    model = PTUPCDRWrapper(data.num_users, data.num_items, embedding_dim=64, device=data.device)
    model.fit(data.cross_train, data.user_to_idx, data.item_to_idx,
              epochs=50, lr=0.001, reg_lambda=1e-4, batch_size=4096,
              positive_threshold=POSITIVE_THRESHOLD)
    metrics = evaluate_cross_domain("PTUPCDR", lambda uid: model.predict(uid), data)
    save_result(
        algo="PTUPCDR", metrics=metrics, dataset_info=data.dataset_info,
        lesson=args.lesson, train_time=time.time() - t0,
        description="PTUPCDR cold-start (per-user hypernetwork mapping)",
    )

    # --- BiTGCF ---
    t0 = time.time()
    model = BiTGCFWrapper(data.num_users, data.num_items, embedding_dim=96, device=data.device)
    model.fit(data.cross_train, data.user_to_idx, data.item_to_idx,
              epochs=150, lr=0.001, reg_lambda=1e-4, batch_size=4096,
              positive_threshold=POSITIVE_THRESHOLD)
    metrics = evaluate_cross_domain("BiTGCF", lambda uid: model.predict(uid), data)
    save_result(
        algo="BiTGCF", metrics=metrics, dataset_info=data.dataset_info,
        lesson=args.lesson, train_time=time.time() - t0,
        description="BiTGCF cold-start (GCN + bidirectional transfer, emb=96, layers=3, epochs=150)",
    )

    # Print summary
    logger.info("\n=== L6 Cold-Start Results (Recall@10) ===")
    for name, m in sorted(results.items(), key=lambda x: x[1].get("recall_at_10", 0), reverse=True):
        logger.info("  %-12s  Recall@10=%.4f", name, m.get("recall_at_10", 0))


if __name__ == "__main__":
    main()
