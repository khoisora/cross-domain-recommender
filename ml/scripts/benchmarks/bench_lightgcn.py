"""LightGCN — single-domain graph benchmark.

Lesson 2: LightGCN uses graph convolution on the user-item bipartite graph.
Expected to outperform MF-BPR and CDR models on the standard benchmark.
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

from ml.scripts.benchmarks.benchmark_common import (
    add_common_args, configure_benchmark, evaluate_cross_domain,
    load_cross_domain_split, save_result, setup_logging, verify_no_leakage,
    POSITIVE_THRESHOLD, K,
)
from ml.models.lightgcn import LightGCN

logger = logging.getLogger(__name__)
ALGO = "LightGCN"


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{ALGO} benchmark")
    add_common_args(parser)
    args = parser.parse_args()

    setup_logging()
    configure_benchmark(args.domain_pair)

    data = load_cross_domain_split(
        domain_pair=args.domain_pair, target_domain=args.target,
        single_domain_item_space=True,
    )
    verify_no_leakage(data)

    # Validation metric for early stopping
    from ml.evaluation.evaluator import evaluate_full_rank
    def val_fn(model):
        m = evaluate_full_rank(ALGO, lambda uid: model.predict(uid), data,
                               eval_user_override=data.eval_user_indices[:500])
        return m.get("ndcg@10", 0)

    model = LightGCN(data.num_users, data.num_items,
                     embedding_dim=96, num_layers=3,
                     device=data.device, dropout=0.1)
    t0 = time.time()
    model.fit(data.target_train, data.user_to_idx, data.item_to_idx,
              epochs=40, lr=0.001, reg_lambda=0.001, batch_size=4096,
              positive_threshold=POSITIVE_THRESHOLD,
              neg_sampling="popularity", neg_popularity_alpha=0.75,
              neg_item_indices=sorted(data.target_item_indices),
              early_stopping_patience=5, val_metric_fn=val_fn, val_every=3)
    train_time = time.time() - t0

    metrics = evaluate_cross_domain(ALGO, lambda uid: model.predict(uid), data)

    save_result(
        algo=ALGO, metrics=metrics, dataset_info=data.dataset_info,
        lesson=args.lesson, train_time=train_time,
        description="PyG LightGCN, emb=96, layers=3, epochs=40, lr=0.001, reg=0.001, dropout=0.1",
    )


if __name__ == "__main__":
    main()
