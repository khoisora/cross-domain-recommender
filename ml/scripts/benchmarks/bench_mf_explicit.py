"""MF Explicit (Surprise SVD) — single-domain benchmark.

Lesson 1: shows that explicit rating prediction (RMSE objective) is a poor
proxy for top-K ranking. MF-BPR should outperform on Recall@10 and NDCG@10.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from ml.scripts.benchmarks.benchmark_common import (
    add_common_args, evaluate_cross_domain,
    load_cross_domain_split, save_result, setup_logging, verify_no_leakage,
)
from ml.models.matrix_factorization import MatrixFactorizationExplicit

ALGO = "MF_Explicit"


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{ALGO} benchmark")
    add_common_args(parser)
    args = parser.parse_args()

    setup_logging()

    data = load_cross_domain_split(target_domain=args.target, single_domain_item_space=True)
    verify_no_leakage(data)

    model = MatrixFactorizationExplicit(
        data.num_users, data.num_items, embedding_dim=128, device="cpu",
    )
    t0 = time.time()
    model.fit(data.target_train, data.user_to_idx, data.item_to_idx,
              epochs=100, lr=0.01, reg_lambda=0.005)
    train_time = time.time() - t0

    metrics = evaluate_cross_domain(ALGO, lambda uid: model.predict(uid), data)

    save_result(
        algo=ALGO, metrics=metrics, dataset_info=data.dataset_info,
        lesson=args.lesson, train_time=train_time,
        description="Surprise SVD, emb=128, epochs=100, lr=0.01, reg=0.005",
    )


if __name__ == "__main__":
    main()
