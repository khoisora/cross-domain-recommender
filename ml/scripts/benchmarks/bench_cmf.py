"""CMF (Collective Matrix Factorization) — cross-domain benchmark.

Lesson 2: CMF jointly factorizes both domains with shared user factors.
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
    POSITIVE_THRESHOLD, add_common_args, evaluate_cross_domain,
    load_cross_domain_split, save_result, setup_logging, verify_no_leakage,
)
from ml.models.cmf import CMF

ALGO = "CMF"


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{ALGO} benchmark")
    add_common_args(parser)
    args = parser.parse_args()

    setup_logging()

    # CDR models need unified item space (movie + game).
    data = load_cross_domain_split(target_domain=args.target, single_domain_item_space=False)
    verify_no_leakage(data)

    model = CMF(data.num_users, data.num_items, embedding_dim=96, device=data.device)
    t0 = time.time()
    # alpha=0.05: higher source weights over-push user emb toward movie space.
    # lr=0.0005: RecBole uses Adam; lr=0.01 diverges the joint loss.
    model.fit(data.cross_train, data.user_to_idx, data.item_to_idx,
              epochs=100, lr=0.0005, reg_lambda=0.0, batch_size=8192,
              alpha=0.05, positive_threshold=POSITIVE_THRESHOLD)
    train_time = time.time() - t0

    metrics = evaluate_cross_domain(ALGO, lambda uid: model.predict(uid), data)

    save_result(
        algo=ALGO, metrics=metrics, dataset_info=data.dataset_info,
        lesson=args.lesson, train_time=train_time,
        description="RecBole-CDR CMF, emb=96, epochs=100, lr=0.0005, alpha=0.05",
    )


if __name__ == "__main__":
    main()
