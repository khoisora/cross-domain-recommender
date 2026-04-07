"""NCF (NeuMF) — single-domain neural benchmark.

Lesson 2: NeuMF combines GMF and MLP branches for non-linear user-item scoring.
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
    POSITIVE_THRESHOLD,
)
from ml.models.ncf import NCF

logger = logging.getLogger(__name__)
ALGO = "NCF"


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

    model = NCF(data.num_users, data.num_items,
                embedding_dim=64, device=data.device)
    t0 = time.time()
    # epochs=150: sweep shows 50 epochs under-trains; lr=0.001 is already correct for Adam.
    model.fit(data.target_train, data.user_to_idx, data.item_to_idx,
              epochs=150, lr=0.001, reg_lambda=0.001, batch_size=4096,
              positive_threshold=POSITIVE_THRESHOLD)
    train_time = time.time() - t0

    metrics = evaluate_cross_domain(ALGO, lambda uid: model.predict(uid), data)

    save_result(
        algo=ALGO, metrics=metrics, dataset_info=data.dataset_info,
        lesson=args.lesson, train_time=train_time,
        description="RecBole NeuMF, emb=64, epochs=150, lr=0.001, reg=0.001",
    )


if __name__ == "__main__":
    main()
