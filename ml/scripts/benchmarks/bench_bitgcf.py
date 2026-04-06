"""BiTGCF — GCN-based cross-domain benchmark.

BiTGCF combines LightGCN-style graph propagation with bi-directional
transfer layers between source and target domain graphs.
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
from ml.models.bitgcf import BiTGCFWrapper

logger = logging.getLogger(__name__)
ALGO = "BiTGCF"


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{ALGO} benchmark")
    add_common_args(parser)
    args = parser.parse_args()

    setup_logging()
    configure_benchmark(args.domain_pair)

    data = load_cross_domain_split(
        domain_pair=args.domain_pair, target_domain=args.target,
        single_domain_item_space=False,
    )
    verify_no_leakage(data)

    model = BiTGCFWrapper(data.num_users, data.num_items,
                          embedding_dim=96, device=data.device)
    t0 = time.time()
    model.fit(data.cross_train, data.user_to_idx, data.item_to_idx,
              epochs=150, lr=0.001, reg_lambda=1e-4, batch_size=4096,
              positive_threshold=POSITIVE_THRESHOLD)
    train_time = time.time() - t0

    metrics = evaluate_cross_domain(ALGO, lambda uid: model.predict(uid), data)

    save_result(
        algo=ALGO, metrics=metrics, dataset_info=data.dataset_info,
        lesson=args.lesson, train_time=train_time,
        description="BiTGCF, emb=96, layers=3, epochs=150, lr=0.001, λ_src=0.8, λ_tgt=0.2, drop=0.1",
    )


if __name__ == "__main__":
    main()
