"""EMCDR — cross-domain mapping benchmark.

Lesson 2: EMCDR trains separate MF per domain, then learns a global MLP mapping
from source user embeddings to target space. Cold-start capable.
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
from ml.models.emcdr import EMCDRWrapper

ALGO = "EMCDR"


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{ALGO} benchmark")
    add_common_args(parser)
    args = parser.parse_args()

    setup_logging()

    data = load_cross_domain_split(target_domain=args.target, single_domain_item_space=False)
    verify_no_leakage(data)

    model = EMCDRWrapper(data.num_users, data.num_items,
                         embedding_dim=64, device=data.device)
    t0 = time.time()
    model.fit(data.cross_train, data.user_to_idx, data.item_to_idx,
              epochs=20, lr=0.001, reg_lambda=1e-4, batch_size=4096,
              positive_threshold=POSITIVE_THRESHOLD)
    train_time = time.time() - t0

    metrics = evaluate_cross_domain(ALGO, lambda uid: model.predict(uid), data)

    save_result(
        algo=ALGO, metrics=metrics, dataset_info=data.dataset_info,
        lesson=args.lesson, train_time=train_time,
        description="RecBole-CDR EMCDR, emb=64, SOURCE:20+TARGET:20+OVERLAP:10, lr=0.001",
    )


if __name__ == "__main__":
    main()
