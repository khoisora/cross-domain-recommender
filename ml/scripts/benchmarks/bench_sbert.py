"""SBERT — in-domain content-based benchmark (cold-start evaluation).

User profile = mean of game item embeddings (target-domain only).
No training required. Evaluated on cold-start users (zero game history)
using the same 80/20 user-split protocol as Lesson 6.

For cold users: no game training items → zero user vector → uniform scores.
This establishes the content-only game baseline. Compare with SBERT-CDR
which uses movie profiles to bridge cross-domain for cold users.
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

import pandas as pd

from ml.scripts.benchmarks.benchmark_common import (
    add_common_args, evaluate_cross_domain,
    load_user_split_cold_start_split, save_result, setup_logging,
    POSITIVE_THRESHOLD, _DOMAIN_PAIR_PATHS,
)
from ml.models.sbert_model import SBERTModel

logger = logging.getLogger(__name__)
ALGO = "SBERT"


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{ALGO} cold-start benchmark")
    add_common_args(parser)
    args = parser.parse_args()

    setup_logging()

    data = load_user_split_cold_start_split(
        domain_pair=args.domain_pair, target_domain=args.target,
    )
    data.device = "cpu"

    data_dir = _DOMAIN_PAIR_PATHS[args.domain_pair][0]
    games_df = pd.read_parquet(data_dir / "games.parquet")

    t0 = time.time()
    model = SBERTModel()
    model.encode_items(games_df, data.item_to_idx)
    # game_train = warm users only; cold users have no game history → zero profile
    model.compute_user_embeddings(
        data.game_train, data.user_to_idx, data.item_to_idx,
        positive_threshold=POSITIVE_THRESHOLD,
    )
    train_time = time.time() - t0

    metrics = evaluate_cross_domain(ALGO, lambda uid: model.predict(uid), data)

    save_result(
        algo=ALGO, metrics=metrics, dataset_info=data.dataset_info,
        lesson=args.lesson, train_time=train_time,
        description="SBERT all-MiniLM-L6-v2, in-domain game profiles, cold-start eval",
    )


if __name__ == "__main__":
    main()
