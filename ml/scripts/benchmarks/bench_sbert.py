"""SBERT — in-domain content-based benchmark.

User profile = mean of game item embeddings (target-domain only).
No training required. Measures how well semantic text similarity serves
as a game recommender when the user has game history.
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

    # L7 uses cold-start split to evaluate on zero-game users
    data = load_user_split_cold_start_split(
        domain_pair=args.domain_pair, target_domain=args.target,
    )
    # Force CPU — unified ID space too large for MPS
    data.device = "cpu"

    # Load game item metadata
    data_dir = _DOMAIN_PAIR_PATHS[args.domain_pair][0]
    games_df = pd.read_parquet(data_dir / "games.parquet")

    # Build item_to_idx for game-only items (SBERT scores over all items but
    # computes user profile from game train interactions for cold users = 0 games)
    t0 = time.time()
    model = SBERTModel()
    model.encode_items(games_df, data.item_to_idx)
    # game_train has only warm users — cold users have no game history
    # SBERT user profile = 0 vector for cold users → uniform scores → random rank
    # This establishes the content-only game baseline for cold-start users
    model.compute_user_embeddings(
        data.game_train, data.user_to_idx, data.item_to_idx,
        positive_threshold=POSITIVE_THRESHOLD,
    )
    train_time = time.time() - t0

    metrics = evaluate_cross_domain(ALGO, lambda uid: model.predict(uid), data)

    save_result(
        algo=ALGO, metrics=metrics, dataset_info=data.dataset_info,
        lesson=args.lesson, train_time=train_time,
        description="SBERT all-MiniLM-L6-v2, in-domain game profiles (cold-start eval)",
    )


if __name__ == "__main__":
    main()
