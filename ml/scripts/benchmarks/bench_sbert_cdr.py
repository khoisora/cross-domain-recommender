"""SBERT-CDR — content-based cross-domain recommender (cold-start evaluation).

User profile strategy (cross-domain):
  - cold users (no game train): mean of movie embeddings → CDR transfer
  - overlap/warm users: weighted blend of movie + game profiles

Evaluated on the same cold-start user split as Lesson 6 (80/20 warm/cold).
Cold users have zero game training history — only movie data available.
SBERT-CDR bridges the gap using shared semantic space (movies + games).

Key finding: SBERT-CDR wins on one_shot_unpopular_target_user subgroup
where collaborative CDR (PTUPCDR) is blind to niche long-tail items.
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
ALGO = "SBERT-CDR"


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{ALGO} cold-start benchmark")
    add_common_args(parser)
    parser.add_argument("--source-weight", type=float, default=0.5,
                        help="Weight for movie profile in overlap-user blend (default=0.5)")
    args = parser.parse_args()

    setup_logging()

    data = load_user_split_cold_start_split(
        domain_pair=args.domain_pair, target_domain=args.target,
    )
    data.device = "cpu"

    data_dir = _DOMAIN_PAIR_PATHS[args.domain_pair][0]
    movies_df = pd.read_parquet(data_dir / "movies.parquet")
    games_df = pd.read_parquet(data_dir / "games.parquet")
    items_df = pd.concat([movies_df, games_df], ignore_index=True)

    t0 = time.time()
    model = SBERTModel()
    model.encode_items(items_df, data.item_to_idx)
    # cold users contribute only movies to cross_train → profile = mean of movie embeddings
    model.compute_cross_domain_user_embeddings(
        data.cross_train, data.user_to_idx, data.item_to_idx,
        source_domain="movie", target_domain="game",
        positive_threshold=POSITIVE_THRESHOLD,
        source_weight=args.source_weight,
    )
    train_time = time.time() - t0

    metrics = evaluate_cross_domain(ALGO, lambda uid: model.predict(uid), data)

    save_result(
        algo=ALGO, metrics=metrics, dataset_info=data.dataset_info,
        lesson=args.lesson, train_time=train_time,
        description=f"SBERT-CDR all-MiniLM-L6-v2, source_weight={args.source_weight}, cold-start eval",
    )


if __name__ == "__main__":
    main()
