"""SBERT-CDR — content-based cross-domain recommender.

User profile = normalised mean of MOVIE item embeddings (source domain).
For overlap users: blends source + target item embeddings with source_weight.

Since movies and games are encoded in the same SBERT semantic space,
"action movie" and "action game" naturally have similar embeddings,
enabling zero-shot cross-domain transfer without any collaborative training.
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
    add_common_args, configure_benchmark, evaluate_cross_domain,
    load_cross_domain_split, save_result, setup_logging, verify_no_leakage,
    POSITIVE_THRESHOLD, _DOMAIN_PAIR_PATHS,
)
from ml.models.sbert_model import SBERTModel

logger = logging.getLogger(__name__)
ALGO = "SBERT-CDR"


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{ALGO} benchmark")
    add_common_args(parser)
    parser.add_argument("--source-weight", type=float, default=0.5,
                        help="Weight for movie profile in overlap-user blend (default=0.5)")
    args = parser.parse_args()

    setup_logging()
    configure_benchmark(args.domain_pair)

    data = load_cross_domain_split(
        domain_pair=args.domain_pair, target_domain=args.target,
        single_domain_item_space=False,
    )
    verify_no_leakage(data)

    # Load item metadata from both domains
    data_dir = _DOMAIN_PAIR_PATHS[args.domain_pair][0]
    movies_df = pd.read_parquet(data_dir / "movies.parquet")
    games_df = pd.read_parquet(data_dir / "games.parquet")
    items_df = pd.concat([movies_df, games_df], ignore_index=True)

    t0 = time.time()
    model = SBERTModel()
    model.encode_items(items_df, data.item_to_idx)
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
        description=f"SBERT-CDR all-MiniLM-L6-v2, source_weight={args.source_weight}",
    )


if __name__ == "__main__":
    main()
