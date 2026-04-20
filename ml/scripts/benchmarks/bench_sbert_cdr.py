"""SBERT-CDR — content-based cross-domain recommender (LLO evaluation).

User profile strategy:
  - source-only (no game train): mean of movie embeddings → CDR transfer
  - overlap users: weighted blend of movie + game profiles
  - target-only: mean of game embeddings (same as plain SBERT)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import pandas as pd

from ml.scripts.benchmarks.benchmark_common import (
    DATA_DIR, POSITIVE_THRESHOLD, add_common_args, evaluate_cross_domain,
    load_cross_domain_split, save_result, setup_logging, verify_no_leakage,
)
from ml.models.sbert_model import SBERTModel

ALGO = "SBERT-CDR"


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{ALGO} benchmark")
    add_common_args(parser)
    parser.add_argument("--source-weight", type=float, default=0.5,
                        help="Weight for movie profile in overlap-user blend (default=0.5)")
    args = parser.parse_args()

    setup_logging()

    data = load_cross_domain_split(target_domain=args.target, single_domain_item_space=False)
    verify_no_leakage(data)

    movies_df = pd.read_parquet(DATA_DIR / "movies.parquet")
    games_df = pd.read_parquet(DATA_DIR / "games.parquet")
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
        description=f"SBERT-CDR all-MiniLM-L6-v2, source_weight={args.source_weight}, LLO eval",
    )


if __name__ == "__main__":
    main()
