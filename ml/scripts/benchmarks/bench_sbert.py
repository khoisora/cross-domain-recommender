"""SBERT — in-domain content-based benchmark (LLO evaluation).

User profile = mean of game item embeddings (target-domain only).
No training required. Evaluated on all users via standard LLO split.
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

ALGO = "SBERT"


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{ALGO} benchmark")
    add_common_args(parser)
    args = parser.parse_args()

    setup_logging()

    data = load_cross_domain_split(target_domain=args.target, single_domain_item_space=True)
    verify_no_leakage(data)

    games_df = pd.read_parquet(DATA_DIR / "games.parquet")

    t0 = time.time()
    model = SBERTModel()
    model.encode_items(games_df, data.item_to_idx)
    model.compute_user_embeddings(
        data.game_train, data.user_to_idx, data.item_to_idx,
        positive_threshold=POSITIVE_THRESHOLD,
    )
    train_time = time.time() - t0

    metrics = evaluate_cross_domain(ALGO, lambda uid: model.predict(uid), data)

    save_result(
        algo=ALGO, metrics=metrics, dataset_info=data.dataset_info,
        lesson=args.lesson, train_time=train_time,
        description="SBERT all-MiniLM-L6-v2, in-domain game profiles, LLO eval",
    )


if __name__ == "__main__":
    main()
