"""Lesson 6: User-split cold-start benchmark.

80% warm users: all game interactions in training.
20% cold users: zero game interactions in training (movies only).

Evaluates all models on cold users only — measures cross-domain transfer
under true zero-game-history conditions.
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

import numpy as np
import pandas as pd

from ml.scripts.benchmarks.benchmark_common import (
    DATA_DIR, POSITIVE_THRESHOLD, add_common_args, evaluate_cross_domain,
    load_user_split_cold_start_split, save_result, setup_logging,
)
from ml.scripts.benchmarks.cooc_rerank import build_movie_game_cooc, wrap_predict_with_cooc
from ml.models.cmf import CMF
from ml.models.emcdr import EMCDRWrapper
from ml.models.id_utils import normalize_id
from ml.models.lightgcn import LightGCN
from ml.models.matrix_factorization_bpr import MatrixFactorizationBPR
from ml.models.ncf import NCF
from ml.models.ptupcdr import PTUPCDRWrapper
from ml.models.sbert_model import SBERTModel

logger = logging.getLogger(__name__)


def popularity_predict_fn(data):
    """Return a predict function that ranks games by warm-user interaction count."""
    counts = data.game_train["item_id"].map(normalize_id).value_counts()
    pop_scores = np.zeros(data.num_items, dtype=np.float64)
    for iid, c in counts.items():
        idx = data.item_to_idx.get(iid)
        if idx is not None:
            pop_scores[idx] = c
    return lambda uid: pop_scores.copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cold-start user-split benchmark")
    add_common_args(parser)
    parser.add_argument("--cold-ratio", type=float, default=0.2)
    args = parser.parse_args()

    setup_logging()

    data = load_user_split_cold_start_split(cold_start_ratio=args.cold_ratio)
    # Force CPU — unified cross-domain ID space is too large for MPS.
    data.device = "cpu"

    logger.info(
        "Dataset: %d eval cold users, %d warm users, %d game train interactions",
        len(data.eval_user_indices),
        data.dataset_info["n_warm_users"],
        data.dataset_info["n_game_interactions_warm"],
    )

    results: dict[str, dict] = {}
    cooc = build_movie_game_cooc(data.movie_train, data.game_train,
                                  rating_threshold=POSITIVE_THRESHOLD)

    def run(algo, predict_fn, desc, train_time=0.0):
        m = evaluate_cross_domain(algo, predict_fn, data)
        results[algo] = m
        save_result(algo=algo, metrics=m, dataset_info=data.dataset_info,
                    lesson=args.lesson, train_time=train_time, description=desc)
        return m

    def run_with_cooc(algo, predict_fn, desc, train_time=0.0):
        run(algo, predict_fn, desc, train_time=train_time)
        cooc_fn = wrap_predict_with_cooc(predict_fn, data, cooc, lam=0.05, max_target_train=0)
        run(f"{algo}_cooc", cooc_fn, f"{desc} + cooc rerank", train_time=train_time)

    # --- Popularity baseline ---
    t0 = time.time()
    pop_fn = popularity_predict_fn(data)
    run_with_cooc("Popularity", pop_fn,
                  "Global game popularity from warm-user interactions",
                  train_time=time.time() - t0)

    # --- MF-BPR ---
    t0 = time.time()
    model = MatrixFactorizationBPR(data.num_users, data.num_items, embedding_dim=64, device=data.device)
    model.fit(data.game_train, data.user_to_idx, data.item_to_idx,
              epochs=60, lr=0.05, reg_lambda=0.01,
              positive_threshold=POSITIVE_THRESHOLD)
    run("MF_BPR", lambda uid: model.predict(uid),
        "MF-BPR cold-start (game-only, warm users only)",
        train_time=time.time() - t0)

    # --- LightGCN ---
    t0 = time.time()
    lgcn_model = LightGCN(data.num_users, data.num_items, embedding_dim=64, device=data.device)
    lgcn_model.fit(data.game_train, data.user_to_idx, data.item_to_idx,
                   epochs=50, lr=0.001, reg_lambda=1e-4, batch_size=4096,
                   positive_threshold=POSITIVE_THRESHOLD)
    lgcn_train_time = time.time() - t0
    run_with_cooc("LightGCN", lambda uid: lgcn_model.predict(uid),
                  "LightGCN cold-start (game-only, warm users only)",
                  train_time=lgcn_train_time)

    # --- NCF ---
    t0 = time.time()
    model = NCF(data.num_users, data.num_items, embedding_dim=64, device=data.device)
    model.fit(data.game_train, data.user_to_idx, data.item_to_idx,
              epochs=50, lr=0.001, reg_lambda=1e-4, batch_size=4096,
              positive_threshold=POSITIVE_THRESHOLD)
    run("NCF", lambda uid: model.predict(uid),
        "NCF cold-start (game-only, warm users only)",
        train_time=time.time() - t0)

    # --- CMF ---
    t0 = time.time()
    cmf_model = CMF(data.num_users, data.num_items, embedding_dim=64, device=data.device)
    cmf_model.fit(data.cross_train, data.user_to_idx, data.item_to_idx,
                  epochs=50, lr=0.001, reg_lambda=0.01, batch_size=4096,
                  positive_threshold=POSITIVE_THRESHOLD)
    run_with_cooc("CMF", lambda uid: cmf_model.predict(uid),
                  "CMF cold-start (joint MF on movies+games)",
                  train_time=time.time() - t0)

    # --- EMCDR ---
    t0 = time.time()
    emcdr_model = EMCDRWrapper(data.num_users, data.num_items, embedding_dim=64, device=data.device)
    emcdr_model.fit(data.cross_train, data.user_to_idx, data.item_to_idx,
                    epochs=50, lr=0.001, reg_lambda=1e-4, batch_size=4096,
                    positive_threshold=POSITIVE_THRESHOLD)
    run_with_cooc("EMCDR", lambda uid: emcdr_model.predict(uid),
                  "EMCDR cold-start (global mapping from movie → game space)",
                  train_time=time.time() - t0)

    # --- PTUPCDR ---
    t0 = time.time()
    ptupcdr_model = PTUPCDRWrapper(data.num_users, data.num_items, embedding_dim=64, device=data.device)
    ptupcdr_model.fit(data.cross_train, data.user_to_idx, data.item_to_idx,
                      epochs=50, lr=0.001, reg_lambda=1e-4, batch_size=4096,
                      positive_threshold=POSITIVE_THRESHOLD)
    run_with_cooc("PTUPCDR", lambda uid: ptupcdr_model.predict(uid),
                  "PTUPCDR cold-start (per-user hypernetwork mapping)",
                  train_time=time.time() - t0)

    # --- SBERT-CDR ---
    movies_df = pd.read_parquet(DATA_DIR / "movies.parquet")
    games_df = pd.read_parquet(DATA_DIR / "games.parquet")
    items_df = pd.concat([movies_df, games_df], ignore_index=True)

    t0 = time.time()
    sbert = SBERTModel()
    sbert.encode_items(items_df, data.item_to_idx)
    sbert.compute_cross_domain_user_embeddings(
        data.cross_train, data.user_to_idx, data.item_to_idx,
        source_domain="movie", target_domain="game",
        positive_threshold=POSITIVE_THRESHOLD,
        source_weight=0.5,
    )
    run_with_cooc("SBERT_CDR", lambda uid: sbert.predict(uid),
                  "SBERT-CDR cold-start: cold users get movie-only SBERT profile",
                  train_time=time.time() - t0)

    logger.info("\n=== L6 Cold-Start Results (Recall@10) ===")
    for name, m in sorted(results.items(), key=lambda x: x[1].get("recall@10", 0), reverse=True):
        logger.info("  %-16s  Recall@10=%.4f", name, m.get("recall@10", 0))


if __name__ == "__main__":
    main()
