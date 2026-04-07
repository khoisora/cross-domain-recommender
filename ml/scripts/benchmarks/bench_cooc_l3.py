"""Lesson 3 cooc ablation: does co-occurrence reranking improve any model on LLO?

Trains all L3 models (same hyperparams as individual bench scripts), then wraps
each with cooc reranking and compares. LLO split on movie_game_overlap dataset
(100% overlap users: movies >= 5, games >= 1).

Hypothesis: cooc helps models that don't already capture movie→game signal
(MF-BPR, NCF, LightGCN) and may hurt or be neutral for mapping CDR models
(EMCDR, PTUPCDR) that already use movie history.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from ml.evaluation.metrics import compute_all_metrics, aggregate_metrics
from ml.scripts.benchmarks.benchmark_common import (
    add_common_args, configure_benchmark, evaluate_cross_domain,
    load_cross_domain_split, save_result, setup_logging, verify_no_leakage,
    POSITIVE_THRESHOLD, K,
)
from ml.scripts.benchmarks.cooc_rerank import build_movie_game_cooc, wrap_predict_with_cooc
from ml.models.lightgcn import LightGCN
from ml.models.matrix_factorization_bpr import MatrixFactorizationBPR
from ml.models.ncf import NCF
from ml.models.cmf import CMF
from ml.models.emcdr import EMCDRWrapper
from ml.models.ptupcdr import PTUPCDRWrapper
from ml.models.bitgcf import BiTGCFWrapper

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="L3 cooc ablation — LLO split")
    add_common_args(parser)
    args = parser.parse_args()

    setup_logging()
    configure_benchmark(args.domain_pair)

    # LightGCN uses single-domain item space; CDR models need unified space.
    # We load both and share user/item maps via the same CrossDomainSplit seed.
    data_sd = load_cross_domain_split(
        domain_pair=args.domain_pair, target_domain=args.target,
        single_domain_item_space=True,
    )
    data_cd = load_cross_domain_split(
        domain_pair=args.domain_pair, target_domain=args.target,
        single_domain_item_space=False,
    )
    verify_no_leakage(data_sd)
    verify_no_leakage(data_cd)

    logger.info("Dataset: %d users, %d movie interactions, %d game interactions",
                data_sd.dataset_info["n_users"],
                data_sd.dataset_info["n_movie_interactions"],
                data_sd.dataset_info["n_game_interactions"])

    # Build cooc from training data — apply to all users (no max_target_train gate)
    cooc = build_movie_game_cooc(data_sd.movie_train, data_sd.game_train,
                                  rating_threshold=POSITIVE_THRESHOLD)

    results: dict[str, dict] = {}

    def run(algo, model, data, desc, cooc_lam=0.05):
        """Eval base model and cooc variant, save both."""
        predict_fn = lambda uid: model.predict(uid)
        m = evaluate_cross_domain(algo, predict_fn, data)
        results[algo] = m
        save_result(algo=algo, metrics=m, dataset_info=data.dataset_info,
                    lesson=args.lesson, train_time=0, description=desc)

        cooc_fn = wrap_predict_with_cooc(predict_fn, data, cooc, lam=cooc_lam)
        m_cooc = evaluate_cross_domain(f"{algo}_cooc", cooc_fn, data)
        results[f"{algo}_cooc"] = m_cooc
        save_result(algo=f"{algo}_cooc", metrics=m_cooc, dataset_info=data.dataset_info,
                    lesson=args.lesson, train_time=0,
                    description=f"{desc} + movie→game co-occurrence rerank (lam={cooc_lam})")

        delta = m_cooc["recall@10"] - m["recall@10"]
        pct = 100 * delta / m["recall@10"] if m["recall@10"] > 0 else float("inf")
        logger.info("%-12s  base=%.4f  cooc=%.4f  Δ=%+.4f (%+.1f%%)",
                    algo, m["recall@10"], m_cooc["recall@10"], delta, pct)

    # --- MF-BPR (single-domain) ---
    t0 = time.time()
    m = MatrixFactorizationBPR(data_sd.num_users, data_sd.num_items,
                                embedding_dim=64, device=data_sd.device)
    m.fit(data_sd.target_train, data_sd.user_to_idx, data_sd.item_to_idx,
          epochs=50, lr=0.001, reg_lambda=0.01, batch_size=4096,
          positive_threshold=POSITIVE_THRESHOLD)
    logger.info("MF-BPR trained in %.1fs", time.time() - t0)
    run("MF_BPR", m, data_sd, "MF-BPR single-domain LLO, L3 overlap dataset")

    # --- NCF (single-domain) ---
    t0 = time.time()
    m = NCF(data_sd.num_users, data_sd.num_items, embedding_dim=64, device=data_sd.device)
    m.fit(data_sd.target_train, data_sd.user_to_idx, data_sd.item_to_idx,
          epochs=50, lr=0.001, reg_lambda=1e-4, batch_size=4096,
          positive_threshold=POSITIVE_THRESHOLD)
    logger.info("NCF trained in %.1fs", time.time() - t0)
    run("NCF", m, data_sd, "NCF single-domain LLO, L3 overlap dataset")

    # --- LightGCN (single-domain) with early stopping ---
    target_mask = np.zeros(data_sd.num_items, dtype=bool)
    for idx in data_sd.target_item_indices:
        target_mask[idx] = True
    val_users = [uid for uid in data_sd.eval_user_indices if data_sd.game_val_relevant.get(uid)]

    def val_ndcg(model):
        per_user = []
        for uid in val_users[:500]:
            relevant = data_sd.game_val_relevant.get(uid, set())
            if not relevant:
                continue
            scores = model.predict(uid).copy()
            scores[~target_mask] = -np.inf
            for iid in data_sd.target_train_seen.get(uid, set()):
                if 0 <= iid < data_sd.num_items:
                    scores[iid] = -np.inf
            top = np.argsort(scores)[::-1][:K].tolist()
            per_user.append(compute_all_metrics(top, relevant, k_values=[K]))
        return aggregate_metrics(per_user).get("ndcg@10", 0.0) if per_user else 0.0

    t0 = time.time()
    lgcn = LightGCN(data_sd.num_users, data_sd.num_items,
                    embedding_dim=96, num_layers=3, device="cpu", dropout=0.1)
    lgcn.fit(data_sd.target_train, data_sd.user_to_idx, data_sd.item_to_idx,
             epochs=50, lr=0.001, reg_lambda=0.001, batch_size=4096,
             positive_threshold=POSITIVE_THRESHOLD,
             neg_sampling="popularity", neg_popularity_alpha=0.75,
             neg_item_indices=np.array(sorted(data_sd.target_item_indices), dtype=np.int64),
             early_stopping_patience=5, val_metric_fn=val_ndcg, val_every=3)
    logger.info("LightGCN trained in %.1fs", time.time() - t0)
    run("LightGCN", lgcn, data_sd, "LightGCN single-domain LLO, L3 overlap dataset, emb=96, layers=3")

    # --- CMF (cross-domain) ---
    t0 = time.time()
    m = CMF(data_cd.num_users, data_cd.num_items, embedding_dim=96, device=data_cd.device)
    m.fit(data_cd.cross_train, data_cd.user_to_idx, data_cd.item_to_idx,
          epochs=40, lr=0.01, reg_lambda=0.0, batch_size=8192,
          positive_threshold=POSITIVE_THRESHOLD)
    logger.info("CMF trained in %.1fs", time.time() - t0)
    run("CMF", m, data_cd, "CMF cross-domain LLO, L3 overlap dataset, emb=96")

    # --- EMCDR (cross-domain) ---
    t0 = time.time()
    m = EMCDRWrapper(data_cd.num_users, data_cd.num_items, embedding_dim=64, device=data_cd.device)
    m.fit(data_cd.cross_train, data_cd.user_to_idx, data_cd.item_to_idx,
          epochs=20, lr=0.001, reg_lambda=1e-4, batch_size=4096,
          positive_threshold=POSITIVE_THRESHOLD)
    logger.info("EMCDR trained in %.1fs", time.time() - t0)
    run("EMCDR", m, data_cd, "EMCDR cross-domain LLO, L3 overlap dataset")

    # --- PTUPCDR (cross-domain) ---
    t0 = time.time()
    m = PTUPCDRWrapper(data_cd.num_users, data_cd.num_items, embedding_dim=64, device=data_cd.device)
    m.fit(data_cd.cross_train, data_cd.user_to_idx, data_cd.item_to_idx,
          epochs=20, lr=0.001, reg_lambda=1e-4, batch_size=4096,
          positive_threshold=POSITIVE_THRESHOLD,
          meta_epochs=30, meta_lr=0.001, n_experts=8)
    logger.info("PTUPCDR trained in %.1fs", time.time() - t0)
    run("PTUPCDR", m, data_cd, "PTUPCDR cross-domain LLO, L3 overlap dataset")

    # --- BiTGCF (cross-domain) ---
    t0 = time.time()
    m = BiTGCFWrapper(data_cd.num_users, data_cd.num_items, embedding_dim=96, device=data_cd.device)
    m.fit(data_cd.cross_train, data_cd.user_to_idx, data_cd.item_to_idx,
          epochs=150, lr=0.001, reg_lambda=1e-4, batch_size=4096,
          positive_threshold=POSITIVE_THRESHOLD)
    logger.info("BiTGCF trained in %.1fs", time.time() - t0)
    run("BiTGCF", m, data_cd, "BiTGCF cross-domain LLO, L3 overlap dataset, emb=96, epochs=150")

    # Summary
    logger.info("\n=== L3 Cooc Ablation (Recall@10) ===")
    for name in ["MF_BPR", "NCF", "LightGCN", "CMF", "EMCDR", "PTUPCDR", "BiTGCF"]:
        base = results.get(name, {}).get("recall@10", 0)
        cooc = results.get(f"{name}_cooc", {}).get("recall@10", 0)
        delta_pct = 100 * (cooc - base) / base if base > 0 else float("inf")
        logger.info("  %-12s  base=%.4f  cooc=%.4f  %+.1f%%", name, base, cooc, delta_pct)


if __name__ == "__main__":
    main()
