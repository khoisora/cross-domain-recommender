"""Focused CMF cold-start sweep: best alpha/lr with varied emb, epochs, reg."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import torch

from ml.scripts.benchmarks.benchmark_common import (
    evaluate_cross_domain, load_user_split_cold_start_split,
    setup_logging, POSITIVE_THRESHOLD,
)
from ml.models._cdr_base import fit_cdr

logger = logging.getLogger(__name__)


class CMFSweep:
    def __init__(self, num_users, num_items, embedding_dim=64, use_sigmoid=True):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.use_sigmoid = use_sigmoid
        self.user_embeddings = None
        self.item_embeddings = None
        self._valid_items = None
        self._valid_users = None

    def fit(self, ratings, user_to_idx, item_to_idx,
            epochs=50, lr=0.001, reg_lambda=0.0, batch_size=4096,
            positive_threshold=4.0, alpha=0.1,
            source_domain="movie", target_domain="game"):
        t0 = time.time()
        model, rb_users, rb_items = fit_cdr(
            "CMF",
            {"embedding_size": self.embedding_dim,
             "alpha": alpha, "lambda": 0.001, "gamma": 0.001},
            ratings, user_to_idx, item_to_idx,
            epochs, lr, reg_lambda, batch_size, positive_threshold,
            source_domain=source_domain, target_domain=target_domain,
        )
        with torch.no_grad():
            u = model.user_embedding.weight.cpu().numpy()
            i = model.item_embedding.weight.cpu().numpy()
        self.user_embeddings = u[rb_users]
        self.item_embeddings = i[rb_items]
        self._valid_items = rb_items != 0
        self._valid_users = rb_users != 0
        return {"train_time": time.time() - t0}

    def predict(self, user_idx, item_indices=None):
        if not self._valid_users[user_idx]:
            n = self.num_items if item_indices is None else len(item_indices)
            return np.full(n, -np.inf, dtype=np.float64)
        user_emb = self.user_embeddings[user_idx]
        items = self.item_embeddings if item_indices is None else self.item_embeddings[item_indices]
        scores = (items @ user_emb).astype(np.float64)
        if self.use_sigmoid:
            scores = 1.0 / (1.0 + np.exp(-scores))
        valid = self._valid_items if item_indices is None else self._valid_items[item_indices]
        scores[~valid] = -np.inf
        return scores


def main():
    setup_logging()
    data = load_user_split_cold_start_split(domain_pair="movie_game")
    data.device = "cpu"

    out_dir = Path("artifacts/sweeps")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    # Best from v1: alpha=0.1, lr=0.001, sigmoid=True
    # Now vary emb_dim, epochs, reg_lambda, and also try alpha=0.05/0.15
    for alpha in [0.05, 0.1, 0.15]:
        for emb_dim in [64, 96, 128]:
            for epochs in [50, 100]:
                for reg_lambda in [0.0, 0.001, 0.01]:
                    configs.append({
                        "alpha": alpha, "emb_dim": emb_dim,
                        "epochs": epochs, "reg_lambda": reg_lambda,
                    })

    logger.info("Running %d configurations", len(configs))
    results = []

    for i, cfg in enumerate(configs):
        tag = f"a{cfg['alpha']}_e{cfg['emb_dim']}_ep{cfg['epochs']}_r{cfg['reg_lambda']}"
        logger.info("[%d/%d] %s", i + 1, len(configs), tag)

        model = CMFSweep(data.num_users, data.num_items,
                         embedding_dim=cfg["emb_dim"], use_sigmoid=True)
        t0 = time.time()
        model.fit(data.cross_train, data.user_to_idx, data.item_to_idx,
                  epochs=cfg["epochs"], lr=0.001, reg_lambda=cfg["reg_lambda"],
                  batch_size=4096, positive_threshold=POSITIVE_THRESHOLD,
                  alpha=cfg["alpha"])
        train_time = time.time() - t0

        metrics = evaluate_cross_domain("CMF_sweep", lambda uid: model.predict(uid), data)

        r = metrics.get("recall@10", 0)
        n = metrics.get("ndcg@10", 0)
        sh = metrics.get("sampled_hr@10", 0)
        sn = metrics.get("sampled_ndcg@10", 0)
        logger.info("  => recall=%.4f ndcg=%.4f s_hr=%.4f s_ndcg=%.4f (%.1fs)",
                    r, n, sh, sn, train_time)

        results.append({"config": cfg, "recall@10": r, "ndcg@10": n,
                        "sampled_hr@10": sh, "sampled_ndcg@10": sn,
                        "train_time": train_time})

    results.sort(key=lambda x: x["recall@10"], reverse=True)
    logger.info("\n=== Top 5 configs by Recall@10 ===")
    for r in results[:5]:
        logger.info("  recall=%.4f ndcg=%.4f | %s", r["recall@10"], r["ndcg@10"], r["config"])

    Path("artifacts/sweeps/cmf_coldstart_v2_summary.json").write_text(
        json.dumps(results[:10], indent=2))


if __name__ == "__main__":
    main()
