"""Hyperparameter sweep for CMF on lesson-6 cold-start.

Sweeps alpha (source weight) and lr, with/without sigmoid in predict.
Saves each config's results to artifacts/sweeps/cmf_coldstart_*.json.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from itertools import product
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
from ml.models.id_utils import normalize_id, normalize_maps

logger = logging.getLogger(__name__)


class CMFSweep:
    """CMF with configurable alpha and optional sigmoid removal."""

    def __init__(self, num_users: int, num_items: int,
                 embedding_dim: int = 64, use_sigmoid: bool = False) -> None:
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
            positive_threshold=4.0, alpha=0.3,
            source_domain="movie", target_domain="game") -> dict:
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

    def predict(self, user_idx: int, item_indices=None) -> np.ndarray:
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
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for lr in [0.001, 0.005, 0.01]:
            for use_sigmoid in [False, True]:
                configs.append({"alpha": alpha, "lr": lr, "use_sigmoid": use_sigmoid})

    logger.info("Running %d configurations", len(configs))

    best_recall = 0
    best_cfg = None

    for i, cfg in enumerate(configs):
        tag = f"a{cfg['alpha']}_lr{cfg['lr']}_{'sig' if cfg['use_sigmoid'] else 'raw'}"
        out_file = out_dir / f"cmf_coldstart_{tag}.json"

        logger.info("[%d/%d] alpha=%.1f lr=%.3f sigmoid=%s",
                    i + 1, len(configs), cfg["alpha"], cfg["lr"], cfg["use_sigmoid"])

        model = CMFSweep(data.num_users, data.num_items,
                         embedding_dim=64, use_sigmoid=cfg["use_sigmoid"])
        t0 = time.time()
        model.fit(data.cross_train, data.user_to_idx, data.item_to_idx,
                  epochs=50, lr=cfg["lr"], batch_size=4096,
                  positive_threshold=POSITIVE_THRESHOLD,
                  alpha=cfg["alpha"])
        train_time = time.time() - t0

        metrics = evaluate_cross_domain("CMF_sweep", lambda uid: model.predict(uid), data)

        result = {
            "config": cfg,
            "train_time": train_time,
            **metrics,
        }
        out_file.write_text(json.dumps(result, indent=2))

        r = metrics.get("recall@10", 0)
        n = metrics.get("ndcg@10", 0)
        sh = metrics.get("sampled_hr@10", 0)
        sn = metrics.get("sampled_ndcg@10", 0)
        logger.info("  => recall=%.4f ndcg=%.4f s_hr=%.4f s_ndcg=%.4f (%.1fs)",
                    r, n, sh, sn, train_time)

        if r > best_recall:
            best_recall = r
            best_cfg = cfg

    logger.info("\n=== Best config: %s  recall@10=%.4f ===", best_cfg, best_recall)


if __name__ == "__main__":
    main()
