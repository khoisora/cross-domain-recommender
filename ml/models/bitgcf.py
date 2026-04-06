"""BiTGCF – Bi-directional Transfer Graph Collaborative Filtering (Liu et al., CIKM 2020).

GCN-based CDR: runs LightGCN-style message passing on both domain graphs,
with transfer layers that blend source/target user+item embeddings at each hop.
Combines graph collaborative filtering with cross-domain knowledge transfer.

Uses vendored recbole-cdr at ml/models/recbole_cdr/.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np
import torch

from ._cdr_base import fit_cdr

logger = logging.getLogger(__name__)


class BiTGCFWrapper:
    """BiTGCF wrapper. Source = movies, target = games."""

    def __init__(self, num_users: int, num_items: int,
                 embedding_dim: int = 96, device: str = "cpu") -> None:
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.device = device
        self._model = None
        self._rb_users: Optional[np.ndarray] = None
        self._rb_items: Optional[np.ndarray] = None
        self._valid_items: Optional[np.ndarray] = None
        self._valid_users: Optional[np.ndarray] = None
        self.user_embeddings: Optional[np.ndarray] = None
        self.item_embeddings: Optional[np.ndarray] = None

    def fit(self, ratings, user_to_idx, item_to_idx,
            epochs: int = 150, lr: float = 0.001, reg_lambda: float = 1e-4,
            batch_size: int = 4096, positive_threshold: float = 4.0,
            source_domain: str = "movie", target_domain: str = "game",
            model_hyperparams: Optional[dict[str, Any]] = None) -> dict[str, float]:
        t0 = time.time()
        extra: dict[str, Any] = {
            "embedding_size": self.embedding_dim,
            "n_layers": 3,
            "reg_weight": reg_lambda,
            "weight_decay": 0.0,
            "lambda_source": 0.8,
            "lambda_target": 0.2,
            "drop_rate": 0.10,
            "connect_way": "mean",
        }
        if model_hyperparams:
            extra.update(model_hyperparams)

        self._model, self._rb_users, self._rb_items = fit_cdr(
            "BiTGCF", extra,
            ratings, user_to_idx, item_to_idx,
            epochs, lr, reg_lambda, batch_size, positive_threshold,
            source_domain=source_domain, target_domain=target_domain,
        )

        self._model.eval()
        with torch.no_grad():
            _, _, target_user_e, target_item_e = self._model.forward()
        u = target_user_e.cpu().numpy()
        i = target_item_e.cpu().numpy()
        self.user_embeddings = u[self._rb_users]
        self.item_embeddings = i[self._rb_items]
        self._valid_items = self._rb_items != 0
        self._valid_users = self._rb_users != 0

        n_valid = self._valid_items.sum()
        logger.info("BiTGCF item mapping: %d/%d valid (%.1f%% unmapped)",
                    n_valid, len(self._rb_items),
                    100 * (1 - n_valid / len(self._rb_items)))

        train_time = time.time() - t0
        logger.info("BiTGCF trained in %.1fs", train_time)
        return {"train_time": train_time}

    def predict(self, user_idx: int,
                item_indices: Optional[np.ndarray] = None) -> np.ndarray:
        if self.user_embeddings is None:
            raise ValueError("Model not trained yet")
        if not self._valid_users[user_idx]:
            n = self.num_items if item_indices is None else len(item_indices)
            return np.full(n, -np.inf, dtype=np.float64)
        user_emb = self.user_embeddings[user_idx]
        items = self.item_embeddings if item_indices is None else self.item_embeddings[item_indices]
        scores = (items @ user_emb).astype(np.float64)
        valid = self._valid_items if item_indices is None else self._valid_items[item_indices]
        scores[~valid] = -np.inf
        return scores

    def get_user_embeddings(self) -> np.ndarray:
        return self.user_embeddings

    def get_item_embeddings(self) -> np.ndarray:
        return self.item_embeddings
