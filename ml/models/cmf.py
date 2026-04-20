"""CMF (Collective Matrix Factorization) – cross-domain recommender.

Paper: Ajit P. Singh et al. "Relational Learning via Collective Matrix Factorization",
       SIGKDD 2008.

Joint factorization with shared user factors across movies + games.
Uses vendored recbole-cdr at ml/models/recbole_cdr/.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from ._cdr_base import fit_cdr

logger = logging.getLogger(__name__)


class CMF:
    """CMF wrapper. Source = movies, target = games."""

    def __init__(self, num_users: int, num_items: int,
                 embedding_dim: int = 64, device: str = "cpu") -> None:
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.device = device
        self._model = None
        self._rb_users: np.ndarray | None = None
        self._rb_items: np.ndarray | None = None
        self._valid_items: np.ndarray | None = None
        self._valid_users: np.ndarray | None = None
        self.user_embeddings: np.ndarray | None = None
        self.item_embeddings: np.ndarray | None = None

    def fit(self, ratings, user_to_idx, item_to_idx,
            epochs: int = 100, lr: float = 0.0005, reg_lambda: float = 0.0,
            batch_size: int = 8192, positive_threshold: float = 4.0,
            alpha: float = 0.05,
            source_domain: str = "movie", target_domain: str = "game") -> dict[str, float]:
        t0 = time.time()
        self._model, self._rb_users, self._rb_items = fit_cdr(
            "CMF",
            {"embedding_size": self.embedding_dim, "alpha": alpha, "lambda": 0.001, "gamma": 0.001},
            ratings, user_to_idx, item_to_idx,
            epochs, lr, reg_lambda, batch_size, positive_threshold,
            source_domain=source_domain, target_domain=target_domain,
        )

        with torch.no_grad():
            u = self._model.user_embedding.weight.cpu().numpy()
            i = self._model.item_embedding.weight.cpu().numpy()
        self.user_embeddings = u[self._rb_users]
        self.item_embeddings = i[self._rb_items]
        self._valid_items = self._rb_items != 0
        self._valid_users = self._rb_users != 0

        train_time = time.time() - t0
        logger.info("CMF trained in %.1fs (%d/%d items mapped)",
                    train_time, self._valid_items.sum(), len(self._rb_items))
        return {"train_time": train_time}

    def predict(self, user_idx: int) -> np.ndarray:
        if self.user_embeddings is None:
            raise ValueError("Model not trained yet")
        if not self._valid_users[user_idx]:
            return np.full(self.num_items, -np.inf, dtype=np.float64)
        user_emb = self.user_embeddings[user_idx]
        # CMF uses sigmoid(dot product) to bound scores in (0, 1), matching the
        # BPR training objective where positive pairs should score close to 1.
        scores = 1.0 / (1.0 + np.exp(-(self.item_embeddings @ user_emb).astype(np.float64)))
        scores[~self._valid_items] = -np.inf
        return scores

    def get_user_embeddings(self) -> np.ndarray:
        return self.user_embeddings

    def get_item_embeddings(self) -> np.ndarray:
        return self.item_embeddings
