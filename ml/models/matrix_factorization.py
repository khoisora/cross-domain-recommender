"""Matrix Factorization using Surprise SVD for explicit ratings.

Prediction: score(u,i) = mu + b_u + b_i + q_i^T * p_u

Optimizes RMSE on raw star ratings (1-5) — baseline to show that explicit
rating prediction is a poor objective for top-K ranking.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader

from .base_recommender import BaseRecommender
from .id_utils import map_user_item_columns, normalize_maps

logger = logging.getLogger(__name__)


class MatrixFactorizationExplicit(BaseRecommender):
    """Surprise SVD with biases for explicit rating prediction."""

    def __init__(self, num_users: int, num_items: int,
                 embedding_dim: int = 64, device: str = "cpu") -> None:
        super().__init__(num_users, num_items, embedding_dim, device)
        self.model = None
        self._user_biases: np.ndarray | None = None
        self._item_biases: np.ndarray | None = None
        self._global_mean: float = 0.0

    def fit(self, ratings: pd.DataFrame, user_to_idx: dict, item_to_idx: dict,
            epochs: int = 100, lr: float = 0.01, reg_lambda: float = 0.005) -> dict[str, float]:
        """Train Surprise SVD on ALL ratings (explicit feedback)."""
        user_to_idx, item_to_idx = normalize_maps(user_to_idx, item_to_idx)
        mapped = map_user_item_columns(ratings, user_to_idx, item_to_idx)
        logger.info("Surprise SVD: %d explicit ratings", len(mapped))

        df = pd.DataFrame({
            "user": mapped["user_id"].astype(str),
            "item": mapped["item_id"].astype(str),
            "rating": mapped["rating"].astype(float),
        })

        reader = Reader(rating_scale=(1, 5))
        trainset = Dataset.load_from_df(df, reader).build_full_trainset()

        t0 = time.time()
        self.model = SVD(n_factors=self.embedding_dim, n_epochs=epochs,
                         lr_all=lr, reg_all=reg_lambda, random_state=42,
                         verbose=False, biased=True)
        self.model.fit(trainset)
        train_time = time.time() - t0

        user_emb = np.zeros((self.num_users, self.embedding_dim))
        item_emb = np.zeros((self.num_items, self.embedding_dim))
        self._user_biases = np.zeros(self.num_users)
        self._item_biases = np.zeros(self.num_items)
        self._global_mean = trainset.global_mean

        for inner_uid in range(trainset.n_users):
            idx = user_to_idx.get(str(trainset.to_raw_uid(inner_uid)))
            if idx is not None:
                user_emb[idx] = self.model.pu[inner_uid]
                self._user_biases[idx] = self.model.bu[inner_uid]

        for inner_iid in range(trainset.n_items):
            idx = item_to_idx.get(str(trainset.to_raw_iid(inner_iid)))
            if idx is not None:
                item_emb[idx] = self.model.qi[inner_iid]
                self._item_biases[idx] = self.model.bi[inner_iid]

        self._set_embeddings(user_emb, item_emb)
        logger.info("MF Explicit trained in %.1fs", train_time)
        return {"train_time": train_time}

    def predict(self, user_idx: int) -> np.ndarray:
        """Score = mu + b_u + b_i + q_i^T * p_u (full SVD with biases)."""
        u = self.user_embeddings[user_idx]
        b_u = self._user_biases[user_idx]
        return self._global_mean + b_u + self._item_biases + self.item_embeddings @ u
