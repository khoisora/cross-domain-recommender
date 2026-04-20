"""Abstract base class for recommenders."""

from __future__ import annotations

import abc

import numpy as np


class BaseRecommender(abc.ABC):
    """Common interface: fit → predict → get_*_embeddings."""

    def __init__(self, num_users: int, num_items: int,
                 embedding_dim: int = 64, device: str = "cpu") -> None:
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.device = device
        self.user_embeddings: np.ndarray | None = None
        self.item_embeddings: np.ndarray | None = None

    @abc.abstractmethod
    def fit(self, ratings, user_to_idx, item_to_idx, **kwargs) -> dict[str, float]:
        ...

    def predict(self, user_idx: int) -> np.ndarray:
        if self.user_embeddings is None or self.item_embeddings is None:
            raise ValueError("Model not trained yet")
        return self.item_embeddings @ self.user_embeddings[user_idx]

    def get_user_embeddings(self) -> np.ndarray:
        if self.user_embeddings is None:
            raise ValueError("Model not trained yet")
        return self.user_embeddings

    def get_item_embeddings(self) -> np.ndarray:
        if self.item_embeddings is None:
            raise ValueError("Model not trained yet")
        return self.item_embeddings

    def _set_embeddings(self, user_emb: np.ndarray, item_emb: np.ndarray) -> None:
        self.user_embeddings = user_emb
        self.item_embeddings = item_emb
