"""PyTorch Dataset utilities for training ML models."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ratingDataset(Dataset):
    """PyTorch Dataset for user-item ratings.

    Used by the two-tower model and matrix factorization training loops.
    Supports negative sampling.
    """

    def __init__(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        num_items: int,
        num_negatives: int = 5,
        positive_threshold: float = 3.5,
    ) -> None:
        """
        Args:
            user_ids: Array of user indices.
            item_ids: Array of item indices.
            ratings: Array of rating values.
            num_items: Total number of items (for negative sampling).
            num_negatives: Number of negative samples per positive.
            positive_threshold: Rating threshold for positive signal.
        """
        self.user_ids = user_ids.astype(np.int64)
        self.item_ids = item_ids.astype(np.int64)
        self.ratings = ratings.astype(np.float32)
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.positive_threshold = positive_threshold

        # Build per-user positive item sets for efficient negative sampling
        self._user_positives: dict[int, set[int]] = {}
        for uid, iid, r in zip(self.user_ids, self.item_ids, self.ratings):
            if r >= positive_threshold:
                self._user_positives.setdefault(int(uid), set()).add(int(iid))

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        user = self.user_ids[idx]
        pos_item = self.item_ids[idx]
        rating = self.ratings[idx]

        # Sample negative items
        user_pos = self._user_positives.get(int(user), set())
        neg_items = []
        attempts = 0
        while len(neg_items) < self.num_negatives and attempts < self.num_negatives * 10:
            neg = np.random.randint(0, self.num_items)
            if neg not in user_pos:
                neg_items.append(neg)
            attempts += 1
        # Pad if we couldn't find enough negatives
        while len(neg_items) < self.num_negatives:
            neg_items.append(np.random.randint(0, self.num_items))

        return {
            "user_id": torch.tensor(user, dtype=torch.long),
            "pos_item_id": torch.tensor(pos_item, dtype=torch.long),
            "neg_item_ids": torch.tensor(neg_items, dtype=torch.long),
            "rating": torch.tensor(rating, dtype=torch.float),
            "label": torch.tensor(
                1.0 if rating >= self.positive_threshold else 0.0,
                dtype=torch.float,
            ),
        }


class EmbeddingLookupDataset(Dataset):
    """Dataset that provides precomputed embeddings for user/item pairs.

    Used when training the two-tower model on top of GraphSAGE embeddings.
    """

    def __init__(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        labels: np.ndarray,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
    ) -> None:
        self.user_ids = user_ids.astype(np.int64)
        self.item_ids = item_ids.astype(np.int64)
        self.labels = labels.astype(np.float32)
        self.user_embeddings = user_embeddings.astype(np.float32)
        self.item_embeddings = item_embeddings.astype(np.float32)

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        uid = self.user_ids[idx]
        iid = self.item_ids[idx]
        return {
            "user_embedding": torch.tensor(self.user_embeddings[uid]),
            "item_embedding": torch.tensor(self.item_embeddings[iid]),
            "label": torch.tensor(self.labels[idx]),
        }
