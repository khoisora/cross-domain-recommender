"""PyTorch Dataset for BPR / pointwise training with negative sampling."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class BPRDataset(Dataset):
    """Dataset for BPR pairwise training: (user, pos_item, neg_item).

    Each __getitem__ returns a positive interaction and one random negative.
    """

    def __init__(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        num_items: int,
        positive_threshold: float = 4.0,
        ratings: np.ndarray | None = None,
    ) -> None:
        self.user_ids = user_ids.astype(np.int64)
        self.item_ids = item_ids.astype(np.int64)
        self.num_items = num_items

        # Build per-user positive item sets for negative sampling
        self._user_positives: dict[int, set[int]] = {}
        if ratings is not None:
            for uid, iid, r in zip(self.user_ids, self.item_ids, ratings):
                if r >= positive_threshold:
                    self._user_positives.setdefault(int(uid), set()).add(int(iid))
        else:
            # All interactions are implicit positives
            for uid, iid in zip(self.user_ids, self.item_ids):
                self._user_positives.setdefault(int(uid), set()).add(int(iid))

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user = self.user_ids[idx]
        pos_item = self.item_ids[idx]

        # Sample one negative item
        user_pos = self._user_positives.get(int(user), set())
        neg = np.random.randint(0, self.num_items)
        while neg in user_pos:
            neg = np.random.randint(0, self.num_items)

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )
