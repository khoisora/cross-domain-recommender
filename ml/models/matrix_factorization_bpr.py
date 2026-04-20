"""Matrix Factorization with BPR (Bayesian Personalized Ranking).

BPR optimizes pairwise ranking: observed items should score higher than
unobserved items. Loss = -log(sigmoid(pos_score - neg_score)).
Much stronger than explicit MF for top-K recommendation tasks.

Uses vectorized mini-batch SGD for efficiency on large datasets.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from .base_recommender import BaseRecommender
from .id_utils import normalize_maps, map_user_item_columns

logger = logging.getLogger(__name__)


class MatrixFactorizationBPR(BaseRecommender):
    """MF with BPR pairwise loss on implicit feedback."""

    def fit(self, ratings: pd.DataFrame, user_to_idx: dict, item_to_idx: dict,
            epochs: int = 100, lr: float = 0.01, reg_lambda: float = 0.01,
            positive_threshold: float = 4.0, batch_size: int = 4096) -> dict[str, float]:
        """Train BPR on positive interactions (rating >= threshold).

        Uses vectorized mini-batch updates: each batch samples one negative
        per positive pair and updates all embeddings in parallel via np.add.at.
        """
        user_to_idx, item_to_idx = normalize_maps(user_to_idx, item_to_idx)

        positive = ratings[ratings["rating"] >= positive_threshold]
        mapped = map_user_item_columns(positive, user_to_idx, item_to_idx)
        logger.info("BPR: %d positives (threshold=%s), %d after ID alignment",
                     len(positive), positive_threshold, len(mapped))

        # Flatten to numpy arrays for fast vectorized mini-batch access
        user_arr = mapped["_uidx"].astype(np.int64).values
        pos_arr = mapped["_iidx"].astype(np.int64).values

        # Per-user positive item sets — negatives are rejection-sampled against these.
        user_items: dict[int, set[int]] = {}
        for u, i in zip(user_arr.tolist(), pos_arr.tolist()):
            user_items.setdefault(u, set()).add(i)

        np.random.seed(42)
        user_emb = np.random.normal(0, 0.01, (self.num_users, self.embedding_dim))
        item_emb = np.random.normal(0, 0.01, (self.num_items, self.embedding_dim))

        n_pairs = len(user_arr)
        logger.info("BPR training: %d pairs, %d epochs, batch=%d", n_pairs, epochs, batch_size)
        t0 = time.time()

        for epoch in range(epochs):
            perm = np.random.permutation(n_pairs)
            epoch_loss = 0.0

            for start in range(0, n_pairs, batch_size):
                end = min(start + batch_size, n_pairs)
                idx = perm[start:end]
                b_users = user_arr[idx]
                b_pos = pos_arr[idx]

                # Sample negatives: start with uniform random, then reject any
                # that collide with the user's known positives. The inner while
                # loop rarely iterates because positives are sparse (<1% of items).
                b_neg = np.random.randint(0, self.num_items, size=len(idx))
                for j in range(len(idx)):
                    while b_neg[j] in user_items.get(int(b_users[j]), set()):
                        b_neg[j] = np.random.randint(0, self.num_items)

                # Vectorized BPR forward pass:
                # score_diff = dot(u, pos) - dot(u, neg) — positive items should
                # score higher than negatives. BPR loss = -log(sigmoid(diff)).
                u_emb = user_emb[b_users]       # (B, D)
                p_emb = item_emb[b_pos]          # (B, D)
                n_emb = item_emb[b_neg]          # (B, D)

                diff = np.sum(u_emb * (p_emb - n_emb), axis=1)  # (B,)
                sig = 1.0 / (1.0 + np.exp(-np.clip(diff, -30, 30)))
                g = (sig - 1.0)[:, np.newaxis]  # (B, 1) — gradient scale factor

                epoch_loss -= np.sum(np.log(sig + 1e-10))

                # SGD update with L2 regularization. np.add.at handles duplicate
                # user/item indices in the same batch (scatter-add semantics).
                u_grad = g * (p_emb - n_emb) + 2 * reg_lambda * u_emb
                p_grad = g * u_emb + 2 * reg_lambda * p_emb
                n_grad = -g * u_emb + 2 * reg_lambda * n_emb

                np.add.at(user_emb, b_users, -lr * u_grad)
                np.add.at(item_emb, b_pos, -lr * p_grad)
                np.add.at(item_emb, b_neg, -lr * n_grad)

            if (epoch + 1) % 10 == 0:
                logger.info("BPR epoch %d/%d, avg loss: %.4f",
                           epoch + 1, epochs, epoch_loss / n_pairs)

        train_time = time.time() - t0
        self._set_embeddings(user_emb, item_emb)
        logger.info("MF BPR trained in %.1fs", train_time)
        return {"final_loss": epoch_loss / n_pairs, "train_time": train_time}
