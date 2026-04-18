"""EMCDR – Embedding and Mapping CDR (Man et al., IJCAI 2017).

Three training phases:
  SOURCE  – train source MF on movie interactions
  TARGET  – train target MF on overlap-user game interactions
  OVERLAP – train mapping MLP (source_emb → target_emb) on overlap users

At inference:
  overlap users → mapping(source_user_emb)
  target-only   → target_user_emb
  source-only   → mapping(source_user_emb) (cold-start)

Uses vendored recbole-cdr at ml/models/recbole_cdr/.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch

from ._cdr_base import fit_cdr

logger = logging.getLogger(__name__)


class EMCDRWrapper:
    """EMCDR wrapper. Source = movies, target = games."""

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
            epochs: int = 20, lr: float = 0.001, reg_lambda: float = 1e-4,
            batch_size: int = 4096, positive_threshold: float = 4.0,
            source_domain: str = "movie", target_domain: str = "game",
            model_hyperparams: dict[str, Any] | None = None) -> dict[str, float]:
        t0 = time.time()
        extra: dict[str, Any] = {
            "train_epochs": ["SOURCE:20", "TARGET:20", "OVERLAP:10"],
            "latent_factor_model": "MF",
            "source_embedding_size": self.embedding_dim,
            "target_embedding_size": self.embedding_dim,
            "reg_weight": reg_lambda,
            "mapping_function": "mlp",
            "mlp_hidden_size": [self.embedding_dim],
        }
        if model_hyperparams:
            extra.update(model_hyperparams)

        self._model, self._rb_users, self._rb_items = fit_cdr(
            "EMCDR", extra,
            ratings, user_to_idx, item_to_idx,
            epochs, lr, reg_lambda, batch_size, positive_threshold,
            source_domain=source_domain, target_domain=target_domain,
        )
        self._model.eval()
        self._precompute_embeddings()

        train_time = time.time() - t0
        logger.info("EMCDR trained in %.1fs", train_time)
        return {"train_time": train_time}

    def _precompute_embeddings(self) -> None:
        """Pre-compute user/item embeddings mapped to our index space."""
        model = self._model
        overlap_n = model.overlapped_num_users
        target_n = model.target_num_users

        with torch.no_grad():
            src_emb = model.source_user_embedding.weight.cpu()
            tgt_emb = model.target_user_embedding.weight.cpu()
            # Apply the learned MLP mapping to project ALL source embeddings
            # into the target space
            mapped = model.mapping(src_emb)

            # RecBole user ID layout:
            #   [0, overlap_n)      → overlap users: use mapped source embedding
            #   [overlap_n, target_n) → target-only users: use target MF embedding
            #   [target_n, ...)      → source-only users: use mapped source embedding
            all_user_emb = mapped.clone()
            all_user_emb[overlap_n:target_n] = tgt_emb[overlap_n:target_n]
            tgt_item_emb = model.target_item_embedding.weight[:target_n].cpu()

        all_user_np = all_user_emb.detach().numpy()
        tgt_item_np = tgt_item_emb.detach().numpy()

        # Scatter RecBole embeddings into our contiguous index space.
        # rb_u/rb_i = 0 means PAD (item/user unknown to RecBole), skip those.
        emb_dim = all_user_np.shape[1]
        self.user_embeddings = np.zeros((len(self._rb_users), emb_dim), dtype=np.float32)
        self.item_embeddings = np.zeros((len(self._rb_items), emb_dim), dtype=np.float32)

        for our_idx, rb_u in enumerate(self._rb_users):
            if rb_u != 0:
                self.user_embeddings[our_idx] = all_user_np[rb_u]

        # Only target-domain items have embeddings (source items aren't scored)
        target_n_int = int(target_n)
        for our_idx, rb_i in enumerate(self._rb_items):
            if rb_i != 0 and rb_i < target_n_int:
                self.item_embeddings[our_idx] = tgt_item_np[rb_i]

        self._valid_items = self._rb_items != 0
        self._valid_users = self._rb_users != 0
        logger.info("EMCDR: %d/%d users valid, %d/%d items valid",
                    self._valid_users.sum(), len(self._rb_users),
                    self._valid_items.sum(), len(self._rb_items))

    def predict(self, user_idx: int, item_indices: np.ndarray | None = None) -> np.ndarray:
        if self.user_embeddings is None:
            raise ValueError("Model not trained yet")
        # Invalid users (unmapped by RecBole) get -inf scores everywhere so
        # they never contaminate evaluation metrics
        if not self._valid_users[user_idx]:
            n = self.num_items if item_indices is None else len(item_indices)
            return np.full(n, -np.inf, dtype=np.float64)
        user_emb = self.user_embeddings[user_idx]
        items = self.item_embeddings if item_indices is None else self.item_embeddings[item_indices]
        scores = (items @ user_emb).astype(np.float64)
        # Mask items that RecBole didn't learn embeddings for (zero-vector items)
        valid = self._valid_items if item_indices is None else self._valid_items[item_indices]
        scores[~valid] = -np.inf
        return scores

    def get_user_embeddings(self) -> np.ndarray:
        return self.user_embeddings

    def get_item_embeddings(self) -> np.ndarray:
        return self.item_embeddings
