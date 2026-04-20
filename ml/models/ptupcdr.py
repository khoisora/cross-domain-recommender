"""PTUPCDR – Personalized Transfer of User Preferences for CDR.

Paper: Yiqing Wu et al. "Personalized Transfer of User Preferences for
       Cross-domain Recommendation", WSDM 2022.

Key idea vs EMCDR:
  EMCDR:   mapping = global_MLP(source_user_embedding[u])
  PTUPCDR: mapping = hypernetwork(preference_u)
             where preference_u = mean(source_item_emb[user u's rated movies])

For few-shot users (k≥1 game interactions):
  final_emb = (1/(1+k)) * mapped_pref + (k/(1+k)) * target_emb

Uses vendored recbole-cdr at ml/models/recbole_cdr/.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ._cdr_base import fit_cdr
from .id_utils import normalize_id

logger = logging.getLogger(__name__)


class _HyperMapper(nn.Module):
    """Mixture-of-experts mapping: preference → target embedding space.

    K expert matrices W_k combined via user-specific gate weights from preference.
    Output = (sum_k gate_k * W_k) @ pref + bias.
    """

    def __init__(self, d: int, n_experts: int = 8) -> None:
        super().__init__()
        # Gate network: preference vector → softmax weights over K experts
        self.gate = nn.Sequential(nn.Linear(d, d * 2), nn.ReLU(), nn.Linear(d * 2, n_experts))
        # K expert linear transforms, each d×d. Scaled init prevents exploding outputs.
        self.experts = nn.Parameter(torch.randn(n_experts, d, d) * (1.0 / d ** 0.5))
        # Residual bias path for identity-like initialization
        self.bias = nn.Linear(d, d, bias=True)

    def forward(self, pref: torch.Tensor) -> torch.Tensor:
        # Soft mixture-of-experts: each user gets a personalized linear transform
        # F = sum_k(gate_k * W_k), then output = F @ pref + bias(pref)
        gate_w = torch.softmax(self.gate(pref), dim=-1)           # (B, K)
        F = torch.einsum("bk,kij->bij", gate_w, self.experts)     # (B, d, d)
        mapped = torch.bmm(F, pref.unsqueeze(-1)).squeeze(-1)      # (B, d)
        return mapped + self.bias(pref)


class PTUPCDRWrapper:
    """PTUPCDR wrapper. Source = movies, target = games."""

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

    def fit(self, ratings, user_to_idx: dict, item_to_idx: dict,
            epochs: int = 20, lr: float = 0.001, reg_lambda: float = 1e-4,
            batch_size: int = 4096, positive_threshold: float = 4.0,
            source_domain: str = "movie", target_domain: str = "game",
            meta_epochs: int = 30, meta_lr: float = 0.001, n_experts: int = 8) -> dict[str, float]:
        t0 = time.time()

        # Phase 1+2: Train source-domain MF (movies) and target-domain MF (games)
        # separately. We reuse EMCDR's training pipeline but skip its OVERLAP
        # mapping phase — PTUPCDR replaces it with the HyperMapper below.
        extra: dict[str, Any] = {
            "train_epochs": [f"SOURCE:{epochs}", f"TARGET:{epochs}"],
            "latent_factor_model": "MF",
            "source_embedding_size": self.embedding_dim,
            "target_embedding_size": self.embedding_dim,
            "reg_weight": reg_lambda,
            "mapping_function": "mlp",
            "mlp_hidden_size": [self.embedding_dim],
        }

        self._model, self._rb_users, self._rb_items = fit_cdr(
            "EMCDR", extra,
            ratings, user_to_idx, item_to_idx,
            epochs, lr, reg_lambda, batch_size, positive_threshold,
            source_domain=source_domain, target_domain=target_domain,
        )
        logger.info("PTUPCDR: source+target MF done in %.1fs", time.time() - t0)

        # Phase 3: Compute per-user preference = mean(source_item_emb[rated movies])
        model = self._model
        model.eval()
        with torch.no_grad():
            src_item_emb_all = model.source_item_embedding.weight.cpu()
            src_user_emb_all = model.source_user_embedding.weight.cpu()
            tgt_user_emb_all = model.target_user_embedding.weight.cpu()
            tgt_item_emb_all = model.target_item_embedding.weight.cpu()

        d = self.embedding_dim
        overlap_n = int(model.overlapped_num_users)
        target_n = int(model.target_num_users)

        # Map source item embeddings from RecBole's internal index space to our
        # contiguous index space. rb_i=0 is RecBole's PAD token (unmapped items).
        src_item_np = src_item_emb_all.detach().numpy()
        src_item_our = np.zeros((len(item_to_idx), d), dtype=np.float32)
        i_valid = (self._rb_items > 0) & (self._rb_items < src_item_np.shape[0])
        src_item_our[i_valid] = src_item_np[self._rb_items[i_valid]]

        # Per-user preference aggregation over source items
        src_ratings = ratings[ratings["domain"] == source_domain].copy()
        src_ratings["_uid"] = src_ratings["user_id"].map(lambda x: user_to_idx.get(normalize_id(x)))
        src_ratings["_iid"] = src_ratings["item_id"].map(lambda x: item_to_idx.get(normalize_id(x)))
        src_ratings = src_ratings.dropna(subset=["_uid", "_iid"])
        src_ratings[["_uid", "_iid"]] = src_ratings[["_uid", "_iid"]].astype(int)

        # Game interaction counts per user for blend weight
        tgt_uid = ratings[ratings["domain"] == target_domain]["user_id"].map(
            lambda x: user_to_idx.get(normalize_id(x)))
        tgt_uid = tgt_uid.dropna().astype(int)
        game_counts = tgt_uid.value_counts().to_dict()

        # Build per-user preference vector = mean of source item embeddings for
        # movies the user rated. This captures the user's movie taste as a dense
        # vector in source embedding space. Falls back to the source user embedding
        # if no valid item embeddings exist (e.g., all items were unmapped).
        n_users = len(user_to_idx)
        pref_np = np.zeros((n_users, d), dtype=np.float32)
        for uid_idx, grp in src_ratings.groupby("_uid"):
            embs = src_item_our[grp["_iid"].values]
            valid = np.linalg.norm(embs, axis=1) > 1e-8
            if valid.any():
                pref_np[uid_idx] = embs[valid].mean(axis=0)
            elif self._rb_users[uid_idx] > 0:
                pref_np[uid_idx] = src_user_emb_all[self._rb_users[uid_idx]].detach().numpy()

        logger.info("PTUPCDR: preferences computed (%.1fs)", time.time() - t0)

        # Phase 4: Train HyperMapper on overlap users (users with both movie and
        # game data). The mapper learns to transform source preferences into target
        # embeddings. RecBole assigns IDs [0, overlap_n) to overlap users.
        overlap_mask = (self._rb_users > 0) & (self._rb_users < overlap_n)
        overlap_idx = np.where(overlap_mask)[0]

        mapper = _HyperMapper(d=d, n_experts=n_experts)
        if len(overlap_idx) > 0:
            pref_t = torch.tensor(pref_np[overlap_idx], dtype=torch.float32)
            tgt_emb_t = torch.tensor(
                tgt_user_emb_all[self._rb_users[overlap_idx]].detach().numpy(), dtype=torch.float32
            )
            optimizer = torch.optim.Adam(mapper.parameters(), lr=meta_lr, weight_decay=1e-5)
            bs = min(512, len(overlap_idx))

            for ep in range(meta_epochs):
                perm = torch.randperm(len(overlap_idx))
                total_loss, n_b = 0.0, 0
                for start in range(0, len(overlap_idx), bs):
                    idx = perm[start:start + bs]
                    loss = nn.functional.mse_loss(mapper(pref_t[idx]), tgt_emb_t[idx])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    n_b += 1
                if (ep + 1) % 10 == 0:
                    logger.info("  HyperMapper epoch %d/%d  loss=%.4f", ep + 1, meta_epochs, total_loss / max(n_b, 1))
        else:
            logger.warning("No overlap users — PTUPCDR degrades to preference-mean scoring")

        # Phase 5: Build final user embeddings with blending.
        # Run all user preferences through the trained HyperMapper in chunks
        # to avoid OOM on large user sets.
        mapper.eval()
        with torch.no_grad():
            all_pref = torch.tensor(pref_np, dtype=torch.float32)
            mapped_chunks = []
            for start in range(0, n_users, 1024):
                mapped_chunks.append(mapper(all_pref[start:start + 1024]).detach().numpy())
            mapped_np = np.concatenate(mapped_chunks, axis=0)

        # Blend mapped source preference with target-domain MF embedding.
        # Weight w = 1/(1+k) where k = number of game interactions:
        #   k=0 (cold-start): 100% mapped preference (full transfer from movies)
        #   k=1: 50/50 blend
        #   k→∞: converges to target MF embedding (movies become irrelevant)
        tgt_user_np = tgt_user_emb_all.detach().numpy()
        rb_u_arr = self._rb_users
        k_arr = np.zeros(n_users, dtype=np.float32)
        for u_idx, c in game_counts.items():
            k_arr[u_idx] = c

        has_user = rb_u_arr > 0
        blend_mask = has_user & (k_arr > 0) & (rb_u_arr < target_n)
        mapped_only_mask = has_user & ~blend_mask

        final_user_emb = np.zeros((n_users, d), dtype=np.float32)
        final_user_emb[mapped_only_mask] = mapped_np[mapped_only_mask]
        w = (1.0 / (1.0 + k_arr[blend_mask]))[:, None]
        final_user_emb[blend_mask] = (
            w * mapped_np[blend_mask]
            + (1 - w) * tgt_user_np[rb_u_arr[blend_mask]]
        )

        # Target item embeddings (source items aren't scored)
        tgt_item_np = tgt_item_emb_all.detach().numpy()
        valid_items = (self._rb_items > 0) & (self._rb_items < target_n)
        final_item_emb = np.zeros((len(item_to_idx), d), dtype=np.float32)
        final_item_emb[valid_items] = tgt_item_np[self._rb_items[valid_items]]

        self.user_embeddings = final_user_emb
        self.item_embeddings = final_item_emb
        self._valid_items = valid_items
        # A user is valid for prediction only if they have a RecBole mapping AND
        # a non-zero preference vector (i.e., they rated at least one source item)
        pref_norms = np.linalg.norm(pref_np, axis=1)
        self._valid_users = (self._rb_users != 0) & (pref_norms > 1e-8)

        train_time = time.time() - t0
        logger.info("PTUPCDR ready: %d/%d users, %d/%d items valid, %.1fs",
                    self._valid_users.sum(), n_users, valid_items.sum(), len(item_to_idx), train_time)
        return {"train_time": train_time}

    def predict(self, user_idx: int) -> np.ndarray:
        if self.user_embeddings is None:
            raise ValueError("Model not trained yet")
        if not self._valid_users[user_idx]:
            return np.full(self.num_items, -np.inf, dtype=np.float64)
        user_emb = self.user_embeddings[user_idx]
        scores = (self.item_embeddings @ user_emb).astype(np.float64)
        scores[~self._valid_items] = -np.inf
        return scores

    def get_user_embeddings(self) -> np.ndarray:
        return self.user_embeddings

    def get_item_embeddings(self) -> np.ndarray:
        return self.item_embeddings
