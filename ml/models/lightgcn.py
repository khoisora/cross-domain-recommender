"""LightGCN recommendation model using PyTorch Geometric.

Paper: Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution
       Network for Recommendation", SIGIR 2020.

Uses BPR pairwise loss with popularity-biased negative sampling.
Supports early stopping via validation metric callback.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from torch_geometric.nn.models import LightGCN as _LightGCNModel

from .id_utils import normalize_id, normalize_maps

logger = logging.getLogger(__name__)


class LightGCN:
    """LightGCN wrapper backed by PyTorch Geometric's graph-convolution model."""

    def __init__(self, num_users: int, num_items: int,
                 embedding_dim: int = 64, num_layers: int = 3,
                 device: str = "cpu", dropout: float = 0.0) -> None:
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.device = device
        self.dropout = dropout
        self._model: _LightGCNModel | None = None
        self._edge_index: torch.Tensor | None = None
        self._cached_all_emb: torch.Tensor | None = None

    def fit(self, ratings: pd.DataFrame, user_to_idx: dict, item_to_idx: dict,
            epochs: int = 100, lr: float = 0.001, reg_lambda: float = 1e-4,
            batch_size: int = 2048, positive_threshold: float = 3.5,
            neg_sampling: str = "popularity", neg_popularity_alpha: float = 0.75,
            neg_item_indices: np.ndarray | None = None,
            early_stopping_patience: int = 10,
            val_metric_fn: Callable | None = None,
            val_every: int = 5) -> dict[str, float]:
        """Train LightGCN with BPR loss and optional validation-based early stopping."""
        user_to_idx, item_to_idx = normalize_maps(user_to_idx, item_to_idx)
        self._cached_all_emb = None

        pos = ratings[ratings["rating"] >= positive_threshold].copy()
        logger.info("LightGCN: %d positive interactions", len(pos))

        uid_arr = np.array([user_to_idx[normalize_id(r)] for r in pos["user_id"]], dtype=np.int64)
        iid_arr = np.array([item_to_idx[normalize_id(r)] for r in pos["item_id"]], dtype=np.int64)

        # Build bipartite graph: user nodes [0, num_users), item nodes [num_users, num_users+num_items)
        item_offset = self.num_users
        src = np.concatenate([uid_arr, iid_arr + item_offset])
        dst = np.concatenate([iid_arr + item_offset, uid_arr])
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=self.device)
        self._edge_index = edge_index

        model = _LightGCNModel(
            num_nodes=self.num_users + self.num_items,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        with torch.no_grad():
            torch.nn.init.normal_(model.embedding.weight, std=0.1)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        num_pos = len(uid_arr)
        uid_t = torch.tensor(uid_arr, dtype=torch.long, device=self.device)
        iid_t = torch.tensor(iid_arr, dtype=torch.long, device=self.device)

        # Per-user positive items for collision-free negative sampling
        user_pos: dict[int, set[int]] = {}
        for u, i in zip(uid_arr.tolist(), iid_arr.tolist()):
            user_pos.setdefault(u, set()).add(i)

        # Popularity-biased negative sampling distribution.
        # Items seen more often are sampled more as negatives — this produces
        # harder negatives than uniform sampling, improving ranking quality.
        # Alpha < 1 smooths the distribution (0.75 = sublinear popularity).
        pop_t = None
        if neg_sampling == "popularity":
            pop = np.bincount(iid_arr, minlength=self.num_items).astype(np.float64)
            pop = np.maximum(pop, 1.0) ** neg_popularity_alpha
            # Zero out non-target items so negatives come only from the target domain
            if neg_item_indices is not None:
                mask = np.zeros(self.num_items, dtype=bool)
                mask[neg_item_indices] = True
                pop[~mask] = 0.0
            pop /= pop.sum()
            pop_t = torch.tensor(pop, dtype=torch.float32, device=self.device)

        t0 = time.time()
        best_score = -float("inf")
        best_loss = float("inf")
        patience_counter = 0
        best_state = None
        use_val = val_metric_fn is not None
        self._model = model

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            perm = torch.randperm(num_pos, device=self.device)

            for start in range(0, num_pos, batch_size):
                end = min(start + batch_size, num_pos)
                idx = perm[start:end]
                b = end - start

                batch_uid = uid_t[idx]
                batch_iid = iid_t[idx]

                # Negative sampling
                if pop_t is not None:
                    batch_neg = torch.multinomial(pop_t, num_samples=b, replacement=True)
                else:
                    batch_neg = torch.randint(0, self.num_items, (b,), device=self.device)

                # Fix collisions (avoid sampling known positives)
                for _ in range(10):
                    neg_cpu = batch_neg.cpu().numpy()
                    uid_cpu = batch_uid.cpu().numpy()
                    collisions = [j for j in range(b) if int(neg_cpu[j]) in user_pos.get(int(uid_cpu[j]), ())]
                    if not collisions:
                        break
                    if pop_t is not None:
                        repl = torch.multinomial(pop_t, num_samples=len(collisions), replacement=True)
                    else:
                        repl = torch.randint(0, self.num_items, (len(collisions),), device=self.device)
                    batch_neg[torch.tensor(collisions, device=self.device)] = repl

                optimizer.zero_grad()
                all_emb = model.get_embedding(edge_index)

                u_emb = all_emb[batch_uid]
                pi_emb = all_emb[item_offset + batch_iid]
                ni_emb = all_emb[item_offset + batch_neg]

                # BPR loss via softplus: log(1 + exp(neg_score - pos_score)).
                # Equivalent to -log(sigmoid(pos - neg)) but numerically stable.
                bpr_loss = torch.nn.functional.softplus(
                    (u_emb * ni_emb).sum(-1) - (u_emb * pi_emb).sum(-1)
                ).mean()
                # L2 regularization only on embeddings used in this batch
                # (not all embeddings) to keep gradients sparse
                unique_nodes = torch.cat([batch_uid, item_offset + batch_iid, item_offset + batch_neg]).unique()
                reg_loss = reg_lambda * model.embedding.weight[unique_nodes].pow(2).sum() / unique_nodes.size(0)

                loss = bpr_loss + reg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                self._cached_all_emb = None

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # Early stopping — when val_metric_fn is provided, ONLY validation
            # controls patience and best_state. Loss-based fallback is only used
            # when no validation function is given.
            if use_val:
                if epoch % val_every == 0 or epoch == 1:
                    model.eval()
                    self._cached_all_emb = None
                    score = val_metric_fn(self)
                    if score > best_score + 1e-6:
                        best_score = score
                        patience_counter = 0
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    else:
                        patience_counter += 1
                    logger.info("LightGCN epoch %d/%d  loss=%.6f  val=%.4f  best=%.4f  p=%d/%d",
                               epoch, epochs, avg_loss, score, best_score, patience_counter, early_stopping_patience)
                elif epoch % 5 == 0:
                    logger.info("LightGCN epoch %d/%d  loss=%.6f  (no val this epoch)", epoch, epochs, avg_loss)
            else:
                if avg_loss < best_loss - 1e-6:
                    best_loss = avg_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                if epoch % 5 == 0:
                    logger.info("LightGCN epoch %d/%d  loss=%.6f  best=%.6f  p=%d/%d",
                               epoch, epochs, avg_loss, best_loss, patience_counter, early_stopping_patience)

            if patience_counter >= early_stopping_patience:
                logger.info("LightGCN early stopping at epoch %d", epoch)
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        self._model = model
        logger.info("LightGCN trained in %.1fs", time.time() - t0)
        return {"final_loss": avg_loss, "train_time": time.time() - t0}

    def predict(self, user_idx: int) -> np.ndarray:
        if self._model is None:
            return np.zeros(self.num_items)
        self._model.eval()
        with torch.no_grad():
            # Cache the full GCN embedding (users + items) across predict() calls.
            # Invalidated on each training step via self._cached_all_emb = None.
            if self._cached_all_emb is None:
                self._cached_all_emb = self._model.get_embedding(self._edge_index)
            all_emb = self._cached_all_emb
            u_emb = all_emb[user_idx]
            # Item nodes start at offset num_users in the bipartite graph
            return (all_emb[self.num_users:] @ u_emb).cpu().numpy()

    def get_user_embeddings(self) -> np.ndarray:
        if self._model is None:
            return np.zeros((self.num_users, self.embedding_dim))
        self._model.eval()
        with torch.no_grad():
            return self._model.get_embedding(self._edge_index)[:self.num_users].cpu().numpy()

    def get_item_embeddings(self) -> np.ndarray:
        if self._model is None:
            return np.zeros((self.num_items, self.embedding_dim))
        self._model.eval()
        with torch.no_grad():
            return self._model.get_embedding(self._edge_index)[self.num_users:].cpu().numpy()
