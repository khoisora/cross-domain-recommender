"""NCF (NeuMF) using RecBole with full model inference.

Paper: Xiangnan He et al. "Neural Collaborative Filtering", WWW 2017.

Scoring uses the full NeuMF forward pass (GMF + MLP + predict layer),
not just GMF dot product.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import NeuMF as RecBoleNeuMF
from recbole.trainer import Trainer

from .id_utils import map_user_item_columns, normalize_id, normalize_maps

logger = logging.getLogger(__name__)


class NCF:
    """NeuMF wrapper using RecBole with full model inference."""

    def __init__(self, num_users: int, num_items: int,
                 embedding_dim: int = 64, device: str = "cpu") -> None:
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.device = device
        self.model = None
        self.user_embeddings: np.ndarray | None = None
        self.item_embeddings: np.ndarray | None = None
        self.user_to_idx: dict | None = None
        self.item_to_idx: dict | None = None
        self.dataset = None
        self._idx_to_recbole_uid: dict | None = None
        self._idx_to_recbole_iid: dict | None = None
        self._our_item_indices: np.ndarray | None = None
        self._recbole_item_ids: torch.Tensor | None = None

    def fit(self, ratings: pd.DataFrame, user_to_idx: dict, item_to_idx: dict,
            epochs: int = 50, lr: float = 0.001, reg_lambda: float = 0.001,
            batch_size: int = 4096, positive_threshold: float = 3.5) -> dict[str, float]:
        """Train NeuMF using RecBole."""
        t0 = time.time()
        user_to_idx, item_to_idx = normalize_maps(user_to_idx, item_to_idx)
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx

        # Vectorized filter + ID alignment for the RecBole .inter file.
        pos = ratings[ratings["rating"] >= positive_threshold]
        mapped = map_user_item_columns(pos, user_to_idx, item_to_idx)
        inter_df = pd.DataFrame({
            "user_id:token": mapped["user_id"].map(normalize_id),
            "item_id:token": mapped["item_id"].map(normalize_id),
            "rating:float": 1.0,
        })
        logger.info("NCF: %d positive interactions for RecBole", len(inter_df))

        temp_dir = tempfile.mkdtemp()
        ds_name = "ncf_temp"
        ds_dir = os.path.join(temp_dir, ds_name)
        os.makedirs(ds_dir)
        inter_df.to_csv(os.path.join(ds_dir, f"{ds_name}.inter"), sep="\t", index=False)

        config = Config(model="NeuMF", dataset=ds_name, config_dict={
            "model": "NeuMF", "dataset": ds_name, "data_path": temp_dir,
            "load_col": {"inter": ["user_id", "item_id", "rating"]},
            "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id", "RATING_FIELD": "rating",
            "mf_embedding_size": self.embedding_dim, "mlp_embedding_size": self.embedding_dim,
            "mlp_hidden_size": [self.embedding_dim * 2, self.embedding_dim, self.embedding_dim // 2],
            "dropout_prob": 0.2, "mf_train": True, "mlp_train": True,
            "epochs": epochs, "train_batch_size": batch_size,
            "learning_rate": lr, "reg_weight": reg_lambda,
            "eval_args": {"split": {"RS": [0.99, 0.005, 0.005]}, "order": "RO"},
            "metrics": ["Recall", "NDCG"], "topk": [10], "valid_metric": "ndcg@10",
            "eval_step": epochs + 1,
            "user_inter_num_interval": "[1,inf)", "item_inter_num_interval": "[1,inf)",
            "device": self.device, "show_progress": False, "state": "INFO",
        })

        self.dataset = create_dataset(config)
        train_data, valid_data, _ = data_preparation(config, self.dataset)
        self.model = RecBoleNeuMF(config, train_data.dataset).to(config["device"])
        Trainer(config, self.model).fit(train_data, valid_data, show_progress=False, verbose=False)

        self._build_index_maps()
        self._extract_embeddings()

        shutil.rmtree(temp_dir, ignore_errors=True)
        train_time = time.time() - t0
        logger.info("NCF trained in %.1fs", train_time)
        return {"train_time": train_time}

    def _build_index_maps(self):
        """Map our contiguous indices ↔ RecBole internal token IDs (0 = PAD)."""
        def translate(token2id: dict, our_map: dict) -> dict[int, int]:
            return {
                idx: rb_id for token, rb_id in token2id.items()
                if rb_id != 0 and (idx := our_map.get(normalize_id(token))) is not None
            }

        self._idx_to_recbole_uid = translate(self.dataset.field2token_id["user_id"], self.user_to_idx)
        self._idx_to_recbole_iid = translate(self.dataset.field2token_id["item_id"], self.item_to_idx)

        # Pre-sorted arrays for batch scoring: our_item_indices[i] ↔ recbole_item_ids[i].
        sorted_pairs = sorted(self._idx_to_recbole_iid.items())
        self._our_item_indices = np.array([p[0] for p in sorted_pairs])
        self._recbole_item_ids = torch.tensor(
            [p[1] for p in sorted_pairs], dtype=torch.long,
        ).to(next(self.model.parameters()).device)

    def _extract_embeddings(self):
        """Extract GMF-branch embeddings for downstream analysis (not scoring)."""
        self.user_embeddings = np.zeros((self.num_users, self.embedding_dim))
        self.item_embeddings = np.zeros((self.num_items, self.embedding_dim))
        if self.model is None:
            return
        self.model.eval()
        with torch.no_grad():
            u_emb = self.model.user_mf_embedding.weight.cpu().numpy()
            i_emb = self.model.item_mf_embedding.weight.cpu().numpy()

        for idx, rb_id in self._idx_to_recbole_uid.items():
            if rb_id < u_emb.shape[0]:
                self.user_embeddings[idx] = u_emb[rb_id]
        for idx, rb_id in self._idx_to_recbole_iid.items():
            if rb_id < i_emb.shape[0]:
                self.item_embeddings[idx] = i_emb[rb_id]

    def predict(self, user_idx: int) -> np.ndarray:
        """Score using full NeuMF forward pass (GMF + MLP + predict layer)."""
        if self.model is None or self._idx_to_recbole_uid is None:
            return np.zeros(self.num_items)

        rb_uid = self._idx_to_recbole_uid.get(user_idx)
        if rb_uid is None:
            return np.zeros(self.num_items)

        self.model.eval()
        with torch.no_grad():
            device = next(self.model.parameters()).device
            n_items = len(self._recbole_item_ids)
            user_t = torch.full((n_items,), rb_uid, dtype=torch.long, device=device)
            raw_scores = self.model.forward(user_t, self._recbole_item_ids).detach().cpu().numpy()

        scores = np.full(self.num_items, -np.inf, dtype=np.float32)
        scores[self._our_item_indices] = raw_scores.astype(np.float32)
        return scores

    def get_user_embeddings(self) -> np.ndarray:
        return self.user_embeddings if self.user_embeddings is not None else np.zeros((self.num_users, self.embedding_dim))

    def get_item_embeddings(self) -> np.ndarray:
        return self.item_embeddings if self.item_embeddings is not None else np.zeros((self.num_items, self.embedding_dim))
