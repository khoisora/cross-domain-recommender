"""SBERT (Sentence-BERT) content-based item embedding model.

Encodes item metadata (title, domain, categories, description) using a
pre-trained sentence-transformer to produce dense semantic embeddings.
Similarity is cosine similarity (dot product of L2-normalised vectors).

User embeddings are built as the L2-normalised mean of their positively
rated items' embeddings — no model training required.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _parse_text_field(value: str | None, max_items: int = 8) -> str:
    """Parse a JSON list or plain comma string into a short readable string."""
    if not value:
        return ""
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return ", ".join(str(x).strip() for x in parsed[:max_items] if x)
    except (json.JSONDecodeError, TypeError):
        pass
    return str(value).strip()


class SBERTModel:
    """Content-based similarity via Sentence-BERT embeddings.

    All embeddings are L2-normalised so cosine similarity reduces to dot product.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._encoder = None
        self.item_embeddings: np.ndarray | None = None
        self.user_embeddings: np.ndarray | None = None
        self.num_items: int = 0
        self.num_users: int = 0
        self.embedding_dim: int = 384

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SBERT model '%s'…", self.model_name)
            self._encoder = SentenceTransformer(self.model_name)
            self.embedding_dim = self._encoder.get_sentence_embedding_dimension()
            logger.info("SBERT embedding dim: %d", self.embedding_dim)
        return self._encoder

    def _build_item_text(self, row: dict) -> str:
        domain = (row.get("domain") or "").strip().capitalize()
        title  = (row.get("title")  or "").strip()
        cats   = _parse_text_field(row.get("categories"), max_items=6)
        feats  = _parse_text_field(row.get("features"), max_items=4)
        desc   = (row.get("description") or "").strip()[:300]

        parts: list[str] = []
        if domain:
            parts.append(f"{domain}:")
        if title:
            parts.append(f"{title}.")
        if cats:
            parts.append(f"Genres: {cats}.")
        if feats:
            parts.append(f"Tags: {feats}.")
        if desc:
            parts.append(desc)
        return " ".join(parts) or title

    def encode_items(
        self,
        items_df: pd.DataFrame,
        item_to_idx: dict[str, int],
        batch_size: int = 64,
    ) -> None:
        """Encode item metadata → L2-normalised item embedding matrix."""
        encoder = self._get_encoder()
        num_items = len(item_to_idx)
        self.num_items = num_items

        ext_ids = items_df["external_id"].astype(str)
        idxs = ext_ids.map(item_to_idx)
        matched = items_df[idxs.notna()]
        matched_idxs = idxs.dropna().astype(int).values

        texts: list[str] = [""] * num_items
        for pos, (_, row) in zip(matched_idxs, matched.iterrows()):
            texts[pos] = self._build_item_text(row.to_dict())
        n_matched = len(matched_idxs)

        logger.info("Encoding %d/%d items with SBERT '%s' (batch=%d)…",
                    n_matched, num_items, self.model_name, batch_size)
        embs = encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self.item_embeddings = embs.astype(np.float32)
        logger.info("Item embeddings: %s", self.item_embeddings.shape)

    def compute_user_embeddings(
        self,
        ratings: pd.DataFrame,
        user_to_idx: dict[str, int],
        item_to_idx: dict[str, int],
        positive_threshold: float = 4.0,
    ) -> None:
        """Build user embeddings as L2-normalised mean of positive item embeddings."""
        if self.item_embeddings is None:
            raise RuntimeError("Call encode_items() before compute_user_embeddings()")

        num_users = len(user_to_idx)
        self.num_users = num_users
        dim = self.item_embeddings.shape[1]
        accum = np.zeros((num_users, dim), dtype=np.float64)

        pos = ratings[ratings["rating"] >= positive_threshold]
        u_idx = pos["user_id"].astype(str).map(user_to_idx)
        i_idx = pos["item_id"].astype(str).map(item_to_idx)
        valid = u_idx.notna() & i_idx.notna()
        logger.info("Building user embeddings from %d positive ratings…", int(valid.sum()))

        np.add.at(accum, u_idx[valid].astype(int).values, self.item_embeddings[i_idx[valid].astype(int).values])

        norms = np.linalg.norm(accum, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.user_embeddings = (accum / norms).astype(np.float32)
        logger.info("User embeddings: %s", self.user_embeddings.shape)

    def compute_cross_domain_user_embeddings(
        self,
        cross_train: pd.DataFrame,
        user_to_idx: dict[str, int],
        item_to_idx: dict[str, int],
        source_domain: str = "movie",
        target_domain: str = "game",
        positive_threshold: float = 4.0,
        source_weight: float = 0.5,
    ) -> None:
        """Build cross-domain user profiles blending source and target signals.

        source-only users (cold-start): mean of source item embeddings → CDR transfer
        target-only users: mean of target item embeddings (same as plain SBERT)
        overlap users: weighted blend of source + target means
        """
        if self.item_embeddings is None:
            raise RuntimeError("Call encode_items() before compute_cross_domain_user_embeddings()")

        num_users = len(user_to_idx)
        self.num_users = num_users
        dim = self.item_embeddings.shape[1]

        src_accum = np.zeros((num_users, dim), dtype=np.float64)
        tgt_accum = np.zeros((num_users, dim), dtype=np.float64)
        src_counts = np.zeros(num_users, dtype=np.int32)
        tgt_counts = np.zeros(num_users, dtype=np.int32)

        pos_all = cross_train[cross_train["rating"] >= positive_threshold]
        u_idx = pos_all["user_id"].astype(str).map(user_to_idx)
        i_idx = pos_all["item_id"].astype(str).map(item_to_idx)
        valid = u_idx.notna() & i_idx.notna()
        logger.info("Building CDR user profiles from %d positive ratings…", int(valid.sum()))

        valid_rows = pos_all[valid]
        u = u_idx[valid].astype(int).values
        i = i_idx[valid].astype(int).values
        domain = valid_rows["domain"].astype(str).values

        src_mask = domain == source_domain
        tgt_mask = domain == target_domain
        np.add.at(src_accum, u[src_mask], self.item_embeddings[i[src_mask]])
        np.add.at(tgt_accum, u[tgt_mask], self.item_embeddings[i[tgt_mask]])
        np.add.at(src_counts, u[src_mask], 1)
        np.add.at(tgt_counts, u[tgt_mask], 1)

        # Compute per-user profiles based on which domains they have activity in.
        # All three cases produce un-normalized embeddings that get L2-normalized
        # at the end so cosine similarity = dot product at prediction time.
        user_emb = np.zeros((num_users, dim), dtype=np.float64)
        has_src = src_counts > 0
        has_tgt = tgt_counts > 0

        # Source-only users: their profile is entirely from movie embeddings
        mask_src_only = has_src & ~has_tgt
        user_emb[mask_src_only] = src_accum[mask_src_only] / src_counts[mask_src_only, np.newaxis]

        # Target-only users: profile from game embeddings only
        mask_tgt_only = has_tgt & ~has_src
        user_emb[mask_tgt_only] = tgt_accum[mask_tgt_only] / tgt_counts[mask_tgt_only, np.newaxis]

        # Overlap users: weighted blend of source and target mean embeddings
        mask_both = has_src & has_tgt
        src_mean = np.where(src_counts[:, np.newaxis] > 0, src_accum / np.maximum(src_counts[:, np.newaxis], 1), 0)
        tgt_mean = np.where(tgt_counts[:, np.newaxis] > 0, tgt_accum / np.maximum(tgt_counts[:, np.newaxis], 1), 0)
        user_emb[mask_both] = (
            source_weight * src_mean[mask_both] +
            (1 - source_weight) * tgt_mean[mask_both]
        )

        # L2-normalize so predict() can use simple dot product for cosine similarity
        norms = np.linalg.norm(user_emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.user_embeddings = (user_emb / norms).astype(np.float32)

        logger.info(
            "User profile breakdown: source-only=%d, target-only=%d, overlap=%d, no-history=%d",
            mask_src_only.sum(), mask_tgt_only.sum(), mask_both.sum(),
            (~has_src & ~has_tgt).sum(),
        )

    def predict(self, user_idx: int) -> np.ndarray:
        """Return cosine-similarity scores for all items."""
        if self.item_embeddings is None or self.user_embeddings is None:
            return np.zeros(self.num_items, dtype=np.float32)
        return self.item_embeddings @ self.user_embeddings[user_idx]
