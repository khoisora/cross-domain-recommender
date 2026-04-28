"""Hybrid recommendation engine — games-only output.

Routing by game-history depth:
  cold_start (0 games) — CDR models that transfer movie signal:
    Row 1: CMF + cooc
    Row 2: EMCDR + cooc
    Row 3: PTUPCDR + cooc
    Row 4: Cooc standalone
    Row 5: SBERT-CDR hidden gems
  warm (1+ games) — SDR models that use game history:
    Row 1: LightGCN + cooc
    Row 2: MF-BPR + cooc
    Row 3: NeuMF + cooc
    Row 4: Cooc standalone
    Row 5: SBERT-CDR hidden gems
"""

from __future__ import annotations

import logging

import numpy as np

from backend.demo.store import DemoStore

logger = logging.getLogger(__name__)

# Lesson 8 default cooc blend weight. Controls how much movie→game co-occurrence
# signal is mixed into the base model's scores.
COOC_LAM = 0.05


def _normalise(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1]. Different models produce scores on
    different scales (dot product vs cosine vs count); normalizing before display
    makes cross-model comparison meaningful in the frontend."""
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def _top_k_excluding(scores: np.ndarray, exclude: set[int], k: int) -> list[tuple[int, float]]:
    order = np.argsort(-scores)
    results = []
    for idx in order:
        idx = int(idx)
        if idx in exclude:
            continue
        results.append((idx, float(scores[idx])))
        if len(results) >= k:
            break
    return results


def _classify_segment(game_count: int) -> str:
    if game_count == 0:
        return "cold_start"
    return "warm"


SEGMENT_EXPLAINER = {
    "cold_start": "Cold-start user (0 game ratings) — CDR models transfer your movie preferences into game recommendations.",
    "warm":       "Warm user (1+ game ratings) — single-domain models leverage your game history, boosted by movie co-occurrence.",
}


class HybridRecommender:
    """Produces recommendation rows — one per algorithm/strategy."""

    def __init__(self, store: DemoStore) -> None:
        self.store = store

    # ── Primary entry point ─────────────────────────────────────────────

    def recommend_rows(self, user_ext_id: str, k_per_row: int = 15) -> list[dict]:
        s = self.store
        rated = s.get_user_rated_items(user_ext_id)
        game_count = sum(1 for r in rated if r.get("domain") == "game")
        segment = _classify_segment(game_count)

        ctx = _Context(
            store=s,
            user_ext_id=user_ext_id,
            sd_user_idx=s.sd_user_to_idx.get(user_ext_id),
            cd_user_idx=s.cd_user_to_idx.get(user_ext_id),
            rated_sd=s.get_user_rated_sd_indices(user_ext_id) | s.blocked_sd,
            rated_cd=s.get_user_rated_cd_indices(user_ext_id) | s.blocked_cd,
            cooc_sd=s.compute_cooc_scores(user_ext_id),
            seen_ext=set(),
            k=k_per_row,
            segment=segment,
        )

        rows: list[dict] = []

        if segment == "cold_start":
            # CDR path: transfer movie signal → game recommendations
            row = self._cdr_games(c=ctx, model="cmf", title="Recommended for You")
            if row:
                rows.append(row)
            row = self._cdr_games(c=ctx, model="emcdr", title="Based on Your Movie Taste")
            if row:
                rows.append(row)
            row = self._cdr_games(c=ctx, model="ptupcdr", title="You Might Also Like")
            if row:
                rows.append(row)
            row = self._cooc_standalone_row(ctx)
            if row:
                rows.append(row)
            row = self._hidden_gems_row(ctx)
            if row:
                rows.append(row)
        else:
            # SDR path: leverage game history
            row = self._lightgcn_games(ctx)
            if row:
                row["title"] = "Top Picks for You"
                rows.append(row)
            row = self._mf_bpr_games(ctx)
            if row:
                rows.append(row)
            row = self._neumf_games(ctx)
            if row:
                rows.append(row)
            row = self._cooc_standalone_row(ctx)
            if row:
                rows.append(row)
            row = self._hidden_gems_row(ctx)
            if row:
                rows.append(row)

        return rows

    def segment_info(self, user_ext_id: str) -> dict:
        """Return the user's routing segment + explainer. Used by the API layer."""
        rated = self.store.get_user_rated_items(user_ext_id)
        game_count = sum(1 for r in rated if r.get("domain") == "game")
        movie_count = sum(1 for r in rated if r.get("domain") == "movie")
        segment = _classify_segment(game_count)
        return {
            "segment": segment,
            "explainer": SEGMENT_EXPLAINER[segment],
            "game_count": game_count,
            "movie_count": movie_count,
        }

    # ── Individual model rows ───────────────────────────────────────────

    def _resolve_sd_user_vec(self, c: _Context, user_emb: np.ndarray, item_emb: np.ndarray) -> np.ndarray | None:
        """Trained user vector if known, else rating-weighted mean of rated game item vectors."""
        if c.sd_user_idx is not None and c.sd_user_idx < user_emb.shape[0]:
            return user_emb[c.sd_user_idx]
        return self._foldin_user_vec(c, item_emb, space="sd")

    def _resolve_cd_user_vec(self, c: _Context, user_emb: np.ndarray, item_emb: np.ndarray) -> np.ndarray | None:
        """Trained user vector if known, else rating-weighted mean of rated item vectors (movies + games)."""
        if c.cd_user_idx is not None and c.cd_user_idx < user_emb.shape[0]:
            return user_emb[c.cd_user_idx]
        return self._foldin_user_vec(c, item_emb, space="cd")

    def _foldin_user_vec(self, c: _Context, item_emb: np.ndarray, space: str) -> np.ndarray | None:
        s = c.store
        idx_map = s.sd_item_to_idx if space == "sd" else s.cd_item_to_idx
        ratings = s.user_ratings.get(c.user_ext_id, [])
        profile = np.zeros(item_emb.shape[1], dtype=np.float32)
        total_w = 0.0
        for r in ratings:
            if space == "sd" and r.get("domain") != "game":
                continue
            i = idx_map.get(r["item_id"])
            if i is None or i >= item_emb.shape[0]:
                continue
            w = float(r.get("rating", 0)) / 5.0
            if w <= 0:
                continue
            profile += item_emb[i] * w
            total_w += w
        if total_w < 1e-9:
            return None
        return profile / total_w

    def _lightgcn_games(self, c: _Context) -> dict | None:
        """LightGCN score over game catalog + cooc."""
        s = c.store
        user_vec = self._resolve_sd_user_vec(c, s.lgcn_user, s.lgcn_item)
        if user_vec is None:
            return None
        scores = s.lgcn_item @ user_vec
        self._apply_cooc_sd(scores, c.cooc_sd)
        norm = _normalise(scores)
        items = _top_k_excluding(norm, c.rated_sd, c.k + 10)
        return {
            "key": "lightgcn_cooc",
            "title": "Top Picks for You",
            "subtitle": "LightGCN graph convolution + movie→game co-occurrence",
            "model_tag": "LightGCN + Co-occurrence",
            "items": self._enrich_sd(items, c.seen_ext, c.k,
                                     "LightGCN graph score + movie→game co-occurrence"),
        }

    def _mf_bpr_games(self, c: _Context) -> dict | None:
        """MF-BPR score over game catalog + cooc."""
        s = c.store
        if s.mf_bpr_user is None:
            return None
        user_vec = self._resolve_sd_user_vec(c, s.mf_bpr_user, s.mf_bpr_item)
        if user_vec is None:
            return None
        scores = s.mf_bpr_item @ user_vec
        self._apply_cooc_sd(scores, c.cooc_sd)
        norm = _normalise(scores)
        items = _top_k_excluding(norm, c.rated_sd, c.k + 10)
        return {
            "key": "mf_bpr",
            "title": "You Might Also Like",
            "subtitle": "Matrix Factorization with BPR pairwise ranking + co-occurrence",
            "model_tag": "MF-BPR + Co-occurrence",
            "items": self._enrich_sd(items, c.seen_ext, c.k,
                                     "MF-BPR collaborative signal + movie→game co-occurrence"),
        }

    def _neumf_games(self, c: _Context) -> dict | None:
        """NeuMF score over game catalog + cooc."""
        s = c.store
        if s.neumf_user is None:
            return None
        user_vec = self._resolve_sd_user_vec(c, s.neumf_user, s.neumf_item)
        if user_vec is None:
            return None
        scores = s.neumf_item @ user_vec
        self._apply_cooc_sd(scores, c.cooc_sd)
        norm = _normalise(scores)
        items = _top_k_excluding(norm, c.rated_sd, c.k + 10)
        return {
            "key": "neumf",
            "title": "Neural Collaborative Picks",
            "subtitle": "NeuMF deep collaborative filtering + co-occurrence",
            "model_tag": "NeuMF + Co-occurrence",
            "items": self._enrich_sd(items, c.seen_ext, c.k,
                                     "NeuMF neural scoring + movie→game co-occurrence"),
        }

    def _cdr_games(self, c: _Context, model: str, title: str) -> dict | None:
        """CMF, EMCDR, or PTUPCDR score over game subset of CD catalog + cooc."""
        s = c.store
        if model == "cmf":
            user_emb, item_emb = s.cmf_user, s.cmf_item
            label = "CMF"
            subtitle = "Collective Matrix Factorization — shared user factors across movies & games + cooc"
        elif model == "ptupcdr":
            user_emb, item_emb = s.ptupcdr_user, s.ptupcdr_item
            label = "PTUPCDR"
            subtitle = "Few-shot movie-to-game transfer via PTUPCDR hypernetwork + cooc"
        else:
            user_emb, item_emb = s.emcdr_user, s.emcdr_item
            label = "EMCDR"
            subtitle = "Global movie-to-game mapping via EMCDR + cooc"
        if user_emb is None:
            return None
        user_vec = self._resolve_cd_user_vec(c, user_emb, item_emb)
        if user_vec is None:
            return None

        scores = item_emb @ user_vec
        n_items = len(scores)
        non_game_mask = np.ones(n_items, dtype=bool)
        for gi in s.cd_game_indices:
            if gi < n_items:
                non_game_mask[gi] = False
        scores[non_game_mask] = -1e9
        self._apply_cooc_cd(scores, c.cooc_sd)

        norm = _normalise(scores)
        items = _top_k_excluding(norm, c.rated_cd, c.k + 10)
        return {
            "key": f"cdr_{model}",
            "title": title,
            "subtitle": subtitle,
            "model_tag": f"{label} + Co-occurrence",
            "items": self._enrich_cd(items, c.seen_ext, c.k,
                                     f"Movie→game transfer via {label} + cooc"),
        }

    def _cooc_standalone_row(self, c: _Context) -> dict | None:
        """Raw cooc ranking — fast, training-free, updates instantly on new ratings."""
        if not c.cooc_sd:
            return None
        s = c.store
        scores = np.zeros(s.sd_num_items, dtype=np.float32)
        for ext_id, bonus in c.cooc_sd.items():
            idx = s.sd_item_to_idx.get(ext_id)
            if idx is not None and idx < len(scores):
                scores[idx] = bonus
        norm = _normalise(scores)
        items = _top_k_excluding(norm, c.rated_sd, c.k + 10)
        return {
            "key": "cooc",
            "title": "Players Who Watched Your Movies Also Played",
            "subtitle": "Raw movie→game co-occurrence — updates instantly when you rate a movie",
            "model_tag": "Co-occurrence",
            "items": self._enrich_sd(items, c.seen_ext, c.k,
                                     "Users who liked your movies also played this game"),
        }

    def _hidden_gems_row(self, c: _Context) -> dict | None:
        """SBERT-CDR on bottom-50%-popularity games, boosted by cooc.

        Lesson 7 finds SBERT's advantage is specifically on unpopular target
        items — ranking over the full catalog just returns head items. Capping
        to the tail + applying cooc matches the 'niche path' from the summary.
        """
        s = c.store
        profile = self._movie_sbert_profile(c)
        if profile is None:
            return None

        # Popularity threshold — bottom 50% of games by rating_count
        game_pops = sorted(
            it.get("rating_count", 0) or 0
            for it in s.items_list
            if it.get("domain") == "game"
        )
        if not game_pops:
            return None
        pop_threshold = game_pops[len(game_pops) // 2]

        scores = s.content_emb @ profile
        # Mask non-games and popular games
        mask_out = np.zeros(len(scores), dtype=bool)
        for idx in range(len(scores)):
            if idx not in s.cd_game_indices:
                mask_out[idx] = True
                continue
            meta = s.cd_items_by_idx.get(idx, {})
            if (meta.get("rating_count") or 0) >= pop_threshold:
                mask_out[idx] = True
        scores[mask_out] = -1e9
        # Universal cooc boost — rescues weak semantic scores (L8)
        self._apply_cooc_cd(scores, c.cooc_sd)

        norm = _normalise(scores)
        items = _top_k_excluding(norm, c.rated_cd, c.k + 10)
        if not items:
            return None
        return {
            "key": "hidden_gems",
            "title": "Hidden Gems",
            "subtitle": "SBERT-CDR on low-popularity games + cooc — the niche path where content signal wins",
            "model_tag": "SBERT-CDR + Co-occurrence",
            "items": self._enrich_cd(items, c.seen_ext, c.k,
                                     "Semantic match from your movies + co-occurrence lift"),
        }

    # ── Helpers ─────────────────────────────────────────────────────────

    def _movie_sbert_profile(self, c: _Context) -> np.ndarray | None:
        """Pure SBERT-CDR profile: unit-norm mean of the user's latest 5 liked
        movie SBERT vectors (rating >= 4). Returns None if the user has no
        qualifying movies — pure cross-domain transfer needs source signal."""
        s = c.store
        liked_movies = [
            (s.cd_item_to_idx.get(r["item_id"]), r["rating"])
            for r in s.get_user_rated_items(c.user_ext_id)
            if r.get("domain") == "movie"
            and r.get("rating", 0) >= 4.0
            and r["item_id"] in s.cd_item_to_idx
        ]
        recent = liked_movies[-5:]
        if not recent:
            return None
        profile = np.zeros(s.content_emb.shape[1], dtype=np.float64)
        n = 0
        for idx, rating in recent:
            if idx is not None and idx < s.content_emb.shape[0]:
                profile += s.content_emb[idx] * (rating / 5.0)
                n += 1
        if n == 0:
            return None
        profile /= n
        norm = np.linalg.norm(profile)
        if norm < 1e-9:
            return None
        return (profile / norm).astype(np.float32)

    def _apply_cooc_sd(self, scores: np.ndarray, cooc: dict[str, float]) -> None:
        s = self.store
        for ext_id, bonus in cooc.items():
            idx = s.sd_item_to_idx.get(ext_id)
            if idx is not None and idx < len(scores):
                scores[idx] += COOC_LAM * bonus

    def _apply_cooc_cd(self, scores: np.ndarray, cooc: dict[str, float]) -> None:
        s = self.store
        for ext_id, bonus in cooc.items():
            idx = s.cd_item_to_idx.get(ext_id)
            if idx is not None and idx < len(scores):
                scores[idx] += COOC_LAM * bonus

    def _enrich_sd(self, items, seen, k, reason):
        return self._enrich(items, seen, k, reason, self.store.sd_items_by_idx)

    def _enrich_cd(self, items, seen, k, reason):
        return self._enrich(items, seen, k, reason, self.store.cd_items_by_idx)

    def _enrich(self, items, seen, k, reason, items_by_idx):
        enriched = []
        for idx, score in items:
            meta = items_by_idx.get(idx, {})
            ext_id = meta.get("external_id", "")
            if not ext_id or ext_id in seen:
                continue
            seen.add(ext_id)
            enriched.append({
                "idx": meta.get("idx", idx),
                "external_id": ext_id,
                "title": meta.get("title", "Unknown"),
                "domain": meta.get("domain", ""),
                "image_url": meta.get("image_url") or "",
                "description": (meta.get("description") or "")[:150],
                "avg_rating": meta.get("avg_rating"),
                "rating_count": meta.get("rating_count") or 0,
                "score": round(float(score), 4),
                "reason": reason,
            })
            if len(enriched) >= k:
                break
        return enriched


class _Context:
    """Per-request computed context shared across row builders."""

    __slots__ = (
        "store", "user_ext_id", "sd_user_idx", "cd_user_idx",
        "rated_sd", "rated_cd",
        "cooc_sd", "seen_ext", "k", "segment",
    )

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
