"""Hybrid recommendation engine — one row per algorithm/strategy.

Produces 5 recommendation rows:
  1. LightGCN + cooc — "Top Picks for You" (best overall, game graph + movie transfer)
  2. EMCDR / PTUPCDR — "Based on Your Movie Taste" (CDR cross-domain transfer)
  3. Cooc standalone — "Players Who Watched Your Movies Also Played" (behavioral)
  4. SBERT — "Similar in Theme" (content-based, great for niche items)
  5. Popularity — "Trending Games" (strong baseline, always available)

Fast refresh: after a new rating, rows 3 (cooc) and 4 (SBERT) update instantly.
Rows 1-2 require model retraining (background batch job).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from backend.demo.store import DemoStore

logger = logging.getLogger(__name__)


def _normalise(scores: np.ndarray) -> np.ndarray:
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


class HybridRecommender:
    """Produces recommendation rows — one per algorithm/strategy."""

    def __init__(self, store: DemoStore) -> None:
        self.store = store

    def recommend_rows(
        self,
        user_ext_id: str,
        k_per_row: int = 15,
    ) -> list[dict]:
        s = self.store
        sd_user_idx = s.sd_user_to_idx.get(user_ext_id)
        cd_user_idx = s.cd_user_to_idx.get(user_ext_id)
        rated_sd = s.get_user_rated_sd_indices(user_ext_id) | s.blocked_sd
        rated_cd = s.get_user_rated_cd_indices(user_ext_id) | s.blocked_cd
        all_seen_ext: set[str] = set()  # cross-row dedup by external_id

        rows: list[dict] = []

        # ── Row 1: LightGCN + cooc — "Top Picks for You" ──
        if sd_user_idx is not None and sd_user_idx < s.lgcn_user.shape[0]:
            scores = s.lgcn_item @ s.lgcn_user[sd_user_idx]
            # Add cooc bonus
            cooc_scores = s.compute_cooc_scores(user_ext_id)
            for ext_id, bonus in cooc_scores.items():
                idx = s.sd_item_to_idx.get(ext_id)
                if idx is not None and idx < len(scores):
                    scores[idx] += 0.05 * bonus
            norm = _normalise(scores)
            items = _top_k_excluding(norm, rated_sd, k_per_row + 10)
            enriched = self._enrich_sd(items, all_seen_ext, k_per_row,
                                        "Recommended by LightGCN + movie co-occurrence")
            rows.append({
                "key": "lightgcn_cooc",
                "title": "Top Picks for You",
                "subtitle": "Graph-based collaborative filtering enhanced with movie-game co-occurrence signal",
                "items": enriched,
            })

        # ── Row 2: CDR — "Based on Your Movie Taste" ──
        # Use PTUPCDR if user has game history, EMCDR for cold-start
        cdr_user_emb = None
        cdr_item_emb = None
        cdr_label = ""
        has_games = any(r.get("domain") == "game" for r in s.get_user_rated_items(user_ext_id))
        if has_games and s.ptupcdr_user is not None and cd_user_idx is not None:
            cdr_user_emb = s.ptupcdr_user
            cdr_item_emb = s.ptupcdr_item
            cdr_label = "PTUPCDR (personalized movie-to-game transfer)"
        elif s.emcdr_user is not None and cd_user_idx is not None:
            cdr_user_emb = s.emcdr_user
            cdr_item_emb = s.emcdr_item
            cdr_label = "EMCDR (movie-to-game mapping)"

        if cdr_user_emb is not None and cd_user_idx < cdr_user_emb.shape[0]:
            scores = cdr_item_emb @ cdr_user_emb[cd_user_idx]
            # Zero out non-game items (CDR space has both movies and games)
            non_game = set(range(len(scores))) - s.cd_game_indices
            scores[list(non_game)] = -1e9
            norm = _normalise(scores)
            items = _top_k_excluding(norm, rated_cd, k_per_row + 10)
            enriched = self._enrich_cd(items, all_seen_ext, k_per_row,
                                        f"Transferred from your movie preferences via {cdr_label}")
            rows.append({
                "key": "cdr_transfer",
                "title": "Based on Your Movie Taste",
                "subtitle": f"Cross-domain transfer: {cdr_label}",
                "items": enriched,
            })

        # ── Row 3: Cooc standalone — "Players Who Watched Your Movies Also Played" ──
        # This row refreshes INSTANTLY when user rates a new movie
        cooc_scores = s.compute_cooc_scores(user_ext_id)
        if cooc_scores:
            scores_arr = np.zeros(s.sd_num_items, dtype=np.float32)
            for ext_id, bonus in cooc_scores.items():
                idx = s.sd_item_to_idx.get(ext_id)
                if idx is not None and idx < len(scores_arr):
                    scores_arr[idx] = bonus
            norm = _normalise(scores_arr)
            items = _top_k_excluding(norm, rated_sd, k_per_row + 10)
            enriched = self._enrich_sd(items, all_seen_ext, k_per_row,
                                        "Users who watched your movies also played this game")
            rows.append({
                "key": "cooc",
                "title": "Players Who Watched Your Movies Also Played",
                "subtitle": "Cross-domain co-occurrence — updates instantly after you rate a movie",
                "items": enriched,
            })

        # ── Row 4: SBERT — "Similar in Theme" ──
        # Uses content embeddings — also refreshes fast (mean of item vectors)
        if cd_user_idx is not None:
            recent_liked = [(s.cd_item_to_idx.get(r["item_id"]), r["rating"])
                           for r in s.get_user_rated_items(user_ext_id)
                           if r["rating"] >= 4.0 and r["item_id"] in s.cd_item_to_idx]
            if recent_liked:
                profile = np.zeros(s.content_emb.shape[1], dtype=np.float64)
                n = 0
                for idx, rating in recent_liked:
                    if idx is not None and idx < s.content_emb.shape[0]:
                        profile += s.content_emb[idx] * (rating / 5.0)
                        n += 1
                if n > 0:
                    profile /= n
                    norm_p = np.linalg.norm(profile)
                    if norm_p > 0:
                        profile /= norm_p
                    scores = s.content_emb @ profile.astype(np.float32)
                    # Zero out non-game items
                    non_game = set(range(len(scores))) - s.cd_game_indices
                    scores[list(non_game)] = -1e9
                    norm = _normalise(scores)
                    items = _top_k_excluding(norm, rated_cd, k_per_row + 10)
                    enriched = self._enrich_cd(items, all_seen_ext, k_per_row,
                                               "Content similarity to your rated items (SBERT)")
                    rows.append({
                        "key": "sbert",
                        "title": "Similar in Theme",
                        "subtitle": "Semantic text similarity — great for discovering niche games",
                        "items": enriched,
                    })

        # ── Row 5: Popularity — "Trending Games" ──
        popular = sorted(
            [(it.get("sd_idx"), it.get("rating_count", 0))
             for it in s.items_list
             if it.get("domain") == "game" and it.get("sd_idx") is not None],
            key=lambda x: -x[1],
        )
        pop_items = [(idx, float(count)) for idx, count in popular
                     if idx not in rated_sd][:k_per_row * 3]
        norm_pop = [(idx, _normalise(np.array([s for _, s in pop_items]))[i])
                    for i, (idx, _) in enumerate(pop_items)]
        enriched = self._enrich_sd(norm_pop, all_seen_ext, k_per_row,
                                    "Popular with many gamers")
        rows.append({
            "key": "popular",
            "title": "Trending Games",
            "subtitle": "Most played games — a strong baseline for new users",
            "items": enriched,
        })

        return rows

    def _enrich_sd(self, items: list[tuple[int, float]], seen: set[str],
                   k: int, reason: str) -> list[dict]:
        """Enrich items from single-domain (game-only) index space."""
        return self._enrich(items, seen, k, reason, self.store.sd_items_by_idx)

    def _enrich_cd(self, items: list[tuple[int, float]], seen: set[str],
                   k: int, reason: str) -> list[dict]:
        """Enrich items from cross-domain (unified) index space."""
        return self._enrich(items, seen, k, reason, self.store.cd_items_by_idx)

    def _enrich(self, items: list[tuple[int, float]], seen: set[str],
                k: int, reason: str, items_by_idx: dict) -> list[dict]:
        enriched = []
        for idx, score in items:
            meta = items_by_idx.get(idx, {})
            ext_id = meta.get("external_id", "")
            if ext_id in seen:
                continue
            seen.add(ext_id)
            enriched.append({
                "idx": idx,
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
