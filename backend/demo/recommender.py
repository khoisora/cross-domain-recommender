"""Hybrid recommendation engine — one row per algorithm/strategy.

Lesson-aligned row layout:
  Row 1 (primary)  — segment-routed headline pick:
                       cold_start (0 games)   → EMCDR + cooc
                       one_shot   (1-2 games) → PTUPCDR + cooc
                       warm       (3+ games)  → LightGCN + cooc
  Row 2 (secondary collaborative) — complementary contrast:
                       cold_start → (skipped, Row 1 is already CDR)
                       one_shot   → LightGCN + cooc
                       warm       → EMCDR + cooc
  Row 3 — Cooc standalone (instant-refresh explainable)
  Row 4 — Hidden Gems (SBERT-CDR on low-popularity games + cooc)
  Row 5 — SBERT movies (in-domain content similarity from movie profile)
  Row 6 — LightGCN movies + reverse cooc
  Row 7 — Reverse cooc standalone
  Row 8 — Trending games
  Row 9 — Trending movies

Design rules from the lesson plan:
  • Cooc is universal post-processing (Lesson 8) — applied to every ranking row
    except Popularity (Lesson 8: cooc adds noise on a popularity baseline).
  • Headline row routes by game-history depth (Lesson 6 routing rule) —
    single-domain graph models collapse to popularity at true cold-start, so
    EMCDR/PTUPCDR take the front page when game signal is missing.
  • SBERT's validated strength is the long tail (Lesson 7), not overall ranking,
    so the SBERT games row filters to bottom-50% popularity.
  • SBERT profile uses movies only → pure SBERT-CDR (Lesson 7), not a mixed
    profile that drifts back toward in-domain.
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
    """Map user's game-rating count to the Lesson 6/8 routing segment."""
    if game_count == 0:
        return "cold_start"
    if game_count <= 2:
        return "one_shot"
    return "warm"


SEGMENT_EXPLAINER = {
    "cold_start": "Cold-start user (0 game ratings) — EMCDR maps your movie preferences into game space.",
    "one_shot":   "One-shot user (1–2 games) — PTUPCDR's few-shot blend combines movie transfer with your limited game signal.",
    "warm":       "Warm user (3+ games) — LightGCN graph convolution on your rating history, boosted by movie co-occurrence.",
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
            rated_movie_idx=self._rated_movie_indices(user_ext_id),
            cooc_sd=s.compute_cooc_scores(user_ext_id),       # {game_ext_id: score}
            rev_cooc=s.compute_reverse_cooc_scores(user_ext_id),  # {movie_ext_id: score}
            seen_ext=set(),
            k=k_per_row,
            segment=segment,
        )

        rows: list[dict] = []

        # Row 1 — segment-routed primary pick
        primary = self._primary_row(ctx)
        if primary:
            rows.append(primary)

        # Row 2 — secondary collaborative contrast (per segment)
        secondary = self._secondary_row(ctx)
        if secondary:
            rows.append(secondary)

        # Row 3 — Cooc standalone (instant refresh, explainable)
        row = self._cooc_standalone_row(ctx)
        if row:
            rows.append(row)

        # Row 4 — Hidden Gems (SBERT-CDR on bottom-50% popularity + cooc)
        row = self._hidden_gems_row(ctx)
        if row:
            rows.append(row)

        # Row 5 — SBERT movies (in-domain content similarity)
        row = self._sbert_movies_row(ctx)
        if row:
            rows.append(row)

        # Row 6 — LightGCN movies + reverse cooc
        row = self._lightgcn_movies_row(ctx)
        if row:
            rows.append(row)

        # Row 7 — Reverse cooc standalone
        row = self._reverse_cooc_row(ctx)
        if row:
            rows.append(row)

        # Row 8 — Popular games (baseline; cooc intentionally NOT applied — L8)
        rows.append(self._popular_games_row(ctx))

        # Row 9 — Popular movies
        row = self._popular_movies_row(ctx)
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

    # ── Row builders: collaborative primary/secondary ───────────────────

    def _primary_row(self, c: _Context) -> dict | None:
        if c.segment == "warm":
            row = self._lightgcn_games(c)
            if row:
                row["title"] = "Top Picks for You"
                row["subtitle"] = "LightGCN graph convolution + movie→game co-occurrence"
                return row
            # Warm user not in SD training — fall back to CDR
            return self._cdr_games(c, "ptupcdr", title="Top Picks for You")
        if c.segment == "one_shot":
            return self._cdr_games(c, "ptupcdr", title="Top Picks for You")
        return self._cdr_games(c, "emcdr", title="Top Picks for You")

    def _secondary_row(self, c: _Context) -> dict | None:
        if c.segment == "warm":
            # Show pure cross-domain alternative for warm users
            return self._cdr_games(c, "emcdr", title="Based on Your Movie Taste")
        if c.segment == "one_shot":
            # Contrast PTUPCDR headline with graph-based LightGCN
            row = self._lightgcn_games(c)
            if row:
                row["title"] = "Graph-Based Collaborative Picks"
                row["subtitle"] = "LightGCN on the rating graph (contrast to the few-shot CDR primary)"
                return row
            return None
        return None  # cold_start: Row 1 is already EMCDR — skip duplicate

    # ── Individual model rows ───────────────────────────────────────────

    def _lightgcn_games(self, c: _Context) -> dict | None:
        """LightGCN score over game catalog + cooc."""
        s = c.store
        if c.sd_user_idx is None or c.sd_user_idx >= s.lgcn_user.shape[0]:
            return None
        scores = s.lgcn_item @ s.lgcn_user[c.sd_user_idx]
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

    def _cdr_games(self, c: _Context, model: str, title: str) -> dict | None:
        """EMCDR or PTUPCDR score over game subset of CD catalog + cooc."""
        s = c.store
        if model == "ptupcdr":
            user_emb, item_emb = s.ptupcdr_user, s.ptupcdr_item
            label = "PTUPCDR"
            subtitle = "Few-shot movie-to-game transfer via PTUPCDR hypernetwork + cooc"
        else:
            user_emb, item_emb = s.emcdr_user, s.emcdr_item
            label = "EMCDR"
            subtitle = "Global movie-to-game mapping via EMCDR + cooc"
        if user_emb is None or c.cd_user_idx is None or c.cd_user_idx >= user_emb.shape[0]:
            return None

        scores = item_emb @ user_emb[c.cd_user_idx]
        # Mask non-game items: CDR embeddings live in the unified (movie+game)
        # space, but we only want to surface game recommendations here.
        non_game_mask = np.ones(len(scores), dtype=bool)
        non_game_mask[list(s.cd_game_indices)] = False
        scores[non_game_mask] = -1e9
        # Universal cooc post-processing (Lesson 8)
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

    def _sbert_movies_row(self, c: _Context) -> dict | None:
        """In-domain SBERT on movies from the same movie profile."""
        s = c.store
        profile = self._movie_sbert_profile(c)
        if profile is None:
            return None
        scores = s.content_emb @ profile
        # Mask out games (keep only movies)
        game_mask = np.zeros(len(scores), dtype=bool)
        game_mask[list(s.cd_game_indices)] = True
        scores[game_mask] = -1e9
        norm = _normalise(scores)
        items = _top_k_excluding(norm, c.rated_cd, c.k + 10)
        if not items:
            return None
        return {
            "key": "sbert_movies",
            "title": "Movies Similar in Theme",
            "subtitle": "In-domain SBERT content similarity to your latest movie ratings",
            "model_tag": "SBERT (in-domain)",
            "items": self._enrich_cd(items, c.seen_ext, c.k,
                                     "Content similarity to your rated movies (SBERT)"),
        }

    def _lightgcn_movies_row(self, c: _Context) -> dict | None:
        """Single-domain LightGCN on movies + reverse cooc (game→movie)."""
        s = c.store
        if s.lgcn_movie_user is None:
            return None
        mi = s.cd_user_to_idx.get(c.user_ext_id)
        if mi is None or mi >= s.lgcn_movie_user.shape[0]:
            return None
        scores = s.lgcn_movie_item @ s.lgcn_movie_user[mi]
        for ext_id, bonus in c.rev_cooc.items():
            idx = s.movie_item_to_idx.get(ext_id)
            if idx is not None and idx < len(scores):
                scores[idx] += COOC_LAM * bonus
        norm = _normalise(scores)
        items = _top_k_excluding(norm, c.rated_movie_idx, c.k + 10)
        enriched = self._enrich(items, c.seen_ext, c.k,
                                "LightGCN on movie ratings + game→movie co-occurrence",
                                s.movie_items_by_idx)
        if not enriched:
            return None
        return {
            "key": "lightgcn_movies",
            "title": "Top Movie Picks",
            "subtitle": "LightGCN on your movie history + game→movie reverse co-occurrence",
            "model_tag": "LightGCN (movies) + Reverse Co-occurrence",
            "items": enriched,
        }

    def _reverse_cooc_row(self, c: _Context) -> dict | None:
        """Raw game→movie cooc from user's game history."""
        s = c.store
        if not c.rev_cooc:
            return None
        scores = np.zeros(len(s.movie_item_to_idx), dtype=np.float32)
        for ext_id, bonus in c.rev_cooc.items():
            idx = s.movie_item_to_idx.get(ext_id)
            if idx is not None and idx < len(scores):
                scores[idx] = bonus
        norm = _normalise(scores)
        items = _top_k_excluding(norm, c.rated_movie_idx, c.k + 10)
        enriched = self._enrich(items, c.seen_ext, c.k,
                                "Gamers who played your games also watched this movie",
                                s.movie_items_by_idx)
        if not enriched:
            return None
        return {
            "key": "reverse_cooc",
            "title": "Movies Fans of Your Games Also Watched",
            "subtitle": "Reverse game→movie co-occurrence",
            "model_tag": "Reverse Co-occurrence",
            "items": enriched,
        }

    def _popular_games_row(self, c: _Context) -> dict:
        """Popularity baseline — L8 shows cooc adds noise here, so leave it raw."""
        s = c.store
        popular = sorted(
            [(it.get("sd_idx"), it.get("rating_count", 0))
             for it in s.items_list
             if it.get("domain") == "game" and it.get("sd_idx") is not None],
            key=lambda x: -x[1],
        )
        pop_items = [(idx, float(cnt)) for idx, cnt in popular
                     if idx not in c.rated_sd][:c.k * 3]
        counts = np.array([v for _, v in pop_items], dtype=np.float32) if pop_items else np.zeros(1)
        norm = _normalise(counts)
        items = [(idx, float(norm[i])) for i, (idx, _) in enumerate(pop_items)]
        return {
            "key": "popular_games",
            "title": "Trending Games",
            "subtitle": "Most-rated games — a strong cold-start baseline (Lesson 6)",
            "model_tag": "Popularity Baseline",
            "items": self._enrich_sd(items, c.seen_ext, c.k, "Popular with many gamers"),
        }

    def _popular_movies_row(self, c: _Context) -> dict | None:
        s = c.store
        pop_movies = sorted(
            [(s.movie_item_to_idx.get(it.get("external_id", "")), it.get("rating_count", 0))
             for it in s.items_list
             if it.get("domain") == "movie" and it.get("external_id") in s.movie_item_to_idx],
            key=lambda x: -x[1],
        )
        pop_m_items = [(idx, float(cnt)) for idx, cnt in pop_movies
                       if idx is not None and idx not in c.rated_movie_idx][:c.k * 3]
        if not pop_m_items:
            return None
        counts = np.array([v for _, v in pop_m_items], dtype=np.float32)
        norm = _normalise(counts)
        items = [(idx, float(norm[i])) for i, (idx, _) in enumerate(pop_m_items)]
        enriched = self._enrich(items, c.seen_ext, c.k, "Popular with many viewers",
                                s.movie_items_by_idx)
        if not enriched:
            return None
        return {
            "key": "popular_movies",
            "title": "Trending Movies",
            "subtitle": "Most-rated movies in the catalog",
            "model_tag": "Popularity Baseline",
            "items": enriched,
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

    def _rated_movie_indices(self, user_ext_id: str) -> set[int]:
        s = self.store
        return {
            s.movie_item_to_idx[r["item_id"]]
            for r in s.get_user_rated_items(user_ext_id)
            if r.get("item_id") in s.movie_item_to_idx
        }

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
        "rated_sd", "rated_cd", "rated_movie_idx",
        "cooc_sd", "rev_cooc", "seen_ext", "k", "segment",
    )

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
