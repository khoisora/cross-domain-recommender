"""Item-to-item explanation graph service.

Builds local subgraphs centered on a recommended item, showing
content-similar neighbors and optionally the user's rated history
items that contributed to the recommendation.
"""

from __future__ import annotations

import logging

import numpy as np

from backend.demo.store import DemoStore

logger = logging.getLogger(__name__)


class GraphService:
    """Builds explanation graphs for recommended items."""

    def __init__(self, store: DemoStore) -> None:
        self.store = store

    def build_item_graph(
        self,
        item_idx: int,
        user_ext_id: str | None = None,
        top_k: int = 9,
    ) -> dict:
        """Build a local explanation graph centered on an item.

        Returns:
            {nodes: [...], edges: [...], explanation_text: str}
        """
        s = self.store
        center_meta = s.items_by_idx.get(item_idx, {})
        if not center_meta:
            return {"nodes": [], "edges": [], "explanation_text": "Item not found."}

        nodes = []
        edges = []
        node_ids = set()

        # Center node — include user's own rating if they rated this item
        center_rating = 0.0
        center_ts = None
        if user_ext_id:
            for r in s.user_ratings.get(user_ext_id, []):
                if s.item_to_idx.get(r["item_id"]) == item_idx:
                    center_rating = r["rating"]
                    center_ts = r.get("timestamp")
        nodes.append(self._make_node(item_idx, "center", rating=center_rating, timestamp=center_ts))
        node_ids.add(item_idx)

        max_nodes = 10
        target_rated_movies = 3  # Ensure at least 3 rated movies

        # First collect user's rated movies (prioritize movies over games)
        rated_movies_to_add = []
        if user_ext_id:
            recent = s.get_recent_liked_items(user_ext_id, n=15, threshold=3.0)
            rating_timestamps = {}
            for r in s.user_ratings.get(user_ext_id, []):
                idx = s.item_to_idx.get(r["item_id"])
                if idx is not None:
                    rating_timestamps[idx] = r.get("timestamp")

            # Filter to movies first, then games, and check similarity
            movie_candidates = []
            game_candidates = []
            for ri, rating in recent:
                if ri in node_ids:
                    continue
                sim = float(s.content_emb[item_idx] @ s.content_emb[ri])
                if sim > 0.15:  # Similarity threshold
                    ts = rating_timestamps.get(ri)
                    meta = s.items_by_idx.get(ri, {})
                    candidate = {
                        "idx": ri,
                        "rating": rating,
                        "timestamp": ts,
                        "sim": sim,
                        "domain": meta.get("domain", "unknown")
                    }
                    if meta.get("domain") == "movie":
                        movie_candidates.append(candidate)
                    else:
                        game_candidates.append(candidate)

            # Sort by similarity (descending) and take top movies first
            movie_candidates.sort(key=lambda x: x["sim"], reverse=True)
            game_candidates.sort(key=lambda x: x["sim"], reverse=True)

            # Take up to 3 movies, then fill with games if needed
            selected_rated = movie_candidates[:target_rated_movies]
            if len(selected_rated) < target_rated_movies:
                needed = target_rated_movies - len(selected_rated)
                selected_rated.extend(game_candidates[:needed])

            rated_movies_to_add = selected_rated

        # Reserve space for rated movies
        reserved_slots = len(rated_movies_to_add)
        available_for_similar = max_nodes - reserved_slots

        # Add content-similar neighbors (leave space for rated movies)
        sim_indices = s.content_sim_indices[item_idx][:top_k]
        sim_scores = s.content_sim_scores[item_idx][:top_k]

        neighbor_titles = []
        added_similar = 0
        for ni, ns in zip(sim_indices, sim_scores):
            if added_similar >= available_for_similar:
                break
            ni = int(ni)
            ns = float(ns)
            if ni not in node_ids and ns > 0.1:
                nodes.append(self._make_node(ni, "related"))
                node_ids.add(ni)
                edges.append({
                    "source": item_idx,
                    "target": ni,
                    "weight": round(ns, 3),
                    "label": f"similarity: {ns:.2f}",
                })
                n_meta = s.items_by_idx.get(ni, {})
                neighbor_titles.append(n_meta.get("title", "Unknown"))
                added_similar += 1

        # Now add the rated movies (guaranteed slots)
        for candidate in rated_movies_to_add:
            ri = candidate["idx"]
            rating = candidate["rating"]
            ts = candidate["timestamp"]
            sim = candidate["sim"]
            
            # Skip if already added as similar item
            if ri in node_ids:
                continue
                
            nodes.append(self._make_node(ri, "rated", rating=rating, timestamp=ts))
            node_ids.add(ri)
            edges.append({
                "source": ri,
                "target": item_idx,
                "weight": round(sim, 3),
                "label": f"your rating: {rating:.0f}★",
            })

        # Build shared attributes for explanation text
        center_genres = set(g.strip().lower() for g in center_meta.get("genres", "").split(",") if g.strip())
        shared_genres = set()
        for ni in [int(x) for x in sim_indices[:5]]:
            n_meta = s.items_by_idx.get(ni, {})
            n_genres = set(g.strip().lower() for g in n_meta.get("genres", "").split(",") if g.strip())
            shared_genres.update(center_genres & n_genres)

        genre_str = ", ".join(g.title() for g in sorted(shared_genres)[:5]) if shared_genres else "similar themes"
        neighbor_str = ", ".join(neighbor_titles[:3]) if neighbor_titles else "related items"

        explanation_text = (
            f'"{center_meta.get("title", "This item")}" is connected to {neighbor_str} '
            f"because they share {genre_str}."
        )

        return {
            "nodes": nodes,
            "edges": edges,
            "explanation_text": explanation_text,
        }

    def get_item_explanation(
        self,
        item_idx: int,
        user_ext_id: str | None = None,
    ) -> dict:
        """Generate a structured explanation for why an item is recommended."""
        s = self.store
        meta = s.items_by_idx.get(item_idx, {})
        if not meta:
            return {"reasons": [], "summary": "Item not found."}

        reasons = []

        # Content similarity reasons
        sim_indices = s.content_sim_indices[item_idx][:5]
        sim_scores = s.content_sim_scores[item_idx][:5]

        if user_ext_id:
            rated_items = s.get_user_rated_items(user_ext_id)
            rated_ext_ids = {r["item_id"] for r in rated_items}
            high_rated = {r["item_id"]: r["rating"] for r in rated_items if r["rating"] >= 3.5}

            # Check if content-similar items are in user's history
            for ni, ns in zip(sim_indices, sim_scores):
                ni = int(ni)
                n_ext = s.idx_to_item.get(ni, "")
                if n_ext in high_rated:
                    n_meta = s.items_by_idx.get(ni, {})
                    reasons.append({
                        "type": "content_match",
                        "text": f'Similar to "{n_meta.get("title", "")}" which you rated {high_rated[n_ext]:.0f}/5',
                        "weight": float(ns),
                        "source_item_idx": ni,
                    })

            # LightGCN collaborative reason
            user_idx = s.user_to_idx.get(user_ext_id)
            if user_idx is not None:
                lgcn_score = float(s.lgcn_item[item_idx] @ s.lgcn_user[user_idx])
                if lgcn_score > 0:
                    reasons.append({
                        "type": "collaborative",
                        "text": "Matches your long-term collaborative taste profile",
                        "weight": lgcn_score,
                    })

            # MF reason
            if user_idx is not None:
                mf_score = float(
                    s.mf_item[item_idx] @ s.mf_user[user_idx]
                    + s.mf_item_bias[item_idx]
                    + s.mf_user_bias[user_idx]
                    + s.mf_global_bias
                )
                if mf_score > 2.5:
                    reasons.append({
                        "type": "mf_predicted",
                        "text": f"Predicted rating: {min(mf_score, 5.0):.1f}/5 based on your latest preferences",
                        "weight": mf_score / 5.0,
                    })

        # Genre explanation
        genres = meta.get("genres", "")
        if genres:
            reasons.append({
                "type": "genre",
                "text": f"Genres: {genres}",
                "weight": 0.5,
            })

        summary = f'Recommended because it aligns with your preferences'
        if reasons:
            top_reason = max(reasons, key=lambda r: r.get("weight", 0))
            summary = top_reason["text"]

        return {"reasons": reasons[:6], "summary": summary}

    def _make_node(self, idx: int, node_type: str, rating: float = 0, timestamp: str = None) -> dict:
        meta = self.store.items_by_idx.get(idx, {})
        node = {
            "id": idx,
            "label": meta.get("title", f"Item {idx}")[:40],
            "domain": meta.get("domain", "movie"),
            "type": node_type,
            "genres": meta.get("genres", ""),
            "tags": meta.get("tags", ""),
            "image_url": meta.get("image_url", ""),
            "avg_rating": meta.get("avg_rating"),
        }
        if rating > 0:
            node["user_rating"] = rating
        if timestamp:
            node["rating_timestamp"] = str(timestamp)
        return node
