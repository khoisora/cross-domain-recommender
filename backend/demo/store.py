"""In-memory artifact store for the demo portal.

Loads precomputed embeddings from artifacts/demo/:
  Single-domain (game items only): LightGCN
  Cross-domain (all items): EMCDR, PTUPCDR, SBERT
  Co-occurrence: movie→game behavioral associations
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEMO_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts" / "demo"

# Runtime blocklist — hide non-game items like consoles, gift cards
_BLOCK_KEYWORDS = [
    "500gb console", "1tb console", "console [old", "console [new",
    "[digital code]", "digital code]", "gift card", "membership [digital",
    "membership [online", "playstation plus", "ps plus", "xbox live gold",
    "xbox game pass", "psn card", "eshop card", "nintendo eshop",
    "xbox gift", "store credit", "month membership",
    "ps4 console", "ps5 console", "xbox one console", "xbox one s ",
    "xbox one x ", "xbox series", "nintendo switch console",
    "playstation 4 pro console", "playstation vr",
]


class DemoStore:
    """Singleton in-memory store for all demo artifacts."""

    _instance: DemoStore | None = None

    def __init__(self) -> None:
        self.loaded = False

    @classmethod
    def get(cls) -> "DemoStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        if self.loaded:
            return
        logger.info("Loading demo artifacts from %s", DEMO_DIR)

        # --- Mappings ---
        # Single-domain (game items only, for LightGCN/NCF)
        self.sd_user_to_idx: dict[str, int] = json.loads((DEMO_DIR / "user_to_idx.json").read_text())
        self.sd_item_to_idx: dict[str, int] = json.loads((DEMO_DIR / "item_to_idx.json").read_text())
        self.sd_idx_to_item: dict[int, str] = {int(k): v for k, v in json.loads((DEMO_DIR / "idx_to_item.json").read_text()).items()}

        # Cross-domain (all items, for EMCDR/PTUPCDR/SBERT)
        self.cd_user_to_idx: dict[str, int] = json.loads((DEMO_DIR / "cross_domain_user_to_idx.json").read_text())
        self.cd_item_to_idx: dict[str, int] = json.loads((DEMO_DIR / "cross_domain_item_to_idx.json").read_text())
        self.cd_idx_to_item: dict[int, str] = {int(k): v for k, v in json.loads((DEMO_DIR / "cross_domain_idx_to_item.json").read_text()).items()}

        # --- Item catalog ---
        self.items_list: list[dict] = json.loads((DEMO_DIR / "items.json").read_text())
        self.items_by_ext: dict[str, dict] = {c["external_id"]: c for c in self.items_list}

        # Build idx lookups for both spaces
        self.sd_items_by_idx: dict[int, dict] = {}
        self.cd_items_by_idx: dict[int, dict] = {}
        for c in self.items_list:
            if c.get("sd_idx") is not None:
                self.sd_items_by_idx[c["sd_idx"]] = c
            if c.get("cd_idx") is not None:
                self.cd_items_by_idx[c["cd_idx"]] = c

        # --- Sample users ---
        self.sample_users: list[dict] = json.loads((DEMO_DIR / "users.json").read_text())
        self.sample_users_by_id: dict[int, dict] = {u["id"]: u for u in self.sample_users}

        # --- Model embeddings ---
        # Single-domain (game-only item space)
        self.lgcn_user: np.ndarray = np.load(DEMO_DIR / "lightgcn_user.npy")
        self.lgcn_item: np.ndarray = np.load(DEMO_DIR / "lightgcn_item.npy")

        # Cross-domain (unified item space)
        self.emcdr_user: np.ndarray | None = None
        self.emcdr_item: np.ndarray | None = None
        if (DEMO_DIR / "emcdr_user.npy").exists():
            self.emcdr_user = np.load(DEMO_DIR / "emcdr_user.npy")
            self.emcdr_item = np.load(DEMO_DIR / "emcdr_item.npy")

        self.ptupcdr_user: np.ndarray | None = None
        self.ptupcdr_item: np.ndarray | None = None
        if (DEMO_DIR / "ptupcdr_user.npy").exists():
            self.ptupcdr_user = np.load(DEMO_DIR / "ptupcdr_user.npy")
            self.ptupcdr_item = np.load(DEMO_DIR / "ptupcdr_item.npy")

        # Content embeddings (cross-domain space)
        self.content_emb: np.ndarray = np.load(DEMO_DIR / "content_emb.npy")
        self.content_sim_indices: np.ndarray = np.load(DEMO_DIR / "content_sim_indices.npy")
        self.content_sim_scores: np.ndarray = np.load(DEMO_DIR / "content_sim_scores.npy")

        # LightGCN movie-domain (single-domain collaborative on movies)
        self.lgcn_movie_user: np.ndarray | None = None
        self.lgcn_movie_item: np.ndarray | None = None
        self.movie_item_to_idx: dict[str, int] = {}
        self.movie_idx_to_item: dict[int, str] = {}
        if (DEMO_DIR / "lightgcn_movie_user.npy").exists():
            self.lgcn_movie_user = np.load(DEMO_DIR / "lightgcn_movie_user.npy")
            self.lgcn_movie_item = np.load(DEMO_DIR / "lightgcn_movie_item.npy")
            self.movie_item_to_idx = json.loads((DEMO_DIR / "movie_item_to_idx.json").read_text())
            self.movie_idx_to_item = {int(k): v for k, v in json.loads((DEMO_DIR / "movie_idx_to_item.json").read_text()).items()}
            logger.info("Loaded LightGCN-movies: user=%s, item=%s",
                        self.lgcn_movie_user.shape, self.lgcn_movie_item.shape)

        # Co-occurrence matrices
        self.cooc: dict[str, dict[str, float]] = json.loads((DEMO_DIR / "cooc.json").read_text())
        self.reverse_cooc: dict[str, dict[str, float]] = {}
        if (DEMO_DIR / "reverse_cooc.json").exists():
            self.reverse_cooc = json.loads((DEMO_DIR / "reverse_cooc.json").read_text())
            logger.info("Loaded reverse cooc (game→movie): %d games", len(self.reverse_cooc))

        # --- Train ratings ---
        raw_ratings: list[dict] = json.loads((DEMO_DIR / "train_ratings.json").read_text())
        self.user_ratings: dict[str, list[dict]] = {}
        for r in raw_ratings:
            self.user_ratings.setdefault(r["user_id"], []).append(r)

        self.sd_num_users = self.lgcn_user.shape[0]
        self.sd_num_items = self.lgcn_item.shape[0]
        self.cd_num_items = self.content_emb.shape[0]

        # --- Blocklist ---
        self.blocked_sd: set[int] = set()
        self.blocked_cd: set[int] = set()
        for item in self.items_list:
            t = item.get("title", "").lower()
            if item.get("domain") == "game" and any(kw in t for kw in _BLOCK_KEYWORDS):
                if item.get("sd_idx") is not None:
                    self.blocked_sd.add(item["sd_idx"])
                if item.get("cd_idx") is not None:
                    self.blocked_cd.add(item["cd_idx"])

        # Game indices in CD space (from training data, not DB domain labels)
        game_ids: list[str] = json.loads((DEMO_DIR / "game_item_ids.json").read_text())
        game_id_set = set(game_ids)
        self.cd_game_indices: set[int] = set()
        for gid in game_id_set:
            cd_idx = self.cd_item_to_idx.get(gid)
            if cd_idx is not None:
                self.cd_game_indices.add(cd_idx)

        # Lookup by DB idx (for backward compat with frontend)
        self.items_by_db_idx: dict[int, dict] = {c["idx"]: c for c in self.items_list if "idx" in c}

        # Movie items by movie-domain model idx
        self.movie_items_by_idx: dict[int, dict] = {}
        for c in self.items_list:
            ext_id = c.get("external_id", "")
            mi = self.movie_item_to_idx.get(ext_id)
            if mi is not None:
                self.movie_items_by_idx[mi] = c

        self.loaded = True
        logger.info(
            "Store loaded: SD=%d users, %d items | CD=%d items | content=%s | cooc=%d movies",
            self.sd_num_users, self.sd_num_items, self.cd_num_items,
            self.content_emb.shape, len(self.cooc),
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    def get_user_rated_items(self, user_ext_id: str) -> list[dict]:
        return self.user_ratings.get(user_ext_id, [])

    def get_user_rated_sd_indices(self, user_ext_id: str) -> set[int]:
        indices = set()
        for r in self.user_ratings.get(user_ext_id, []):
            idx = self.sd_item_to_idx.get(r["item_id"])
            if idx is not None:
                indices.add(idx)
        return indices

    def get_user_rated_cd_indices(self, user_ext_id: str) -> set[int]:
        indices = set()
        for r in self.user_ratings.get(user_ext_id, []):
            idx = self.cd_item_to_idx.get(r["item_id"])
            if idx is not None:
                indices.add(idx)
        return indices

    def get_user_movie_ids(self, user_ext_id: str, threshold: float = 4.0) -> list[str]:
        """Return list of movie item IDs the user liked (for cooc reranking)."""
        return [r["item_id"] for r in self.user_ratings.get(user_ext_id, [])
                if r.get("domain") == "movie" and r["rating"] >= threshold]

    def add_rating(self, user_ext_id: str, item_ext_id: str, rating: float, domain: str = "") -> None:
        ratings = self.user_ratings.setdefault(user_ext_id, [])
        for r in ratings:
            if r["item_id"] == item_ext_id:
                r["rating"] = rating
                return
        ratings.append({"user_id": user_ext_id, "item_id": item_ext_id,
                        "rating": rating, "domain": domain})

    def search_items(self, query: str, limit: int = 20) -> list[dict]:
        q = query.lower()
        return [it for it in self.items_list if q in it.get("title", "").lower()][:limit]

    def get_user_game_ids(self, user_ext_id: str, threshold: float = 4.0) -> list[str]:
        """Return list of game item IDs the user liked (for reverse cooc)."""
        return [r["item_id"] for r in self.user_ratings.get(user_ext_id, [])
                if r.get("domain") == "game" and r["rating"] >= threshold]

    def compute_reverse_cooc_scores(self, user_ext_id: str) -> dict[str, float]:
        """Compute game→movie cooc: recommend movies based on game history."""
        game_ids = self.get_user_game_ids(user_ext_id)
        if not game_ids:
            return {}
        movie_scores: dict[str, float] = {}
        for gid in game_ids:
            for mid, score in self.reverse_cooc.get(gid, {}).items():
                movie_scores[mid] = movie_scores.get(mid, 0) + score
        return movie_scores

    def compute_cooc_scores(self, user_ext_id: str) -> dict[str, float]:
        """Compute cooc bonus for game items based on user's movie history.

        Returns {game_item_ext_id: score} — fast, training-free, real-time.
        """
        movie_ids = self.get_user_movie_ids(user_ext_id)
        if not movie_ids:
            return {}
        game_scores: dict[str, float] = {}
        for mid in movie_ids:
            for gid, score in self.cooc.get(mid, {}).items():
                game_scores[gid] = game_scores.get(gid, 0) + score
        return game_scores
