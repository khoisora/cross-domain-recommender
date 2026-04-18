"""SQLite database for persistent storage of users, ratings, and items.

Tables:
  users       – user accounts (sample + created)
  ratings     – user-item ratings (persistent across restarts)
  items       – cached item metadata (loaded from artifacts on init)
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "demo.db"


class DemoDB:
    """SQLite database for the demo portal."""

    _instance: DemoDB | None = None

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    @classmethod
    def get(cls) -> "DemoDB":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _create_tables(self) -> None:
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                external_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                avatar TEXT DEFAULT '👤',
                taste_summary TEXT DEFAULT '',
                is_sample INTEGER DEFAULT 0,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            );

            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                item_idx INTEGER NOT NULL,
                item_id TEXT NOT NULL,
                rating REAL NOT NULL,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, item_idx)
            );

            CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(user_id);
            CREATE INDEX IF NOT EXISTS idx_ratings_item ON ratings(item_idx);
        """)
        self.conn.commit()
        logger.info("Database tables ready at %s", self.db_path)

    # ── User operations ─────────────────────────────────────────────────

    def upsert_sample_users(self, sample_users: list[dict]) -> None:
        """Insert or update sample users from artifacts."""
        cur = self.conn.cursor()
        for u in sample_users:
            cur.execute("""
                INSERT INTO users (id, external_id, name, avatar, taste_summary, is_sample)
                VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(external_id) DO UPDATE SET
                    name=excluded.name, avatar=excluded.avatar,
                    taste_summary=excluded.taste_summary, is_sample=1
            """, (u["id"], u["external_id"], u["name"], u["avatar"], u["taste_summary"]))
        self.conn.commit()
        logger.info("Upserted %d sample users", len(sample_users))

    def create_user(self, name: str) -> dict:
        """Create a new user with the given name. Returns user dict."""
        cur = self.conn.cursor()
        ext_id = f"custom_{int(time.time() * 1000)}"
        avatar = "👤"
        cur.execute("""
            INSERT INTO users (external_id, name, avatar, taste_summary, is_sample)
            VALUES (?, ?, ?, ?, 0)
        """, (ext_id, name, avatar, f"{name}'s personal recommendations"))
        self.conn.commit()
        user_id = cur.lastrowid
        return {
            "id": user_id,
            "external_id": ext_id,
            "name": name,
            "avatar": avatar,
            "taste_summary": f"{name}'s personal recommendations",
            "is_sample": False,
            "total_ratings": 0,
            "avg_rating": 0.0,
        }

    def create_user_with_ext_id(self, ext_id: str, name: str) -> dict:
        """Create a user with a specific external_id (for seeding from ratings store)."""
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO users (external_id, name, avatar, taste_summary, is_sample)
            VALUES (?, ?, '', '', 0)
        """, (ext_id, name))
        self.conn.commit()
        return {"id": cur.lastrowid, "external_id": ext_id, "name": name}

    def get_user(self, user_id: int) -> dict | None:
        """Get user by ID."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        if not row:
            return None
        return dict(row)

    def get_user_by_external_id(self, ext_id: str) -> dict | None:
        """Get user by external ID."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM users WHERE external_id = ?", (ext_id,))
        row = cur.fetchone()
        if not row:
            return None
        return dict(row)

    def list_users(self) -> list[dict]:
        """List all users."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM users ORDER BY is_sample DESC, id ASC")
        return [dict(r) for r in cur.fetchall()]

    # ── Rating operations ───────────────────────────────────────────────

    def add_rating(self, user_id: int, item_idx: int, item_id: str, rating: float) -> None:
        """Add or update a rating."""
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO ratings (user_id, item_idx, item_id, rating)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, item_idx) DO UPDATE SET
                rating=excluded.rating, created_at=strftime('%s', 'now')
        """, (user_id, item_idx, item_id, rating))
        self.conn.commit()

    def get_user_ratings(self, user_id: int) -> list[dict]:
        """Get all ratings for a user, ordered by most recent."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT item_idx, item_id, rating, created_at
            FROM ratings WHERE user_id = ? ORDER BY created_at DESC
        """, (user_id,))
        return [dict(r) for r in cur.fetchall()]

    def get_user_rating_for_item(self, user_id: int, item_idx: int) -> float | None:
        """Get a specific user's rating for an item."""
        cur = self.conn.cursor()
        cur.execute("SELECT rating FROM ratings WHERE user_id = ? AND item_idx = ?", (user_id, item_idx))
        row = cur.fetchone()
        return row["rating"] if row else None

    def get_user_rating_count(self, user_id: int) -> int:
        """Count user's total ratings."""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) as cnt FROM ratings WHERE user_id = ?", (user_id,))
        return cur.fetchone()["cnt"]

    def get_all_runtime_ratings(self) -> list[dict]:
        """Get all ratings not from artifact seeding (i.e. submitted at runtime)."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT r.user_id, u.external_id as user_id,
                   r.item_idx, r.item_id, r.rating, r.created_at
            FROM ratings r
            JOIN users u ON u.id = r.user_id
            ORDER BY r.created_at ASC
        """)
        return [dict(r) for r in cur.fetchall()]

    def seed_ratings_from_artifacts(self, sample_users: list[dict], item_to_idx: dict[str, int]) -> None:
        """Seed ratings from sample user artifacts (only if not already seeded)."""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) as cnt FROM ratings WHERE user_id <= ?", (len(sample_users),))
        existing = cur.fetchone()["cnt"]
        if existing > 0:
            logger.info("Ratings already seeded (%d existing), skipping", existing)
            return

        count = 0
        for u in sample_users:
            for r in u.get("ratings", []):
                ext_id = r["item_id"]
                idx = item_to_idx.get(ext_id)
                if idx is not None:
                    cur.execute("""
                        INSERT OR IGNORE INTO ratings (user_id, item_idx, item_id, rating)
                        VALUES (?, ?, ?, ?)
                    """, (u["id"], idx, ext_id, r["rating"]))
                    count += 1
        self.conn.commit()
        logger.info("Seeded %d ratings from sample users", count)
