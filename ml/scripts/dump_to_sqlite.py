#!/usr/bin/env python3
"""Dump processed Parquet data into SQLite for the backend API.

Run directly or schedule as a cron job:
    0 3 * * * cd /path/to/MoviesGamesRecommender && python -m ml.scripts.dump_to_sqlite
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "amazon_2023" / "processed"
DB_PATH = PROJECT_ROOT / "ml" / "data" / "amazon_2023" / "movies_games.db"

_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_ratings_user        ON ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_ratings_item        ON ratings(item_id);
CREATE INDEX IF NOT EXISTS idx_items_domain        ON items(domain);
CREATE INDEX IF NOT EXISTS idx_items_title         ON items(title);
CREATE INDEX IF NOT EXISTS idx_items_idx           ON items(idx);
CREATE INDEX IF NOT EXISTS idx_items_external_id   ON items(external_id);
CREATE INDEX IF NOT EXISTS idx_users_id            ON users(id);
CREATE INDEX IF NOT EXISTS idx_users_external_id   ON users(external_id);
"""

_AVATARS = ["🎬", "🎮", "🌟", "🎭", "🎯", "🏆", "🎪", "🎸", "🎲", "🎃"]


def _taste_summary(movie_ratings: int, game_ratings: int) -> str:
    total = movie_ratings + game_ratings
    if total == 0:
        return "Diverse Taste"
    ratio = movie_ratings / total
    if ratio > 0.8:
        return "Movie Buff"
    if ratio > 0.6:
        return "Mostly Movies"
    if ratio < 0.2:
        return "Hardcore Gamer"
    if ratio < 0.4:
        return "Mostly Games"
    return "Movie & Game Fan"


def _to_sql(df: pd.DataFrame, table: str, conn: sqlite3.Connection) -> None:
    df.to_sql(table, conn, if_exists="replace", index=False, method="multi", chunksize=2000)
    logger.info("  %s: %d rows", table, len(df))


def main() -> None:
    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(f"Processed data not found at {PROCESSED_DIR}")

    logger.info("Loading parquet files…")
    ratings = pd.read_parquet(
        PROCESSED_DIR / "ratings.parquet",
        columns=["user_id", "item_id", "rating", "domain", "timestamp"],
    )
    movies = pd.read_parquet(PROCESSED_DIR / "movies.parquet")
    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing to %s…", DB_PATH)

    # ── Items ────────────────────────────────────────────────────────────────
    items = pd.concat([movies, games], ignore_index=True)[
        ["external_id", "domain", "title", "description",
         "categories", "features", "main_category", "price", "image_url"]
    ].copy()

    # Precompute per-item rating stats
    item_stats = (
        ratings.groupby("item_id")
        .agg(rating_count=("rating", "count"), avg_rating=("rating", "mean"))
        .reset_index()
        .rename(columns={"item_id": "external_id"})
    )
    items = items.merge(item_stats, on="external_id", how="left")
    items["rating_count"] = items["rating_count"].fillna(0).astype(int)
    items["avg_rating"] = items["avg_rating"].round(4)

    # Assign sequential idx (sorted by rating_count DESC so popular items get low idx)
    items = items.sort_values("rating_count", ascending=False).reset_index(drop=True)
    items["idx"] = items.index

    # ── Users ────────────────────────────────────────────────────────────────
    # Precompute per-user stats
    user_stats = (
        ratings.groupby("user_id")
        .agg(
            total_ratings=("rating", "count"),
            avg_rating=("rating", "mean"),
            movie_ratings=("domain", lambda x: int((x == "movie").sum())),
            game_ratings=("domain", lambda x: int((x == "game").sum())),
        )
        .reset_index()
        .rename(columns={"user_id": "external_id"})
    )
    user_stats["avg_rating"] = user_stats["avg_rating"].round(4)
    user_stats = user_stats.sort_values("total_ratings", ascending=False).reset_index(drop=True)
    user_stats["id"] = user_stats.index
    user_stats["created_at"] = pd.Timestamp.now().isoformat()
    user_stats["avatar"] = user_stats["id"].apply(lambda i: _AVATARS[i % len(_AVATARS)])
    user_stats["name"] = user_stats["external_id"].apply(
        lambda uid: f"User {str(uid)[:8].upper()}"
    )
    user_stats["taste_summary"] = user_stats.apply(
        lambda r: _taste_summary(r["movie_ratings"], r["game_ratings"]), axis=1
    )

    # ── Write to SQLite ──────────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    try:
        _to_sql(items, "items", conn)
        _to_sql(user_stats, "users", conn)

        ratings["timestamp"] = ratings["timestamp"].astype(str)
        _to_sql(ratings, "ratings", conn)

        conn.commit()
        # Create indices after tables exist
        conn.executescript(_SCHEMA)
        conn.commit()

        c = conn.cursor()
        logger.info(
            "Done — users: %d, items: %d, ratings: %d",
            c.execute("SELECT COUNT(*) FROM users").fetchone()[0],
            c.execute("SELECT COUNT(*) FROM items").fetchone()[0],
            c.execute("SELECT COUNT(*) FROM ratings").fetchone()[0],
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
