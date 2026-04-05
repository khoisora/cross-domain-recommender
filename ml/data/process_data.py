"""Amazon review dataset processing: raw JSONL → parquet for movie_game pair.

Phase 0: k-core >= 10 in both domains (movies AND games).
No cohort variants, no genre/overlap filters — those are added per-lesson.
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal, Optional

import pandas as pd

try:
    from .item_dedup import deduplicate_dataset
except ImportError:
    from ml.data.item_dedup import deduplicate_dataset

logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
AMAZON_DATASET_PATH = PROJECT_ROOT / "ml" / "data" / "amazon_2023"
RAW_DATA_DIR = AMAZON_DATASET_PATH / "raw"
OUTPUT_DIR = AMAZON_DATASET_PATH / "processed"

REVIEW_FILES = {
    "movies": RAW_DATA_DIR / "movies_reviews_2023.jsonl.gz",
    "games": RAW_DATA_DIR / "games_reviews_2023.jsonl.gz",
}
META_FILES = {
    "movies": RAW_DATA_DIR / "movies_meta_2023.jsonl.gz",
    "games": RAW_DATA_DIR / "games_meta_2023.jsonl.gz",
}

# K-core thresholds: users with >= 10 interactions in each domain
MIN_USER_INTERACTIONS = 10
# Item thresholds per domain (from old repo)
MOVIE_MIN_ITEM_INTERACTIONS = 20
GAME_MIN_ITEM_INTERACTIONS = 10


# ═══════════════════════════════════════════════════════════════════════════
# JSONL parsing
# ═══════════════════════════════════════════════════════════════════════════

def _parse_review(line: str, domain: str) -> Optional[dict]:
    """Parse a single JSONL review line."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    user_id = obj.get("user_id") or obj.get("reviewerID")
    asin = obj.get("parent_asin") or obj.get("asin")
    rating = obj.get("rating") if obj.get("rating") is not None else obj.get("overall")
    if not user_id or not asin or rating is None:
        return None

    # Parse timestamp
    ts_raw = obj.get("timestamp") or obj.get("unixReviewTime")
    ts = None
    if ts_raw:
        try:
            if isinstance(ts_raw, (int, float)):
                epoch = ts_raw / 1000 if ts_raw > 1e12 else float(ts_raw)
                ts = datetime.fromtimestamp(epoch)
            else:
                ts = datetime.fromisoformat(str(ts_raw))
        except (ValueError, OSError):
            pass

    return {
        "user_id": str(user_id),
        "item_id": str(asin),
        "rating": float(rating),
        "review_text": str(obj.get("text") or obj.get("reviewText") or "")[:5000] or None,
        "timestamp": ts,
        "domain": domain,
    }


def _parse_meta(line: str, domain: str) -> Optional[dict]:
    """Parse a single JSONL metadata line."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    asin = obj.get("parent_asin") or obj.get("asin")
    title = obj.get("title")
    if not asin or not title:
        return None

    # Description
    desc = obj.get("description")
    description = " ".join(str(d) for d in desc)[:2000] if isinstance(desc, list) else (str(desc)[:2000] if desc else None)

    # Categories
    cats = obj.get("categories") or obj.get("category")
    if isinstance(cats, list):
        flat = []
        for c in cats:
            flat.extend(c) if isinstance(c, list) else flat.append(str(c))
        categories = ", ".join(flat)
    elif isinstance(cats, str):
        categories = cats
    else:
        categories = None

    # Main category
    main_category = obj.get("main_category")

    # Features
    feat = obj.get("features") or obj.get("feature")
    features = ", ".join(str(f) for f in feat)[:1000] if isinstance(feat, list) else (str(feat)[:1000] if feat else None)

    # Image URL
    images = obj.get("images") or obj.get("imageURL") or obj.get("imUrl")
    image_url = None
    if isinstance(images, list) and images:
        first = images[0]
        image_url = first.get("large") or first.get("hi_res") if isinstance(first, dict) else str(first)
    elif isinstance(images, str):
        image_url = images

    return {
        "external_id": str(asin),
        "domain": domain,
        "title": str(title)[:512],
        "description": description,
        "categories": categories,
        "features": features,
        "main_category": str(main_category) if main_category else None,
        "price": str(obj.get("price", "")) if obj.get("price") else None,
        "image_url": str(image_url)[:1024] if image_url else None,
    }


def _iter_jsonl(filepath: Path, domain: str, parser) -> Iterator[dict]:
    """Iterate over a JSONL(.gz) file, yielding parsed records."""
    opener = gzip.open if filepath.suffix == ".gz" else open
    count = errors = 0
    with opener(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = parser(line, domain)
            if record:
                count += 1
                yield record
            else:
                errors += 1
            if count % 500_000 == 0:
                logger.info("  parsed %dk from %s", count // 1000, filepath.name)
    logger.info("Finished %s: %d records, %d errors", filepath.name, count, errors)


def _load_reviews(filepath: Path, domain: str) -> pd.DataFrame:
    """Load reviews from JSONL to DataFrame."""
    records = list(_iter_jsonl(filepath, domain, _parse_review))
    df = pd.DataFrame(records)
    logger.info("Loaded %d %s reviews", len(df), domain)
    return df


def _load_metadata(filepath: Path, domain: str) -> pd.DataFrame:
    """Load item metadata from JSONL to DataFrame."""
    records = list(_iter_jsonl(filepath, domain, _parse_meta))
    df = pd.DataFrame(records)
    logger.info("Loaded %d %s metadata records", len(df), domain)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Game item filter (only actual games, not accessories/consoles)
# ═══════════════════════════════════════════════════════════════════════════

def _is_actual_game(categories: Optional[str]) -> bool:
    """Check if item is an actual game (not accessories, consoles, etc)."""
    if not categories:
        return False
    segments = [str(s).strip() for s in str(categories).split(",")]
    return "Games" in segments


# ═══════════════════════════════════════════════════════════════════════════
# K-core filtering
# ═══════════════════════════════════════════════════════════════════════════

def _kcore_filter(
    ratings: pd.DataFrame,
    domain: str,
    min_user: int,
    min_item: int,
) -> pd.DataFrame:
    """Single-pass k-core: keep users with >= min_user and items with >= min_item interactions."""
    before = len(ratings)
    user_counts = ratings["user_id"].value_counts()
    ratings = ratings[ratings["user_id"].isin(user_counts[user_counts >= min_user].index)]
    item_counts = ratings["item_id"].value_counts()
    ratings = ratings[ratings["item_id"].isin(item_counts[item_counts >= min_item].index)]
    logger.info(
        "%s k-core (u>=%d, i>=%d): %d -> %d ratings (%.1f%% kept)",
        domain, min_user, min_item, before, len(ratings), 100 * len(ratings) / before if before else 0,
    )
    return ratings


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def build_movie_game_dataset(
    force_reprocess: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build the movie_game processed dataset with k-core >= 10 both domains.

    Returns (ratings_df, items_df, metadata_dict).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ratings_path = OUTPUT_DIR / "ratings.parquet"
    metadata_path = OUTPUT_DIR / "dataset_metadata.json"

    # Check for existing processed data
    if not force_reprocess and ratings_path.exists() and metadata_path.exists():
        logger.info("Loading existing processed dataset from %s", OUTPUT_DIR)
        ratings = pd.read_parquet(ratings_path)
        movies = pd.read_parquet(OUTPUT_DIR / "movies.parquet")
        games = pd.read_parquet(OUTPUT_DIR / "games.parquet")
        items = pd.concat([movies, games], ignore_index=True)
        with open(metadata_path) as f:
            metadata = json.load(f)
        logger.info(
            "Loaded: %d ratings, %d users, %d items",
            len(ratings), ratings["user_id"].nunique(), ratings["item_id"].nunique(),
        )
        return ratings, items, metadata

    # Verify raw files exist
    for name, path in {**REVIEW_FILES, **META_FILES}.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing raw file: {path}")

    # 1. Load raw reviews + metadata
    movie_reviews = _load_reviews(REVIEW_FILES["movies"], "movie")
    game_reviews = _load_reviews(REVIEW_FILES["games"], "game")
    ratings = pd.concat([movie_reviews, game_reviews], ignore_index=True)
    ratings = ratings.drop_duplicates(subset=["user_id", "item_id"], keep="last")

    movie_meta = _load_metadata(META_FILES["movies"], "movie")
    game_meta = _load_metadata(META_FILES["games"], "game")

    # Filter games to actual games (not accessories/consoles)
    before = len(game_meta)
    game_meta = game_meta[game_meta["categories"].apply(_is_actual_game)].copy()
    logger.info("Game item filter: %d -> %d (removed %d non-game items)", before, len(game_meta), before - len(game_meta))

    items = pd.concat([movie_meta, game_meta], ignore_index=True)
    items = items.drop_duplicates(subset=["external_id"], keep="last")

    # Keep only ratings for items with metadata
    valid_items = set(items["external_id"])
    ratings = ratings[ratings["item_id"].isin(valid_items)].copy()

    # 2. Deduplicate (merge DVD/Blu-ray/digital variants)
    logger.info("Deduplicating items...")
    ratings, items = deduplicate_dataset(ratings, items)

    # 3. K-core filtering per domain
    movie_ratings = _kcore_filter(
        ratings[ratings["domain"] == "movie"].copy(),
        "movie", MIN_USER_INTERACTIONS, MOVIE_MIN_ITEM_INTERACTIONS,
    )
    game_ratings = _kcore_filter(
        ratings[ratings["domain"] == "game"].copy(),
        "game", MIN_USER_INTERACTIONS, GAME_MIN_ITEM_INTERACTIONS,
    )

    # 4. Keep only overlap users (users in BOTH domains after k-core)
    movie_users = set(movie_ratings["user_id"])
    game_users = set(game_ratings["user_id"])
    overlap_users = movie_users & game_users
    logger.info("Overlap users (both domains after k-core): %d", len(overlap_users))

    movie_ratings = movie_ratings[movie_ratings["user_id"].isin(overlap_users)].copy()
    game_ratings = game_ratings[game_ratings["user_id"].isin(overlap_users)].copy()
    filtered_ratings = pd.concat([movie_ratings, game_ratings], ignore_index=True)

    # Sync items
    valid_item_ids = set(filtered_ratings["item_id"])
    items = items[items["external_id"].isin(valid_item_ids)].copy()

    # Stats
    n_users = filtered_ratings["user_id"].nunique()
    n_items = filtered_ratings["item_id"].nunique()
    n_ratings = len(filtered_ratings)
    n_movies = len(movie_ratings)
    n_games = len(game_ratings)

    logger.info("Final dataset: %d users, %d items, %d ratings", n_users, n_items, n_ratings)
    logger.info("  movie: %d ratings, %d items", n_movies, movie_ratings["item_id"].nunique())
    logger.info("  game:  %d ratings, %d items", n_games, game_ratings["item_id"].nunique())

    # 5. Save parquet files
    movies_df = items[items["domain"] == "movie"].copy()
    games_df = items[items["domain"] == "game"].copy()

    filtered_ratings.to_parquet(ratings_path, compression="snappy")
    movie_ratings.to_parquet(OUTPUT_DIR / "movie_ratings.parquet", compression="snappy")
    game_ratings.to_parquet(OUTPUT_DIR / "game_ratings.parquet", compression="snappy")
    movies_df.to_parquet(OUTPUT_DIR / "movies.parquet", compression="snappy")
    games_df.to_parquet(OUTPUT_DIR / "games.parquet", compression="snappy")

    # Cross-domain ratings (overlap users only — same as all ratings here)
    filtered_ratings.to_parquet(OUTPUT_DIR / "cross_domain_ratings.parquet", compression="snappy")

    metadata = {
        "processing_date": datetime.now().isoformat(),
        "pair": "movie_game",
        "cohort_filter": "k-core >= 10 (both movies and games)",
        "total_users": n_users,
        "total_items": n_items,
        "total_ratings": n_ratings,
        "n_movie_ratings": n_movies,
        "n_game_ratings": n_games,
        "n_movie_items": int(movies_df["external_id"].nunique()),
        "n_game_items": int(games_df["external_id"].nunique()),
        "overlap_users": int(len(overlap_users)),
        "user_k_core": MIN_USER_INTERACTIONS,
        "movie_item_k_core": MOVIE_MIN_ITEM_INTERACTIONS,
        "game_item_k_core": GAME_MIN_ITEM_INTERACTIONS,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved processed dataset to %s", OUTPUT_DIR)
    return filtered_ratings, items, metadata


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Build processed movie_game parquet dataset.")
    parser.add_argument("--force-reprocess", action="store_true", help="Force re-processing even if parquets exist")
    args = parser.parse_args()

    ratings, items, metadata = build_movie_game_dataset(force_reprocess=args.force_reprocess)
    logger.info("Done. %d users, %d ratings", metadata["total_users"], metadata["total_ratings"])
