"""Amazon review dataset processing: raw JSONL → parquet for movie_game pair.

Pipeline:
  1. Parse raw JSONL.gz reviews + metadata for movies and games
  2. Filter game metadata to actual games (exclude accessories, consoles, etc.)
  3. Deduplicate items (merge DVD/Blu-ray/digital/platform variants → one canonical item)
  4. K-core filter: users with >= 10 total interactions (across both domains),
     movie items >= 20 interactions, game items >= 10 interactions
  5. Save parquet files + metadata JSON

Phase 0 / Lessons 1–2: users >= 10 total interactions, NO overlap-user filter.
Users active in only one domain are kept. Overlap-user filtering is introduced
in Lesson 3 cohort variants.

Output files in ml/data/amazon_2023/processed/:
  - ratings.parquet        — all interactions (movie + game)
  - movie_ratings.parquet   — movie interactions only
  - game_ratings.parquet    — game interactions only
  - movies.parquet          — movie item metadata
  - games.parquet           — game item metadata
  - dataset_metadata.json   — processing parameters and statistics
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal, Optional

import numpy as np
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

SEED = 42

# K-core thresholds
# Users: >= 10 total interactions across both domains (not per-domain)
MIN_USER_INTERACTIONS = 10
# Item thresholds per domain (movies have more data, so higher threshold)
MOVIE_MIN_ITEM_INTERACTIONS = 20
GAME_MIN_ITEM_INTERACTIONS = 10


# ═══════════════════════════════════════════════════════════════════════════
# JSONL parsing
# ═══════════════════════════════════════════════════════════════════════════

def _parse_review(line: str, domain: str) -> Optional[dict]:
    """Parse a single JSONL review line.

    Handles both Amazon 2023 format (user_id, parent_asin, rating)
    and older SNAP format (reviewerID, asin, overall).
    """
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    # Amazon 2023 uses "user_id"/"parent_asin"/"rating";
    # older SNAP format uses "reviewerID"/"asin"/"overall"
    user_id = obj.get("user_id") or obj.get("reviewerID")
    asin = obj.get("parent_asin") or obj.get("asin")
    rating = obj.get("rating") if obj.get("rating") is not None else obj.get("overall")
    if not user_id or not asin or rating is None:
        return None

    # Parse timestamp — Amazon 2023 uses millisecond epoch (>1e12),
    # SNAP uses second epoch. Both formats handled here.
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
    """Check if item is an actual game (not accessories, consoles, etc).

    The Amazon Video_Games category includes controllers, headsets, gift cards,
    and console hardware. We keep only items with "Games" as a standalone
    category segment (e.g. "Video Games, PlayStation 4, Games" → True,
    "Video Games, Accessories, Controllers" → False).
    """
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
    """Single-pass k-core: keep users with >= min_user and items with >= min_item interactions.

    K-core filtering removes sparse users and items that don't have enough
    interactions for meaningful evaluation. Single-pass (user filter → item filter)
    is sufficient here because we apply overlap-user filtering afterward.
    """
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
    min_user_interactions: int = MIN_USER_INTERACTIONS,
    sample_users: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build the movie_game processed dataset.

    Pipeline steps:
      1. Load raw JSONL reviews + metadata for both domains
      2. Filter game items to actual games (exclude accessories, consoles)
      3. Deduplicate items (merge format/platform variants)
      4. Item k-core: movies >= 20, games >= 10 interactions
      5. Optional user k-core (min_user_interactions, default 10; set 0 to skip)
      6. Optional random user sampling (sample_users, preserving overlap ratio)
      7. Save parquet files + metadata JSON

    Caches results: skips processing if output parquets already exist.

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

    # 3. Item k-core filtering per domain (remove items with too few interactions)
    movie_ratings = ratings[ratings["domain"] == "movie"].copy()
    game_ratings = ratings[ratings["domain"] == "game"].copy()

    # Movie items: keep only items with >= 20 interactions
    movie_item_counts = movie_ratings["item_id"].value_counts()
    valid_movie_items = set(movie_item_counts[movie_item_counts >= MOVIE_MIN_ITEM_INTERACTIONS].index)
    before_m = len(movie_ratings)
    movie_ratings = movie_ratings[movie_ratings["item_id"].isin(valid_movie_items)].copy()
    logger.info("Movie item k-core (>=%d): %d -> %d ratings", MOVIE_MIN_ITEM_INTERACTIONS, before_m, len(movie_ratings))

    # Game items: keep only items with >= 10 interactions
    game_item_counts = game_ratings["item_id"].value_counts()
    valid_game_items = set(game_item_counts[game_item_counts >= GAME_MIN_ITEM_INTERACTIONS].index)
    before_g = len(game_ratings)
    game_ratings = game_ratings[game_ratings["item_id"].isin(valid_game_items)].copy()
    logger.info("Game item k-core (>=%d): %d -> %d ratings", GAME_MIN_ITEM_INTERACTIONS, before_g, len(game_ratings))

    # Recombine after item filtering
    filtered_ratings = pd.concat([movie_ratings, game_ratings], ignore_index=True)

    # 4. User k-core (optional — set min_user_interactions=0 to skip)
    if min_user_interactions > 0:
        before_total = len(filtered_ratings)
        user_counts = filtered_ratings["user_id"].value_counts()
        valid_users = set(user_counts[user_counts >= min_user_interactions].index)
        filtered_ratings = filtered_ratings[filtered_ratings["user_id"].isin(valid_users)].copy()
        logger.info(
            "User k-core (>=%d total): %d -> %d ratings (%d users kept)",
            min_user_interactions, before_total, len(filtered_ratings), len(valid_users),
        )

    # 5. Random user sampling (optional — preserves overlap ratio)
    if sample_users is not None:
        movie_r = filtered_ratings[filtered_ratings["domain"] == "movie"]
        game_r = filtered_ratings[filtered_ratings["domain"] == "game"]
        mu = set(movie_r["user_id"])
        gu = set(game_r["user_id"])
        overlap = sorted(mu & gu)
        movie_only = sorted(mu - gu)
        game_only = sorted(gu - mu)
        total = len(overlap) + len(movie_only) + len(game_only)

        rng = np.random.RandomState(SEED)
        n_ov = max(1, int(sample_users * len(overlap) / total))
        n_mo = max(1, int(sample_users * len(movie_only) / total))
        n_go = max(1, int(sample_users * len(game_only) / total))

        sampled = set()
        sampled.update(rng.choice(overlap, min(n_ov, len(overlap)), replace=False))
        sampled.update(rng.choice(movie_only, min(n_mo, len(movie_only)), replace=False))
        sampled.update(rng.choice(game_only, min(n_go, len(game_only)), replace=False))

        before_sample = filtered_ratings["user_id"].nunique()
        filtered_ratings = filtered_ratings[filtered_ratings["user_id"].isin(sampled)].copy()
        logger.info("User sampling: %d -> %d users", before_sample, filtered_ratings["user_id"].nunique())

    # Sync items — remove items with no remaining ratings
    valid_item_ids = set(filtered_ratings["item_id"])
    items = items[items["external_id"].isin(valid_item_ids)].copy()

    # Split back into per-domain views for saving
    movie_ratings = filtered_ratings[filtered_ratings["domain"] == "movie"].copy()
    game_ratings = filtered_ratings[filtered_ratings["domain"] == "game"].copy()

    # Stats
    n_users = filtered_ratings["user_id"].nunique()
    n_items = filtered_ratings["item_id"].nunique()
    n_ratings = len(filtered_ratings)
    n_movies = len(movie_ratings)
    n_games = len(game_ratings)

    movie_users = set(movie_ratings["user_id"])
    game_users = set(game_ratings["user_id"])
    overlap_users = movie_users & game_users
    overlap_pct = round(100 * len(overlap_users) / n_users, 1) if n_users else 0

    logger.info("Final dataset: %d users, %d items, %d ratings (%.1f%% overlap)", n_users, n_items, n_ratings, overlap_pct)
    logger.info("  movie: %d ratings, %d items, %d users", n_movies, movie_ratings["item_id"].nunique(), len(movie_users))
    logger.info("  game:  %d ratings, %d items, %d users", n_games, game_ratings["item_id"].nunique(), len(game_users))
    logger.info("  overlap users: %d (%.1f%%)", len(overlap_users), overlap_pct)

    # 5. Save parquet files
    movies_df = items[items["domain"] == "movie"].copy()
    games_df = items[items["domain"] == "game"].copy()

    filtered_ratings.to_parquet(ratings_path, compression="snappy")
    movie_ratings.to_parquet(OUTPUT_DIR / "movie_ratings.parquet", compression="snappy")
    game_ratings.to_parquet(OUTPUT_DIR / "game_ratings.parquet", compression="snappy")
    movies_df.to_parquet(OUTPUT_DIR / "movies.parquet", compression="snappy")
    games_df.to_parquet(OUTPUT_DIR / "games.parquet", compression="snappy")

    filtered_ratings.to_parquet(OUTPUT_DIR / "cross_domain_ratings.parquet", compression="snappy")

    cohort_parts = []
    if min_user_interactions > 0:
        cohort_parts.append(f"users >= {min_user_interactions} total interactions")
    else:
        cohort_parts.append("no user k-core filter")
    if sample_users:
        cohort_parts.append(f"sampled to ~{sample_users // 1000}K users")
    cohort_filter = ", ".join(cohort_parts)

    metadata = {
        "processing_date": datetime.now().isoformat(),
        "pair": "movie_game",
        "cohort_filter": cohort_filter,
        "total_users": n_users,
        "total_items": n_items,
        "total_ratings": n_ratings,
        "n_movie_ratings": n_movies,
        "n_game_ratings": n_games,
        "n_movie_items": int(movies_df["external_id"].nunique()),
        "n_game_items": int(games_df["external_id"].nunique()),
        "n_movie_users": len(movie_users),
        "n_game_users": len(game_users),
        "overlap_users": int(len(overlap_users)),
        "overlap_pct": overlap_pct,
        "movie_only_users": len(movie_users - game_users),
        "game_only_users": len(game_users - movie_users),
        "user_k_core": min_user_interactions,
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
    parser.add_argument("--min-user-interactions", type=int, default=MIN_USER_INTERACTIONS,
                        help="Min total interactions per user (0 to skip)")
    parser.add_argument("--sample-users", type=int, default=None,
                        help="Subsample to N users preserving overlap ratio")
    args = parser.parse_args()

    ratings, items, metadata = build_movie_game_dataset(
        force_reprocess=args.force_reprocess,
        min_user_interactions=args.min_user_interactions,
        sample_users=args.sample_users,
    )
    logger.info("Done. %d users, %d ratings", metadata["total_users"], metadata["total_ratings"])
