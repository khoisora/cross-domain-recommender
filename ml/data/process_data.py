"""Amazon review dataset processing: raw JSONL → parquet for movie_game pair.

Pipeline:
  1. Parse raw JSONL.gz reviews + metadata for movies and games
  2. Filter game metadata to actual games (exclude accessories, consoles)
  3. Deduplicate items (merge DVD/Blu-ray/digital/platform variants)
  4. Convert to implicit: keep ratings >= POSITIVE_THRESHOLD
  5. Item k-core: movies >= 20, games >= 10 interactions
  6. Optional user k-core (min_user_interactions, default 10)
  7. Optional overlap filter (min_movie_ratings, min_game_ratings)

Output files (written to --output-dir):
  - ratings.parquet          — all interactions (movie + game)
  - movies.parquet / games.parquet — item metadata per domain
  - dataset_metadata.json     — processing parameters and statistics
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import datetime
from pathlib import Path
from collections.abc import Iterator

import pandas as pd

try:
    from .item_dedup import deduplicate_dataset
except ImportError:
    from ml.data.item_dedup import deduplicate_dataset

logger = logging.getLogger(__name__)

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

POSITIVE_THRESHOLD = 4
MIN_USER_INTERACTIONS = 10
MOVIE_MIN_ITEM_INTERACTIONS = 20
GAME_MIN_ITEM_INTERACTIONS = 10


# ── JSONL parsing ────────────────────────────────────────────────────────────

def _parse_review(line: str, domain: str) -> dict | None:
    """Parse a JSONL review. Handles Amazon 2023 and older SNAP formats."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    user_id = obj.get("user_id") or obj.get("reviewerID")
    asin = obj.get("parent_asin") or obj.get("asin")
    rating = obj.get("rating") if obj.get("rating") is not None else obj.get("overall")
    if not user_id or not asin or rating is None:
        return None

    # Amazon 2023 uses ms epoch (>1e12); SNAP uses seconds.
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


def _parse_meta(line: str, domain: str) -> dict | None:
    """Parse a JSONL metadata line."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    asin = obj.get("parent_asin") or obj.get("asin")
    title = obj.get("title")
    if not asin or not title:
        return None

    desc = obj.get("description")
    description = " ".join(str(d) for d in desc)[:2000] if isinstance(desc, list) else (str(desc)[:2000] if desc else None)

    cats = obj.get("categories") or obj.get("category")
    if isinstance(cats, list):
        flat: list[str] = []
        for c in cats:
            flat.extend(c) if isinstance(c, list) else flat.append(str(c))
        categories = ", ".join(flat)
    elif isinstance(cats, str):
        categories = cats
    else:
        categories = None

    feat = obj.get("features") or obj.get("feature")
    features = ", ".join(str(f) for f in feat)[:1000] if isinstance(feat, list) else (str(feat)[:1000] if feat else None)

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
        "main_category": str(obj.get("main_category")) if obj.get("main_category") else None,
        "price": str(obj.get("price", "")) if obj.get("price") else None,
        "image_url": str(image_url)[:1024] if image_url else None,
    }


def _iter_jsonl(filepath: Path, domain: str, parser) -> Iterator[dict]:
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


def _load(filepath: Path, domain: str, parser) -> pd.DataFrame:
    df = pd.DataFrame(list(_iter_jsonl(filepath, domain, parser)))
    logger.info("Loaded %d %s records from %s", len(df), domain, filepath.name)
    return df


def _is_actual_game(categories: str | None) -> bool:
    """True if "Games" appears as a standalone category segment (excludes
    accessories, controllers, consoles)."""
    if not categories:
        return False
    return "Games" in [s.strip() for s in str(categories).split(",")]


# ── Main pipeline ────────────────────────────────────────────────────────────

def build_movie_game_dataset(
    force_reprocess: bool = False,
    min_user_interactions: int = MIN_USER_INTERACTIONS,
    min_movie_ratings: int = 0,
    min_game_ratings: int = 0,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build the movie_game processed dataset. Cached via output parquets."""
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ratings_path = out_dir / "ratings.parquet"
    metadata_path = out_dir / "dataset_metadata.json"

    if not force_reprocess and ratings_path.exists() and metadata_path.exists():
        logger.info("Loading existing processed dataset from %s", out_dir)
        ratings = pd.read_parquet(ratings_path)
        movies = pd.read_parquet(out_dir / "movies.parquet")
        games = pd.read_parquet(out_dir / "games.parquet")
        items = pd.concat([movies, games], ignore_index=True)
        with open(metadata_path) as f:
            metadata = json.load(f)
        logger.info("Loaded: %d ratings, %d users, %d items",
                    len(ratings), ratings["user_id"].nunique(), ratings["item_id"].nunique())
        return ratings, items, metadata

    for path in {**REVIEW_FILES, **META_FILES}.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing raw file: {path}")

    # 1. Load reviews + metadata
    ratings = pd.concat(
        [_load(REVIEW_FILES["movies"], "movie", _parse_review),
         _load(REVIEW_FILES["games"], "game", _parse_review)],
        ignore_index=True,
    ).drop_duplicates(subset=["user_id", "item_id"], keep="last")

    movie_meta = _load(META_FILES["movies"], "movie", _parse_meta)
    game_meta = _load(META_FILES["games"], "game", _parse_meta)

    # 2. Game item filter — drop accessories, consoles, gift cards.
    before = len(game_meta)
    game_meta = game_meta[game_meta["categories"].apply(_is_actual_game)].copy()
    logger.info("Game item filter: %d -> %d (removed %d non-game items)",
                before, len(game_meta), before - len(game_meta))

    items = pd.concat([movie_meta, game_meta], ignore_index=True) \
        .drop_duplicates(subset=["external_id"], keep="last")
    ratings = ratings[ratings["item_id"].isin(set(items["external_id"]))].copy()

    # 3. Deduplicate format/platform variants
    logger.info("Deduplicating items...")
    ratings, items = deduplicate_dataset(ratings, items)

    # 4. Implicit feedback: rating >= 4 = positive. Lower ratings dropped entirely
    # (absence is a stronger negative signal than a 3-star review).
    before_implicit = len(ratings)
    ratings = ratings[ratings["rating"] >= POSITIVE_THRESHOLD].copy()
    logger.info("Implicit conversion (rating >= %d): %d -> %d ratings",
                POSITIVE_THRESHOLD, before_implicit, len(ratings))

    # 5. Per-domain item k-core
    def _item_kcore(df: pd.DataFrame, min_count: int, name: str) -> pd.DataFrame:
        counts = df["item_id"].value_counts()
        keep = set(counts[counts >= min_count].index)
        out = df[df["item_id"].isin(keep)].copy()
        logger.info("%s item k-core (>=%d): %d -> %d ratings", name, min_count, len(df), len(out))
        return out

    filtered = pd.concat([
        _item_kcore(ratings[ratings["domain"] == "movie"], MOVIE_MIN_ITEM_INTERACTIONS, "Movie"),
        _item_kcore(ratings[ratings["domain"] == "game"], GAME_MIN_ITEM_INTERACTIONS, "Game"),
    ], ignore_index=True)

    # 6. User k-core (across both domains)
    if min_user_interactions > 0:
        counts = filtered["user_id"].value_counts()
        keep = set(counts[counts >= min_user_interactions].index)
        before = len(filtered)
        filtered = filtered[filtered["user_id"].isin(keep)].copy()
        logger.info("User k-core (>=%d): %d -> %d ratings (%d users)",
                    min_user_interactions, before, len(filtered), len(keep))

    # 7. Overlap filter — users with enough interactions in BOTH domains.
    # Ensures CDR models have cross-domain signal to learn from.
    if min_movie_ratings > 0 or min_game_ratings > 0:
        movie_per_user = filtered[filtered["domain"] == "movie"].groupby("user_id").size()
        game_per_user = filtered[filtered["domain"] == "game"].groupby("user_id").size()
        overlap = (set(movie_per_user[movie_per_user >= min_movie_ratings].index) &
                   set(game_per_user[game_per_user >= min_game_ratings].index))
        before = filtered["user_id"].nunique()
        filtered = filtered[filtered["user_id"].isin(overlap)].copy()
        logger.info("Overlap filter (movies>=%d, games>=%d): %d -> %d users",
                    min_movie_ratings, min_game_ratings, before, filtered["user_id"].nunique())

    # Sync items with remaining ratings
    items = items[items["external_id"].isin(set(filtered["item_id"]))].copy()

    # Stats
    movie_ratings = filtered[filtered["domain"] == "movie"]
    game_ratings = filtered[filtered["domain"] == "game"]
    movie_users = set(movie_ratings["user_id"])
    game_users = set(game_ratings["user_id"])
    overlap_users = movie_users & game_users
    n_users = filtered["user_id"].nunique()
    overlap_pct = round(100 * len(overlap_users) / n_users, 1) if n_users else 0

    logger.info("Final: %d users, %d items, %d ratings (%.1f%% overlap)",
                n_users, filtered["item_id"].nunique(), len(filtered), overlap_pct)
    logger.info("  movie: %d ratings, %d items, %d users",
                len(movie_ratings), movie_ratings["item_id"].nunique(), len(movie_users))
    logger.info("  game:  %d ratings, %d items, %d users",
                len(game_ratings), game_ratings["item_id"].nunique(), len(game_users))

    # Save
    movies_df = items[items["domain"] == "movie"]
    games_df = items[items["domain"] == "game"]
    filtered.to_parquet(ratings_path, compression="snappy")
    movies_df.to_parquet(out_dir / "movies.parquet", compression="snappy")
    games_df.to_parquet(out_dir / "games.parquet", compression="snappy")

    cohort = f"users >= {min_user_interactions} total interactions" if min_user_interactions > 0 else "no user k-core"
    if min_movie_ratings > 0 or min_game_ratings > 0:
        cohort += f", overlap users (movies >= {min_movie_ratings}, games >= {min_game_ratings})"

    metadata = {
        "processing_date": datetime.now().isoformat(),
        "pair": "movie_game",
        "cohort_filter": cohort,
        "total_users": n_users,
        "total_items": filtered["item_id"].nunique(),
        "total_ratings": len(filtered),
        "n_movie_ratings": len(movie_ratings),
        "n_game_ratings": len(game_ratings),
        "n_movie_items": int(movies_df["external_id"].nunique()),
        "n_game_items": int(games_df["external_id"].nunique()),
        "n_movie_users": len(movie_users),
        "n_game_users": len(game_users),
        "overlap_users": len(overlap_users),
        "overlap_pct": overlap_pct,
        "movie_only_users": len(movie_users - game_users),
        "game_only_users": len(game_users - movie_users),
        "user_k_core": min_user_interactions,
        "movie_item_k_core": MOVIE_MIN_ITEM_INTERACTIONS,
        "game_item_k_core": GAME_MIN_ITEM_INTERACTIONS,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved processed dataset to %s", out_dir)
    return filtered, items, metadata


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Build processed movie_game parquet dataset.")
    parser.add_argument("--force-reprocess", action="store_true")
    parser.add_argument("--min-user-interactions", type=int, default=MIN_USER_INTERACTIONS)
    parser.add_argument("--min-movie-ratings", type=int, default=0)
    parser.add_argument("--min-game-ratings", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    _, _, metadata = build_movie_game_dataset(
        force_reprocess=args.force_reprocess,
        min_user_interactions=args.min_user_interactions,
        min_movie_ratings=args.min_movie_ratings,
        min_game_ratings=args.min_game_ratings,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    logger.info("Done. %d users, %d ratings", metadata["total_users"], metadata["total_ratings"])
