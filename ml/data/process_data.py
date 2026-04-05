"""Amazon review dataset ingestion and normalization.

Supports the Amazon Reviews 2023 dataset format (McAuley Lab):
- Movies_and_TV reviews
- Video_Games reviews
- Books reviews (for movie↔book cross-domain experiments)

Dataset files are expected to be JSONL (one JSON object per line),
either plain or gzip-compressed.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Literal, Optional

import pandas as pd

try:
    from .item_dedup import deduplicate_dataset
except ImportError:
    from ml.data.item_dedup import deduplicate_dataset

logger = logging.getLogger(__name__)

# Local file paths (assuming data exists in data/amazon_2023/raw)
# Use absolute path from script location
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
# Common path components
AMAZON_DATASET_PATH = PROJECT_ROOT / "ml" / "data" / "amazon_2023"

RAW_DATA_DIR = AMAZON_DATASET_PATH / "raw"
OUTPUT_DIR = AMAZON_DATASET_PATH / "processed"

AMAZON_REVIEW_FILES = {
    "movies": RAW_DATA_DIR / "movies_reviews_2023.jsonl.gz",
    "games": RAW_DATA_DIR / "games_reviews_2023.jsonl.gz",
    # SNAP default filenames — download into ``raw/`` as-is
    "books": RAW_DATA_DIR / "reviews_Books_5.json.gz",
}

AMAZON_META_FILES = {
    "movies": RAW_DATA_DIR / "movies_meta_2023.jsonl.gz",
    "games": RAW_DATA_DIR / "games_meta_2023.jsonl.gz",
    "books": RAW_DATA_DIR / "meta_Books.json.gz",
}

# Download URLs — Stanford SNAP “5-core” review + metadata (same family as movies/games).
# Place downloaded files under ``raw/`` using the names in AMAZON_REVIEW_FILES / AMAZON_META_FILES,
# or gzip-decode JSON lines into ``.jsonl.gz`` as used elsewhere in this project.
AMAZON_REVIEW_URLS = {
    "movies": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz",
    "games": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz",
    "books": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz",
}

AMAZON_META_URLS = {
    "movies": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Movies_and_TV.json.gz",
    "games": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Video_Games.json.gz",
    "books": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz",
}

# Separate processed output for movie↔book (does not overwrite movie↔game ``processed/``).
OUTPUT_DIR_MOVIE_BOOK = AMAZON_DATASET_PATH / "processed_movie_book"

# Transfer-focused dataset variants (overlap-user subsets for CDR experiments).
OUTPUT_DIR_TRANSFER_LOOSE = AMAZON_DATASET_PATH / "processed_transfer_loose"
OUTPUT_DIR_TRANSFER_STRICT = AMAZON_DATASET_PATH / "processed_transfer_strict"

TRANSFER_TIER_DIRS: dict[str, Path] = {
    "loose": OUTPUT_DIR_TRANSFER_LOOSE,
    "strict": OUTPUT_DIR_TRANSFER_STRICT,
}


def check_local_file(file_path: Path) -> bool:
    """Check if local file exists and is accessible."""
    if file_path.exists():
        logger.info(f"Found local file: {file_path}")
        return True
    else:
        logger.error(f"Local file not found: {file_path}")
        return False


def filter_users_and_items_with_min_reviews(
    ratings: pd.DataFrame,
    domain: str,
    min_user_ratings: int = None,
    min_item_ratings: int = None,
) -> pd.DataFrame:
    """Apply domain-specific k-core filtering (single pass).

    Default thresholds:
        movie: users ≥ 10 interactions, items ≥ 20 interactions
        game:  users ≥  2 interactions, items ≥ 10 interactions
    """
    if domain == "movie":
        min_user_ratings = min_user_ratings or 10
        min_item_ratings = min_item_ratings or 20
    elif domain == "game":
        min_user_ratings = min_user_ratings or 2
        min_item_ratings = min_item_ratings or 10
    elif domain == "book":
        # Books: huge catalog — slightly looser user threshold, stricter item threshold than games.
        min_user_ratings = min_user_ratings or 5
        min_item_ratings = min_item_ratings or 15
    else:
        min_user_ratings = min_user_ratings or 2
        min_item_ratings = min_item_ratings or 10

    original_count = len(ratings)

    user_counts = ratings["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    ratings = ratings[ratings["user_id"].isin(valid_users)]

    item_counts = ratings["item_id"].value_counts()
    valid_items = item_counts[item_counts >= min_item_ratings].index
    ratings = ratings[ratings["item_id"].isin(valid_items)]

    logger.info(
        "%s k-core (u≥%d, i≥%d): %d → %d ratings (%.1f%% retained)",
        domain, min_user_ratings, min_item_ratings,
        original_count, len(ratings), len(ratings) / original_count * 100,
    )
    return ratings


# Keywords indicating non-transferable movie content (no game genre equivalent).
# Sports is intentionally excluded — sports games (FIFA, NBA 2K, Madden) exist.
_NONTRANSFERABLE_MOVIE_CATEGORY_KEYWORDS = [
    "exercise", "workout", "fitness", "aerobics", "yoga", "pilates",
    "dance instruction", "weight loss",          # fitness DVDs
    "opera",                                      # classical opera
    "classical music",                            # classical music concerts
]
_NONTRANSFERABLE_MOVIE_MAIN_CATEGORIES = {"Sports & Outdoors"}


def _filter_nontransferable_movies(items_df: pd.DataFrame) -> pd.DataFrame:
    """Remove movie items with no game-genre equivalent (fitness DVDs, opera/classical).

    Keeps sports movies (sports games exist), documentaries, musicals, drama, etc.
    Games are passed through unchanged.
    """
    movie_mask = items_df["domain"] == "movie"
    movies = items_df[movie_mask]

    cats_lower = movies["categories"].fillna("").str.lower()
    main_cat = movies["main_category"].fillna("")

    is_nontransferable = main_cat.isin(_NONTRANSFERABLE_MOVIE_MAIN_CATEGORIES) | \
        cats_lower.apply(lambda c: any(kw in c for kw in _NONTRANSFERABLE_MOVIE_CATEGORY_KEYWORDS))

    removed = int(is_nontransferable.sum())
    logger.info(
        "Genre filter: removed %d non-transferable movie items (fitness/opera/classical)", removed
    )

    keep_ids = set(movies[~is_nontransferable]["external_id"]) | \
               set(items_df[~movie_mask]["external_id"])
    return items_df[items_df["external_id"].isin(keep_ids)].copy()


def filter_items_by_overlap_users(
    dom_a_ratings: pd.DataFrame,
    dom_b_ratings: pd.DataFrame,
    items_df: pd.DataFrame,
    label_a: str = "A",
    label_b: str = "B",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove items not rated by any cross-domain (overlap) user.

    Items exclusively rated by domain-only users carry no CDR signal because
    the model can never leverage source-domain preferences for those items.
    """
    overlap_users = set(dom_a_ratings["user_id"]) & set(dom_b_ratings["user_id"])
    if not overlap_users:
        logger.warning("No overlap users found — skipping overlap-user item filter")
        return dom_a_ratings, dom_b_ratings, items_df

    logger.info("Overlap users: %d", len(overlap_users))

    a_overlap_items = set(
        dom_a_ratings[dom_a_ratings["user_id"].isin(overlap_users)]["item_id"]
    )
    b_overlap_items = set(
        dom_b_ratings[dom_b_ratings["user_id"].isin(overlap_users)]["item_id"]
    )

    before_a = dom_a_ratings["item_id"].nunique()
    before_b = dom_b_ratings["item_id"].nunique()

    dom_a_ratings = dom_a_ratings[dom_a_ratings["item_id"].isin(a_overlap_items)].copy()
    dom_b_ratings = dom_b_ratings[dom_b_ratings["item_id"].isin(b_overlap_items)].copy()

    keep_ids = a_overlap_items | b_overlap_items
    items_df = items_df[items_df["external_id"].isin(keep_ids)].copy()

    logger.info(
        "Overlap-user item filter: %s %d→%d items, %s %d→%d items",
        label_a,
        before_a,
        dom_a_ratings["item_id"].nunique(),
        label_b,
        before_b,
        dom_b_ratings["item_id"].nunique(),
    )
    return dom_a_ratings, dom_b_ratings, items_df

def filter_transfer_focused_users(
    dom_a_ratings: pd.DataFrame,
    dom_b_ratings: pd.DataFrame,
    tier: Literal["loose", "strict"] = "loose",
    source_min: int = 10,
    target_min: int = 1,
    target_max: Optional[int] = None,
    label_source: str = "movie",
    label_target: str = "game",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only overlap users with rich source history and sparse target history.

    Tiers (defaults for movie_game, source=movie, target=game):
        loose:  source >= 10, target >= 1       (all usable overlap users)
        strict: source >= 10, 1 <= target <= 3  (strongest transfer regime)

    Returns filtered (source_ratings, target_ratings) — items are NOT pruned here
    so both model families evaluate on the same target catalog.
    """
    if tier == "strict" and target_max is None:
        target_max = 3

    source_counts = dom_a_ratings.groupby("user_id").size()
    target_counts = dom_b_ratings.groupby("user_id").size()

    source_ok = set(source_counts[source_counts >= source_min].index)
    target_ok = set(target_counts[target_counts >= target_min].index)
    if target_max is not None:
        target_ok &= set(target_counts[target_counts <= target_max].index)

    keep_users = source_ok & target_ok

    before_a = dom_a_ratings["user_id"].nunique()
    before_b = dom_b_ratings["user_id"].nunique()

    dom_a_ratings = dom_a_ratings[dom_a_ratings["user_id"].isin(keep_users)].copy()
    dom_b_ratings = dom_b_ratings[dom_b_ratings["user_id"].isin(keep_users)].copy()

    logger.info(
        "Transfer-focused user filter (tier=%s, %s≥%d, %d≤%s≤%s): "
        "%s users %d→%d, %s users %d→%d, overlap=%d",
        tier,
        label_source, source_min,
        target_min, label_target, target_max if target_max is not None else "∞",
        label_source, before_a, dom_a_ratings["user_id"].nunique(),
        label_target, before_b, dom_b_ratings["user_id"].nunique(),
        len(keep_users),
    )
    return dom_a_ratings, dom_b_ratings


@dataclass
class Rawrating:
    """A single parsed rating from the Amazon dataset."""

    user_id: str
    item_id: str  # ASIN
    rating: float
    review_text: Optional[str]
    timestamp: Optional[datetime]
    domain: Literal["movie", "game", "book"]


@dataclass
class RawItemMeta:
    """Parsed item metadata from the Amazon metadata files."""

    external_id: str  # ASIN
    domain: Literal["movie", "game", "book"]
    title: str
    description: Optional[str]
    categories: Optional[str]
    features: Optional[str]
    main_category: Optional[str]
    price: Optional[str]
    image_url: Optional[str]


def parse_review_line(line: str, domain: Literal["movie", "game", "book"]) -> Optional[Rawrating]:
    """Parse a single JSONL review line into a Rawrating."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    user_id = obj.get("user_id") or obj.get("reviewerID")
    asin = obj.get("parent_asin") or obj.get("asin")
    rating = obj.get("rating") if obj.get("rating") is not None else obj.get("overall")
    text = obj.get("text") or obj.get("reviewText")
    ts_raw = obj.get("timestamp") or obj.get("unixReviewTime")

    if not user_id or not asin or rating is None:
        return None

    _AMAZON_LAUNCH = datetime(1994, 1, 1)
    _TS_CEILING = datetime(2035, 1, 1)
    ts = None
    if ts_raw:
        try:
            if isinstance(ts_raw, (int, float)):
                epoch = ts_raw / 1000 if ts_raw > 1e12 else float(ts_raw)
                candidate = datetime.fromtimestamp(epoch)
            else:
                candidate = datetime.fromisoformat(str(ts_raw))
            if _AMAZON_LAUNCH <= candidate <= _TS_CEILING:
                ts = candidate
        except (ValueError, OSError):
            pass

    return Rawrating(
        user_id=str(user_id),
        item_id=str(asin),
        rating=float(rating),
        review_text=str(text)[:5000] if text else None,
        timestamp=ts,
        domain=domain,
    )


def parse_meta_line(line: str, domain: Literal["movie", "game", "book"]) -> Optional[RawItemMeta]:
    """Parse a single JSONL metadata line into a RawItemMeta."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    asin = obj.get("parent_asin") or obj.get("asin")
    title = obj.get("title")
    if not asin or not title:
        return None

    description = None
    desc_list = obj.get("description")
    if isinstance(desc_list, list):
        description = " ".join(str(d) for d in desc_list)[:2000]
    elif isinstance(desc_list, str):
        description = desc_list[:2000]

    features = None
    feat_list = obj.get("features") or obj.get("feature")
    if isinstance(feat_list, list):
        features = ", ".join(str(f) for f in feat_list)[:1000]
    elif isinstance(feat_list, str):
        features = feat_list[:1000]

    main_category = obj.get("main_category")

    categories = None
    cats = obj.get("categories") or obj.get("category")
    if isinstance(cats, list):
        flat = []
        for c in cats:
            if isinstance(c, list):
                flat.extend(c)
            else:
                flat.append(str(c))
        categories = ", ".join(flat)
    elif isinstance(cats, str):
        categories = cats

    images = obj.get("images") or obj.get("imageURL") or obj.get("imUrl")
    image_url = None
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, dict):
            image_url = first.get("large") or first.get("hi_res") or first.get("thumb")
        else:
            image_url = str(first)
    elif isinstance(images, str):
        image_url = images

    return RawItemMeta(
        external_id=str(asin),
        domain=domain,
        title=str(title)[:512],
        description=description,
        categories=categories,
        features=features,
        main_category=str(main_category) if main_category else None,
        price=str(obj.get("price", "")) if obj.get("price") else None,
        image_url=str(image_url)[:1024] if image_url else None,
    )


def iter_jsonl_file(
    filepath: Path, domain: Literal["movie", "game", "book"], file_type: Literal["review", "meta"]
) -> Iterator[Rawrating | RawItemMeta]:
    """Iterate over a JSONL file (plain or gzip), yielding parsed records."""
    parser = parse_review_line if file_type == "review" else parse_meta_line
    opener = gzip.open if filepath.suffix == ".gz" else open
    count = 0
    errors = 0

    with opener(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = parser(line, domain)
            if record:
                count += 1
                if count == 1:
                    logger.info(f"sample record from {filepath.name}: {record}")
                yield record
            else:
                errors += 1
            if count % 100_000 == 0:
                logger.info("Parsed %d records from %s (%d errors)", count, filepath.name, errors)

    logger.info(
        "Finished parsing %s: %d records, %d errors", filepath.name, count, errors
    )


def load_reviews_to_dataframe(
    filepath: Path, domain: Literal["movie", "game", "book"], max_rows: Optional[int] = None
) -> pd.DataFrame:
    """Load review data from a JSONL file into a DataFrame.

    Args:
        filepath: Path to the JSONL(.gz) review file.
        domain: 'movie', 'game', or 'book'.
        max_rows: Optional limit on number of rows to read.

    Returns:
        DataFrame with columns: user_id, item_id,
        rating, review_text, timestamp, domain.
    """
    records = []
    for rating in iter_jsonl_file(filepath, domain, "review"):
        records.append({
            "user_id": rating.user_id,
            "item_id": rating.item_id,
            "rating": rating.rating,
            "review_text": rating.review_text,
            "timestamp": rating.timestamp,
            "domain": rating.domain,
        })
        if max_rows and len(records) >= max_rows:
            break

    df = pd.DataFrame(records)
    logger.info(
        "Loaded %d %s reviews from %s", len(df), domain, filepath.name
    )
    return df


def load_metadata_to_dataframe(
    filepath: Path, domain: Literal["movie", "game", "book"], max_rows: Optional[int] = None
) -> pd.DataFrame:
    """Load item metadata from a JSONL file into a DataFrame."""
    records = []
    for meta in iter_jsonl_file(filepath, domain, "meta"):
        records.append({
            "external_id": meta.external_id,
            "domain": meta.domain,
            "title": meta.title,
            "description": meta.description,
            "categories": meta.categories,
            "features": meta.features,
            "main_category": meta.main_category,
            "price": meta.price,
            "image_url": meta.image_url,
        })
        if max_rows and len(records) >= max_rows:
            break

    df = pd.DataFrame(records)
    logger.info(
        "Loaded %d %s metadata records from %s", len(df), domain, filepath.name
    )
    return df


def _is_actual_game(categories: Optional[str]) -> bool:
    """Check if item is an actual game (not accessories, consoles, subscriptions, etc).

    Matches 'Games' as a standalone segment in the comma-separated categories.
    e.g. 'Video Games, PlayStation 4, Games' -> True
         'Games, Action'                     -> True
         'Video Games, PlayStation 4, Accessories, Controllers' -> False
    """
    if not categories:
        return False
    # ``categories`` can occasionally arrive as non-string (e.g. float/Arrow scalar);
    # coerce to string before splitting to avoid AttributeError.
    segments = [str(s).strip() for s in str(categories).split(",")]
    return "Games" in segments


def _is_actual_book(categories: Optional[str], main_category: Optional[str]) -> bool:
    """Keep typical book ASINs; drop obvious non-book product lines when categories are present."""
    if main_category and str(main_category).strip().lower() in (
        "kindle store", "audible books & originals"
    ):
        return False
    if not categories:
        return True
    segments = [s.strip() for s in categories.split(",")]
    return "Books" in segments


def build_unified_dataset(
    movie_reviews_path: Path,
    game_reviews_path: Path,
    movie_meta_path: Optional[Path] = None,
    game_meta_path: Optional[Path] = None,
    max_reviews_per_domain: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Movie + game unified dataset (backward-compatible wrapper)."""
    return build_unified_two_domain_dataset(
        movie_reviews_path,
        game_reviews_path,
        "movie",
        "game",
        movie_meta_path,
        game_meta_path,
        max_reviews_per_domain,
        second_meta_filter=_is_actual_game,
        second_meta_filter_needs_main_cat=False,
    )


def build_unified_two_domain_dataset(
    first_reviews_path: Path,
    second_reviews_path: Path,
    first_domain: str,
    second_domain: str,
    first_meta_path: Optional[Path] = None,
    second_meta_path: Optional[Path] = None,
    max_reviews_per_domain: Optional[int] = None,
    second_meta_filter: Optional[Callable[..., bool]] = None,
    second_meta_filter_needs_main_cat: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a unified two-domain dataset from Amazon review + meta files.

    Returns:
        (ratings_df, items_df) — unified DataFrames ready for dedup / k-core.
    """
    logger.info(
        "Building unified dataset: %s + %s ...",
        first_domain,
        second_domain,
    )

    first_reviews = load_reviews_to_dataframe(
        first_reviews_path, first_domain, max_rows=max_reviews_per_domain
    )
    second_reviews = load_reviews_to_dataframe(
        second_reviews_path, second_domain, max_rows=max_reviews_per_domain
    )
    ratings = pd.concat([first_reviews, second_reviews], ignore_index=True)

    ratings = ratings.drop_duplicates(subset=["user_id", "item_id"], keep="last")

    items_records: list[pd.DataFrame] = []
    if first_meta_path and first_meta_path.exists():
        items_records.append(load_metadata_to_dataframe(first_meta_path, first_domain))
    if second_meta_path and second_meta_path.exists():
        second_meta = load_metadata_to_dataframe(second_meta_path, second_domain)
        if second_meta_filter is not None:
            before_count = len(second_meta)
            if second_meta_filter_needs_main_cat:
                mask = second_meta.apply(
                    lambda r: second_meta_filter(
                        r.get("categories"), r.get("main_category")
                    ),
                    axis=1,
                )
            else:
                mask = second_meta["categories"].apply(second_meta_filter)
            second_meta = second_meta[mask].copy()
            logger.info(
                "Filtered %s items: %d -> %d",
                second_domain,
                before_count,
                len(second_meta),
            )
        items_records.append(second_meta)

    if not items_records:
        logger.error("No item metadata records found. Cannot proceed without item data.")
        raise SystemExit("No item records found in dataset files.")

    items_df = pd.concat(items_records, ignore_index=True)
    items_df = items_df.drop_duplicates(subset=["external_id"], keep="last")

    valid_items = set(items_df["external_id"])
    ratings_before = len(ratings)
    ratings = ratings[ratings["item_id"].isin(valid_items)].copy()
    logger.info(
        "Filtered ratings after item filtering: %d -> %d",
        ratings_before,
        len(ratings),
    )

    logger.info(
        "Unified dataset: %d ratings, %d items (%d users)",
        len(ratings),
        len(items_df),
        ratings["user_id"].nunique(),
    )
    return ratings, items_df


def build_cross_domain_dataset(
    processed_dir: Optional[Path] = None,
    max_reviews_per_domain: Optional[int] = None,
    force_reprocess: bool = False,
    genre_filter: bool = True,
    overlap_item_filter: bool = True,
    pair: Literal["movie_game", "movie_book"] = "movie_game",
    transfer_tier: Optional[Literal["loose", "strict"]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Build cross-domain Amazon dataset with domain-specific k-core filtering.

    Filtering pipeline:
      1. Genre filter: remove non-transferable movie items (fitness DVDs, opera).
      2. Domain-specific k-core (single pass).
      3. Overlap-user item filter: drop items rated only by domain-exclusive users.
      4. (optional) Transfer-focused user filter: keep only source-rich /
         target-sparse overlap users for CDR-favorable benchmarking.

    Args:
        processed_dir: Output directory (auto-resolved from pair + transfer_tier if None).
        pair: ``movie_game`` (default) or ``movie_book`` — selects raw files and k-core rules.
        transfer_tier: ``None`` (default, no user filter), ``"loose"``
            (source≥10, target≥1), or ``"strict"`` (source≥10, 1≤target≤3).

    Returns:
        (ratings_df, items_df, density) tuple
    """
    if processed_dir is None:
        if transfer_tier is not None:
            processed_dir = TRANSFER_TIER_DIRS[transfer_tier]
        else:
            processed_dir = OUTPUT_DIR if pair == "movie_game" else OUTPUT_DIR_MOVIE_BOOK
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    secondary = "game" if pair == "movie_game" else "book"
    logger.info(
        "Building cross-domain dataset (pair=%s, secondary=%s) → %s",
        pair,
        secondary,
        processed_dir,
    )

    ratings_parquet_file = processed_dir / "ratings.parquet"
    movies_parquet_file = processed_dir / "movies.parquet"
    games_parquet_file = processed_dir / "games.parquet"
    books_parquet_file = processed_dir / "books.parquet"
    secondary_items_parquet = games_parquet_file if pair == "movie_game" else books_parquet_file
    metadata_file = processed_dir / "dataset_metadata.json"

    expected_params_base = {
        "pair": pair,
        "max_reviews_per_domain": max_reviews_per_domain,
        "genre_filter": genre_filter,
        "overlap_item_filter": overlap_item_filter,
        "transfer_tier": transfer_tier,
        "min_rating": 4,
        "movie_min_user_ratings": 10,
        "movie_min_item_ratings": 20,
    }
    if pair == "movie_game":
        expected_params = {
            **expected_params_base,
            "game_min_user_ratings": 2,
            "game_min_item_ratings": 10,
        }
    else:
        expected_params = {
            **expected_params_base,
            "book_min_user_ratings": 5,
            "book_min_item_ratings": 15,
        }

    if metadata_file.exists() and not force_reprocess:
        parquet_ok = (
            ratings_parquet_file.exists()
            and movies_parquet_file.exists()
            and secondary_items_parquet.exists()
        )
        if parquet_ok:
            logger.info("Loading existing processed dataset...")
            ratings = pd.read_parquet(ratings_parquet_file)
            movies = pd.read_parquet(movies_parquet_file)
            secondary_df = pd.read_parquet(secondary_items_parquet)
            items = pd.concat([movies, secondary_df], ignore_index=True)

            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            params_match = all(
                metadata.get(key) == expected_value
                for key, expected_value in expected_params.items()
            )

            if pair == "movie_game":
                _source_file_map = {
                    "movies_reviews": AMAZON_REVIEW_FILES["movies"],
                    "games_reviews": AMAZON_REVIEW_FILES["games"],
                    "movies_meta": AMAZON_META_FILES["movies"],
                    "games_meta": AMAZON_META_FILES["games"],
                }
            else:
                _source_file_map = {
                    "movies_reviews": AMAZON_REVIEW_FILES["movies"],
                    "books_reviews": AMAZON_REVIEW_FILES["books"],
                    "movies_meta": AMAZON_META_FILES["movies"],
                    "books_meta": AMAZON_META_FILES["books"],
                }
            saved_mtimes = metadata.get("source_mtimes", {})
            mtimes_match = all(
                saved_mtimes.get(name) == (path.stat().st_mtime if path.exists() else None)
                for name, path in _source_file_map.items()
            )

            if params_match and mtimes_match:
                n_users = ratings["user_id"].nunique()
                n_items = ratings["item_id"].nunique()
                n_ratings = len(ratings)
                density = n_ratings / (n_users * n_items) if n_users and n_items else 0.0

                logger.info("Loaded existing dataset with matching parameters:")
                logger.info("  Users: %s", f"{n_users:,}")
                logger.info("  Items: %s", f"{n_items:,}")
                logger.info("  Ratings: %s", f"{n_ratings:,}")
                logger.info("  Density: %f", density)
                logger.info("  Processing date: %s", metadata.get("processing_date"))

                return ratings, items, density
            if not params_match:
                logger.info("Parameters changed, reprocessing dataset...")
            else:
                logger.info("Source files changed, reprocessing dataset...")

    if pair == "movie_game":
        required_files = [
            AMAZON_REVIEW_FILES["movies"],
            AMAZON_REVIEW_FILES["games"],
            AMAZON_META_FILES["movies"],
            AMAZON_META_FILES["games"],
        ]
    else:
        required_files = [
            AMAZON_REVIEW_FILES["movies"],
            AMAZON_REVIEW_FILES["books"],
            AMAZON_META_FILES["movies"],
            AMAZON_META_FILES["books"],
        ]

    missing_files = [f for f in required_files if not check_local_file(f)]
    if missing_files:
        logger.error("Missing required files:")
        for f in missing_files:
            logger.error("  - %s", f)
        raise FileNotFoundError(
            "Please download SNAP Amazon files into ml/data/amazon_2023/raw/ "
            "(see AMAZON_REVIEW_URLS / AMAZON_META_URLS in process_data.py)."
        )

    if pair == "movie_game":
        ratings, items = build_unified_dataset(
            AMAZON_REVIEW_FILES["movies"],
            AMAZON_REVIEW_FILES["games"],
            AMAZON_META_FILES["movies"],
            AMAZON_META_FILES["games"],
            max_reviews_per_domain,
        )
    else:
        ratings, items = build_unified_two_domain_dataset(
            AMAZON_REVIEW_FILES["movies"],
            AMAZON_REVIEW_FILES["books"],
            "movie",
            "book",
            AMAZON_META_FILES["movies"],
            AMAZON_META_FILES["books"],
            max_reviews_per_domain,
            second_meta_filter=_is_actual_book,
            second_meta_filter_needs_main_cat=True,
        )
    
    # ── Deduplication ────────────────────────────────────────────────────────
    logger.info("Applying item deduplication...")
    logger.info(f"Before dedup: {len(ratings)} ratings, {len(items)} items")
    ratings, items = deduplicate_dataset(ratings, items)
    logger.info(f"After dedup: {len(ratings)} ratings, {len(items)} items")

    # ── Genre filter: remove non-transferable movie items ─────────────────
    if genre_filter:
        before_items = len(items)
        items = _filter_nontransferable_movies(items)
        valid_item_ids = set(items["external_id"])
        ratings = ratings[ratings["item_id"].isin(valid_item_ids)].copy()
        logger.info("After genre filter: %d items (removed %d), %d ratings",
                    len(items), before_items - len(items), len(ratings))

    # ── Rating filter: keep only positive interactions (≥4 stars) ────────
    before_rating_filter = len(ratings)
    ratings = ratings[ratings["rating"] >= 4.0].copy()
    logger.info("Rating filter (≥4): %d → %d ratings (%.1f%% retained)",
                before_rating_filter, len(ratings), 100 * len(ratings) / before_rating_filter)

    # ── Domain-specific k-core ────────────────────────────────────────────
    movie_ratings = filter_users_and_items_with_min_reviews(
        ratings[ratings["domain"] == "movie"].copy(), domain="movie"
    )
    secondary_ratings = filter_users_and_items_with_min_reviews(
        ratings[ratings["domain"] == secondary].copy(), domain=secondary
    )

    # ── Overlap-user item filter ──────────────────────────────────────────
    if overlap_item_filter:
        movie_ratings, secondary_ratings, items = filter_items_by_overlap_users(
            movie_ratings,
            secondary_ratings,
            items,
            label_a="movie",
            label_b=secondary,
        )

    # ── Transfer-focused user filter (CDR-favorable subset) ───────────
    if transfer_tier is not None:
        movie_ratings, secondary_ratings = filter_transfer_focused_users(
            movie_ratings,
            secondary_ratings,
            tier=transfer_tier,
            label_source="movie",
            label_target=secondary,
        )

    filtered_ratings = pd.concat([movie_ratings, secondary_ratings], ignore_index=True)

    # Sync items to only those present in filtered ratings
    valid_items = set(filtered_ratings["item_id"])
    filtered_items = items[items["external_id"].isin(valid_items)].copy()
    
    # Calculate density
    n_users = filtered_ratings['user_id'].nunique()
    n_items = filtered_ratings['item_id'].nunique()
    n_ratings = len(filtered_ratings)
    density = n_ratings / (n_users * n_items)
    
    # Log statistics
    logger.info(f"Final Cross-Domain 2023 Dataset Statistics:")
    logger.info(f"  Users: {n_users:,}")
    logger.info(f"  Items: {n_items:,}")
    logger.info(f"  Ratings: {n_ratings:,}")
    logger.info(f"  Density: {density:.6f}")
    
    # Domain distribution
    domain_stats = filtered_ratings.groupby('domain').agg({
        'user_id': 'nunique',
        'item_id': 'nunique', 
        'rating': 'count',
    }).rename(columns={
        'user_id': 'users',
        'item_id': 'items',
        'rating': 'ratings'
    })
    
    logger.info(f"Final Domain Distribution:")
    for domain, stats in domain_stats.iterrows():
        logger.info(f"  {domain}: {stats['users']:,} users, {stats['items']:,} items, {stats['ratings']:,} ratings")
    
    # Cross-domain overlap
    movie_users = set(filtered_ratings[filtered_ratings["domain"] == "movie"]["user_id"])
    secondary_users = set(filtered_ratings[filtered_ratings["domain"] == secondary]["user_id"])
    overlap_users = movie_users & secondary_users
    all_users_union = movie_users | secondary_users
    pct = len(overlap_users) / len(all_users_union) * 100 if all_users_union else 0.0
    logger.info(
        "Cross-domain users: %s (%.1f%% of all users)",
        f"{len(overlap_users):,}",
        pct,
    )
    
    # Save processed dataset as parquet
    logger.info("Saving processed dataset for future use...")
    
    # Split items by domain
    movies = filtered_items[filtered_items["domain"] == "movie"].copy()
    secondary_items = filtered_items[filtered_items["domain"] == secondary].copy()

    movie_ratings = filtered_ratings[filtered_ratings["domain"] == "movie"].copy()
    secondary_ratings_save = filtered_ratings[filtered_ratings["domain"] == secondary].copy()
    cross_domain_ratings = filtered_ratings[filtered_ratings["user_id"].isin(overlap_users)].copy()

    movie_ratings_file = processed_dir / "movie_ratings.parquet"
    secondary_ratings_file = (
        processed_dir / "game_ratings.parquet"
        if pair == "movie_game"
        else processed_dir / "book_ratings.parquet"
    )
    cross_domain_file = processed_dir / "cross_domain_ratings.parquet"

    logger.info("Converting to parquet format...")
    filtered_ratings.to_parquet(ratings_parquet_file, compression="snappy")
    movie_ratings.to_parquet(movie_ratings_file, compression="snappy")
    secondary_ratings_save.to_parquet(secondary_ratings_file, compression="snappy")
    cross_domain_ratings.to_parquet(cross_domain_file, compression="snappy")
    movies.to_parquet(movies_parquet_file, compression="snappy")
    secondary_items.to_parquet(secondary_items_parquet, compression="snappy")

    if pair == "movie_game":
        source_mtimes = {
            "movies_reviews": AMAZON_REVIEW_FILES["movies"].stat().st_mtime
            if AMAZON_REVIEW_FILES["movies"].exists()
            else None,
            "games_reviews": AMAZON_REVIEW_FILES["games"].stat().st_mtime
            if AMAZON_REVIEW_FILES["games"].exists()
            else None,
            "movies_meta": AMAZON_META_FILES["movies"].stat().st_mtime
            if AMAZON_META_FILES["movies"].exists()
            else None,
            "games_meta": AMAZON_META_FILES["games"].stat().st_mtime
            if AMAZON_META_FILES["games"].exists()
            else None,
        }
        source_data = {
            "movies_reviews": str(AMAZON_REVIEW_FILES["movies"]),
            "games_reviews": str(AMAZON_REVIEW_FILES["games"]),
            "movies_meta": str(AMAZON_META_FILES["movies"]),
            "games_meta": str(AMAZON_META_FILES["games"]),
        }
        split_stats = {
            "movie_ratings": len(movie_ratings),
            "game_ratings": len(secondary_ratings_save),
            "cross_domain_ratings": len(cross_domain_ratings),
            "cross_domain_users": len(overlap_users),
        }
    else:
        source_mtimes = {
            "movies_reviews": AMAZON_REVIEW_FILES["movies"].stat().st_mtime
            if AMAZON_REVIEW_FILES["movies"].exists()
            else None,
            "books_reviews": AMAZON_REVIEW_FILES["books"].stat().st_mtime
            if AMAZON_REVIEW_FILES["books"].exists()
            else None,
            "movies_meta": AMAZON_META_FILES["movies"].stat().st_mtime
            if AMAZON_META_FILES["movies"].exists()
            else None,
            "books_meta": AMAZON_META_FILES["books"].stat().st_mtime
            if AMAZON_META_FILES["books"].exists()
            else None,
        }
        source_data = {
            "movies_reviews": str(AMAZON_REVIEW_FILES["movies"]),
            "books_reviews": str(AMAZON_REVIEW_FILES["books"]),
            "movies_meta": str(AMAZON_META_FILES["movies"]),
            "books_meta": str(AMAZON_META_FILES["books"]),
        }
        split_stats = {
            "movie_ratings": len(movie_ratings),
            "book_ratings": len(secondary_ratings_save),
            "cross_domain_ratings": len(cross_domain_ratings),
            "cross_domain_users": len(overlap_users),
        }

    metadata = {
        **expected_params,
        "processing_date": datetime.now().isoformat(),
        "total_users": n_users,
        "total_items": n_items,
        "total_ratings": n_ratings,
        "density": density,
        "cross_domain_users": len(overlap_users),
        "cross_domain_percentage": pct,
        "domain_stats": domain_stats.to_dict(),
        "split_stats": split_stats,
        "source_mtimes": source_mtimes,
        "source_data": source_data,
        "output_files": {
            "ratings_parquet": str(ratings_parquet_file),
            "movie_ratings_parquet": str(movie_ratings_file),
            "secondary_ratings_parquet": str(secondary_ratings_file),
            "cross_domain_ratings_parquet": str(cross_domain_file),
            "movies_parquet": str(movies_parquet_file),
            "secondary_items_parquet": str(secondary_items_parquet),
            "metadata": str(metadata_file),
        },
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved processed dataset to %s", processed_dir)
    logger.info("  - ratings.parquet: %s ratings (all)", f"{len(filtered_ratings):,}")
    logger.info("  - movie_ratings.parquet: %s", f"{len(movie_ratings):,}")
    logger.info("  - %s: %s", secondary_ratings_file.name, f"{len(secondary_ratings_save):,}")
    logger.info(
        "  - cross_domain_ratings.parquet: %s ratings (%s users)",
        f"{len(cross_domain_ratings):,}",
        f"{len(overlap_users):,}",
    )
    logger.info("  - movies.parquet: %s items", f"{len(movies):,}")
    logger.info("  - %s: %s items", secondary_items_parquet.name, f"{len(secondary_items):,}")
    logger.info("  - dataset_metadata.json")
    
    return filtered_ratings, filtered_items, density


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build processed Amazon cross-domain parquet.")
    parser.add_argument(
        "--pair",
        choices=["movie_game", "movie_book"],
        default="movie_game",
        help="movie_game → ml/data/amazon_2023/processed; "
        "movie_book → ml/data/amazon_2023/processed_movie_book",
    )
    parser.add_argument("--force-reprocess", action="store_true")
    parser.add_argument("--max-reviews-per-domain", type=int, default=None)
    parser.add_argument(
        "--transfer-tier",
        choices=["loose", "strict"],
        default=None,
        help="Build a transfer-focused CDR subset: "
        "loose (source≥10, target≥1) or strict (source≥10, 1≤target≤3). "
        "Outputs to processed_transfer_loose/ or processed_transfer_strict/.",
    )
    args = parser.parse_args()

    try:
        if args.transfer_tier is not None:
            out = TRANSFER_TIER_DIRS[args.transfer_tier]
        elif args.pair == "movie_game":
            out = OUTPUT_DIR
        else:
            out = OUTPUT_DIR_MOVIE_BOOK

        ratings, items, density = build_cross_domain_dataset(
            processed_dir=out,
            max_reviews_per_domain=args.max_reviews_per_domain,
            force_reprocess=args.force_reprocess,
            genre_filter=True,
            overlap_item_filter=True,
            pair=args.pair,
            transfer_tier=args.transfer_tier,
        )

        logger.info("Successfully built dataset (%s, tier=%s) → %s",
                     args.pair, args.transfer_tier, out)
        logger.info("Final density: %f", density)

    except Exception as e:
        logger.error("Failed to build dataset: %s", e)
        import traceback

        traceback.print_exc()
