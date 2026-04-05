"""Item deduplication for cross-format / cross-platform variants.

Movies: DVD vs Blu-ray vs Digital vs 4K UHD → single canonical item.
Games: PS3/PS4/PS5/Xbox/Switch/PC variants → single canonical item.

The module normalises titles, groups duplicates, picks a canonical
representative (most ratings + best metadata), and remaps all
ratings to the canonical item.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Patterns to strip format/platform info from titles.
# e.g., "The Matrix [Blu-ray]" → "The Matrix"
#        "Halo (Xbox One)" → "Halo"
_BRACKET_CONTENT = re.compile(r"\s*\[[^\]]*\]", re.IGNORECASE)
_PAREN_CONTENT = re.compile(r"\s*\([^)]*\)", re.IGNORECASE)


def normalise_title(title: str) -> str:
    """Strip format / platform suffixes to get the canonical title."""
    t = _BRACKET_CONTENT.sub("", title.strip())
    t = _PAREN_CONTENT.sub("", t)
    t = re.sub(r"\s*[,;]\s*", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.rstrip(" -–:")


def _pick_canonical(group: list[dict]) -> dict:
    """Pick the best representative from a group of duplicate items.

    Prefers: most ratings → longest description → longest categories.
    """
    return max(
        group,
        key=lambda x: (
            x.get("rating_count", 0),
            len(str(x.get("description", ""))),
            len(str(x.get("categories", ""))),
        ),
    )


def build_dedup_mapping(
    items_df: pd.DataFrame,
) -> tuple[dict[str, str], pd.DataFrame]:
    """Build a mapping from duplicate external_ids to canonical external_ids.

    Args:
        items_df: DataFrame with columns [external_id, domain, title, ...].

    Returns:
        (ext_id_remap, deduped_items_df)
        - ext_id_remap: {old_ext_id: canonical_ext_id} for ALL items
          (identity for non-duplicates).
        - deduped_items_df: items_df with duplicates removed, keeping only
          the canonical representative per group.
    """
    items_df = items_df.copy()
    items_df["_norm_title"] = items_df["title"].apply(
        lambda t: normalise_title(str(t))
    )

    ext_id_remap: dict[str, str] = {}
    canonical_ids: set[str] = set()
    merge_count = 0

    for (norm_title, domain), group_df in items_df.groupby(["_norm_title", "domain"]):
        group = group_df.to_dict("records")
        canonical = _pick_canonical(group)
        canon_id = canonical["external_id"]
        canonical_ids.add(canon_id)

        for item in group:
            ext_id_remap[item["external_id"]] = canon_id

        if len(group) > 1:
            merge_count += 1
            variants = [g["title"][:60] for g in group if g["external_id"] != canon_id]
            logger.debug(
                "Merged %d variants of '%s' → %s: %s",
                len(group) - 1, norm_title, canon_id, variants,
            )

    deduped_df = items_df[items_df["external_id"].isin(canonical_ids)].copy()
    deduped_df["title"] = deduped_df["_norm_title"]
    deduped_df.drop(columns=["_norm_title"], inplace=True)

    logger.info(
        "Item dedup: %d items → %d unique (%d merge groups, %d items merged away)",
        len(items_df), len(deduped_df), merge_count,
        len(items_df) - len(deduped_df),
    )
    return ext_id_remap, deduped_df


def remap_ratings(
    ratings: pd.DataFrame,
    ext_id_remap: dict[str, str],
) -> pd.DataFrame:
    """Remap item external IDs in ratings to canonical IDs.

    When a user has ratings for multiple variants of the same item
    (e.g., DVD + Blu-ray), keep the highest rating. Rationale: the user's
    best experience with that content is the most informative signal.
    """
    df = ratings.copy()
    before = len(df)

    df["item_id"] = df["item_id"].map(ext_id_remap).fillna(df["item_id"])
    df = df.sort_values("rating", ascending=False)
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first")
    df = df.reset_index(drop=True)

    logger.info(
        "Rating remap: %d → %d (removed %d duplicate user-item pairs)",
        before, len(df), before - len(df),
    )
    return df


def deduplicate_dataset(
    ratings: pd.DataFrame,
    items_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full deduplication pipeline: normalise, group, merge, remap.

    Args:
        ratings: raw ratings DataFrame.
        items_df: item metadata DataFrame.

    Returns:
        (deduped_ratings, deduped_items_df)
    """
    rating_counts = ratings.groupby("item_id").size().to_dict()
    items_df = items_df.copy()
    items_df["rating_count"] = (
        items_df["external_id"].map(rating_counts).fillna(0).astype(int)
    )

    ext_id_remap, deduped_items = build_dedup_mapping(items_df)
    deduped_ratings = remap_ratings(ratings, ext_id_remap)

    valid_items = set(deduped_items["external_id"])
    before = len(deduped_ratings)
    deduped_ratings = deduped_ratings[
        deduped_ratings["item_id"].isin(valid_items)
    ].copy()

    if len(deduped_ratings) < before:
        logger.info(
            "Filtered %d ratings for items without metadata",
            before - len(deduped_ratings),
        )

    return deduped_ratings, deduped_items
