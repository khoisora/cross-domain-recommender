"""Item deduplication for cross-format / cross-platform variants.

Movies: DVD vs Blu-ray vs Digital vs 4K UHD → single canonical item.
Games: PS3/PS4/PS5/Xbox/Switch/PC variants → single canonical item.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

_BRACKET_CONTENT = re.compile(r"\s*\[[^\]]*\]", re.IGNORECASE)
_PAREN_CONTENT = re.compile(r"\s*\([^)]*\)", re.IGNORECASE)


def _normalise_title(title: str) -> str:
    """Strip format/platform suffixes to get the canonical title."""
    t = _BRACKET_CONTENT.sub("", title.strip())
    t = _PAREN_CONTENT.sub("", t)
    t = re.sub(r"\s*[,;]\s*", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.rstrip(" -–:")


def _pick_canonical(group: list[dict]) -> dict:
    """Pick the best representative: most ratings → longest description → longest categories."""
    return max(
        group,
        key=lambda x: (
            x.get("rating_count", 0),
            len(str(x.get("description", ""))),
            len(str(x.get("categories", ""))),
        ),
    )


def _build_dedup_mapping(items_df: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame]:
    """Return (ext_id_remap, deduped_items_df). Remap is identity for non-duplicates."""
    items_df = items_df.copy()
    items_df["_norm_title"] = items_df["title"].apply(lambda t: _normalise_title(str(t)))

    ext_id_remap: dict[str, str] = {}
    canonical_ids: set[str] = set()
    merge_count = 0

    for (_norm_title, _domain), group_df in items_df.groupby(["_norm_title", "domain"]):
        group = group_df.to_dict("records")
        canon_id = _pick_canonical(group)["external_id"]
        canonical_ids.add(canon_id)
        for item in group:
            ext_id_remap[item["external_id"]] = canon_id
        if len(group) > 1:
            merge_count += 1

    deduped_df = items_df[items_df["external_id"].isin(canonical_ids)].copy()
    deduped_df["title"] = deduped_df["_norm_title"]
    deduped_df.drop(columns=["_norm_title"], inplace=True)

    logger.info(
        "Item dedup: %d items → %d unique (%d merge groups, %d items merged away)",
        len(items_df), len(deduped_df), merge_count, len(items_df) - len(deduped_df),
    )
    return ext_id_remap, deduped_df


def _remap_ratings(ratings: pd.DataFrame, ext_id_remap: dict[str, str]) -> pd.DataFrame:
    """Remap item IDs; when a user rated multiple variants, keep highest rating."""
    df = ratings.copy()
    before = len(df)
    df["item_id"] = df["item_id"].map(ext_id_remap).fillna(df["item_id"])
    df = df.sort_values("rating", ascending=False)
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first").reset_index(drop=True)
    logger.info("Rating remap: %d → %d (removed %d duplicate user-item pairs)",
                before, len(df), before - len(df))
    return df


def deduplicate_dataset(
    ratings: pd.DataFrame,
    items_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full deduplication pipeline: normalise titles, merge variants, remap ratings."""
    rating_counts = ratings.groupby("item_id").size().to_dict()
    items_df = items_df.copy()
    items_df["rating_count"] = items_df["external_id"].map(rating_counts).fillna(0).astype(int)

    ext_id_remap, deduped_items = _build_dedup_mapping(items_df)
    deduped_ratings = _remap_ratings(ratings, ext_id_remap)

    valid_items = set(deduped_items["external_id"])
    before = len(deduped_ratings)
    deduped_ratings = deduped_ratings[deduped_ratings["item_id"].isin(valid_items)].copy()
    if len(deduped_ratings) < before:
        logger.info("Filtered %d ratings for items without metadata", before - len(deduped_ratings))

    return deduped_ratings, deduped_items
