"""Consistent string IDs for user/item keys across models.

Amazon dataset IDs come in as mixed types (int, float, string) depending on
how pandas reads them. This module ensures all ID lookups use the same
canonical string form, preventing KeyError mismatches between data loading
and model prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_id(x) -> str:
    """Map any hashable raw id to a canonical string key."""
    if x is None:
        raise ValueError("user/item id is None")
    if isinstance(x, float) and np.isnan(x):
        raise ValueError("user/item id is NaN")
    return str(x).strip()


def normalize_maps(
    user_to_idx: dict,
    item_to_idx: dict,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return copies with string keys for consistent lookups."""
    u = {normalize_id(k): v for k, v in user_to_idx.items()}
    i = {normalize_id(k): v for k, v in item_to_idx.items()}
    return u, i


def _safe_lookup(v, mapping: dict[str, int]):
    if pd.isna(v):
        return None
    try:
        return mapping.get(normalize_id(v))
    except ValueError:
        return None


def map_user_item_columns(
    df: pd.DataFrame,
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    *,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> pd.DataFrame:
    """Add _uidx, _iidx columns with int indices; drop rows with unknown ids."""
    out = df.copy()
    out["_uidx"] = out[user_col].map(lambda v: _safe_lookup(v, user_to_idx))
    out["_iidx"] = out[item_col].map(lambda v: _safe_lookup(v, item_to_idx))
    return out.dropna(subset=["_uidx", "_iidx"])
