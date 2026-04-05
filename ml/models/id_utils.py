"""Consistent string IDs for user/item keys across models.

Amazon dataset IDs come in as mixed types (int, float, string) depending on
how pandas reads them. This module ensures all ID lookups use the same
canonical string form, preventing KeyError mismatches between data loading
and model prediction.
"""

from __future__ import annotations

import numpy as np


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
