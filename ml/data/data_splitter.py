"""Train/validation/test splitting utilities.

The primary split used in benchmarks is leave-last-out, implemented in
benchmark_common.py (_leave_last_out). This module provides additional
split strategies for experimentation.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def leave_last_out_split(
    ratings: pd.DataFrame,
    user_col: str = "user_id",
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user leave-last-out split sorted by timestamp.

    For each user:
      n <= 2  → all in train (not enough for test)
      n >= 3  → last = test, second-to-last = val, rest = train
    """
    df = ratings.copy()
    df["_sort"] = df[timestamp_col].fillna(pd.Timestamp.min) if not df[timestamp_col].isna().all() else range(len(df))
    df = df.sort_values([user_col, "_sort"])

    train_rows, val_rows, test_rows = [], [], []
    for _, group in df.groupby(user_col):
        n = len(group)
        if n <= 2:
            train_rows.append(group)
        else:
            train_rows.append(group.iloc[:n - 2])
            val_rows.append(group.iloc[n - 2:n - 1])
            test_rows.append(group.iloc[n - 1:])

    cols = [c for c in df.columns if c != "_sort"]
    def _concat(rows):
        return pd.concat(rows, ignore_index=True).drop(columns=["_sort"], errors="ignore") if rows else pd.DataFrame(columns=cols)

    train, val, test = _concat(train_rows), _concat(val_rows), _concat(test_rows)
    logger.info("Leave-last-out: train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test
