"""Train/validation/test splitting utilities.

Supports temporal splits (preferred for recommender systems) and
random splits as fallback.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _per_user_split(
    ratings: pd.DataFrame,
    user_col: str,
    timestamp_col: str,
    split_fn,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort each user's ratings by time and apply split_fn(group) → (train, val, test).

    split_fn should return (train_df, val_df | None, test_df | None).
    Rows without timestamps are treated as oldest.
    """
    df = ratings.copy()
    df["_sort_key"] = (
        df[timestamp_col].fillna(pd.Timestamp.min)
        if not df[timestamp_col].isna().all()
        else range(len(df))
    )
    df = df.sort_values([user_col, "_sort_key"])

    train_rows, val_rows, test_rows = [], [], []
    for _, group in df.groupby(user_col):
        t, v, te = split_fn(group)
        train_rows.append(t)
        if v is not None:
            val_rows.append(v)
        if te is not None:
            test_rows.append(te)

    out_cols = [c for c in df.columns if c != "_sort_key"]

    def _concat(rows):
        return (
            pd.concat(rows, ignore_index=True).drop(columns=["_sort_key"])
            if rows
            else pd.DataFrame(columns=out_cols)
        )

    return _concat(train_rows), _concat(val_rows), _concat(test_rows)


def temporal_split(
    ratings: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ratings by timestamp into train/val/test sets.

    Args:
        ratings: DataFrame with rating data.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for test.
        timestamp_col: Column name containing timestamps.

    Returns:
        (train_df, val_df, test_df)
    """
    df = ratings.copy()
    has_ts = df[timestamp_col].notna()
    df_with_ts = df[has_ts].sort_values(timestamp_col)
    df_no_ts = df[~has_ts]

    n = len(df_with_ts)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train_ts = n - n_val - n_test

    val = df_with_ts.iloc[n_train_ts : n_train_ts + n_val]
    test = df_with_ts.iloc[n_train_ts + n_val :]
    train = pd.concat([df_no_ts, df_with_ts.iloc[:n_train_ts]], ignore_index=True)

    logger.info(
        "Temporal split: train=%d, val=%d, test=%d", len(train), len(val), len(test)
    )
    return train, val, test


def random_split(
    ratings: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ratings randomly into train/val/test sets.

    Args:
        ratings: DataFrame with rating data.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for test.
        seed: Random seed for reproducibility.

    Returns:
        (train_df, val_df, test_df)
    """
    df = ratings.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test = df.iloc[:n_test]
    val = df.iloc[n_test : n_test + n_val]
    train = df.iloc[n_test + n_val :]

    logger.info(
        "Random split: train=%d, val=%d, test=%d", len(train), len(val), len(test)
    )
    return train, val, test


def leave_last_out_split(
    ratings: pd.DataFrame,
    user_col: str = "user_id",
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leave-last-out split: for each user, hold out their last
    rating for test and second-last for validation.

    This is the standard evaluation protocol for sequential recommender
    systems.

    Args:
        ratings: DataFrame with rating data.
        user_col: Column name for user identifier.
        timestamp_col: Column name containing timestamps.

    Returns:
        (train_df, val_df, test_df)
    """
    def _split(group):
        n = len(group)
        if n <= 2:
            return group, None, None
        return group.iloc[:n - 2], group.iloc[n - 2:n - 1], group.iloc[n - 1:]

    train, val, test = _per_user_split(ratings, user_col, timestamp_col, _split)
    logger.info(
        "Leave-last-out split: train=%d, val=%d, test=%d", len(train), len(val), len(test)
    )
    return train, val, test


def per_user_temporal_split(
    ratings: pd.DataFrame,
    user_col: str = "user_id",
    timestamp_col: str = "timestamp",
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user temporal split: for each user, sort by timestamp
    and assign the last test_ratio fraction to test, the preceding
    val_ratio fraction to validation, and the rest to train.

    Each user contributes multiple test items, giving the evaluation
    more signal than leave-last-out (which only uses 1 item).

    Args:
        ratings: DataFrame with rating data.
        user_col: Column name for user identifier.
        timestamp_col: Column name containing timestamps.
        test_ratio: Fraction of each user's ratings for the test set.
        val_ratio: Fraction of each user's ratings for the validation set.

    Returns:
        (train_df, val_df, test_df)
    """
    def _split(group):
        n = len(group)
        if n < 5:
            return group, None, None
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train = 1
            n_val = max(1, (n - 1) // 2)
        return (
            group.iloc[:n_train],
            group.iloc[n_train:n_train + n_val],
            group.iloc[n_train + n_val:],
        )

    train, val, test = _per_user_split(ratings, user_col, timestamp_col, _split)
    logger.info(
        "Per-user temporal split (test=%.0f%%, val=%.0f%%): train=%d, val=%d, test=%d",
        test_ratio * 100, val_ratio * 100, len(train), len(val), len(test),
    )
    return train, val, test
