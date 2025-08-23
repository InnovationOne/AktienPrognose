# filename: src/utils/splits.py
from __future__ import annotations

from typing import Tuple
import pandas as pd


def time_split(
    df: pd.DataFrame,
    train_end: str,
    test_start: str,
    test_end: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal train/test split on a DatetimeIndex.
    Args:
        df: Full DataFrame with a DatetimeIndex.
        train_end: Inclusive end date for the training window (YYYY-MM-DD).
        test_start: Inclusive start date for the test window (YYYY-MM-DD).
        test_end: Optional inclusive end date for the test window; if None, uses df.index.max().
    Returns: (train_df, test_df) as copies to avoid mutating the original.
    Raises:
        TypeError: If df.index is not a DatetimeIndex.
        ValueError: If the requested windows are inconsistent (train_end < min or test_start > max)."""
    # Many time-based ops require a proper DatetimeIndex.
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("time_split requires a DataFrame with a DatetimeIndex.")

    # Normalize string inputs to Timestamps for robust slicing and comparisons.
    train_end_ts = pd.to_datetime(train_end)
    test_start_ts = pd.to_datetime(test_start)
    test_end_ts = pd.to_datetime(test_end) if test_end is not None else df.index.max()

    if train_end_ts < df.index.min():
        raise ValueError("train_end is before the start of the data.")
    if test_start_ts > df.index.max():
        raise ValueError("test_start is after the end of the data.")
    if train_end_ts >= test_start_ts:
        raise ValueError("train_end must be strictly before test_start to avoid overlap.")

    # Use .loc slicing which is inclusive for DatetimeIndex with exact timestamps.
    train_df = df.loc[:train_end_ts].copy()
    test_df = df.loc[test_start_ts:test_end_ts].copy()
    return train_df, test_df


def split_Xy(
    df: pd.DataFrame,
    y_reg: str = "y_return_next_pct",
    y_clf: str = "y_direction_next",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split a labeled dataset into features X and targets y (regression + classification).
    Args:
        df: Input DataFrame containing features and both targets.
        y_reg: Column name of the regression target.
        y_clf: Column name of the classification target (binary 0/1).
    Returns:
        X: Feature matrix (all columns except the targets).
        yr: Regression target as a pandas Series (float).
        yc: Classification target as a pandas Series (int8).
    Raises: KeyError: If target columns are missing."""
    missing = [c for c in (y_reg, y_clf) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing target column(s): {missing}")

    # Keep features pure (no leakage from target columns).
    X = df.drop(columns=[y_reg, y_clf])

    # Preserve original dtypes (float for regression).
    yr = df[y_reg]
    yc = df[y_clf].astype("int8")
    return X, yr, yc
