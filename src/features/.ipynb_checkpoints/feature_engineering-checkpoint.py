# filename: src/features/feature_engineering.py
"""Feature engineering for monthly market & macro data.
This module constructs:
- Technical indicators on a market price series (monthly returns, SMA, momentum, volatility).
- Macro transforms (FedFunds month-over-month delta in bps; CPI YoY inflation in %).
- One-step-ahead prediction targets (next-month return in %; binary up/down direction).
Design principles:
- Deterministic, leakage-free features: every feature at time t uses data available at or
  before t; targets refer to t+1.
- Minimal hidden mutation: helpers return new DataFrames instead of mutating inputs.
- Clear failure modes: explicit validation for required columns and DatetimeIndex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# ----------- Config holder (optional convenience) -----------
@dataclass(frozen=True)
class FeatureSpec:
    """Configuration for input column names and engineered feature labels.
    Attributes:
        market_col: Name of the market price level column.
        cpi_col: CPI level column (e.g., CPIAUCSL).
        fedfunds_col: Effective federal funds rate in percent.
        unemployment_col: Optional macro level (percent).
        vix_col: Optional macro level (index).
        epu_col: Optional macro level (index).
        fsi_col: Optional macro level (index).
        gold_col: Optional macro level (USD/oz).
        wti_col: Optional macro level (USD/bbl).
        usdeur_col: Optional FX level (USD per EUR).

        ret_lag1, sma3, sma12, mom3, vol6: Technical feature names.
        fed_delta_bps, infl_yoy: Macro transform names.
        y_ret_next, y_dir_next: Target column names."""
    market_col: str = "SP500"           # Must match merged dataset. Used as the base series.
    cpi_col: str = "CPI"                # CPI level for YoY inflation transform.
    fedfunds_col: str = "FedFundsRate"  # Policy rate in percent.

    # Macro columns.
    unemployment_col: str = "UnemploymentRate"
    vix_col: str = "VIX"
    epu_col: str = "EPU_US"
    fsi_col: str = "FSI"
    gold_col: str = "Gold_USD_oz"
    wti_col: str = "WTI_Spot"
    usdeur_col: str = "USD_per_EUR"

    # Technical feature names.
    ret_lag1: str = "Return_Lag1"
    sma3: str = "3M_SMA_Return"
    sma12: str = "12M_SMA_Return"
    mom3: str = "3M_Momentum"
    vol6: str = "Volatility_6M"

    # Macro transforms
    fed_delta_bps: str = "FedFunds_Delta_bps"
    infl_yoy: str = "Inflation_YoY_pct"

    # Targets
    y_ret_next: str = "y_return_next_pct"
    y_dir_next: str = "y_direction_next"


# ----------- Helpers -----------
def _require_col(df: pd.DataFrame, col: str) -> None:
    """Raise a clear error if a required column is missing.
    Args:
        df: Input DataFrame.
        col: Required column name.
    Raises: KeyError: If the column is absent."""
    if col not in df.columns:
        raise KeyError(
            f"Required column missing: '{col}'. Available (first 10): {list(df.columns)[:10]}..."
        )


def _ensure_datetime_index_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with a sorted DatetimeIndex (no in-place mutation).
    Args: df: Input DataFrame.
    Returns: A new DataFrame with DatetimeIndex sorted ascending.
    Raises: TypeError: If the index is not a DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    if df.index.is_monotonic_increasing:
        return df.copy()  # Preserve original object.
    return df.sort_index()  # Deterministic ordering across runs/platforms.


def _require_numeric(s: pd.Series, name: str) -> None:
    """Validate that a series is numeric (or coercible), else raise.
    Args:
        s: Series to validate.
        name: Human-readable name for error messages.
    Raises: TypeError: If dtype is not numeric and cannot be coerced."""
    if pd.api.types.is_numeric_dtype(s):
        return
    try:
        pd.to_numeric(s.dropna(), errors="raise")
    except Exception as e:
        raise TypeError(f"Column '{name}' must be numeric. Current dtype: {s.dtype}") from e


def monthly_return_pct(price: pd.Series) -> pd.Series:
    """Compute simple monthly returns in percent: "100 * (P_t / P_{t-1} - 1)".
    Args: price: Price level series indexed by month-end dates.
    Returns: Series of monthly returns in percent."""
    _require_numeric(price, name=price.name or "price")
    return 100.0 * (price / price.shift(1) - 1.0)


def delta_bps(series_pct: pd.Series) -> pd.Series:
    """Month-over-month change expressed in basis points.
    If `series_pct` is in percent (e.g., FEDFUNDS), then: "delta_bps = (pct_t - pct_{t-1}) * 100".
    Args: series_pct: Percent-valued series.
    Returns: Basis-point changes."""
    _require_numeric(series_pct, name=series_pct.name or "series")
    return (series_pct - series_pct.shift(1)) * 100.0


def yoy_pct(level_series: pd.Series) -> pd.Series:
    """Year-over-year percentage change for index levels (e.g., CPI).
    Computed as: "100 * (X_t / X_{t-12} - 1)".
    Args: level_series: Level series with monthly frequency.
    Returns: YoY percent changes."""
    _require_numeric(level_series, name=level_series.name or "level_series")
    return 100.0 * (level_series / level_series.shift(12) - 1.0)


# ----------- Technical features for the market series -----------
def add_technical_features(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Add technical indicators on the market series.
    Features computed at month t (using data up to and including t):
      - Return_Lag1    : previous month's return (percent)
      - 3M_SMA_Return  : 3-month SMA of returns (percent)
      - 12M_SMA_Return : 12-month SMA of returns (percent)
      - 3M_Momentum    : 3-month cumulative return (percent)
      - Volatility_6M  : 6-month realized volatility of returns (stdev, percent)
    Args:
        df: Input frame with at least `spec.market_col`.
        spec: FeatureSpec with column names and output labels.
    Returns: A new DataFrame with added technical feature columns.
    Raises:
        KeyError: If the market column is missing.
        TypeError: If index is not a DatetimeIndex or series are non-numeric."""
    df = _ensure_datetime_index_sorted(df)
    _require_col(df, spec.market_col)

    out = df.copy()  # Do not mutate caller's DataFrame.
    price = out[spec.market_col]
    _require_numeric(price, spec.market_col)

    r = monthly_return_pct(price)  # Base series for several indicators.

    # Technical indicators
    out[spec.ret_lag1] = r.shift(1)  # Use t-1 return to predict at t (informative lag).
    out[spec.sma3] = r.rolling(window=3, min_periods=3).mean()  # Short-term trend filter.
    out[spec.sma12] = r.rolling(window=12, min_periods=12).mean()  # Long-term trend proxy.
    out[spec.mom3] = 100.0 * (price / price.shift(3) - 1.0)  # 3M cumulative performance signal.
    out[spec.vol6] = r.rolling(window=6, min_periods=6).std(ddof=1)  # Realized 6M volatility (sample stdev).

    return out


# ----------- Macro transforms -----------
def add_macro_transforms(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Compute macro transforms required by the thesis spec.
    Constructs:
      - FedFunds_Delta_bps: MoM change of FEDFUNDS in basis points.
      - Inflation_YoY_pct : YoY change derived from CPIAUCSL.
    Other macro columns (e.g., VIX, EPU, FSI, USD/EUR, Gold, WTI) are kept as levels.
    Args:
        df: Input monthly DataFrame.
        spec: FeatureSpec for column names/labels.
    Returns: New DataFrame with additional macro transform columns (if sources are present)."""
    df = _ensure_datetime_index_sorted(df)
    out = df.copy()

    # Δ Fed Funds in bps
    if spec.fedfunds_col in out.columns:
        out[spec.fed_delta_bps] = delta_bps(out[spec.fedfunds_col])  # Rate impulse proxy.

    # Inflation YoY in %
    if spec.cpi_col in out.columns:
        out[spec.infl_yoy] = yoy_pct(out[spec.cpi_col])  # Standard inflation metric.

    return out


# ----------- Targets (next-month) -----------
def add_targets_next_month(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Create one-step-ahead targets aligned for forecasting t+1.
    Definitions:
      y_return_next_pct(t) = market_return_pct at (t+1)
      y_direction_next(t)  = 1 if y_return_next_pct >= 0 else 0 (nullable Int64)
    Args:
        df: Input DataFrame with at least `spec.market_col`.
        spec: FeatureSpec object.
    Returns: New DataFrame with target columns added."""
    df = _ensure_datetime_index_sorted(df)
    _require_col(df, spec.market_col)

    out = df.copy()
    price = out[spec.market_col]
    _require_numeric(price, spec.market_col)

    r = monthly_return_pct(price)
    out[spec.y_ret_next] = r.shift(-1)  # Next month's return becomes target for t.
    out[spec.y_dir_next] = (out[spec.y_ret_next] >= 0.0).astype("Int64")  # Binary classification target.

    return out


# ----------- Master builder -----------
def build_feature_matrix(df_raw: pd.DataFrame, spec: Optional[FeatureSpec] = None) -> pd.DataFrame:
    """Build a clean, modeling-ready monthly feature matrix.
    Steps:
      1) Validate index/columns and sort time.
      2) Add technical features for the market.
      3) Add macro transforms.
      4) Add next-month targets.
      5) Subset to features+targets and drop rows with any missing values (due to lags/rolls/shift(-1)).
    Args:
        df_raw: Merged monthly DataFrame (market & macro levels).
        spec: Optional FeatureSpec (defaults to FeatureSpec()).
    Returns: DataFrame containing required features and targets with no missing values.
    Raises:
        KeyError: If required columns are missing.
        RuntimeError: If no rows remain after dropping NaNs or non-numeric values appear."""
    spec = spec or FeatureSpec()
    df = _ensure_datetime_index_sorted(df_raw)
    _require_col(df, spec.market_col)

    # Compose features progressively to keep each function single-responsibility.
    df = add_technical_features(df, spec)
    df = add_macro_transforms(df, spec)
    df = add_targets_next_month(df, spec)

    # Candidate feature columns (some may be absent depending on the dataset).
    pass_through_macro: Sequence[str] = (
        spec.unemployment_col,
        spec.vix_col,
        spec.epu_col,
        spec.fsi_col,
        spec.gold_col,
        spec.wti_col,
        spec.usdeur_col,
    )
    feature_cols: List[str] = [
        spec.ret_lag1,
        spec.sma3,
        spec.sma12,
        spec.mom3,
        spec.vol6,
        spec.fed_delta_bps,
        spec.infl_yoy,
        # Pass-through level columns (if present):
        *[c for c in pass_through_macro if c in df.columns],
    ]
    target_cols: List[str] = [spec.y_ret_next, spec.y_dir_next]

    # Select only present required columns to form the modeling dataset.
    present_features = [c for c in feature_cols if c in df.columns]
    required_cols = present_features + target_cols
    df_clean = df[required_cols].copy()  # Avoid chained assignment; explicit copy.

    # Drop rows with any missing required value (from rolling windows or shift(-1)).
    before = len(df_clean)
    df_clean = df_clean.dropna(how="any")  # Ensure dense matrix for standard estimators.
    after = len(df_clean)

    if after == 0:
        raise RuntimeError(
            "After feature engineering, no rows remain. "
            "Check input coverage and rolling window lengths."
        )

    # Enforce numeric dtypes for ML friendliness; fail if coercion introduces NaN.
    df_coerced = df_clean.apply(pd.to_numeric, errors="coerce")  # Guarantees numeric matrix.
    if df_coerced.isna().any().any():
        bad_cols = df_coerced.columns[df_coerced.isna().any()].tolist()
        raise RuntimeError(f"Non-numeric values after coercion in columns: {bad_cols}")

    # Re-attach metadata for auditability.
    df_coerced.attrs["dropped_rows"] = before - after  # Quick diagnostic for data loss due to NaNs.

    return df_coerced