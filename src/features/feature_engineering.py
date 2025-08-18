# src/features/feature_engineering.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np


# ----------- Config holder (optional convenience) -----------
@dataclass(frozen=True)
class FeatureSpec:
    market_col: str = "SP500"          # must match name from data_config.yaml
    cpi_col: str = "CPI"               # CPIAUCSL (level)
    fedfunds_col: str = "FedFundsRate" # FEDFUNDS (percent)
    # optional macro columns (present if configured)
    unemployment_col: str = "UnemploymentRate"
    vix_col: str = "VIX"
    epu_col: str = "EPU_US"
    fsi_col: str = "FSI"
    gold_col: str = "Gold_USD_oz"
    wti_col: str = "WTI_Spot"
    usdeur_col: str = "USD_per_EUR"

    # technical feature names to be created
    ret_lag1: str = "Return_Lag1"
    sma3: str = "3M_SMA_Return"
    sma12: str = "12M_SMA_Return"
    mom3: str = "3M_Momentum"
    vol6: str = "Volatility_6M"

    # macro transforms
    fed_delta_bps: str = "FedFunds_Delta_bps"
    infl_yoy: str = "Inflation_YoY_pct"

    # targets
    y_ret_next: str = "y_return_next_pct"
    y_dir_next: str = "y_direction_next"


# ----------- Helpers -----------
def _require_col(df: pd.DataFrame, col: str) -> None:
    """Raise a clear error if a required column is missing."""
    if col not in df.columns:
        raise KeyError(f"Required column missing: '{col}'. "
                       f"Available: {list(df.columns)[:10]}...")


def _assert_datetime_index(df: pd.DataFrame) -> None:
    """Ensure a DatetimeIndex, sorted ascending."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        df.sort_index(inplace=True)


def monthly_return_pct(price: pd.Series) -> pd.Series:
    """
    Compute simple monthly returns in percent: 100*(P_t/P_{t-1}-1).
    Using percent simplifies later interpretation and matches BA tables.
    """
    return 100.0 * (price / price.shift(1) - 1.0)


def delta_bps(series_pct: pd.Series) -> pd.Series:
    """
    Month-over-month change expressed in basis points.
    If series is in percent (e.g., FEDFUNDS), then delta_bps = (pct_t - pct_{t-1}) * 100.
    """
    return (series_pct - series_pct.shift(1)) * 100.0


def yoy_pct(level_series: pd.Series) -> pd.Series:
    """
    Year-over-year percentage change for index levels (e.g., CPI):
    100 * (X_t / X_{t-12} - 1)
    """
    return 100.0 * (level_series / level_series.shift(12) - 1.0)


# ----------- Technical features for the market series -----------
def add_technical_features(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """
    Add SMA(3/12) of monthly returns, 3M momentum, 6M volatility, and return lag1.
    All features are computed at month t using only data up to and including t
    (valid to predict t+1 without leakage).
    """
    _assert_datetime_index(df)
    _require_col(df, spec.market_col)

    out = df.copy()
    price = out[spec.market_col]

    # Base monthly return at t (percent)
    r = monthly_return_pct(price)

    # Technical indicators:
    out[spec.ret_lag1] = r.shift(1)                             # previous month's return
    out[spec.sma3]     = r.rolling(window=3, min_periods=3).mean()
    out[spec.sma12]    = r.rolling(window=12, min_periods=12).mean()
    out[spec.mom3]     = 100.0 * (price / price.shift(3) - 1.0) # 3M cumulative return
    out[spec.vol6]     = r.rolling(window=6, min_periods=6).std(ddof=1)  # realized 6M vol

    return out


# ----------- Macro transforms -----------
def add_macro_transforms(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """
    Compute macro transforms required by the thesis spec:
      - FedFunds_Delta_bps: MoM change of FEDFUNDS in basis points
      - Inflation_YoY_pct: YoY change derived from CPIAUCSL
    Other macro columns are kept as levels (VIX, EPU, FSI, USD/EUR, Gold, WTI).
    """
    _assert_datetime_index(df)
    out = df.copy()

    # Δ Fed Funds in bps
    if spec.fedfunds_col in out.columns:
        out[spec.fed_delta_bps] = delta_bps(out[spec.fedfunds_col])

    # Inflation YoY in %
    if spec.cpi_col in out.columns:
        out[spec.infl_yoy] = yoy_pct(out[spec.cpi_col])

    return out


# ----------- Targets (next-month) -----------
def add_targets_next_month(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """
    Construct targets aligned for one-step-ahead prediction:
      y_return_next_pct(t) = market_return_pct at (t+1)
      y_direction_next(t)  = 1 if y_return_next_pct >= 0 else 0
    This ensures features at t are used to predict returns at t+1 (no leakage).
    """
    _assert_datetime_index(df)
    _require_col(df, spec.market_col)

    out = df.copy()
    price = out[spec.market_col]
    r = monthly_return_pct(price)

    out[spec.y_ret_next] = r.shift(-1)
    out[spec.y_dir_next] = (out[spec.y_ret_next] >= 0.0).astype("Int64")  # nullable integer

    return out


# ----------- Master builder -----------
def build_feature_matrix(df_raw: pd.DataFrame, spec: Optional[FeatureSpec] = None) -> pd.DataFrame:
    """
    End-to-end builder:
      1) Validate index/columns
      2) Add technical features for the market
      3) Add macro transforms
      4) Add next-month targets
      5) Drop rows that cannot be used (NaNs from lags/rolls/shift(-1))
    Returns a clean, modeling-ready monthly DataFrame.
    """
    spec = spec or FeatureSpec()
    _assert_datetime_index(df_raw)
    _require_col(df_raw, spec.market_col)

    df = df_raw.copy()
    df = add_technical_features(df, spec)
    df = add_macro_transforms(df, spec)
    df = add_targets_next_month(df, spec)

    # Drop rows with missing features or target due to lags/rolls/shift(-1)
    # Keep macro levels as-is; the early part of the sample will have NaNs for long windows.
    feature_cols = [
        spec.ret_lag1, spec.sma3, spec.sma12, spec.mom3, spec.vol6,
        spec.fed_delta_bps, spec.infl_yoy,
        # pass-through macro levels if present:
        spec.unemployment_col, spec.vix_col, spec.epu_col, spec.fsi_col,
        spec.gold_col, spec.wti_col, spec.usdeur_col,
    ]
    present = [c for c in feature_cols if c in df.columns]
    target_cols = [spec.y_ret_next, spec.y_dir_next]

    # Build a mask of required columns (features + targets)
    required_cols = present + target_cols
    df_clean = df[required_cols].copy()

    # Drop rows with any missing required value
    before = len(df_clean)
    df_clean = df_clean.dropna(how="any")
    after = len(df_clean)

    if after == 0:
        raise RuntimeError("After feature engineering, no rows remain. "
                           "Check input coverage and rolling window lengths.")

    # Optional: enforce numeric dtypes for ML friendliness
    df_clean = df_clean.apply(pd.to_numeric, errors="coerce")
    if df_clean.isna().any().any():
        # If coercion introduced NaNs, fail early with a clear message
        bad_cols = df_clean.columns[df_clean.isna().any()].tolist()
        raise RuntimeError(f"Non-numeric values after coercion in columns: {bad_cols}")

    # Info summary (as attributes)
    df_clean.attrs["dropped_rows"] = before - after
    return df_clean