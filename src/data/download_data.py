# src/data/download_data.py
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

# Keep logs readable and avoid noisy 3rd-party warnings.
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)  # Module-level logger enables selective control.


# ---------- Utilities ----------
def _load_yaml(p: Path) -> Dict:
    """Load a YAML config file.
    Args: p: Path to YAML.
    Returns: Parsed YAML as a dict (empty dict if file is blank).
    Raises:
        FileNotFoundError: If the config path does not exist.
        RuntimeError: If YAML parsing fails."""
    if not Path(p).exists():
        # Fail fast with a clear path to help users fix misconfigured runs.
        raise FileNotFoundError(f"Config not found: {p}")
    try:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        # Wrap to provide filename context while preserving original exception for debugging.
        raise RuntimeError(f"Failed to parse YAML config: {p}") from e



def _require_keys(obj: Dict, keys: List[str], ctx: str) -> None:
    """Validate presence of required keys in mappings.
    Args:
        obj: Mapping to validate.
        keys: Required key names.
        ctx: Human-readable context for error messages.
    Raises: KeyError: If a required key is missing."""
    # Early schema validation avoids cryptic downstream attribute/index errors.
    for k in keys:
        if k not in obj:
            raise KeyError(f"Missing key '{k}' in {ctx}")


def _parse_and_validate_dates(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse date strings and enforce start <= end.
    Args:
        start: Inclusive start date (e.g., '2008-01-01').
        end: Inclusive end date (e.g., '2025-06-30').
    Returns: Tuple of normalized pd.Timestamp (naive UTC-like) for start and end.
    Raises: ValueError: If parsing fails or start > end."""
    # Ensures deterministic clipping and guards against typos like swapped bounds.
    start_ts = pd.to_datetime(start, utc=False)
    end_ts = pd.to_datetime(end, utc=False)
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ValueError(f"Invalid date(s): start='{start}', end='{end}'")
    if start_ts > end_ts:
        raise ValueError(f"Start date {start_ts.date()} is after end date {end_ts.date()}")
    return start_ts.tz_localize(None), end_ts.tz_localize(None)


def _resample_monthly(df: pd.DataFrame, how: str, col_name: str) -> pd.DataFrame:
    """Resample a DatetimeIndex to calendar month with explicit aggregation.
    Args:
        df: Time-indexed DataFrame (DatetimeIndex).
        how: Aggregation strategy: 'last' (month-end) or 'mean' (monthly average).
        col_name: Output column name.
    Returns: Monthly resampled single-column DataFrame named "col_name".
    Raises:
        ValueError: If "how" is unsupported.
        RuntimeError: If input is empty.
        TypeError: If index is not a DatetimeIndex."""
    # Monthly frequency aligns with downstream monthly modeling. Using 'M' gives month-end timestamps.    
    if how not in {"last", "mean"}:
        raise ValueError(f"Unsupported agg '{how}' for {col_name}. Use 'last' or 'mean'.")
    if df.empty:
        raise RuntimeError(f"Empty frame before resampling for {col_name}.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"Index must be DatetimeIndex for {col_name}.")

    # Month-end anchor ('M') ensures consistent alignment across sources.
    out = df.resample("M").last() if how == "last" else df.resample("M").mean()

    # Preserve a stable, human-friendly column name for later merges/analysis.
    out.columns = [col_name]
    return out


def _fetch_equity_yahoo(symbol: str, start: str, end: str, agg: str, col_name: str) -> pd.DataFrame:
    """Download adjusted equity prices from Yahoo Finance and resample to monthly.
    Args:
        symbol: Ticker symbol (e.g., 'AAPL').
        start: Start date (inclusive).
        end: End date (inclusive).
        agg: 'last' or 'mean' monthly aggregation.
        col_name: Output column name.
    Returns: Monthly DataFrame with one column named "col_name".
    Raises:
        ImportError: If yfinance is missing.
        RuntimeError: If download fails or data is empty."""
    # We prefer split/dividend-adjusted prices for economically meaningful returns. Resampling avoids mixing daily/weekly series in monthly models.
    try:
        import yfinance as yf 
    except Exception as e:
        raise ImportError("yfinance is required to load equities from Yahoo.") from e

    try:
        # auto_adjust=True returns total-return-like 'Close' in newer versions.
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as e:
        raise RuntimeError(f"Yahoo download failed for {symbol}: {e}") from e

    if df.empty:
        raise RuntimeError(f"No Yahoo data for symbol '{symbol}' in window {start}..{end}.")

    # Different yfinance versions expose either 'Adj Close' or adjusted 'Close'; picks whichever exists.
    base_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if base_col not in df.columns:
        raise RuntimeError(
            f"Expected 'Adj Close' or 'Close' in Yahoo response for {symbol}; got {list(df.columns)}"
        )

    s = df[[base_col]].copy()
    s.index = pd.to_datetime(s.index) # Normalize to naive timestamps for consistent resampling.
    s = _resample_monthly(s, agg, col_name=col_name)
    return s


def _fetch_fred(series_id: str, start: str, end: str, api_key: str, agg: str, col_name: str) -> pd.DataFrame:
    """Load a FRED series and convert to monthly using the specified aggregator.
    Args:
        series_id: FRED series identifier (e.g., 'DGS10').
        start: Start date (inclusive).
        end: End date (inclusive).
        api_key: FRED API key.
        agg: Aggregation ('last' for month-end, 'mean' for monthly average).
        col_name: Output column name.
    Returns: Monthly DataFrame with one column named "col_name".
    Raises:
        ImportError: If pandas_datareader is missing.
        RuntimeError: If fetch fails or data is empty."""
    # FRED provides weekly/daily/monthly series -> Normalize to a common monthly grid.    
    try:
        from pandas_datareader import data as pdr
    except Exception as e:
        raise ImportError("pandas_datareader is required to load FRED data.") from e

    try:
        s = pdr.DataReader(series_id, "fred", start=start, end=end, api_key=api_key)
    except Exception as e:
        raise RuntimeError(f"FRED download failed for {series_id}: {e}") from e

    if s.empty:
        raise RuntimeError(f"No FRED data for series '{series_id}' in window {start}..{end}.")

    # Normalize to a single-column DataFrame with a stable, human-friendly name.
    if isinstance(s, pd.Series):
        df = s.to_frame(name=col_name)
    else:
        df = s.rename(columns={s.columns[0]: col_name})

    df.index = pd.to_datetime(df.index)
    df = _resample_monthly(df, agg, col_name=col_name)
    return df


def _get_fred_api_key(cfg: Dict) -> str:
    """Extract the FRED API key from the loaded config.
    Args: cfg: Parsed YAML configuration.
    Returns: API key string.
    Raises:
        TypeError: If cfg is not a dict.
        RuntimeError: If key is missing or not a string."""
    # Keep secrets explicit and in one place.    
    if not isinstance(cfg, dict):
        raise TypeError("Config must be a dict.")
    auth = cfg.get("auth", {})
    key = auth.get("fred_api_key")
    if not key or not isinstance(key, str):
        raise RuntimeError("Missing FRED API key in config (auth.fred_api_key).")
    return key


def _enforce_bounds(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Clip a time-indexed DataFrame to [start, end] inclusive.
    Args:
        df: Input DataFrame with DatetimeIndex.
        start: Inclusive start date (string).
        end: Inclusive end date (string).
    Returns: Clipped DataFrame."""
    # Prevents look-ahead leakage outside the configured backtest window.    
    if df.empty:
        return df
    idx = pd.to_datetime(df.index)
    return df.loc[(idx >= pd.to_datetime(start)) & (idx <= pd.to_datetime(end))]


def _series_label(name: Optional[str], default: str) -> str:
    """Resolve the output label for a series.
    Args:
        name: Optional human-friendly name from config.
        default: Fallback label (e.g., symbol or series_id).
    Returns: Chosen label (prefers 'name' if valid)."""
    # Stable, readable column names simplify downstream analysis and plotting.    
    return name if (name and isinstance(name, str)) else default


def _assert_unique_columns(df: pd.DataFrame) -> None:
    """Ensure no duplicate column names after concatenation.
    Args: df: DataFrame to check.
    Raises: RuntimeError: If duplicate column names are detected."""
    # Duplicate names silently shadow analysis assumptions (ambiguous joins/plots).    
    cols = list(df.columns)
    if len(cols) != len(set(cols)):
        dupes = pd.Series(cols).value_counts()
        dupes = dupes[dupes > 1].index.tolist()
        raise RuntimeError(
            f"Duplicate column names after merge: {dupes}. "
            "Give unique 'name' values in your config."
        )


# ---------- Main driver ----------
def download_all_data(config_path: Path, out_path: Path) -> None:
    """End-to-end loader for monthly dataset (Yahoo equities + FRED macro).
    Steps:
      1) Load config
      2) Fetch Yahoo equities and FRED macro with per-series monthly aggregation
      3) Merge on a monthly DatetimeIndex
      4) Enforce hard date bounds
      5) Save as Parquet
    Design choices:
      * Monthly frequency aligns with monthly targets/features.
      * Month-end clipping prevents leakage from future months.
      * Robust exceptions provide actionable messages when a source fails.
    Args:
        config_path: Path to YAML configuration file.
        out_path: Target Parquet file path.
    Raises:
        RuntimeError: If nothing can be merged or writing fails.
        ValueError: If date bounds are invalid."""
    cfg = _load_yaml(config_path)

    # Validate minimal schema early.
    _require_keys(cfg, ["dataset"], "config")
    ds = cfg["dataset"]
    _require_keys(ds, ["start_date", "end_date"], "dataset")

    # Parse and validate dates.
    start_ts, end_ts = _parse_and_validate_dates(ds["start_date"], ds["end_date"])
    start, end = start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")

    fred_key = _get_fred_api_key(cfg)

    frames: List[pd.DataFrame] = []
    problems: List[str] = []

    # --- Equities (Yahoo) ---
    for i, eq in enumerate(ds.get("equities", []), start=1):
        try:
            _require_keys(eq, ["symbol", "source", "agg"], f"dataset.equities[{i}]")
            if eq["source"] != "yahoo":
                raise NotImplementedError("Only source=yahoo implemented for equities.")
            col_name = _series_label(eq.get("name"), f"{eq['symbol']}_adjclose")
            logger.info(f"[EQ] {col_name}: Yahoo {eq['symbol']} {start}..{end} (agg={eq['agg']})")
            frames.append(
                _fetch_equity_yahoo(eq["symbol"], start, end, agg=eq["agg"], col_name=col_name)
            )
        except Exception as e:
            # Collect failures but do not abort the entire pipeline if others succeed.
            problems.append(f"Equity {eq.get('symbol')}: {e}")

    # --- Macro (FRED) ---
    for j, m in enumerate(ds.get("macro", []), start=1):
        try:
            _require_keys(m, ["source", "series_id", "agg"], f"dataset.macro[{j}]")
            col_name = _series_label(m.get("name"), m["series_id"])

            if m["source"] == "fred":
                logger.info(
                    f"[MACRO] {col_name}: FRED {m['series_id']} {start}..{end} (agg={m['agg']})"
                )
                try:
                    frames.append(
                        _fetch_fred(
                            m["series_id"],
                            start,
                            end,
                            api_key=fred_key,
                            agg=m["agg"],
                            col_name=col_name,
                        )
                    )
                except Exception as fred_err:
                    # Optional fallback (e.g., GLD via Yahoo).
                    fb = m.get("fallback")
                    if fb and fb.get("source") == "yahoo" and "symbol" in fb:
                        logger.warning(
                            f"FRED failed for {m['series_id']} ({fred_err}); "
                            f"trying fallback Yahoo {fb['symbol']}"
                        )
                        frames.append(
                            _fetch_equity_yahoo(
                                fb["symbol"],
                                start,
                                end,
                                agg=fb.get("agg", "mean"),
                                col_name=col_name,
                            )
                        )
                    else:
                        raise fred_err
            else:
                raise NotImplementedError(
                    "Only source=fred implemented for macro (plus optional Yahoo fallback)."
                )

        except Exception as e:
            problems.append(f"Macro {m.get('series_id') or m.get('name')}: {e}")

    # Report source-level problems but continue if we have usable data.
    if problems:
        logger.error("Some data sources failed:\n - " + "\n - ".join(problems))

    if not frames:
        # Nothing to merge -> do not silently create an empty artifact.
        raise RuntimeError("No data frames to merge — all sources failed or none configured.")

    # --- Merge all series into a single monthly index ---
    df = pd.concat(frames, axis=1).sort_index()  # Aligns on time index; preserves month-end anchors.
    _assert_unique_columns(df)  # Prevents ambiguous downstream references.

    # --- Enforce hard bounds ---
    df = _enforce_bounds(df, start, end)

    # --- Structural sanity checks with clear messages ---
    if df.empty:
        raise RuntimeError("Merged dataset is empty after clipping — check sources and date window.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Merged dataset must have a DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        # Deterministic ordering for reproducible modeling/IO.
        df = df.sort_index()

    # Surface data quality early. This helps decide imputation/exclusion strategies.
    na_rates = df.isna().mean().sort_values(ascending=False)
    top_na = na_rates.head(5)
    logger.info(f"Dataset shape: {df.shape}; top-5 missing rates:\n{top_na.to_string()}")

    # --- Persist artifact ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path)
    except Exception as e:
        # Provide path context and keep original traceback for debugging.
        raise RuntimeError(f"Failed to write Parquet to {out_path}: {e}") from e

    logger.info(f"Saved parquet: {out_path}  -> shape={df.shape}")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create an argument parser for CLI usage.
    Returns: Configured ArgumentParser."""
    # Separate builder keeps '__main__' block tidy and testable.    
    ap = argparse.ArgumentParser(
        description="Download and aggregate monthly data (Yahoo + FRED)."
    )
    ap.add_argument(
        "--config",
        type=str,
        default="config/data_config.yaml",
        help="Path to YAML config describing sources and date window.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="artifacts/data/raw_data.parquet",
        help="Target Parquet file for raw monthly dataset.",
    )
    return ap


if __name__ == "__main__":
    # Standard CLI entry point for reproducible, scriptable runs.
    parser = _build_arg_parser()
    args = parser.parse_args()
    download_all_data(Path(args.config), Path(args.out))