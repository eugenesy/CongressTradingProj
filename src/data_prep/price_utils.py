"""
price_utils.py
==============
Utilities for loading parquet price data and computing:
  - Forward excess returns (stock vs SPY) at standard horizons
  - Backward technical features (as of T-1 before filing)

All price lookups use `close` (split-adjusted by provider) for returns,
which avoids NaN gaps in adjClose from 2025-05-22 onward.

Horizon anchor: Filed date (NOT Traded date).
This matches the copy-trading use case — an investor can only act after
they see the filing, so all returns are measured from the filing date.
"""

import logging
import os
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Forward return horizons (approximate trading days from Filed date)
# ---------------------------------------------------------------------------
HORIZONS = {
    "1M":  21,
    "2M":  42,
    "3M":  63,
    "6M":  126,
    "8M":  168,
    "12M": 252,
    "18M": 378,
    "24M": 504,
}

# Number of trading days before Filing to use for technical features
LOOKBACK_DAYS = 20   # Volatility window
RSI_PERIOD    = 14

PRICE_FEATURE_COLS = [
    "stock_ret_1d", "stock_ret_5d", "stock_ret_10d", "stock_ret_20d",
    "stock_vol_20d", "stock_rsi_14", "stock_vol_ratio",
    "spy_ret_1d",   "spy_ret_5d",   "spy_ret_10d",   "spy_ret_20d",
    "spy_vol_20d",  "spy_rsi_14",   "spy_vol_ratio",
]


# ---------------------------------------------------------------------------
# Parquet loading (cached per ticker — each ticker loaded at most once)
# ---------------------------------------------------------------------------

_PARQUET_CACHE: dict[str, Optional[pd.DataFrame]] = {}


def load_parquet(ticker: str, parquet_dir: str) -> Optional[pd.DataFrame]:
    """
    Load and cache a ticker's parquet file.
    Returns a DataFrame with a DatetimeIndex named 'date', or None if missing.
    The index is sorted ascending.
    """
    key = (parquet_dir, ticker)
    if key not in _PARQUET_CACHE:
        path = Path(parquet_dir) / f"{ticker}.parquet"
        if not path.exists():
            logger.debug("Parquet not found: %s", path)
            _PARQUET_CACHE[key] = None
        else:
            try:
                df = pd.read_parquet(path)
                # Ensure datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                _PARQUET_CACHE[key] = df
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)
                _PARQUET_CACHE[key] = None
    return _PARQUET_CACHE[key]


def clear_cache():
    """Free all cached parquet DataFrames (call between runs if memory is tight)."""
    _PARQUET_CACHE.clear()


# ---------------------------------------------------------------------------
# Price lookup helpers
# ---------------------------------------------------------------------------

def _get_price_on_or_after(df: pd.DataFrame, date: pd.Timestamp,
                            col: str = "close",
                            max_skip: int = 5) -> Optional[float]:
    """
    Return the first available price on `date` or within `max_skip` trading
    days after (handles weekends, holidays).
    Returns None if no price found within the window.
    """
    end = date + timedelta(days=max_skip * 2)
    window = df.loc[date:end, col]
    if len(window) == 0:
        return None
    return float(window.iloc[0])


def _get_price_at_offset(df: pd.DataFrame, base_date: pd.Timestamp,
                          offset_td: int, col: str = "close",
                          max_skip: int = 5) -> Optional[float]:
    """
    Return the price approximately `offset_td` trading days after `base_date`.
    Uses the closest available date within a 5-day tolerance window.
    """
    # Find the position of base_date in the index
    idx = df.index.searchsorted(base_date)
    target_pos = idx + offset_td
    if target_pos >= len(df):
        return None
    return float(df[col].iloc[target_pos])


# ---------------------------------------------------------------------------
# Forward return computation (vectorised over the full dataset)
# ---------------------------------------------------------------------------

def compute_excess_returns(df: pd.DataFrame, parquet_dir: str) -> pd.DataFrame:
    """
    For each row in df (must have columns: Matched_Ticker, Filed),
    compute excess returns at each horizon vs SPY.

    Adds columns:
        Excess_Return_1M, Excess_Return_2M, ..., Excess_Return_24M
        Stock_Return_6M, SPY_Return_6M  (6M and 12M kept for diagnostics)
        Has_Return_{horizon}  (bool: was price available for this horizon)

    Filed date is the anchor (copy-trader acts on filing, not trade date).
    Uses close for all return calculations.
    """
    spy_df = load_parquet("SPY", parquet_dir)
    if spy_df is None:
        raise FileNotFoundError("SPY.parquet not found — required for benchmark returns")

    results: list[dict] = []

    # Pre-load all unique tickers (avoids repeated disk access)
    unique_tickers = df["Matched_Ticker"].dropna().unique()
    logger.info("Pre-loading %d unique ticker parquets...", len(unique_tickers))
    for t in unique_tickers:
        load_parquet(t, parquet_dir)  # warms the cache

    logger.info("Computing forward excess returns for %d rows...", len(df))

    for _, row in df.iterrows():
        ticker = row["Matched_Ticker"]
        filed_dt = pd.Timestamp(row["Filed"])

        stock_df = load_parquet(ticker, parquet_dir) if pd.notna(ticker) else None

        row_result: dict = {}
        for horizon, td in HORIZONS.items():
            if stock_df is None:
                row_result[f"Excess_Return_{horizon}"] = np.nan
                row_result[f"Has_Return_{horizon}"] = False
                continue

            # Base price: first available on or after Filed date
            p0_stock = _get_price_on_or_after(stock_df, filed_dt)
            p0_spy   = _get_price_on_or_after(spy_df,   filed_dt)

            if p0_stock is None or p0_spy is None or p0_stock == 0 or p0_spy == 0:
                row_result[f"Excess_Return_{horizon}"] = np.nan
                row_result[f"Has_Return_{horizon}"] = False
                continue

            # Forward price at horizon
            p1_stock = _get_price_at_offset(stock_df, filed_dt, td)
            p1_spy   = _get_price_at_offset(spy_df,   filed_dt, td)

            if p1_stock is None or p1_spy is None:
                row_result[f"Excess_Return_{horizon}"] = np.nan
                row_result[f"Has_Return_{horizon}"] = False
                continue

            stock_ret = (p1_stock - p0_stock) / p0_stock
            spy_ret   = (p1_spy   - p0_spy)   / p0_spy
            excess    = stock_ret - spy_ret

            row_result[f"Excess_Return_{horizon}"] = excess
            row_result[f"Has_Return_{horizon}"] = True

        # Keep raw returns for a couple of horizons for diagnostics
        row_result["_Stock_Return_6M"] = row_result.get("Excess_Return_6M", np.nan)
        results.append(row_result)

    return pd.DataFrame(results, index=df.index)


# ---------------------------------------------------------------------------
# Technical feature computation (vectorised per ticker)
# ---------------------------------------------------------------------------

def _compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Compute RSI for the last observation given a price array."""
    if len(prices) < period + 1:
        return 50.0  # neutral when insufficient history
    deltas = np.diff(prices[-(period + 1):])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_features_for_ticker(
    ticker_df: pd.DataFrame,
    date: pd.Timestamp,
    lookback: int = LOOKBACK_DAYS,
) -> Optional[np.ndarray]:
    """
    Compute 7 technical features for one ticker as of T-1 before `date`.

    Features (all backward-looking — no look-ahead):
      0: ret_1d   — 1-day return ending T-1
      1: ret_5d   — 5-day return ending T-1
      2: ret_10d  — 10-day return ending T-1
      3: ret_20d  — 20-day return ending T-1
      4: vol_20d  — 20-day return volatility (annualised)
      5: rsi_14   — RSI(14) / 100  (normalised to [0, 1])
      6: vol_ratio — today volume / 20d avg volume

    Returns None if insufficient history (< lookback + 5 trading days before date).
    """
    # All rows strictly before the filing date (T-1 and earlier)
    hist = ticker_df.loc[:date - timedelta(days=1)]

    min_needed = lookback + RSI_PERIOD + 2
    if len(hist) < min_needed:
        return None

    closes  = hist["close"].values.astype(float)
    volumes = hist["volume"].values.astype(float)

    if closes[-1] == 0 or closes[-1] != closes[-1]:  # 0 or NaN
        return None

    def safe_ret(n: int) -> float:
        if len(closes) < n + 1 or closes[-n - 1] == 0:
            return 0.0
        return float((closes[-1] - closes[-n - 1]) / closes[-n - 1])

    log_rets = np.diff(np.log(np.where(closes > 0, closes, np.nan)))
    log_rets = log_rets[~np.isnan(log_rets)]
    vol_20d = float(np.std(log_rets[-lookback:]) * np.sqrt(252)) if len(log_rets) >= lookback else 0.0

    vol_ratio = 1.0
    if len(volumes) >= lookback and np.mean(volumes[-lookback:]) > 0:
        vol_ratio = float(volumes[-1] / np.mean(volumes[-lookback:]))

    return np.array([
        safe_ret(1),
        safe_ret(5),
        safe_ret(10),
        safe_ret(20),
        vol_20d,
        _compute_rsi(closes, RSI_PERIOD) / 100.0,
        min(vol_ratio, 10.0),   # cap at 10x to avoid extreme outliers
    ], dtype=np.float32)


def compute_price_features(df: pd.DataFrame, parquet_dir: str) -> dict[str, np.ndarray]:
    """
    For each row in df (must have: transaction_id, Matched_Ticker, Filed),
    compute 14 price features (7 stock + 7 SPY) as of T-1 before Filing.

    Returns:
        Dict mapping transaction_id -> np.ndarray of shape (14,)

    Rows where features cannot be computed (insufficient history) are omitted
    from the returned dict — the caller should filter accordingly.
    """
    spy_df = load_parquet("SPY", parquet_dir)
    if spy_df is None:
        raise FileNotFoundError("SPY.parquet not found")

    feature_map: dict[str, np.ndarray] = {}
    skipped = 0

    for _, row in df.iterrows():
        tx_id  = str(row["transaction_id"])
        ticker = row["Matched_Ticker"]
        filed  = pd.Timestamp(row["Filed"])

        stock_df = load_parquet(ticker, parquet_dir) if pd.notna(ticker) else None
        if stock_df is None:
            skipped += 1
            continue

        stock_feats = _compute_features_for_ticker(stock_df, filed)
        spy_feats   = _compute_features_for_ticker(spy_df,   filed)

        if stock_feats is None or spy_feats is None:
            skipped += 1
            continue

        feature_map[tx_id] = np.concatenate([stock_feats, spy_feats])

    logger.info("Price features computed: %d rows, %d skipped (insufficient history)",
                len(feature_map), skipped)
    return feature_map
