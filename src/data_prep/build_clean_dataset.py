"""
build_clean_dataset.py
======================
Single-pass clean pipeline for the congressional trading dataset.

Replaces the old 9-script pipeline with a single, readable script.

Usage:
    python scripts/build_v2/build_clean_dataset.py [--parquet-dir data/parquet]
                                                    [--output-dir data/processed]
                                                    [--input experiments/signal_isolation/new_v5_transactions_with_committee_indsutry.csv]
                                                    [--min-year 2015]
                                                    [--skip-price-features]

Outputs (in --output-dir):
    ml_dataset_v2.csv          — clean ML-ready dataset (one row per transaction)
    price_sequences_v2.pt      — dict {transaction_id: np.ndarray(14,)} price features
    ticker_normalizer_report.csv — unmatched ticker summary
    build_v2_summary.txt        — full build run statistics

Pipeline steps:
    1. Load & validate raw CSV
    2. Filter to equity transactions, Filed >= min_year
    3. Clean / standardize fields
    4. Ticker normalization (conservative, parquet-verified)
    5. Compute forward excess returns vs SPY (close, Filed anchor)
    6. Compute backward technical price features (T-1 before Filing)
    7. Quality filter & output
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Allow running from repo root or scripts/build_v2/
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data_prep.ticker_normalizer import TickerNormalizer
from src.data_prep.price_utils import (
    compute_excess_returns,
    compute_price_features,
    clear_cache,
    HORIZONS,
    PRICE_FEATURE_COLS,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EQUITY_TICKER_TYPES = {
    "", "ST", "Stock",       # standard equities (House and Senate naming differ)
    "GS", "CS", "HN",        # other standard security types
    "OT",                    # other (mostly equities)
    "AB",                    # unknown but frequently present
    None,
}

# STOCK Act max allowed filing delay: 45 days.
# We keep up to 120 days to be generous (some late filers correct filings),
# but flag the ones that exceed 45 days.
FILING_GAP_MAX_DAYS = 120
FILING_GAP_FLAG_DAYS = 45

# Minimum price history (trading days) before filing to compute features
MIN_HISTORY_DAYS = 30

# ---------------------------------------------------------------------------
# Step 1: Load raw CSV
# ---------------------------------------------------------------------------

def load_raw(path: str) -> pd.DataFrame:
    logger.info("Loading raw transactions from: %s", path)
    # Handle UTF-8 BOM (present in the new v5 file)
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    logger.info("  Loaded %d rows, %d columns", len(df), len(df.columns))
    logger.info("  Columns: %s", df.columns.tolist())
    return df


# ---------------------------------------------------------------------------
# Step 2: Filter to tractable equity transactions
# ---------------------------------------------------------------------------

def filter_transactions(df: pd.DataFrame, min_year: int = 2015) -> pd.DataFrame:
    n0 = len(df)
    log_lines = []

    # Parse dates (fail-safe: coerce bad values to NaT)
    df["Filed"]  = pd.to_datetime(df["Filed"],  errors="coerce")
    df["Traded"] = pd.to_datetime(df["Traded"], errors="coerce")

    # Drop rows with missing dates
    before = len(df)
    df = df.dropna(subset=["Filed", "Traded"])
    log_lines.append(f"Dropped {before - len(df):>6,} rows: missing Filed or Traded date")

    # Date range filter
    before = len(df)
    df = df[df["Filed"].dt.year >= min_year]
    log_lines.append(f"Dropped {before - len(df):>6,} rows: Filed before {min_year}")

    # Drop future filings (sanity check)
    before = len(df)
    today = pd.Timestamp.today()
    df = df[df["Filed"] <= today]
    log_lines.append(f"Dropped {before - len(df):>6,} rows: Filed date in the future")

    # Filing gap = days between trade and filing
    df["Filing_Gap"] = (df["Filed"] - df["Traded"]).dt.days

    # Drop negative gaps (filed before trade — data error)
    before = len(df)
    df = df[df["Filing_Gap"] >= 0]
    log_lines.append(f"Dropped {before - len(df):>6,} rows: negative Filing_Gap")

    # Drop extreme late filers (> FILING_GAP_MAX_DAYS days)
    before = len(df)
    df = df[df["Filing_Gap"] <= FILING_GAP_MAX_DAYS]
    log_lines.append(f"Dropped {before - len(df):>6,} rows: Filing_Gap > {FILING_GAP_MAX_DAYS} days")

    # Flag late filers (> 45 days) — keep them but mark
    df["Late_Filing"] = df["Filing_Gap"] > FILING_GAP_FLAG_DAYS

    # Drop Exchange transactions (not a directional bet)
    before = len(df)
    df = df[~df["Transaction"].str.strip().str.lower().eq("exchange")]
    log_lines.append(f"Dropped {before - len(df):>6,} rows: Exchange transactions")

    # Normalise transaction type: Sale (Full) / Sale (Partial) -> Sale
    df["Transaction"] = df["Transaction"].str.strip()
    df["Transaction"] = df["Transaction"].replace({
        "Sale (Full)":    "Sale",
        "Sale (Partial)": "Sale",
        "SALE":           "Sale",
        "PURCHASE":       "Purchase",
    })
    # Keep only Purchase and Sale
    before = len(df)
    df = df[df["Transaction"].isin(["Purchase", "Sale"])]
    log_lines.append(f"Dropped {before - len(df):>6,} rows: unrecognised Transaction type")

    # Non-equity ticker types (options, bonds, crypto, etc.)
    if "TickerType" in df.columns:
        before = len(df)
        non_equity_types = {"OP", "Stock Option", "Corporate Bond", "Cryptocurrency",
                            "OI", "OL", "SA", "CT", "PS"}
        df = df[~df["TickerType"].isin(non_equity_types)]
        log_lines.append(f"Dropped {before - len(df):>6,} rows: non-equity TickerType")

    # Require non-empty Ticker
    before = len(df)
    df = df[df["Ticker"].notna() & (df["Ticker"].str.strip() != "")]
    log_lines.append(f"Dropped {before - len(df):>6,} rows: missing Ticker")

    # Require non-empty BioGuideID (needed for politician node identity)
    before = len(df)
    df = df[df["BioGuideID"].notna() & (df["BioGuideID"].str.strip() != "")]
    log_lines.append(f"Dropped {before - len(df):>6,} rows: missing BioGuideID")

    for line in log_lines:
        logger.info("  Filter: %s", line)
    logger.info("  After filters: %d rows (%.1f%% of raw)", len(df), len(df) / n0 * 100)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3: Standardise fields
# ---------------------------------------------------------------------------

def standardise_fields(df: pd.DataFrame) -> pd.DataFrame:
    # Party: expand abbreviations
    df["Party"] = df["Party"].str.strip().replace({
        "R": "Republican", "D": "Democrat", "I": "Independent",
    })

    # Chamber: strip whitespace
    df["Chamber"] = df["Chamber"].str.strip()

    # State: strip whitespace
    if "State" in df.columns:
        df["State"] = df["State"].str.strip()

    # Trade size: standardise dollar-range strings
    if "Trade_Size_USD" in df.columns:
        df["Trade_Size_USD"] = _standardise_trade_size(df["Trade_Size_USD"])

    # Assign sequential transaction IDs (zero-padded to 6 digits)
    df.insert(0, "transaction_id", [f"T{i:06d}" for i in range(len(df))])

    # Convert dates back to string (YYYY-MM-DD) for downstream CSV compatibility
    df["Filed_str"]  = df["Filed"].dt.strftime("%Y-%m-%d")
    df["Traded_str"] = df["Traded"].dt.strftime("%Y-%m-%d")

    return df


_TRADE_SIZE_MAP = {
    # Exact dollar amounts from some filings — bucket them
    r"^\$0\b":                              "$0 - $1,000",
    r"^\$1[,\s]?001\s*[-–]\s*\$15[,\s]?000":  "$1,001 - $15,000",
    r"^\$15[,\s]?001\s*[-–]\s*\$50[,\s]?000": "$15,001 - $50,000",
    r"^\$50[,\s]?001\s*[-–]\s*\$100[,\s]?000":"$50,001 - $100,000",
    r"^\$100[,\s]?001\s*[-–]\s*\$250[,\s]?000":"$100,001 - $250,000",
    r"^\$250[,\s]?001\s*[-–]\s*\$500[,\s]?000":"$250,001 - $500,000",
    r"^\$500[,\s]?001\s*[-–]\s*\$1[,\s]?000[,\s]?000":"$500,001 - $1,000,000",
    r"^\$1[,\s]?000[,\s]?001\s*[-–]\s*\$5[,\s]?000[,\s]?000":"$1,000,001 - $5,000,000",
    r"Over\s+\$5[,\s]?000[,\s]?000":       "Over $5,000,000",
}


def _standardise_trade_size(series: pd.Series) -> pd.Series:
    """Pass through well-formed ranges; bucket malformed/exact amounts."""
    import re
    result = series.copy().fillna("Unknown")
    for pat, replacement in _TRADE_SIZE_MAP.items():
        mask = result.str.contains(pat, regex=True, na=False)
        result[mask] = replacement
    return result


# ---------------------------------------------------------------------------
# Step 4: Ticker normalisation
# ---------------------------------------------------------------------------

def normalise_tickers(df: pd.DataFrame, parquet_dir: str,
                      report_path: str) -> tuple[pd.DataFrame, TickerNormalizer]:
    normalizer = TickerNormalizer(parquet_dir=parquet_dir)
    normalizer.fit()

    df["Matched_Ticker"] = normalizer.transform(df["Ticker"])
    normalizer.print_report(df["Ticker"])

    # Save unmatched report
    unmatched = normalizer.get_unmatched(df["Ticker"])
    unmatched.to_csv(report_path, index=False)
    logger.info("Ticker normalizer unmatched report saved: %s", report_path)

    before = len(df)
    df = df[df["Matched_Ticker"].notna()].copy()
    logger.info("Dropped %d rows with unmatched tickers; %d remain",
                before - len(df), len(df))
    return df.reset_index(drop=True), normalizer


# ---------------------------------------------------------------------------
# Step 5 & 6: Prices — vectorised computation
# ---------------------------------------------------------------------------

def compute_returns_vectorised(df: pd.DataFrame, parquet_dir: str) -> pd.DataFrame:
    """
    Vectorised forward return computation.
    Avoids the slow row-by-row loop in price_utils.compute_excess_returns
    by loading each ticker's parquet once and doing all date lookups at once.
    """
    import warnings
    from src.data_prep.price_utils import load_parquet, HORIZONS

    spy_df = load_parquet("SPY", parquet_dir)
    if spy_df is None:
        raise FileNotFoundError("SPY.parquet not found")

    # Preload all unique tickers
    unique_tickers = df["Matched_Ticker"].unique()
    logger.info("Pre-loading %d ticker parquets...", len(unique_tickers))
    ticker_dfs = {}
    for t in unique_tickers:
        tdf = load_parquet(t, parquet_dir)
        if tdf is not None:
            ticker_dfs[t] = tdf

    logger.info("Computing excess returns for %d rows...", len(df))
    t0 = time.time()

    all_results = {f"Excess_Return_{h}": np.full(len(df), np.nan) for h in HORIZONS}
    for h in HORIZONS:
        all_results[f"Stock_Return_{h}"] = np.full(len(df), np.nan)
        all_results[f"SPY_Return_{h}"]   = np.full(len(df), np.nan)

    # Group by ticker for efficiency
    for ticker, grp in df.groupby("Matched_Ticker"):
        tdf = ticker_dfs.get(ticker)
        if tdf is None:
            continue
        closes_stock = tdf["close"]
        closes_spy   = spy_df["close"]

        for i, (idx, row) in enumerate(grp.iterrows()):
            filed = row["Filed"]

            # Base prices (first trading day on or after filing)
            def get_base(closes, date):
                loc = closes.index.searchsorted(date)
                if loc >= len(closes):
                    return None
                # Scan up to 5 trading days forward
                for j in range(5):
                    if loc + j < len(closes):
                        v = closes.iloc[loc + j]
                        if v > 0:
                            return v, loc + j
                return None

            base_stock = get_base(closes_stock, filed)
            base_spy   = get_base(closes_spy,   filed)

            if base_stock is None or base_spy is None:
                continue

            p0_stock, base_pos_stock = base_stock
            p0_spy,   base_pos_spy   = base_spy

            for horizon, td in HORIZONS.items():
                fwd_pos_stock = base_pos_stock + td
                fwd_pos_spy   = base_pos_spy   + td
                if fwd_pos_stock >= len(closes_stock) or fwd_pos_spy >= len(closes_spy):
                    continue
                p1_stock = closes_stock.iloc[fwd_pos_stock]
                p1_spy   = closes_spy.iloc[fwd_pos_spy]
                if p1_stock <= 0 or p1_spy <= 0:
                    continue
                sr = (p1_stock - p0_stock) / p0_stock
                br = (p1_spy   - p0_spy)   / p0_spy
                all_results[f"Excess_Return_{horizon}"][idx] = sr - br
                all_results[f"Stock_Return_{horizon}"][idx] = sr
                all_results[f"SPY_Return_{horizon}"][idx]   = br

    logger.info("Returns computed in %.1fs", time.time() - t0)

    for col, arr in all_results.items():
        df[col] = arr

    return df


def compute_features_vectorised(df: pd.DataFrame, parquet_dir: str) -> dict:
    """
    Compute backward technical features per transaction.
    Returns {transaction_id: np.ndarray(14,)}.
    """
    from src.data_prep.price_utils import (
        load_parquet, _compute_features_for_ticker
    )

    spy_df = load_parquet("SPY", parquet_dir)
    if spy_df is None:
        raise FileNotFoundError("SPY.parquet not found")

    logger.info("Computing price features for %d rows...", len(df))
    t0 = time.time()
    feature_map = {}
    skipped = 0

    for ticker, grp in df.groupby("Matched_Ticker"):
        tdf = load_parquet(ticker, parquet_dir)
        if tdf is None:
            skipped += len(grp)
            continue
        for _, row in grp.iterrows():
            tx_id = row["transaction_id"]
            filed = row["Filed"]
            sf = _compute_features_for_ticker(tdf,    filed)
            bf = _compute_features_for_ticker(spy_df, filed)
            if sf is None or bf is None:
                skipped += 1
                continue
            feature_map[tx_id] = np.concatenate([sf, bf])

    logger.info("Features computed: %d rows, %d skipped; %.1fs",
                len(feature_map), skipped, time.time() - t0)
    return feature_map


# ---------------------------------------------------------------------------
# Step 7: Quality filter & select final columns
# ---------------------------------------------------------------------------

ML_COLUMNS = [
    "transaction_id",
    "BioGuideID",
    "Chamber",
    "Party",
    "State",
    "Ticker",          # original raw ticker (for reference)
    "Matched_Ticker",  # verified parquet-matched ticker (used for price lookups)
    "TickerType",
    "Transaction",     # Purchase / Sale
    "Trade_Size_USD",
    "Traded",          # as datetime
    "Filed",           # as datetime
    "Filing_Gap",      # days between trade and filing
    "Late_Filing",     # bool: Filed > 45 days after Traded
    # Forward returns (label columns)
    "Excess_Return_1M",
    "Excess_Return_2M",
    "Excess_Return_3M",
    "Excess_Return_6M",
    "Excess_Return_8M",
    "Excess_Return_12M",
    "Excess_Return_18M",
    "Excess_Return_24M",
    # Diagnostic / raw return columns for all horizons
    *[f"Stock_Return_{h}" for h in HORIZONS],
    *[f"SPY_Return_{h}" for h in HORIZONS],
]

# If the raw input provides committee/industry metadata, include them in the
# final ML dataset so downstream experiments can use them.
EXTRA_META_COLS = ["Company", "Name", "Committees", "Industry", "Sector"]
ML_COLUMNS += EXTRA_META_COLS


def build_final_dataset(df: pd.DataFrame, feature_map: dict,
                        min_horizon: str = "6M") -> pd.DataFrame:
    """
    Keep only rows where:
      - Excess_Return for the primary horizon is available (not NaN)
      - transaction_id has a computed price feature vector

    Also adds a 'Has_Price_Features' column for completeness.
    """
    # Keep rows with valid primary return label
    return_col = f"Excess_Return_{min_horizon}"
    if return_col in df.columns:
        before = len(df)
        df = df[df[return_col].notna()].copy()
        logger.info("Dropped %d rows: no %s return (future trades / data gap)",
                    before - len(df), min_horizon)

    df["Has_Price_Features"] = df["transaction_id"].isin(feature_map)
    logger.info("Rows with price features: %d / %d",
                df["Has_Price_Features"].sum(), len(df))

    # Select output columns (only those that exist)
    out_cols = [c for c in ML_COLUMNS + ["Has_Price_Features"] if c in df.columns]
    df = df[out_cols].copy()

    # Ensure dates are stored as strings in the CSV
    for col in ["Traded", "Filed"]:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, feature_map: dict, output_dir: str) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("BUILD V2 SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total rows in output: {len(df):,}")
    lines.append(f"Rows with price features: {df['Has_Price_Features'].sum():,}")
    lines.append(f"Date range (Filed): {df['Filed'].min()} — {df['Filed'].max()}")
    lines.append(f"Unique politicians (BioGuideID): {df['BioGuideID'].nunique():,}")
    lines.append(f"Unique tickers (Matched): {df['Matched_Ticker'].nunique():,}")
    lines.append("")
    lines.append("Transaction breakdown:")
    for k, v in df["Transaction"].value_counts().items():
        lines.append(f"  {k:12s}: {v:>8,}")
    lines.append("")
    lines.append("Chamber breakdown:")
    for k, v in df["Chamber"].value_counts().items():
        lines.append(f"  {k:12s}: {v:>8,}")
    lines.append("")
    lines.append("Party breakdown:")
    for k, v in df["Party"].value_counts().items():
        lines.append(f"  {k:14s}: {v:>8,}")
    lines.append("")
    lines.append("Return label availability:")
    for h in HORIZONS:
        col = f"Excess_Return_{h}"
        if col in df.columns:
            n = df[col].notna().sum()
            pct = n / len(df) * 100
            pos = (df[col] > 0).sum()
            pos_pct = pos / n * 100 if n > 0 else 0
            lines.append(f"  {h:4s}: {n:>8,} rows ({pct:5.1f}%)  |  "
                         f"Beat SPY: {pos:>7,} ({pos_pct:4.1f}%)")
    lines.append("")
    lines.append("Late filings (> 45 days): "
                 f"{df['Late_Filing'].sum():,} ({df['Late_Filing'].mean()*100:.1f}%)")
    lines.append("=" * 60)

    summary = "\n".join(lines)
    print(summary)

    summary_path = os.path.join(output_dir, "build_v2_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    logger.info("Summary saved: %s", summary_path)
    return summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build clean congressional trading ML dataset (v2)")
    parser.add_argument(
        "--input",
        default="experiments/signal_isolation/new_v5_transactions_with_committee_indsutry.csv",
        help="Path to raw v5 transactions CSV (supports Committees/Industry/Sector columns)",
    )
    parser.add_argument("--parquet-dir", default="data/parquet",
                        help="Directory containing ticker parquet files")
    parser.add_argument("--output-dir", default="data/processed",
                        help="Output directory for the final dataset")
    parser.add_argument("--min-year", type=int, default=2015,
                        help="Exclude filings before this year (default: 2015)")
    parser.add_argument("--skip-price-features", action="store_true",
                        help="Skip backward price feature computation (faster, for testing)")
    parser.add_argument("--primary-horizon", default="6M",
                        choices=list(HORIZONS.keys()),
                        help="Primary return horizon used for quality filtering (default: 6M)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load
    # ------------------------------------------------------------------
    df = load_raw(args.input)

    # ------------------------------------------------------------------
    # Step 2: Filter
    # ------------------------------------------------------------------
    logger.info("Step 2: Filtering transactions...")
    df = filter_transactions(df, min_year=args.min_year)

    # ------------------------------------------------------------------
    # Step 3: Standardise
    # ------------------------------------------------------------------
    logger.info("Step 3: Standardising fields...")
    df = standardise_fields(df)

    # ------------------------------------------------------------------
    # Step 4: Ticker normalisation
    # ------------------------------------------------------------------
    logger.info("Step 4: Normalising tickers...")
    report_path = os.path.join(args.output_dir, "ticker_normalizer_report.csv")
    df, normalizer = normalise_tickers(df, args.parquet_dir, report_path)

    # ------------------------------------------------------------------
    # Step 5: Forward returns
    # ------------------------------------------------------------------
    logger.info("Step 5: Computing forward excess returns...")
    df = compute_returns_vectorised(df, args.parquet_dir)

    # ------------------------------------------------------------------
    # Step 6: Backward price features
    # ------------------------------------------------------------------
    feature_map: dict = {}
    if not args.skip_price_features:
        logger.info("Step 6: Computing backward price features...")
        feature_map = compute_features_vectorised(df, args.parquet_dir)
        feat_path = os.path.join(args.output_dir, "price_sequences_v2.pt")
        torch.save(feature_map, feat_path)
        logger.info("Price features saved: %s (%d entries)", feat_path, len(feature_map))
    else:
        logger.info("Step 6: Skipped (--skip-price-features)")

    # ------------------------------------------------------------------
    # Step 7: Build final dataset
    # ------------------------------------------------------------------
    logger.info("Step 7: Building final ML dataset...")
    df_final = build_final_dataset(df, feature_map, min_horizon=args.primary_horizon)

    csv_path = os.path.join(args.output_dir, "ml_dataset_v2.csv")
    df_final.to_csv(csv_path, index=False)
    logger.info("Dataset saved: %s  (%d rows, %d columns)", csv_path, len(df_final), len(df_final.columns))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_summary(df_final, feature_map, args.output_dir)
    logger.info("Total build time: %.1f seconds", time.time() - t_start)


if __name__ == "__main__":
    main()
