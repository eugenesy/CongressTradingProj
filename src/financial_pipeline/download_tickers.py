import pandas as pd
import pickle
import os
from tqdm import tqdm
from datetime import datetime

# Import utility functions
from src.financial_pipeline.utils import load_checkpoint, save_checkpoint, get_data_path

# Config - These could ideally be passed as arguments or loaded from a central config
CSV_FILE = get_data_path('processed', 'v5_transactions_with_approp_ticker.csv')
OUT_PKL = get_data_path('processed', 'all_tickers_historical_data.pkl')
FAILED_REPORT = get_data_path('processed', 'failed_tickers_report.txt')
PARQUET_DIR = get_data_path('parquet')
START_DATE = datetime(2012, 7, 30)
END_DATE = datetime(2025, 10, 4)
CHECKPOINT_INTERVAL = 1000
SAMPLE_TICKER = 'AAPL'  # Example ticker for preview

def _load_ticker_from_parquet(ticker, start_date, end_date):
    """Load historical data for a single ticker from parquet file"""
    parquet_file = os.path.join(PARQUET_DIR, f'{ticker}.parquet')

    if not os.path.exists(parquet_file):
        return None, "File not found"

    try:
        df = pd.read_parquet(parquet_file)

        # Filter to date range
        df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]

        if len(df_filtered) == 0:
            return {}, "No data in date range"

        # Convert to dictionary format matching original structure
        ticker_dict = {}
        for date, row in df_filtered.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            ticker_dict[date_str] = {
                # Original columns
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': int(row['volume']),
                # NEW columns from parquet
                'adjClose': row['adjClose'],
                'unadjustedVolume': int(row['unadjustedVolume']),
                'change': row['change'],
                'changePercent': row['changePercent'],
                'vwap': row['vwap'],
                'label': row['label'],
                'changeOverTime': row['changeOverTime']
            }
        return ticker_dict, None

    except Exception as e:
        return None, str(e)

def _save_failed_report(failed_tickers):
    """Save a detailed report of all failed tickers"""
    with open(FAILED_REPORT, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("FAILED TICKERS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        by_type = {}
        for ticker, reason in failed_tickers:
            if reason not in by_type:
                by_type[reason] = []
            by_type[reason].append(ticker)

        f.write(f"SUMMARY:\n")
        f.write(f"Total failed: {len(failed_tickers)}\n\n")

        for reason, tickers in by_type.items():
            f.write(f"{reason}: {len(tickers)}\n")

        f.write("\n" + "=" * 60 + "\n\n")

        for reason, tickers in sorted(by_type.items()):
            f.write(f"\n{reason.upper()} ({len(tickers)} tickers):\n")
            f.write("-" * 60 + "\n")
            for ticker in sorted(tickers):
                f.write(f"  {ticker}\n")
            f.write("\n")

    print(f"\nFailed tickers report saved to: {FAILED_REPORT}")

def _show_data_preview():
    """Display sample historical data before starting full download"""
    print("\nLoading sample data for preview...")
    parquet_file = os.path.join(PARQUET_DIR, f'{SAMPLE_TICKER}.parquet')

    try:
        if os.path.exists(parquet_file):
            df = pd.read_parquet(parquet_file)
            df_filtered = df[(df.index >= START_DATE) & (df.index <= END_DATE)]
            df_preview = df_filtered.head(3).reset_index()
            df_preview = df_preview[['date', 'open', 'high', 'low', 'close', 'volume',
                                     'adjClose', 'change', 'changePercent', 'vwap']]

            print("\nSample data format:")
            print(df_preview)
            print("\nAll available columns:", df.columns.tolist())
            print(f"Date range in parquet: {df.index.min()} to {df.index.max()}")
            print(f"Total days available: {len(df)}")
            return True
        else:
            print(f"Sample file not found: {parquet_file}")
            return False
    except Exception as e:
        print(f"Preview error: {e}")
        return False

def download_all_tickers_historical(
    csv_file=str(CSV_FILE),
    out_pkl=str(OUT_PKL),
    failed_report=str(FAILED_REPORT),
    parquet_dir=str(PARQUET_DIR),
    start_date=START_DATE,
    end_date=END_DATE,
    checkpoint_interval=CHECKPOINT_INTERVAL,
    sample_ticker=SAMPLE_TICKER
):
    print("Starting download of all tickers historical data...")

    if not _show_data_preview():
        print("Could not retrieve sample data. Check parquet directory and files.")
        return False

    # input("\nPress Enter to start loading all tickers or Ctrl+C to abort...")

    df = pd.read_csv(csv_file)
    tickers = df['Appropriate_Ticker'].dropna().unique()
    all_data = load_checkpoint(out_pkl) or {}
    processed = set(all_data.keys())

    print(f"\nTotal unique tickers to process: {len(tickers)}")
    print(f"Already processed: {len(processed)}")
    print(f"Remaining: {len(tickers) - len(processed)}\n")

    stats = {
        'success': 0,
        'not_found': 0,
        'no_data': 0,
        'error': 0
    }
    failed_tickers = []

    pbar = tqdm(
        [t for t in tickers if t not in processed],
        desc="Loading Tickers",
        unit="ticker"
    )

    for ticker in pbar:
        pbar.set_postfix_str(ticker)

        ticker_dict, error = _load_ticker_from_parquet(ticker, start_date, end_date)

        if ticker_dict is not None:
            all_data[ticker] = ticker_dict
            if len(ticker_dict) > 0:
                days_count = len(ticker_dict)
                pbar.write(f"✓ {ticker}: Loaded {days_count} days")
                stats['success'] += 1
            else:
                pbar.write(f"⚠ {ticker}: No data in date range")
                stats['no_data'] += 1
                failed_tickers.append((ticker, "No data in date range"))
        else:
            all_data[ticker] = {}
            if "File not found" in error:
                pbar.write(f"✗ {ticker}: Parquet file not found")
                stats['not_found'] += 1
                failed_tickers.append((ticker, "File not found"))
            else:
                pbar.write(f"✗ {ticker}: Error - {error}")
                stats['error'] += 1
                failed_tickers.append((ticker, f"Error: {error}"))

        if (len(all_data) % checkpoint_interval == 0) and (len(all_data) > len(processed)):
            save_checkpoint(all_data, out_pkl)

    save_checkpoint(all_data, out_pkl)

    if failed_tickers:
        _save_failed_report(failed_tickers)

    print("\n" + "=" * 60)
    print("LOADING COMPLETE!")
    print("=" * 60)
    print(f"Total tickers processed: {len(all_data)}")
    print(f"✓ Successfully loaded: {stats['success']}")
    print(f"✗ File not found: {stats['not_found']}")
    print(f"⚠ No data in range: {stats['no_data']}")
    print(f"✗ Errors: {stats['error']}")
    print(f"\nOutput file: {out_pkl}")

    if failed_tickers:
        print(f"\n⚠ WARNING: {len(failed_tickers)} tickers failed")
        print(f"See detailed report: {FAILED_REPORT}")
        print(f"\nTop 10 failed tickers:")
        for ticker, reason in failed_tickers[:10]:
            print(f"  {ticker}: {reason}")
        if len(failed_tickers) > 10:
            print(f"  ... and {len(failed_tickers) - 10} more (see report file)")

    if stats['success'] > 0:
        sample_ticker = [t for t, data in all_data.items() if len(data) > 0][0]
        sample_date = list(all_data[sample_ticker].keys())[0]
        print(f"\nSample entry ({sample_ticker} on {sample_date}):")
        for key, value in all_data[sample_ticker][sample_date].items():
            print(f"  {key}: {value}")
    return True