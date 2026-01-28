import pandas as pd
import pickle
import os
from tqdm import tqdm
from datetime import datetime

# Import utility functions
from src.financial_pipeline.utils import load_checkpoint, save_checkpoint, get_data_path

# Config - updated to ml_dataset_clean.csv
CSV_FILE = get_data_path('raw', 'ml_dataset_clean.csv')
OUT_PKL = get_data_path('processed', 'all_tickers_historical_data.pkl')
FAILED_REPORT = get_data_path('processed', 'failed_tickers_report.txt')
PARQUET_DIR = get_data_path('parquet')
START_DATE = datetime(2012, 7, 30)
END_DATE = datetime(2025, 10, 4)
CHECKPOINT_INTERVAL = 1000
SAMPLE_TICKER = 'AAPL'

def _load_ticker_from_parquet(ticker, start_date, end_date):
    """Load historical data for a single ticker from parquet file"""
    parquet_file = os.path.join(PARQUET_DIR, f'{ticker}.parquet')

    if not os.path.exists(parquet_file):
        return None, "File not found"

    try:
        df = pd.read_parquet(parquet_file)
        df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]

        if len(df_filtered) == 0:
            return {}, "No data in date range"

        ticker_dict = {}
        for date, row in df_filtered.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            ticker_dict[date_str] = {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': int(row['volume']),
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
    with open(FAILED_REPORT, 'w') as f:
        f.write(f"FAILED TICKERS REPORT - {datetime.now()}\n\n")
        for ticker, reason in failed_tickers:
            f.write(f"{ticker}: {reason}\n")
    print(f"\nFailed tickers report saved to: {FAILED_REPORT}")

def _show_data_preview():
    print("\nLoading sample data for preview...")
    parquet_file = os.path.join(PARQUET_DIR, f'{SAMPLE_TICKER}.parquet')
    try:
        if os.path.exists(parquet_file):
            df = pd.read_parquet(parquet_file)
            print(f"Sample data loaded. Date range: {df.index.min()} to {df.index.max()}")
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
        print("Could not retrieve sample data. Check parquet directory.")
        return False

    df = pd.read_csv(csv_file)
    # Check for correct ticker column
    ticker_col = 'Ticker' if 'Ticker' in df.columns else 'Appropriate_Ticker'
    tickers = df[ticker_col].dropna().unique()
    
    all_data = load_checkpoint(out_pkl) or {}
    processed = set(all_data.keys())

    print(f"\nTotal tickers to process: {len(tickers)}")
    print(f"Remaining: {len(tickers) - len(processed)}")

    failed_tickers = []
    pbar = tqdm([t for t in tickers if t not in processed], desc="Loading Tickers")

    for ticker in pbar:
        ticker_dict, error = _load_ticker_from_parquet(ticker, start_date, end_date)
        if ticker_dict is not None:
            all_data[ticker] = ticker_dict
        else:
            all_data[ticker] = {}
            failed_tickers.append((ticker, error))

        if len(all_data) % checkpoint_interval == 0:
            save_checkpoint(all_data, out_pkl)

    save_checkpoint(all_data, out_pkl)
    if failed_tickers:
        _save_failed_report(failed_tickers)

    print("LOADING COMPLETE!")
    return True