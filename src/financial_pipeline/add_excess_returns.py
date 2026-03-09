import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm

# Import utility functions
from src.financial_pipeline.utils import load_checkpoint, save_checkpoint, load_csv_with_path, save_csv_with_path

# Configuration
INPUT_CSV = '../data/v6_transactions.csv'
OUTPUT_CSV = '../data/v7_transactions.csv'
CHECKPOINT_FILE = '../data/excess_returns_checkpoint.pkl'
CHECKPOINT_INTERVAL = 100  # Save every 100 rows processed

# Required columns
COLUMNS_NEEDED = [
    'SPY_TradeDate', 
    'SPY_1Month', 'SPY_2Months', 'SPY_3Months', 
    'SPY_6Months', 'SPY_8Months', 'SPY_12Months', 
    'SPY_18Months', 'SPY_24Months',
    'Close_TradeDate', 
    'Close_1Month', 'Close_2Months', 'Close_3Months', 
    'Close_6Months', 'Close_8Months', 'Close_12Months', 
    'Close_18Months', 'Close_24Months'
]

NEW_COLUMNS = [
    'Stock_Return_1M', 'SPY_Return_1M', 'Excess_Return_1M', 'Beat_SPY_6pct_1M',
    'Stock_Return_2M', 'SPY_Return_2M', 'Excess_Return_2M', 'Beat_SPY_6pct_2M',
    'Stock_Return_3M', 'SPY_Return_3M', 'Excess_Return_3M', 'Beat_SPY_6pct_3M',
    'Stock_Return_6M', 'SPY_Return_6M', 'Excess_Return_6M', 'Beat_SPY_6pct_6M',
    'Stock_Return_8M', 'SPY_Return_8M', 'Excess_Return_8M', 'Beat_SPY_6pct_8M',
    'Stock_Return_12M', 'SPY_Return_12M', 'Excess_Return_12M', 'Beat_SPY_6pct_12M',
    'Stock_Return_18M', 'SPY_Return_18M', 'Excess_Return_18M', 'Beat_SPY_6pct_18M',
    'Stock_Return_24M', 'SPY_Return_24M', 'Excess_Return_24M', 'Beat_SPY_6pct_24M'
]

def _calculate_returns(row):
    """Calculate returns for all periods in a vectorized way"""
    results = {}

    period_map = {
        '1M': '1Month',
        '2M': '2Months',
        '3M': '3Months',
        '6M': '6Months',
        '8M': '8Months',
        '12M': '12Months',
        '18M': '18Months',
        '24M': '24Months'
    }

    for period_short, period_long in period_map.items():
        stock_close = row[f'Close_{period_long}']
        spy_close = row[f'SPY_{period_long}']
        stock_base = row['Close_TradeDate']
        spy_base = row['SPY_TradeDate']

        stock_return = (stock_close - stock_base) / stock_base if (stock_base and stock_base != 0) else np.nan
        spy_return = (spy_close - spy_base) / spy_base if (spy_base and spy_base != 0) else np.nan
        excess_return = stock_return - spy_return if not (np.isnan(stock_return) or np.isnan(spy_return)) else np.nan

        results.update({
            f'Stock_Return_{period_short}': stock_return,
            f'SPY_Return_{period_short}': spy_return,
            f'Excess_Return_{period_short}': excess_return,
            f'Beat_SPY_6pct_{period_short}': excess_return > 0.06 if not np.isnan(excess_return) else False
        })

    return pd.Series(results)

def add_excess_returns(
    input_csv=INPUT_CSV,
    output_csv=OUTPUT_CSV,
    checkpoint_file=CHECKPOINT_FILE,
    checkpoint_interval=CHECKPOINT_INTERVAL
):
    print("Loading data...")
    try:
        df = load_csv_with_path(input_csv)
    except FileNotFoundError:
        raise SystemExit("Error: Input file not found. Check the INPUT_CSV path.")

    missing = set(COLUMNS_NEEDED) - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    for col in NEW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    processed = load_checkpoint(checkpoint_file) or set()

    print("Calculating excess returns...")
    pbar = tqdm(total=len(df), desc="Processing rows")

    batch_size = checkpoint_interval
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size].copy()

        batch = batch[~batch.index.isin(processed)]
        if batch.empty:
            pbar.update(batch_size)
            continue

        results = batch.apply(_calculate_returns, axis=1)

        for col in NEW_COLUMNS:
            df.loc[results.index, col] = results[col]

        processed.update(batch.index.tolist())
        pbar.update(len(batch))

        if (i + batch_size) % checkpoint_interval == 0:
            save_csv_with_path(df, output_csv, index=False)
            save_checkpoint(processed, checkpoint_file)

    pbar.close()
    save_csv_with_path(df, output_csv, index=False)
    save_checkpoint(processed, checkpoint_file)
    print("Processing complete!")
    print(f"Final output saved to {output_csv}")
    return df
