import pandas as pd
import numpy as np
from tqdm import tqdm
from src.financial_pipeline.utils import load_checkpoint, save_checkpoint, get_data_path

# Config - Input: v3, Output: v4
INPUT_CSV = get_data_path('processed', 'ml_dataset_v3.csv')
OUTPUT_CSV = get_data_path('processed', 'ml_dataset_v4.csv')
CHECKPOINT_FILE = get_data_path('processed', 'excess_returns_checkpoint.pkl')
CHECKPOINT_INTERVAL = 100

def _calculate_returns(row):
    results = {}
    period_map = {
        '1M': '1Month', '2M': '2Months', '3M': '3Months', '6M': '6Months',
        '8M': '8Months', '12M': '12Months', '18M': '18Months', '24M': '24Months'
    }
    for period_short, period_long in period_map.items():
        stock_close = row.get(f'Close_{period_long}')
        spy_close = row.get(f'SPY_{period_long}')
        stock_base = row.get('Close_TradeDate')
        spy_base = row.get('SPY_TradeDate')

        stock_return = (stock_close - stock_base) / stock_base if (stock_base and stock_base != 0) else np.nan
        spy_return = (spy_close - spy_base) / spy_base if (spy_base and spy_base != 0) else np.nan
        excess_return = stock_return - spy_return if not (np.isnan(stock_return) or np.isnan(spy_return)) else np.nan

        results[f'Excess_Return_{period_short}'] = excess_return
    return pd.Series(results)

def add_excess_returns(
    input_csv=INPUT_CSV,
    output_csv=OUTPUT_CSV,
    checkpoint_file=CHECKPOINT_FILE,
    checkpoint_interval=CHECKPOINT_INTERVAL
):
    print("Loading data for excess returns...")
    df = pd.read_csv(input_csv)
    
    new_cols = [f'Excess_Return_{p}' for p in ['1M','2M','3M','6M','8M','12M','18M','24M']]
    for col in new_cols:
        if col not in df.columns: df[col] = np.nan

    processed = load_checkpoint(checkpoint_file) or set()
    pbar = tqdm(total=len(df), desc="Processing rows")
    
    batch_size = checkpoint_interval
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size].copy()
        batch = batch[~batch.index.isin(processed)]
        if batch.empty:
            pbar.update(batch_size)
            continue

        results = batch.apply(_calculate_returns, axis=1)
        for col in results.columns:
            df.loc[results.index, col] = results[col]

        processed.update(batch.index.tolist())
        pbar.update(len(batch))

        if (i + batch_size) % checkpoint_interval == 0:
            df.to_csv(output_csv, index=False)
            save_checkpoint(processed, checkpoint_file)

    pbar.close()
    df.to_csv(output_csv, index=False)
    save_checkpoint(processed, checkpoint_file)
    print(f"Excess returns saved to {output_csv}")
    return df