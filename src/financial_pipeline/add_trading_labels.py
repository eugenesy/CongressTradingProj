import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm

# Import utility functions
from src.financial_pipeline.utils import load_checkpoint, save_checkpoint, load_csv_with_path, save_csv_with_path

# Configuration
INPUT_CSV = '../data/v7_transactions.csv'
OUTPUT_CSV = '../data/v8_transactions.csv'
CHECKPOINT_FILE = '../data/monthly_label_checkpoint.pkl'
CHECKPOINT_INTERVAL = 1000

def _create_monthly_labels(row):
    """Determine labels for each month based on transaction type and excess returns"""
    transaction = row['Transaction']
    labels = {'Label_3M': np.nan, 'Label_6M': np.nan}

    if pd.isna(transaction):
        return labels

    for period in ['3M', '6M']:
        excess_return = row.get(f'Excess_Return_{period}', np.nan)

        if pd.isna(excess_return):
            labels[f'Label_{period}'] = np.nan
            continue

        excess_above_6 = excess_return > 0.06
        labels[f'Label_{period}'] = 1 if excess_above_6 else 0

    return pd.Series(labels)

def add_trading_labels(
    input_csv=INPUT_CSV,
    output_csv=OUTPUT_CSV,
    checkpoint_file=CHECKPOINT_FILE,
    checkpoint_interval=CHECKPOINT_INTERVAL
):
    print("Loading dataset...")
    df = load_csv_with_path(input_csv, low_memory=False)

    for period in ['1M', '3M', '6M']:
        col_name = f'Label_{period}'
        if col_name not in df.columns:
            df[col_name] = np.nan

    processed = load_checkpoint(checkpoint_file) or set()

    print("Calculating monthly labels...")
    pbar = tqdm(total=len(df), desc="Processing rows")

    for idx in df.index:
        if idx in processed:
            pbar.update(1)
            continue

        try:
            new_labels = _create_monthly_labels(df.loc[idx])
            for col in new_labels.index:
                df.at[idx, col] = new_labels[col]
        except KeyError as e:
            print(f"\nMissing column: {e}")
        except Exception as e:
            print(f"\nError processing row {idx}: {e}")

        processed.add(idx)
        pbar.update(1)

        if len(processed) % checkpoint_interval == 0:
            save_csv_with_path(df, output_csv, index=False)
            save_checkpoint(processed, checkpoint_file)

    pbar.close()
    save_csv_with_path(df, output_csv, index=False)
    save_checkpoint(processed, checkpoint_file)
    print("\nLabeling complete!")
    print(f"Final output saved to {output_csv}")
    return df