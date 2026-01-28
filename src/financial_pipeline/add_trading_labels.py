import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm

# Import utility functions
from src.financial_pipeline.utils import load_checkpoint, save_checkpoint, load_csv_with_path, save_csv_with_path, get_data_path

# Configuration
# Input: v4 (Post-Excess Returns)
# Output: v4_labeled (Side branch or intermediate if inserted back into pipeline)
INPUT_CSV = get_data_path('processed', 'ml_dataset_v4.csv')
OUTPUT_CSV = get_data_path('processed', 'ml_dataset_v4_labeled.csv')
CHECKPOINT_FILE = get_data_path('processed', 'monthly_label_checkpoint.pkl')
CHECKPOINT_INTERVAL = 1000

def _create_monthly_labels(row):
    """Determine labels for each month based on transaction type and excess returns"""
    transaction = row.get('Transaction', np.nan)
    labels = {'Label_3M': np.nan, 'Label_6M': np.nan}

    if pd.isna(transaction):
        return labels

    for period in ['3M', '6M']:
        excess_return = row.get(f'Excess_Return_{period}', np.nan)

        if pd.isna(excess_return):
            labels[f'Label_{period}'] = np.nan
            continue

        # Logic: If return > 6% (alpha), Label = 1, else 0
        excess_above_6 = excess_return > 0.06
        labels[f'Label_{period}'] = 1 if excess_above_6 else 0

    return pd.Series(labels)

def add_trading_labels(
    input_csv=INPUT_CSV,
    output_csv=OUTPUT_CSV,
    checkpoint_file=CHECKPOINT_FILE,
    checkpoint_interval=CHECKPOINT_INTERVAL
):
    print("Loading dataset for labeling...")
    # Use pandas directly with the path object/string
    df = pd.read_csv(input_csv, low_memory=False)

    for period in ['1M', '3M', '6M']:
        col_name = f'Label_{period}'
        if col_name not in df.columns:
            df[col_name] = np.nan

    processed = load_checkpoint(checkpoint_file) or set()

    print("Calculating monthly labels...")
    pbar = tqdm(total=len(df), desc="Processing rows")

    # Use a copy to avoid SettingWithCopy warnings if slice
    for idx in df.index:
        if idx in processed:
            pbar.update(1)
            continue

        try:
            new_labels = _create_monthly_labels(df.loc[idx])
            for col in new_labels.index:
                df.at[idx, col] = new_labels[col]
        except KeyError as e:
            # print(f"\nMissing column: {e}") # Reduce noise
            pass
        except Exception as e:
            print(f"\nError processing row {idx}: {e}")

        processed.add(idx)
        pbar.update(1)

        if len(processed) % checkpoint_interval == 0:
            df.to_csv(output_csv, index=False)
            save_checkpoint(processed, checkpoint_file)

    pbar.close()
    df.to_csv(output_csv, index=False)
    save_checkpoint(processed, checkpoint_file)
    print("\nLabeling complete!")
    print(f"Final output saved to {output_csv}")
    return df

if __name__ == "__main__":
    add_trading_labels()