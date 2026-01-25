import pandas as pd
import numpy as np
import re

# Import utility functions
from src.financial_pipeline.utils import load_csv_with_path, save_csv_with_path

# Configuration
INPUT_CSV = '../data/v8_transactions_final_cleaned.csv'
OUTPUT_CSV = '../data/v9_transactions.csv'

# Standardized trade size ranges
TRADE_SIZE_RANGES = [
    '1 - 1000',
    '1001 - 15000',
    '15001 - 50000',
    '50001 - 100000',
    '100001 - 250000',
    '250001 - 500000',
    '500001 - 1000000',
    '1000001 - 5000000',
    '5000001 - 25000000',
    '25000001 - 50000000'
]

def _standardize_trade_size(value):
    """Map trade size to standardized ranges"""
    if pd.isna(value):
        return 'Unknown'

    try:
        if value in TRADE_SIZE_RANGES:
            return value

        try:
            first_part = value.replace(',', '').split(' - ')[0].split('.')[0]
            amount = float(first_part)
        except:
            numbers = re.findall(r'\d+', value)
            if numbers:
                amount = float(numbers[0])
            else:
                return 'Unknown'

        if amount <= 1000:
            return '1 - 1000'
        elif amount <= 15000:
            return '1001 - 15000'
        elif amount <= 50000:
            return '15001 - 50000'
        elif amount <= 100000:
            return '50001 - 100000'
        elif amount <= 250000:
            return '100001 - 250000'
        elif amount <= 500000:
            return '250001 - 500000'
        elif amount <= 1000000:
            return '500001 - 1000000'
        elif amount <= 5000000:
            return '1000001 - 5000000'
        elif amount <= 25000000:
            return '5000001 - 25000000'
        else:
            return '25000001 - 50000000'
    except Exception as e:
        print(f"Error standardizing '{value}': {e}")
        return 'Unknown'

def add_transaction_ids_and_standardize(
    input_csv=INPUT_CSV,
    output_csv=OUTPUT_CSV
):
    print("Loading transaction data...")
    df = load_csv_with_path(input_csv)

    id_count = len(df)
    id_digits = len(str(id_count + 10000))
    df['ID'] = [f"{i:0{id_digits}d}" for i in range(1, id_count + 1)]

    print("Standardizing trade sizes...")
    df['Standardized_Trade_Size'] = df['Trade_Size_USD'].apply(_standardize_trade_size)

    print("Calculating filing gaps...")
    # Convert to datetime if not already
    df['Filed_DT'] = pd.to_datetime(df['Filed'], errors='coerce')
    df['Traded_DT'] = pd.to_datetime(df['Traded'], errors='coerce')
    
    # Calculate gap in days
    df['Filing_Gap'] = (df['Filed_DT'] - df['Traded_DT']).dt.days
    
    # Filter out negative gaps per user request
    initial_len = len(df)
    df = df[df['Filing_Gap'] >= 0].copy()
    removed = initial_len - len(df)
    if removed > 0:
        print(f"  Removed {removed} rows with negative Filing_Gap (impossible dates).")

    # Clean up temporary columns
    df = df.drop(columns=['Filed_DT', 'Traded_DT'])
    
    # Summary of gaps
    median_gap = df['Filing_Gap'].median()
    print(f"  Filing Gap computed (Median: {median_gap:.1f} days)")

    size_counts = df['Standardized_Trade_Size'].value_counts()
    print("\nStandardized Trade Size Distribution:")
    for size, count in size_counts.items():
        print(f"  {size}: {count}")

    cols = df.columns.tolist()
    cols = ['ID'] + [col for col in cols if col != 'ID']
    df = df[cols]

    save_csv_with_path(df, output_csv, index=False)
    print(f"Added {id_count} unique IDs and standardized trade sizes")
    print(f"Output saved to {output_csv}")
    return df