"""
Script to build price sequences for the Temporal Graph Network.
Generates 'data/price_sequences.pt' containing aligned price history tensors.
"""

import pandas as pd
import numpy as np
import torch
import pickle
from tqdm import tqdm
from datetime import timedelta
import sys
import os
from pathlib import Path

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import PRICE_PATH, PROCESSED_DATA_DIR

# Configuration
# Fix: Ensure PROCESSED_DATA_DIR is a Path object before using / operator
TRANSACTIONS_PATH = Path(PROCESSED_DATA_DIR) / "ml_dataset_final.csv"
OUTPUT_PATH = "data/price_sequences.pt"
SEQUENCE_LENGTH = 14  # Days of history to encode

def load_data():
    """Load transactions and historical price data."""
    print(f"Loading transactions from {TRANSACTIONS_PATH}...")
    if not TRANSACTIONS_PATH.exists():
        raise FileNotFoundError(f"Transaction file not found: {TRANSACTIONS_PATH}")
    
    df = pd.read_csv(TRANSACTIONS_PATH, parse_dates=['Filed', 'Traded'])
    
    print(f"Loading historical prices from {PRICE_PATH}...")
    if not os.path.exists(PRICE_PATH):
        raise FileNotFoundError(f"Price file not found: {PRICE_PATH}")
        
    with open(PRICE_PATH, 'rb') as f:
        price_data = pickle.load(f)
        
    return df, price_data

def build_sequences(df, price_data):
    """
    Construct a tensor of price sequences aligned with transactions.
    
    Returns:
        torch.Tensor: Shape (num_transactions, sequence_length)
    """
    sequences = {}
    skipped = 0
    
    # Ensure transaction_id exists
    if 'transaction_id' not in df.columns:
        print("Warning: 'transaction_id' column missing. Using index.")
        df['transaction_id'] = df.index + 1

    print(f"Building {SEQUENCE_LENGTH}-day price sequences...")
    
    if 'SPY' in price_data:
        print("Verified: Price data contains SPY benchmark.")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Aligning Prices"):
        tid = int(row['transaction_id'])
        ticker = row['Ticker']
        
        # Use Traded Date to capture signal leading up to the action
        ref_date = row['Traded']
        if pd.isna(ref_date):
            ref_date = row['Filed']
            
        if pd.isna(ticker) or ticker not in price_data:
            skipped += 1
            sequences[tid] = torch.zeros(SEQUENCE_LENGTH, dtype=torch.float32)
            continue
            
        hist = price_data[ticker]
        
        # Extract sequence
        seq_values = []
        current_d = ref_date
        days_found = 0
        attempts = 0
        
        # Look back up to 45 days to find 14 trading days
        while days_found < SEQUENCE_LENGTH and attempts < 45:
            d_str = current_d.strftime('%Y-%m-%d')
            if d_str in hist:
                val = hist[d_str].get('changePercent', 0.0)
                seq_values.insert(0, val)
                days_found += 1
            
            current_d -= timedelta(days=1)
            attempts += 1
            
        # Pad if insufficient data
        if len(seq_values) < SEQUENCE_LENGTH:
            padding = [0.0] * (SEQUENCE_LENGTH - len(seq_values))
            seq_values = padding + seq_values
            
        sequences[tid] = torch.tensor(seq_values, dtype=torch.float32)

    print(f"Sequences built. Skipped/Zeroed {skipped} transactions due to missing data.")
    return sequences

def main():
    try:
        df, price_data = load_data()
        sequences_map = build_sequences(df, price_data)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        
        print(f"Saving sequence map to {OUTPUT_PATH}...")
        torch.save(sequences_map, OUTPUT_PATH)
        print("✅ Done.")
        
    except Exception as e:
        print(f"❌ Error in build_prices: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()