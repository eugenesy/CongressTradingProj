
import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
from datetime import timedelta

# Paths
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRANSACTIONS_PATH = str(PROJECT_ROOT / "data" / "processed" / "ml_dataset_reduced_attributes.csv")
STOCK_DATA_DIR = str(PROJECT_ROOT / "data" / "parquet")
SPY_PATH = str(PROJECT_ROOT / "data" / "parquet" / "SPY.parquet")
OUTPUT_PATH = str(PROJECT_ROOT / "data" / "price_sequences.pt")

# Configuration
# We use Filing Date as the reference point (when market learns about the trade)
# Features are computed using data BEFORE the Filing Date

def compute_rsi(prices, period=14):
    """Compute Relative Strength Index."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_features(df, end_date):
    """
    Compute engineered features for a stock/SPY as of end_date.
    
    Returns a vector of features:
    - Return_1d, Return_5d, Return_10d, Return_20d
    - Volatility_20d
    - RSI_14
    - Volume_Ratio (Vol / Avg_Vol_20d)
    """
    # Get data up to end_date
    hist = df.loc[:end_date]
    
    if len(hist) < 21:  # Need at least 21 days for 20d features
        return None
    
    close = hist['close']
    volume = hist['volume']
    
    # Returns
    ret_1d = (close.iloc[-1] / close.iloc[-2] - 1) if len(close) >= 2 else 0
    ret_5d = (close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0
    ret_10d = (close.iloc[-1] / close.iloc[-11] - 1) if len(close) >= 11 else 0
    ret_20d = (close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else 0
    
    # Volatility (Std of daily returns over 20 days)
    daily_ret = close.pct_change().iloc[-20:]
    vol_20d = daily_ret.std() if len(daily_ret) >= 10 else 0
    
    # RSI
    rsi_series = compute_rsi(close)
    rsi_14 = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50  # Neutral if missing
    rsi_14 = rsi_14 / 100.0  # Normalize to [0, 1]
    
    # Volume Ratio
    avg_vol_20d = volume.iloc[-20:].mean()
    vol_ratio = volume.iloc[-1] / (avg_vol_20d + 1e-10)
    vol_ratio = np.log1p(vol_ratio)  # Log to compress outliers
    
    return np.array([ret_1d, ret_5d, ret_10d, ret_20d, vol_20d, rsi_14, vol_ratio], dtype=np.float32)

def load_data():
    print(f"Loading transactions from {TRANSACTIONS_PATH}...")
    df = pd.read_csv(TRANSACTIONS_PATH)
    
    df['Traded'] = pd.to_datetime(df['Traded'])
    df['Filed'] = pd.to_datetime(df['Filed'])
    
    df = df.dropna(subset=['Ticker', 'Filed', 'transaction_id'])
    df['transaction_id'] = df['transaction_id'].astype(int)
    
    print(f"Loaded {len(df)} transactions.")
    return df

def load_spy():
    print(f"Loading SPY data from {SPY_PATH}...")
    spy_df = pd.read_parquet(SPY_PATH)
    spy_df.index = pd.to_datetime(spy_df.index)
    # Ensure required columns
    for col in ['close', 'volume']:
        if col not in spy_df.columns:
            spy_df[col] = 0.0
    spy_df = spy_df.fillna(0.0)
    return spy_df

def process_features(df, spy_df):
    results = {}  # TransactionID -> Tensor (14,) = 7 Stock + 7 SPY features
    
    grouped = df.groupby('Ticker')
    skipped = 0
    
    print(f"Processing {len(grouped)} unique tickers...")
    
    for ticker, group in tqdm(grouped):
        stock_path = os.path.join(STOCK_DATA_DIR, f"{ticker}.parquet")
        if not os.path.exists(stock_path):
            skipped += len(group)
            continue
            
        try:
            stock_df = pd.read_parquet(stock_path)
            stock_df.index = pd.to_datetime(stock_df.index)
            for col in ['close', 'volume']:
                if col not in stock_df.columns:
                    stock_df[col] = 0.0
            stock_df = stock_df.fillna(0.0)
        except Exception as e:
            print(f"Error loading {ticker}: {e}")
            skipped += len(group)
            continue
            
        for _, row in group.iterrows():
            tid = row['transaction_id']
            filed_date = row['Filed']
            
            if pd.isna(filed_date):
                continue
            
            # Use data up to day BEFORE filing (avoid same-day info)
            end_date = filed_date - timedelta(days=1)
            
            # Compute Stock Features
            stock_feats = compute_features(stock_df, end_date)
            if stock_feats is None:
                continue
            
            # Compute SPY Features
            spy_feats = compute_features(spy_df, end_date)
            if spy_feats is None:
                continue
            
            # Combine: 7 Stock + 7 SPY = 14 features
            combined = np.concatenate([stock_feats, spy_feats])
            
            # Handle any NaN/Inf
            combined = np.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=-1.0)
            
            results[tid] = torch.tensor(combined, dtype=torch.float32)

    print(f"Skipped {skipped} due to missing data.")
    print(f"Generated {len(results)} feature vectors.")
    return results

def main():
    if not os.path.exists("data"):
        os.makedirs("data")
        
    df = load_data()
    spy = load_spy()
    
    feat_data = process_features(df, spy)
    
    print(f"Saving to {OUTPUT_PATH}...")
    torch.save(feat_data, OUTPUT_PATH)
    print("Done.")

if __name__ == "__main__":
    main()
