"""
path_dependent_study.py
=======================
Baseline Modelling Module. Implements non-GNN tabular approaches (e.g. XGBoost, RF).

Refactored/Audited: 2026-03-20
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def study_max_excess_return(trades_path, parquet_dir, sample_size=3000):
    print(f"Loading trades from {trades_path}...")
    df = pd.read_csv(trades_path)
    # Ensure correct types
    df['Filed'] = pd.to_datetime(df['Filed'])
    
    # Load SPY for continuous benchmark
    spy_path = Path(parquet_dir) / "SPY.parquet"
    if not spy_path.exists():
        print("Error: SPY.parquet missing.")
        return
    spy_df = pd.read_parquet(spy_path)
    spy_df.index = pd.to_datetime(spy_df.index)
    spy_df = spy_df.sort_index()

    # Sample rows to run reasonably fast
    sub_df = df.dropna(subset=['Ticker', 'Filed']).sample(n=sample_size, random_state=42)
    
    horizons = {'3M': 63, '6M': 126, '12M': 252} # approx trading days
    results_by_horizon = {h: [] for h in horizons}

    print(f"Processing {sample_size} sample trades...")
    for idx, row in tqdm(sub_df.iterrows(), total=len(sub_df)):
        ticker = str(row['Ticker']).strip().upper()
        trade_date = row['Filed']
        
        tick_path = Path(parquet_dir) / f"{ticker}.parquet"
        if not tick_path.exists():
             continue
             
        try:
            tick_df = pd.read_parquet(tick_path)
            tick_df.index = pd.to_datetime(tick_df.index)
            tick_df = tick_df.sort_index()
            
            # Sub-frame for trade starting timeline
            stock_sub = tick_df.loc[trade_date:]
            spy_sub = spy_df.loc[trade_date:]
            
            if stock_sub.empty or spy_sub.empty:
                continue

            # Start point
            p0 = stock_sub['close'].iloc[0]
            s0 = spy_sub['close'].iloc[0]
            
            for h_label, days in horizons.items():
                if len(stock_sub) < days:
                    continue # not enough history
                
                s_window = stock_sub.head(days)
                spy_window = spy_sub.head(days)
                
                # Returns from start date
                r_stock = s_window['close'] / p0 - 1
                r_spy = spy_window['close'] / s0 - 1
                
                excess = r_stock - r_spy
                excess = excess.dropna()
                if excess.empty:
                    continue
                max_excess = excess.max()
                
                results_by_horizon[h_label].append(max_excess)
                
        except Exception as e:
            continue

    print("\n" + "="*70)
    print("CONTINUOUS 'MAX' EXCESS RETURN PATHS RESULTS: ")
    print("="*70)
    
    for h, vals in results_by_horizon.items():
        if not vals:
             continue
        v = np.array(vals)
        print(f"\nHorizon: {h} (n={len(v)})")
        print(f"  Mean Maximum Excess: {np.mean(v):.4f}")
        print(f"  Std Maximum Excess:  {np.std(v):.4f}")
        print(f"  25% Quantile (Q1):   {np.percentile(v, 25):.4f}")
        print(f"  50% Quantile (Median):{np.percentile(v, 50):.4f}")
        print(f"  75% Quantile (Q3):   {np.percentile(v, 75):.4f}")
        print(f"  90% Quantile:        {np.percentile(v, 90):.4f}")
        
if __name__ == "__main__":
    study_max_excess_return("data/processed/ml_dataset_v2.csv", "data/parquet")
