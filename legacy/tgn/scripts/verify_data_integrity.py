#!/usr/bin/env python3
"""Quick diagnostic to verify TGN data integrity."""
import torch
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPE_DIR = os.path.dirname(os.path.dirname(BASE_DIR))  # Go up to grape/
sys.path.insert(0, GRAPE_DIR)

from clean_tgn.src.temporal_data import TemporalGraphBuilder

def main():
    print("=" * 60)
    print("DATA INTEGRITY CHECK")
    print("=" * 60)
    
    # Load transactions
    TX_PATH = "/data1/user_syeugene/fintech/grape/data/processed/ml_dataset_reduced_attributes.csv"
    import pandas as pd
    transactions = pd.read_csv(TX_PATH)
    print(f"Loaded {len(transactions)} transactions")
    
    # Build data
    builder = TemporalGraphBuilder(transactions, min_freq=1)
    data, x_static, num_parties, num_states, num_chambers = builder.process(horizon='1M', threshold=0.01)
    
    print("\n### 1. LABELS (y) ###")
    y = data.y.numpy()
    print(f"  Shape: {y.shape}")
    print(f"  Unique values: {np.unique(y)}")
    print(f"  Class distribution: 0={np.sum(y==0)} ({100*np.mean(y==0):.1f}%), 1={np.sum(y==1)} ({100*np.mean(y==1):.1f}%)")
    print(f"  Any NaN: {np.any(np.isnan(y))}")
    
    print("\n### 2. MESSAGES (msg) ###")
    msg = data.msg.numpy()
    print(f"  Shape: {msg.shape}")  # Should be (N, 4) for [Amt, Buy, Gap, WinRate]
    print(f"  Column names: [Amt, Buy, Gap, WinRate]")
    for i, name in enumerate(['Amt', 'Buy', 'Gap', 'WinRate']):
        col = msg[:, i]
        print(f"    {name}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}, std={col.std():.4f}, NaN={np.sum(np.isnan(col))}")
    
    print("\n### 3. PRICE SEQUENCES ###")
    if hasattr(data, 'price_seq'):
        price = data.price_seq.numpy()
        print(f"  Shape: {price.shape}")  # Should be (N, 14)
        print(f"  Any NaN: {np.any(np.isnan(price))}")
        print(f"  Any Inf: {np.any(np.isinf(price))}")
        print(f"  All zeros: {np.all(price == 0)}")
        print(f"  Mean per feature: {np.mean(price, axis=0)[:5]}... (showing first 5)")
        print(f"  Std per feature: {np.std(price, axis=0)[:5]}... (showing first 5)")
    else:
        print("  NOT FOUND!")
    
    print("\n### 4. STATIC FEATURES ###")
    static = x_static.numpy()
    print(f"  Shape: {static.shape}")
    print(f"  Unique Party IDs: {len(np.unique(static[:, 0]))}")
    print(f"  Unique State IDs: {len(np.unique(static[:, 1]))}")
    print(f"  Unique Chamber IDs: {len(np.unique(static[:, 2]))}")
    
    print("\n### 5. TEMPORAL INFO ###")
    t = data.t.numpy()
    print(f"  Time range: {t.min():.0f} to {t.max():.0f}")
    print(f"  Time span (days): {(t.max() - t.min()) / (24*3600):.0f}")
    
    print("\n### 6. CORRELATION CHECK ###")
    print("  Checking if features correlate with target...")
    from scipy.stats import pointbiserialr
    for i, name in enumerate(['Amt', 'Buy', 'Gap', 'WinRate']):
        col = msg[:, i]
        # Remove NaN for correlation
        mask = ~np.isnan(col)
        if mask.sum() > 0:
            corr, pval = pointbiserialr(y[mask], col[mask])
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"    {name} <-> y: r={corr:.4f}, p={pval:.4f} {sig}")
    
    # Price features correlation
    if hasattr(data, 'price_seq'):
        price = data.price_seq.numpy()
        price_names = ['Stock_1d', 'Stock_5d', 'Stock_10d', 'Stock_20d', 'Vol_20d', 'RSI', 'VolRatio',
                      'SPY_1d', 'SPY_5d', 'SPY_10d', 'SPY_20d', 'SPY_Vol', 'SPY_RSI', 'SPY_VolRatio']
        for i, name in enumerate(price_names):
            col = price[:, i]
            mask = ~np.isnan(col)
            if mask.sum() > 0:
                corr, pval = pointbiserialr(y[mask], col[mask])
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                print(f"    {name} <-> y: r={corr:.4f}, p={pval:.4f} {sig}")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

if __name__ == "__main__":
    main()
