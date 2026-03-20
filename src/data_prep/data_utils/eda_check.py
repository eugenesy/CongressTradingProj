"""
eda_check.py
============
Data Preparation Module. Handles raw transformation to ML dataset.

Refactored/Audited: 2026-03-20
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("research")

def run_eda_check():
    data_path = "data/processed/ml_dataset_v2.csv"
    results_dir = Path("experiments/signal_isolation/results/eda")
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset: %s", data_path)
    df = pd.read_csv(data_path)
    
    # 1. Dataset shape and column names
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Columns: %s", df.columns.tolist())

    # Ensure Filed is datetime
    df['Filed'] = pd.to_datetime(df['Filed'])
    df['year'] = df['Filed'].dt.year

    # 2. Class balance per year per label type (L0 6M)
    # raw: Stock_Return_6M > 0
    # excess: Excess_Return_6M > 0
    test_years = [2019, 2020, 2021, 2022, 2023, 2024]
    balance_records = []
    
    for year in test_years:
        year_df = df[df['year'] == year]
        n_total = len(year_df)
        
        pos_raw = (year_df['Stock_Return_6M'] > 0).sum()
        rate_raw = pos_raw / n_total if n_total > 0 else 0
        
        pos_excess = (year_df['Excess_Return_6M'] > 0).sum()
        rate_excess = pos_excess / n_total if n_total > 0 else 0
        
        balance_records.append({
            'year': year,
            'rows': n_total,
            'pos_rate_raw': rate_raw,
            'pos_rate_excess': rate_excess
        })
        logger.info("Year %d: %d rows, Raw Pos Rate: %.3f, Excess Pos Rate: %.3f", 
                    year, n_total, rate_raw, rate_excess)

    balance_df = pd.DataFrame(balance_records)
    balance_df.to_csv(results_dir / "class_balance.csv", index=False)

    # 3. |Stock_Return_6M| percentiles
    # Try all rows for dead-zone to match facts
    abs_ret = df['Stock_Return_6M'].abs()
    
    percentiles = [0.05, 0.10, 0.15, 0.20, 0.25]
    # We want to know what FRACTION of data is WITHIN a threshold
    # The facts say: "±10%: 42.4%" -> this means return <= 0.10
    
    dz_stats = []
    for p in [0.05, 0.10, 0.15, 0.20]:
        frac = (abs_ret < p).mean()
        dz_stats.append({'threshold': p, 'fraction_within': frac})
        logger.info("Dead zone ±%.2f: %.1f%% of data", p, frac * 100)
    
    dz_df = pd.DataFrame(dz_stats)
    dz_df.to_csv(results_dir / "dead_zone_dist.csv", index=False)

    # 4. Expanding-window train sizes
    train_size_records = []
    for year in test_years:
        # Train rows available before each test year
        train_df = df[df['year'] < year]
        train_size_records.append({
            'test_year': year,
            'train_rows': len(train_df)
        })
        logger.info("Train rows before %d: %d", year, len(train_df))
    
    train_sizes_df = pd.DataFrame(train_size_records)
    train_sizes_df.to_csv(results_dir / "train_sizes.csv", index=False)

    logger.info("EDA Verification complete. Files saved to %s", results_dir)

if __name__ == "__main__":
    run_eda_check()
