"""
preprocess_continuous.py
========================
Data Preparation Module. Handles raw transformation to ML dataset.

Refactored/Audited: 2026-03-20
"""

import pandas as pd
from run_path_dependent_ml import vectorize_continuous_labels
from pathlib import Path

def main():
    print("⏳ Vectorizing continuous labels once for caching...")
    df_raw = pd.read_csv("data/processed/ml_dataset_v2.csv")
    out_dir = Path("data/processed")
    out_dir.mkdir(exist_ok=True)
    
    labeled_df = vectorize_continuous_labels(df_raw, "data/parquet", horizon_days=126)
    dest = out_dir / "ml_dataset_continuous.csv"
    labeled_df.to_csv(dest, index=False)
    print(f"✅ Cached continuous labels to {dest} ({len(labeled_df)} rows)")

if __name__ == "__main__":
    main()
