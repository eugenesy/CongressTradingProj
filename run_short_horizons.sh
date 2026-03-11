#!/bin/bash

# Extract the start year dynamically from the processed dataset
START_YEAR=$(python -c "import pandas as pd; print(pd.to_datetime(pd.read_csv('data/processed/ml_dataset_clean.csv')['Filed']).dt.year.min())")
END_YEAR=2025

# Define the horizons to run
HORIZONS=("1W" "2W" "1M" "2M" "3M" "4M")

# Ensure the script stops if any command fails
set -e

for HORIZON in "${HORIZONS[@]}"; do
    echo "====================================================="
    echo "Starting training for horizon: $HORIZON ($START_YEAR to $END_YEAR)"
    echo "====================================================="
    
    python scripts/train_gap_tgn.py \
        --horizon "$HORIZON" \
        --start-year "$START_YEAR" \
        --end-year "$END_YEAR" \
        --out-dir "results/experiments/horizon_${HORIZON}"
done

echo "All horizons completed successfully."