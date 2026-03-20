# Clean TGN: Congressional Trading Analysis

This project implements a Temporal Graph Network (TGN) pipeline to analyze and predict the impact of congressional stock trades. The system is designed for high-fidelity financial modeling, focusing on disclosure delays and market context.

## Project Structure

```text
clean_tgn/
├── dataset_package/    # Self-contained data pipeline (Raw -> ML-Ready)
│   ├── scripts/        # Preprocessing & Sequence generation
│   └── data/           # Symlinks to huge parquet files + processed CSVs
├── src/                # TGN Model Architecture & Temporal Logic
│   ├── models_tgn.py   # Core TGN model implementation
│   └── temporal_data.py # Data loading and feature engineering
├── scripts/            # Training and evaluation utilities
│   └── run_baseline_xgb.py # Baseline comparison
└── ARCHITECTURE_EXPLAINED.md # Deep dive into model logic
```

## Key Features

### 1. Informational Anchoring (Filing Date)
Unlike traditional models that use the "Trade Date," this pipeline uses the **Filing Date** as the zero-point for all predictions. This simulates the real-world scenario where the public only learns about a trade weeks after it occurs.

### 2. Feature Engineering
- **Filing Gap**: Captures the delay between trading and disclosure.
- **Price Sequences**: 14-dimensional tensors for every trade, capturing RSI, Volatility, and Returns for both the stock and the S&P 500 benchmark.
- **Historical Win Rate**: Dynamically tracks each congressperson's historical accuracy as a feature for future trades.

### 3. Modular Data Pipeline
The `dataset_package` automates the complex transition from raw transaction lists to PyTorch-ready temporal graphs in 9 reproducible steps.

## Quick Start

### Build the Dataset
```bash
cd dataset_package
python scripts/run_dataset_build.py
```

### run Baseline Model
```bash
python scripts/run_baseline_xgb.py
```

## Verification
The current pipeline produces a cleaned dataset of **~34,266 transactions** with all temporal features verified. "Impossible" disclosure dates have been removed, and feature leakage has been eliminated by strictly partitioning data at the filing timestamp.
