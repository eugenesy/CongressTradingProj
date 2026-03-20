# TGN Dataset Package

This package contains the modular financial feature pipeline and market context (price) sequence generator for the TGN model. It is designed to be self-contained and reproducible.

## Directory Structure
- `scripts/`: Modular preprocessing logic.
    - `run_dataset_build.py`: **Master script** to run the entire pipeline.
    - `financial_pipeline/`: Step-by-step modular scripts for calculating returns and cleaning data.
    - `build_prices.py`: Generates the 14-dim price context sequences.
## Prerequisites & File Manifest

The pipeline assumes a self-contained environment in the `data/` directory. If you are setting this up for the first time, ensure the following files/links are present:

### 1. Source Data (Required to Start)
- **`data/raw/v5_transactions.csv`**: The raw list of congressional trades.
- **`data/parquet/`**: A directory containing historical stock prices in Parquet format (e.g., `AAPL.parquet`, `TSLA.parquet`, `SPY.parquet`).
    - *Tip: These are usually symlinked from your global data store to save space.*

### 2. Generated Assets (Created by Pipeline)
- **`data/processed/all_tickers_historical_data.pkl`**: Cached dictionary of all stock prices.
- **`data/processed/spy_historical_data.pkl`**: Cached benchmark data.
- **`data/processed/ml_dataset_reduced_attributes.csv`**: The final sanitized transaction list for ML.
- **`data/price_sequences.pt`**: The 14-dim context tensors for the TGN model.

### 3. Intermediate Workfiles
- **`data/processed/v6_transactions.csv`** to **`v9_transactions.csv`**: Incremental build files representing different stages (benchmarking, returns, cleaning).
- **`data/processed/*checkpoint.pkl`**: Resume-points for long-running download/calculation tasks.

## How to Run
To rebuild the entire dataset from raw transactions:
```bash
python scripts/run_dataset_build.py
```

## Data Transformation Pipeline
The `run_dataset_build.py` script orchestrates the following 9 steps:

1.  **Download Tickers**: Loads historical data for all relevant stock tickers.
2.  **Download SPY**: Prepares the S&P 500 benchmark data.
3.  **Add Benchmark**: Merges benchmark prices into the transaction records.
4.  **Add Closing Prices**: Retrieves historical closes for the trade date and future horizons (1M to 24M).
5.  **Calculate Excess Returns**: Computes stock performance relative to SPY for all horizons.
6.  **Clean Data**: Sanitizes the dataset (removes duplicates, normalizes strings, handles missing categorical values).
7.  **Standardization & Gaps**: 
    - Generates unique Transaction IDs.
    - Standardizes trade size ranges.
    - **Filing Gap**: Calculates the delay between `Traded` and `Filed` dates (clamped to $\ge 0$).
8.  **Create ML Dataset**: Exports a filtered CSV (`ml_dataset_reduced_attributes.csv`) with purely predictive features.
9.  **Build Price Sequences**: Generates a `.pt` file containing 14-dimensional context tensors (7 Stock features + 7 SPY features) for every trade, anchored to the **Filing Date**.

## Key Design Principles
- **Filing Date Basis**: All predictions and context are anchored to the date the trade was disclosed to the public.
- **No Data Leakage**: Price features are computed using data strictly *before* the filing date.
- **NaN Preservation**: Transactions missing price data are preserved in the CSV to allow the model to handle them dynamically.
- **Impossible Date Filtering**: Transactions with negative filing gaps (disclosure before trade) are automatically removed.
