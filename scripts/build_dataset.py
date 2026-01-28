"""
Entry point script to run the data processing pipeline.
"""

import sys
import os
from pathlib import Path
import subprocess

# Add current script directory to path so we can import from financial_pipeline
SCRIPT_DIR = Path(__file__).parent

from src.financial_pipeline.download_tickers import download_all_tickers_historical
from src.financial_pipeline.download_spy import download_spy_historical
from src.financial_pipeline.add_spy_columns import add_spy_columns
from src.financial_pipeline.add_closing_prices import add_closing_prices_to_transactions
from src.financial_pipeline.add_excess_returns import add_excess_returns
from src.financial_pipeline.clean_data import clean_transaction_data
from src.financial_pipeline.add_transaction_ids import add_transaction_ids_and_standardize
from src.financial_pipeline.create_ml_dataset import create_ml_dataset


def get_local_path(folder, filename=None):
    """Helper to get paths within the dataset_package/data directory."""
    base = SCRIPT_DIR.parent / "data" / folder
    if filename:
        return base / filename
    return base


def main():
    """Main function to orchestrate the data processing workflow."""
    print("🚀 Starting Dataset Build Pipeline (Packaged Version)...")

    # Define data paths using the self-contained package structure
    raw_dir = get_local_path('raw')
    processed_dir = get_local_path('processed')
    parquet_dir = get_local_path('parquet')

    # --- Pipeline File Sequence ---
    # Input: Standardize on ml_dataset_clean.csv (provided by user in raw)
    input_csv = raw_dir / 'ml_dataset_clean.csv'
    
    # Intermediate steps
    ml_v2_csv = processed_dir / 'ml_dataset_v2.csv'  # With SPY columns
    ml_v3_csv = processed_dir / 'ml_dataset_v3.csv'  # With Closing Prices
    ml_v4_csv = processed_dir / 'ml_dataset_v4.csv'  # With Excess Returns
    ml_v5_csv = processed_dir / 'ml_dataset_v5.csv'  # Cleaned (duplicates removed)
    ml_v6_csv = processed_dir / 'ml_dataset_v6.csv'  # With IDs and standardized sizes
    
    # Final Output
    final_ml_csv = processed_dir / 'ml_dataset_final.csv'

    # Artifacts
    all_tickers_pkl = processed_dir / 'all_tickers_historical_data.pkl'
    failed_tickers_txt = processed_dir / 'failed_tickers_report.txt'
    spy_pkl = processed_dir / 'spy_historical_data.pkl'
    spy_parquet = parquet_dir / 'SPY.parquet'
    
    # Checkpoints
    closing_price_checkpoint_pkl = processed_dir / 'closing_price_checkpoint.pkl'
    excess_returns_checkpoint_pkl = processed_dir / 'excess_returns_checkpoint.pkl'
    
    price_sequences_pt = SCRIPT_DIR.parent / "data" / "price_sequences.pt"

    # Step 1: Download all tickers historical data
    if all_tickers_pkl.exists():
        print(f"Skipping Step 1: {all_tickers_pkl.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 1: Downloading tickers...")
        download_all_tickers_historical(
            csv_file=str(input_csv),
            out_pkl=str(all_tickers_pkl),
            failed_report=str(failed_tickers_txt),
            parquet_dir=str(parquet_dir)
        )

    # Step 2: Download SPY historical data
    if spy_pkl.exists():
        print(f"Skipping Step 2: {spy_pkl.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 2: Downloading SPY...")
        download_spy_historical(
            input_csv=str(input_csv),
            spy_pkl=str(spy_pkl),
            spy_parquet=str(spy_parquet)
        )

    # Step 3: Add SPY columns (Input -> V2)
    if ml_v2_csv.exists():
        print(f"Skipping Step 3: {ml_v2_csv.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 3: Adding benchmark columns...")
        add_spy_columns(
            input_csv=str(input_csv),
            output_csv=str(ml_v2_csv),
            spy_pkl=str(spy_pkl)
        )

    # Step 4: Add closing prices (V2 -> V3)
    if ml_v3_csv.exists():
        print(f"Skipping Step 4: {ml_v3_csv.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 4: Adding closing prices...")
        add_closing_prices_to_transactions(
            input_csv=str(ml_v2_csv),
            hist_pkl=str(all_tickers_pkl),
            output_csv=str(ml_v3_csv),
            checkpoint_file=str(closing_price_checkpoint_pkl)
        )

    # Step 5: Add excess returns (V3 -> V4)
    if ml_v4_csv.exists():
        print(f"Skipping Step 5: {ml_v4_csv.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 5: Adding excess returns...")
        add_excess_returns(
            input_csv=str(ml_v3_csv),
            output_csv=str(ml_v4_csv),
            checkpoint_file=str(excess_returns_checkpoint_pkl)
        )

    # Step 6: Clean data (V4 -> V5)
    if ml_v5_csv.exists():
        print(f"Skipping Step 6: {ml_v5_csv.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 6: Cleaning data...")
        clean_transaction_data(
            input_csv=str(ml_v4_csv),
            output_csv=str(ml_v5_csv)
        )

    # Step 7: Add transaction IDs (V5 -> V6)
    if ml_v6_csv.exists():
        print(f"Skipping Step 7: {ml_v6_csv.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 7: Adding Transaction IDs...")
        add_transaction_ids_and_standardize(
            input_csv=str(ml_v5_csv),
            output_csv=str(ml_v6_csv)
        )

    # Step 8: Create ML dataset (V6 -> Final)
    if final_ml_csv.exists():
        print(f"Skipping Step 8: {final_ml_csv.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 8: Finalizing ML Dataset columns...")
        create_ml_dataset(
            input_file=str(ml_v6_csv),
            output_file=str(final_ml_csv)
        )

    # Step 9: Build Price Sequences
    if price_sequences_pt.exists():
        print(f"Skipping Step 9: {price_sequences_pt.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 9: Building Price Sequences (.pt)...")
        try:
            # Note: We pass the *final* CSV to build_prices if it needs it, 
            # though build_prices.py might look for 'ml_dataset_clean.csv' by default.
            # We should ensure build_prices.py uses the right file or we rely on config.
            subprocess.run([sys.executable, str(SCRIPT_DIR / "build_prices.py")], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error building price sequences: {e}")
            sys.exit(1)

    print("\n✅ Dataset Package Build completed successfully!")


if __name__ == "__main__":
    main()