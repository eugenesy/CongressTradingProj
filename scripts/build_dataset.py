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

    # Input: We start with v5_transactions.csv in data/raw/
    v5_transactions_csv = raw_dir / 'v5_transactions.csv'
    
    # Intermediate and Final Outputs in data/processed/
    all_tickers_pkl = processed_dir / 'all_tickers_historical_data.pkl'
    failed_tickers_txt = processed_dir / 'failed_tickers_report.txt'
    spy_pkl = processed_dir / 'spy_historical_data.pkl'
    spy_parquet = parquet_dir / 'SPY.parquet'
    v5_with_benchmark_csv = processed_dir / 'v5_transactions_with_benchmark.csv'
    v6_transactions_csv = processed_dir / 'v6_transactions.csv'
    closing_price_checkpoint_pkl = processed_dir / 'closing_price_addition_checkpoint.pkl'
    v7_transactions_csv = processed_dir / 'v7_transactions.csv'
    excess_returns_checkpoint_pkl = processed_dir / 'excess_returns_checkpoint.pkl'
    v8_transactions_csv = processed_dir / 'v8_transactions.csv'
    monthly_label_checkpoint_pkl = processed_dir / 'monthly_label_checkpoint.pkl'
    v8_cleaned_csv = processed_dir / 'v8_transactions_cleaned.csv'
    v8_final_cleaned_csv = processed_dir / 'v8_transactions_final_cleaned.csv'
    v9_transactions_csv = processed_dir / 'v9_transactions.csv'
    ml_dataset_csv = processed_dir / 'ml_dataset_reduced_attributes.csv'
    price_sequences_pt = SCRIPT_DIR.parent / "data" / "price_sequences.pt"

    # Step 1: Download all tickers historical data
    if all_tickers_pkl.exists():
        print(f"Skipping Step 1: {all_tickers_pkl.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 1: Downloading tickers...")
        download_all_tickers_historical(
            csv_file=str(v5_transactions_csv),
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
            input_csv=str(v5_transactions_csv),
            spy_pkl=str(spy_pkl),
            spy_parquet=str(spy_parquet)
        )

    # Step 3: Add SPY columns
    if v5_with_benchmark_csv.exists():
        print(f"Skipping Step 3: {v5_with_benchmark_csv.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 3: Adding benchmark...")
        add_spy_columns(
            input_csv=str(v5_transactions_csv),
            output_csv=str(v5_with_benchmark_csv),
            spy_pkl=str(spy_pkl)
        )

    # Step 4: Add closing prices
    if v6_transactions_csv.exists():
        print(f"Skipping Step 4: {v6_transactions_csv.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 4: Adding closing prices...")
        add_closing_prices_to_transactions(
            input_csv=str(v5_with_benchmark_csv),
            hist_pkl=str(all_tickers_pkl),
            output_csv=str(v6_transactions_csv),
            checkpoint_file=str(closing_price_checkpoint_pkl)
        )

    # Step 5: Add excess returns
    if v7_transactions_csv.exists():
        print(f"Skipping Step 5: {v7_transactions_csv.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 5: Adding excess returns...")
        add_excess_returns(
            input_csv=str(v6_transactions_csv),
            output_csv=str(v7_transactions_csv),
            checkpoint_file=str(processed_dir / 'excess_returns_checkpoint.pkl')
        )

    # Step 6: Clean data (Relinked from v7 after removing Labeling step)
    if v8_final_cleaned_csv.exists():
        print(f"Skipping Step 6: {v8_final_cleaned_csv.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 6: Cleaning data...")
        clean_transaction_data(
            input_csv=str(v7_transactions_csv),
            output_cleaned_csv=str(v8_cleaned_csv),
            output_final_cleaned_csv=str(v8_final_cleaned_csv)
        )

    # Step 7: Add transaction IDs
    if v9_transactions_csv.exists():
        print(f"Skipping Step 7: {v9_transactions_csv.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 7: Adding Transaction IDs...")
        add_transaction_ids_and_standardize(
            input_csv=str(v8_final_cleaned_csv),
            output_csv=str(v9_transactions_csv)
        )

    # Step 8: Create ML dataset
    if ml_dataset_csv.exists():
        print(f"Skipping Step 8: {ml_dataset_csv.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 8: Creating Final ML Dataset CSV...")
        create_ml_dataset(
            input_file=str(v9_transactions_csv),
            output_file=str(ml_dataset_csv)
        )

    # Step 9: Build Price Sequences
    if price_sequences_pt.exists():
        print(f"Skipping Step 9: {price_sequences_pt.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 9: Building Price Sequences (.pt)...")
        # Run build_prices.py as a subprocess to keep environments separate if needed
        # and ensure it uses the local paths
        try:
            subprocess.run([sys.executable, str(SCRIPT_DIR / "build_prices.py")], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error building price sequences: {e}")
            sys.exit(1)

    print("\n✅ Dataset Package Build completed successfully!")


if __name__ == "__main__":
    main()
