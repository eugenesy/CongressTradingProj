"""
Entry point script to run the data processing pipeline.
"""

import sys
import os
from pathlib import Path
import subprocess

# Add current script directory to path so we can import from financial_pipeline
SCRIPT_DIR = Path(__file__).parent

# Note: Download scripts removed as we use provided pickle
from src.financial_pipeline.add_spy_columns import add_spy_columns
from src.financial_pipeline.add_closing_prices import add_closing_prices_to_transactions
from src.financial_pipeline.add_excess_returns import add_excess_returns
from src.financial_pipeline.clean_data import clean_transaction_data
from src.financial_pipeline.add_transaction_ids import add_transaction_ids_and_standardize
from src.financial_pipeline.create_ml_dataset import create_ml_dataset
from src.financial_pipeline.add_trading_labels import add_trading_labels


def get_local_path(folder, filename=None):
    """Helper to get paths within the dataset_package/data directory."""
    base = SCRIPT_DIR.parent / "data" / folder
    if filename:
        return base / filename
    return base


def main():
    """Main function to orchestrate the data processing workflow."""
    print("🚀 Starting Dataset Build Pipeline (Pickle-Only Version)...")

    # Define data paths
    raw_dir = get_local_path('raw')
    processed_dir = get_local_path('processed')

    # Input Files
    input_csv = raw_dir / 'ml_dataset_clean.csv'
    # UPDATED: Source of truth for all price data
    all_tickers_pkl = raw_dir / 'all_tickers_historical_data.pkl'

    # Check for requirements
    if not all_tickers_pkl.exists():
        print(f"❌ Error: Price data file not found at {all_tickers_pkl}")
        print("Please place 'all_tickers_historical_data.pkl' in data/raw/")
        sys.exit(1)

    # Intermediate Pipeline Files
    ml_v2_csv = processed_dir / 'ml_dataset_v2.csv'  # With SPY columns
    ml_v3_csv = processed_dir / 'ml_dataset_v3.csv'  # With Closing Prices
    ml_v4_csv = processed_dir / 'ml_dataset_v4.csv'  # With Excess Returns
    ml_v4_labeled_csv = processed_dir / 'ml_dataset_v4_labeled.csv' # With Labels
    ml_v5_csv = processed_dir / 'ml_dataset_v5.csv'  # Cleaned
    ml_v6_csv = processed_dir / 'ml_dataset_v6.csv'  # With IDs
    
    # Final Output
    final_ml_csv = processed_dir / 'ml_dataset_final.csv'

    # Checkpoints
    closing_price_checkpoint_pkl = processed_dir / 'closing_price_checkpoint.pkl'
    excess_returns_checkpoint_pkl = processed_dir / 'excess_returns_checkpoint.pkl'
    label_checkpoint_pkl = processed_dir / 'monthly_label_checkpoint.pkl'
    
    price_sequences_pt = SCRIPT_DIR.parent / "data" / "price_sequences.pt"

    # Step 1: Add SPY columns (Input -> V2)
    # Note: We now pull SPY directly from the main tickers pickle
    if ml_v2_csv.exists():
        print(f"Skipping Step 1: {ml_v2_csv.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 1: Adding benchmark (SPY) columns...")
        add_spy_columns(
            input_csv=str(input_csv),
            output_csv=str(ml_v2_csv),
            hist_pkl=str(all_tickers_pkl) # Pass the big pickle
        )

    # Step 2: Add closing prices (V2 -> V3)
    if ml_v3_csv.exists():
        print(f"Skipping Step 2: {ml_v3_csv.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 2: Adding closing prices...")
        add_closing_prices_to_transactions(
            input_csv=str(ml_v2_csv),
            hist_pkl=str(all_tickers_pkl),
            output_csv=str(ml_v3_csv),
            checkpoint_file=str(closing_price_checkpoint_pkl)
        )

    # Step 3: Add excess returns (V3 -> V4)
    if ml_v4_csv.exists():
        print(f"Skipping Step 3: {ml_v4_csv.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 3: Adding excess returns...")
        add_excess_returns(
            input_csv=str(ml_v3_csv),
            output_csv=str(ml_v4_csv),
            checkpoint_file=str(excess_returns_checkpoint_pkl)
        )
    
    # Step 3.5: Add Training Labels (V4 -> V4_Labeled)
    # Useful for analysis, though GAP-TGN often generates labels dynamically
    if ml_v4_labeled_csv.exists():
        print(f"Skipping Step 3.5: {ml_v4_labeled_csv.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 3.5: Generating classification labels...")
        add_trading_labels(
            input_csv=str(ml_v4_csv),
            output_csv=str(ml_v4_labeled_csv),
            checkpoint_file=str(label_checkpoint_pkl)
        )

    # Step 4: Clean data (V4 -> V5)
    if ml_v5_csv.exists():
        print(f"Skipping Step 4: {ml_v5_csv.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 4: Cleaning data...")
        # Note: We continue from v4 (returns) to v5 (cleaned). 
        # v4_labeled is a side artifact.
        clean_transaction_data(
            input_csv=str(ml_v4_csv),
            output_csv=str(ml_v5_csv)
        )

    # Step 5: Add transaction IDs (V5 -> V6)
    if ml_v6_csv.exists():
        print(f"Skipping Step 5: {ml_v6_csv.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 5: Adding Transaction IDs...")
        add_transaction_ids_and_standardize(
            input_csv=str(ml_v5_csv),
            output_csv=str(ml_v6_csv)
        )

    # Step 6: Create ML dataset (V6 -> Final)
    if final_ml_csv.exists():
        print(f"Skipping Step 6: {final_ml_csv.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 6: Finalizing ML Dataset columns...")
        create_ml_dataset(
            input_file=str(ml_v6_csv),
            output_file=str(final_ml_csv)
        )

    # Step 7: Build Price Sequences
    if price_sequences_pt.exists():
        print(f"Skipping Step 7: {price_sequences_pt.relative_to(SCRIPT_DIR.parent)} already exists.")
    else:
        print("Step 7: Building Price Sequences (.pt)...")
        try:
            # We must ensure build_prices.py reads from the correct raw pickle
            # It usually imports config.PRICE_PATH, so it should be fine.
            subprocess.run([sys.executable, str(SCRIPT_DIR / "build_prices.py")], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error building price sequences: {e}")
            sys.exit(1)

    print("\n✅ Dataset Package Build completed successfully!")


if __name__ == "__main__":
    main()