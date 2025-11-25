"""
Entry point script to run the data processing pipeline.
"""

import sys
from pathlib import Path
# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.download_tickers import download_all_tickers_historical
from src.data_pipeline.download_spy import download_spy_historical
from src.data_pipeline.add_spy_columns import add_spy_columns
from src.data_pipeline.add_closing_prices import add_closing_prices_to_transactions
from src.data_pipeline.add_excess_returns import add_excess_returns
from src.data_pipeline.add_trading_labels import add_trading_labels
from src.data_pipeline.clean_data import clean_transaction_data
from src.data_pipeline.add_transaction_ids import add_transaction_ids_and_standardize
from src.utils import get_data_path
import os


def main():
    """Main function to orchestrate the data processing workflow."""
    print("Starting financial data processing workflow...")

    # Define data paths using the new structure
    v5_transactions_csv = get_data_path('processed', 'v5_transactions_with_approp_ticker.csv')
    all_tickers_pkl = get_data_path('processed', 'all_tickers_historical_data.pkl')
    failed_tickers_txt = get_data_path('processed', 'failed_tickers_report.txt')
    spy_pkl = get_data_path('processed', 'spy_historical_data.pkl')
    spy_parquet = get_data_path('parquet', 'SPY.parquet')
    v5_with_benchmark_csv = get_data_path('processed', 'v5_transactions_with_benchmark.csv')
    v6_transactions_csv = get_data_path('processed', 'v6_transactions.csv')
    closing_price_checkpoint_pkl = get_data_path('processed', 'closing_price_addition_checkpoint.pkl')
    v7_transactions_csv = get_data_path('processed', 'v7_transactions.csv')
    excess_returns_checkpoint_pkl = get_data_path('processed', 'excess_returns_checkpoint.pkl')
    v8_transactions_csv = get_data_path('processed', 'v8_transactions.csv')
    monthly_label_checkpoint_pkl = get_data_path('processed', 'monthly_label_checkpoint.pkl')
    v8_cleaned_csv = get_data_path('processed', 'v8_transactions_cleaned.csv')
    v8_final_cleaned_csv = get_data_path('processed', 'v8_transactions_final_cleaned.csv')
    v9_transactions_csv = get_data_path('processed', 'v9_transactions.csv')

    # Step 1: Download all tickers historical data
    if all_tickers_pkl.exists():
        print(f"Skipping Step 1: {all_tickers_pkl} already exists.")
    else:
        download_all_tickers_historical(
            csv_file=str(v5_transactions_csv),
            out_pkl=str(all_tickers_pkl),
            failed_report=str(failed_tickers_txt),
            parquet_dir=str(get_data_path('parquet'))
        )

    # Step 2: Download SPY historical data
    if spy_pkl.exists():
        print(f"Skipping Step 2: {spy_pkl} already exists.")
    else:
        download_spy_historical(
            input_csv=str(v5_transactions_csv),
            spy_pkl=str(spy_pkl),
            spy_parquet=str(spy_parquet)
        )

    # Step 3: Add SPY columns
    if v5_with_benchmark_csv.exists():
        print(f"Skipping Step 3: {v5_with_benchmark_csv} already exists.")
    else:
        add_spy_columns(
            input_csv=str(v5_transactions_csv),
            output_csv=str(v5_with_benchmark_csv),
            spy_pkl=str(spy_pkl)
        )

    # Step 4: Add closing prices
    if v6_transactions_csv.exists():
        print(f"Skipping Step 4: {v6_transactions_csv} already exists.")
    else:
        add_closing_prices_to_transactions(
            input_csv=str(v5_with_benchmark_csv),
            hist_pkl=str(all_tickers_pkl),
            output_csv=str(v6_transactions_csv),
            checkpoint_file=str(closing_price_checkpoint_pkl)
        )

    # Step 5: Add excess returns
    if v7_transactions_csv.exists():
        print(f"Skipping Step 5: {v7_transactions_csv} already exists.")
    else:
        add_excess_returns(
            input_csv=str(v6_transactions_csv),
            output_csv=str(v7_transactions_csv),
            checkpoint_file=str(excess_returns_checkpoint_pkl)
        )

    # Step 6: Add trading labels
    if v8_transactions_csv.exists():
        print(f"Skipping Step 6: {v8_transactions_csv} already exists.")
    else:
        add_trading_labels(
            input_csv=str(v7_transactions_csv),
            output_csv=str(v8_transactions_csv),
            checkpoint_file=str(monthly_label_checkpoint_pkl)
        )

    # Step 7: Clean data
    if v8_final_cleaned_csv.exists():
        print(f"Skipping Step 7: {v8_final_cleaned_csv} already exists.")
    else:
        clean_transaction_data(
            input_csv=str(v8_transactions_csv),
            output_cleaned_csv=str(v8_cleaned_csv),
            output_final_cleaned_csv=str(v8_final_cleaned_csv)
        )

    # Step 8: Add transaction IDs
    if v9_transactions_csv.exists():
        print(f"Skipping Step 8: {v9_transactions_csv} already exists.")
    else:
        add_transaction_ids_and_standardize(
            input_csv=str(v8_final_cleaned_csv),
            output_csv=str(v9_transactions_csv)
        )

    print("\n✅ Financial data processing workflow completed successfully!")


if __name__ == "__main__":
    main()
