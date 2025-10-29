import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.data_processing.download_tickers import download_all_tickers_historical
from scripts.data_processing.download_spy import download_spy_historical
from scripts.data_processing.add_spy_columns import add_spy_columns
from scripts.data_processing.add_closing_prices import add_closing_prices_to_transactions
from scripts.data_processing.add_excess_returns import add_excess_returns
from scripts.data_processing.add_trading_labels import add_trading_labels
from scripts.data_processing.clean_data import clean_transaction_data
from scripts.data_processing.add_transaction_ids import add_transaction_ids_and_standardize
from scripts.data_processing.embedding_enhanced import enhance_embeddings
from scripts.data_exploration.explore_v9_data import explore_v9_data
from scripts.comparison.compare_data import compare_data

def main():
    print("Starting financial data processing workflow...")

    # Define common data paths
    base_data_path = '../data/'
    v5_transactions_with_approp_ticker_csv = os.path.join(base_data_path, 'v5_transactions_with_approp_ticker.csv')
    all_tickers_historical_data_pkl = os.path.join(base_data_path, 'all_tickers_historical_data.pkl')
    failed_tickers_report_txt = os.path.join(base_data_path, 'failed_tickers_report.txt')
    spy_historical_data_pkl = os.path.join(base_data_path, 'spy_historical_data.pkl')
    spy_parquet_path = '/data1/user_syeugene/fintech/apple/data/parquet_files/SPY.parquet'
    v5_transactions_with_benchmark_csv = os.path.join(base_data_path, 'v5_transactions_with_benchmark.csv')
    v6_transactions_csv = os.path.join(base_data_path, 'v6_transactions.csv')
    closing_price_addition_checkpoint_pkl = os.path.join(base_data_path, 'closing_price_addition_checkpoint.pkl')
    v7_transactions_csv = os.path.join(base_data_path, 'v7_transactions.csv')
    excess_returns_checkpoint_pkl = os.path.join(base_data_path, 'excess_returns_checkpoint.pkl')
    v8_transactions_csv = os.path.join(base_data_path, 'v8_transactions.csv')
    monthly_label_checkpoint_pkl = os.path.join(base_data_path, 'monthly_label_checkpoint.pkl')
    v8_transactions_cleaned_csv = os.path.join(base_data_path, 'v8_transactions_cleaned.csv')
    v8_transactions_final_cleaned_csv = os.path.join(base_data_path, 'v8_transactions_final_cleaned.csv')
    v9_transactions_csv = os.path.join(base_data_path, 'v9_transactions.csv')
    transactions_with_embeddings_enhanced_csv = os.path.join(base_data_path, 'transactions_with_embeddings_enhanced.csv')
    comparison_report_file = os.path.join(base_data_path, 'comparison_results/comparison_report.md')

    # Step 1: Download all tickers historical data
    if os.path.exists(all_tickers_historical_data_pkl):
        print(f"Skipping Step 1: {all_tickers_historical_data_pkl} already exists.")
    else:
        download_all_tickers_historical(
            csv_file=v5_transactions_with_approp_ticker_csv,
            out_pkl=all_tickers_historical_data_pkl,
            failed_report=failed_tickers_report_txt,
            parquet_dir='/data1/user_syeugene/fintech/apple/data/parquet_files'
        )

    # Step 2: Download SPY historical data
    if os.path.exists(spy_historical_data_pkl):
        print(f"Skipping Step 2: {spy_historical_data_pkl} already exists.")
    else:
        download_spy_historical(
            input_csv=v5_transactions_with_approp_ticker_csv,
            spy_pkl=spy_historical_data_pkl,
            spy_parquet=spy_parquet_path
        )

    # Step 3: Add SPY columns
    if os.path.exists(v5_transactions_with_benchmark_csv):
        print(f"Skipping Step 3: {v5_transactions_with_benchmark_csv} already exists.")
    else:
        add_spy_columns(
            input_csv=v5_transactions_with_approp_ticker_csv,
            output_csv=v5_transactions_with_benchmark_csv,
            spy_pkl=spy_historical_data_pkl
        )

    # Step 4: Add closing prices to transactions
    if os.path.exists(v6_transactions_csv):
        print(f"Skipping Step 4: {v6_transactions_csv} already exists.")
    else:
        add_closing_prices_to_transactions(
            input_csv=v5_transactions_with_benchmark_csv,
            hist_pkl=all_tickers_historical_data_pkl,
            output_csv=v6_transactions_csv,
            checkpoint_file=closing_price_addition_checkpoint_pkl
        )

    # Step 5: Add excess returns
    if os.path.exists(v7_transactions_csv):
        print(f"Skipping Step 5: {v7_transactions_csv} already exists.")
    else:
        add_excess_returns(
            input_csv=v6_transactions_csv,
            output_csv=v7_transactions_csv,
            checkpoint_file=excess_returns_checkpoint_pkl
        )

    # Step 6: Add trading labels
    if os.path.exists(v8_transactions_csv):
        print(f"Skipping Step 6: {v8_transactions_csv} already exists.")
    else:
        add_trading_labels(
            input_csv=v7_transactions_csv,
            output_csv=v8_transactions_csv,
            checkpoint_file=monthly_label_checkpoint_pkl
        )

    # Step 7: Clean data
    if os.path.exists(v8_transactions_final_cleaned_csv):
        print(f"Skipping Step 7: {v8_transactions_final_cleaned_csv} already exists.")
    else:
        clean_transaction_data(
            input_csv=v8_transactions_csv,
            output_cleaned_csv=v8_transactions_cleaned_csv,
            output_final_cleaned_csv=v8_transactions_final_cleaned_csv
        )

    # Step 8: Add transaction IDs and standardize
    if os.path.exists(v9_transactions_csv):
        print(f"Skipping Step 8: {v9_transactions_csv} already exists.")
    else:
        add_transaction_ids_and_standardize(
            input_csv=v8_transactions_final_cleaned_csv,
            output_csv=v9_transactions_csv
        )

    # Step 9: Data Exploration
    exploration_results_dir = os.path.join(base_data_path, 'exploration_results/')
    if os.path.exists(exploration_results_dir) and len(os.listdir(exploration_results_dir)) > 0:
        print(f"Skipping Step 9: Data exploration results already exist in {exploration_results_dir}.")
    else:
        explore_v9_data(
            input_csv=v9_transactions_csv,
            output_dir=exploration_results_dir
        )

    # Step 10: Enhance embeddings
    if os.path.exists(transactions_with_embeddings_enhanced_csv):
        print(f"Skipping Step 10: {transactions_with_embeddings_enhanced_csv} already exists.")
    else:
        enhance_embeddings(
            input_csv=v9_transactions_csv,
            output_csv=transactions_with_embeddings_enhanced_csv
        )

    # Step 11: Compare initial and final data
    if os.path.exists(comparison_report_file):
        print(f"Skipping Step 11: {comparison_report_file} already exists.")
    else:
        compare_data(
            initial_csv=v5_transactions_with_approp_ticker_csv,
            final_csv=v9_transactions_csv,
            output_dir=os.path.join(base_data_path, 'comparison_results/'),
            report_file=comparison_report_file
        )

    print("Financial data processing workflow completed successfully.")
