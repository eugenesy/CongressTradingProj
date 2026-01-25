import pandas as pd
import numpy as np

# Import utility functions
from src.financial_pipeline.utils import load_csv_with_path, save_csv_with_path

# Configuration
INPUT_CSV = '../data/v8_transactions.csv'
OUTPUT_CLEANED_CSV = '../data/v8_transactions_cleaned.csv'
OUTPUT_FINAL_CLEANED_CSV = '../data/v8_transactions_final_cleaned.csv'

def clean_transaction_data(
    input_csv=INPUT_CSV,
    output_cleaned_csv=OUTPUT_CLEANED_CSV,
    output_final_cleaned_csv=OUTPUT_FINAL_CLEANED_CSV
):
    print("Loading the CSV file...")
    df = load_csv_with_path(input_csv)

    initial_count = len(df)
    df = df.drop_duplicates()
    removed_count = initial_count - len(df)
    print(f"Number of duplicate rows removed: {removed_count}")

    # Note: NaN filtering moved to model side per user request. 
    # We keep all rows here to maintain a complete record.
    df_cleaned = df.copy()
    nan_count = df_cleaned['Excess_Return_1M'].isna().sum()
    if nan_count > 0:
        print(f"Dataset contains {nan_count} rows with NaN Excess_Return_1M (will be handled by model).")
    else:
        print("No NaN values found in Excess_Return_1M column.")

    columns_to_remove = ['Subholding', 'Description', 'District', 'Short']
    df_cleaned = df_cleaned.drop(columns=columns_to_remove, errors='ignore')
    print("Removed columns:", columns_to_remove)

    fill_columns = ['Committees', 'Industry', 'Sector']
    for column in fill_columns:
        if column in df_cleaned.columns:
            blanks_count = df_cleaned[column].isna().sum()
            if blanks_count > 0:
                print(f"Replacing {blanks_count} blank values in '{column}' with 'Unknown'")
                df_cleaned[column] = df_cleaned[column].fillna('Unknown')

    original_count = len(df_cleaned)
    df_cleaned = df_cleaned[df_cleaned['Transaction'] != 'Exchange']
    rows_removed = original_count - len(df_cleaned)
    print(f"Removed {rows_removed} rows with 'Exchange' value in Transaction column")

    df_cleaned['Transaction'] = df_cleaned['Transaction'].replace({'Sale (Full)': 'Sale', 'Sale (Partial)': 'Sale'})

    save_csv_with_path(df_cleaned, output_cleaned_csv, index=False)
    print(f"File saved successfully as '{output_cleaned_csv}'")

    df_cleaned['Party'] = df_cleaned['Party'].replace({'R': 'Republican', 'I': 'Independent', 'D': 'Democrat'})

    save_csv_with_path(df_cleaned, output_final_cleaned_csv, index=False)
    print(f"Successfully saved final cleaned dataframe to {output_final_cleaned_csv}")
    return df_cleaned