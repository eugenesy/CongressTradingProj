import pandas as pd
import numpy as np

# Import utility functions
from src.utils import load_csv_with_path, save_csv_with_path

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

    nan_count = df['Label_1M'].isna().sum()
    if nan_count > 0:
        print(f"Number of NaN values in Label_1M column: {nan_count}")
        nan_percentage = (nan_count / len(df)) * 100
        print(f"Percentage of NaN values in Label_1M column: {nan_percentage:.2f}%")

        original_count = len(df)
        df_cleaned = df.dropna(subset=['Label_1M'])
        removed_count = original_count - len(df_cleaned)
        print(f"Number of rows removed due to NaN in 'Label_1M': {removed_count}")
        print(f"Percentage of data removed: {(removed_count/original_count)*100:.2f}%")
    else:
        df_cleaned = df.copy()
        print("No NaN values found in Label_1M column.")

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