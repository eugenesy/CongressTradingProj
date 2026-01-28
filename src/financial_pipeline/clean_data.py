import pandas as pd
from src.financial_pipeline.utils import get_data_path

# Config - Input: v4, Output: v5
INPUT_CSV = get_data_path('processed', 'ml_dataset_v4.csv')
OUTPUT_CSV = get_data_path('processed', 'ml_dataset_v5.csv')

def clean_transaction_data(
    input_csv=INPUT_CSV,
    output_csv=OUTPUT_CSV
):
    print("Loading data for cleaning...")
    df = pd.read_csv(input_csv)

    initial_count = len(df)
    df = df.drop_duplicates()
    print(f"Duplicates removed: {initial_count - len(df)}")

    cols_to_remove = ['Subholding', 'Description', 'District', 'Short']
    df = df.drop(columns=cols_to_remove, errors='ignore')

    fill_cols = ['Committees', 'Industry', 'Sector']
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    if 'Transaction' in df.columns:
        df = df[df['Transaction'] != 'Exchange']
        df['Transaction'] = df['Transaction'].replace({'Sale (Full)': 'Sale', 'Sale (Partial)': 'Sale'})
        
    if 'Party' in df.columns:
        df['Party'] = df['Party'].replace({'R': 'Republican', 'I': 'Independent', 'D': 'Democrat'})

    df.to_csv(output_csv, index=False)
    print(f"Cleaned data saved to {output_csv}")
    return df