import pandas as pd
import numpy as np
from src.financial_pipeline.utils import get_data_path

# Config - Input: v5, Output: v6
INPUT_CSV = get_data_path('processed', 'ml_dataset_v5.csv')
OUTPUT_CSV = get_data_path('processed', 'ml_dataset_v6.csv')

def add_transaction_ids_and_standardize(
    input_csv=INPUT_CSV,
    output_csv=OUTPUT_CSV
):
    print("Loading data for IDs...")
    df = pd.read_csv(input_csv)

    # Add IDs
    id_count = len(df)
    id_digits = len(str(id_count + 10000))
    df['transaction_id'] = [i for i in range(1, id_count + 1)] # Integer ID usually better for ML
    # Optional: Keep 'ID' string format if legacy needed, but 'transaction_id' is preferred
    
    print("Calculating filing gaps...")
    df['Filed_DT'] = pd.to_datetime(df['Filed'], errors='coerce')
    df['Traded_DT'] = pd.to_datetime(df['Traded'], errors='coerce')
    df['Filing_Gap'] = (df['Filed_DT'] - df['Traded_DT']).dt.days
    
    # Filter negative gaps
    df = df[df['Filing_Gap'] >= 0].copy()
    
    # Cleanup
    df = df.drop(columns=['Filed_DT', 'Traded_DT'])
    
    df.to_csv(output_csv, index=False)
    print(f"IDs added. Saved to {output_csv}")
    return df