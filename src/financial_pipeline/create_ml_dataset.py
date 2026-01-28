import pandas as pd
from src.financial_pipeline.utils import get_data_path

# Config - Input: v6, Output: final
INPUT_CSV = get_data_path('processed', 'ml_dataset_v6.csv')
OUTPUT_CSV = get_data_path('processed', 'ml_dataset_final.csv')

def create_ml_dataset(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        
        # Columns to keep (Dynamic check)
        target_cols = [
            'transaction_id', 'BioGuideID', 'Chamber', 'Filed', 'Party', 'State', 
            'Ticker', 'TickerType', 'Trade_Size_USD', 'Traded', 'Transaction', 'Filing_Gap'
        ]
        # Add return columns
        for h in ['1M','2M','3M','6M','8M','12M','18M','24M']:
            target_cols.append(f'Excess_Return_{h}')
            
        # Filter existing
        existing_cols = [c for c in target_cols if c in df.columns]
        ml_df = df[existing_cols].copy()
        
        ml_df.to_csv(output_file, index=False)
        print(f"Final ML dataset created at {output_file}")
        print(f"Columns: {ml_df.columns.tolist()}")
        
    except Exception as e:
        print(f"Error creating ML dataset: {e}")

if __name__ == "__main__":
    create_ml_dataset(str(INPUT_CSV), str(OUTPUT_CSV))