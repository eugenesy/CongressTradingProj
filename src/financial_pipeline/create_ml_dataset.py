
import pandas as pd
from src.financial_pipeline.utils import get_data_path

def create_ml_dataset(input_file, output_file):
    """
    Creates a new CSV file with selected columns for machine learning.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
    """
    try:
        # Read the input CSV file
        df = pd.read_csv(input_file)

        # Define the columns to keep
        matching_columns = [
            'ID', 'BioGuideID', 'Chamber', 'Filed', 'Party', 'State', 'Ticker',
            'TickerType', 'Trade_Size_USD', 'Traded', 'Transaction', 'Filing_Gap'
        ]
        additional_columns = [
            'Excess_Return_1M', 'Excess_Return_2M', 'Excess_Return_3M', 
            'Excess_Return_6M', 'Excess_Return_8M', 'Excess_Return_12M', 
            'Excess_Return_18M', 'Excess_Return_24M'
        ]
        selected_columns = matching_columns + additional_columns

        # Create a new DataFrame with the selected columns
        ml_df = df[selected_columns].copy()
        
        # Rename ID to transaction_id
        ml_df.rename(columns={'ID': 'transaction_id'}, inplace=True)

        # Save the new DataFrame to a CSV file
        ml_df.to_csv(output_file, index=False)

        print(f"Successfully created '{output_file}' with the following columns:")
        print(ml_df.columns.tolist())

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except KeyError as e:
        print(f"Error: A column was not found - {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Use the new path structure
    input_csv = get_data_path("processed", "v9_transactions.csv")
    output_csv = get_data_path("processed", "ml_dataset_reduced_attributes.csv")

    # Create the new dataset
    create_ml_dataset(str(input_csv), str(output_csv))
