import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from src.utils import load_data, get_data_path

def preprocess_data(input_path, output_path):
    """Preprocesses the raw data and saves the result."""
    df = load_data(input_path)

    # Convert date columns
    df["Filed"] = pd.to_datetime(df["Filed"])
    df["Traded"] = pd.to_datetime(df["Traded"])

    # Create filing gap feature
    df["filing_gap_days"] = (df["Filed"] - df["Traded"]).dt.days

    # Feature engineering for Traded date
    df["Traded_year"] = df["Traded"].dt.year
    df["Traded_month"] = df["Traded"].dt.month
    df["Traded_day"] = df["Traded"].dt.day
    df["Traded_dayofweek"] = df["Traded"].dt.dayofweek

    # Keep Filed date components for rolling window
    df["Filed_year"] = df["Filed"].dt.year
    df["Filed_month"] = df["Filed"].dt.month
    df["Filed_day"] = df["Filed"].dt.day

    # Drop original date columns
    df.drop(["Filed", "Traded"], axis=1, inplace=True)

    # Convert Trade_Size_USD to numerical
    def trade_size_to_midpoint(size_range):
        if isinstance(size_range, str):
            size_range = size_range.replace(",", "")
            if " - " in size_range:
                low, high = size_range.split(" - ")
                high = high.strip().rstrip('.')
                return (float(low) + float(high)) / 2
            else:
                size_range = size_range.strip().rstrip('.')
                return float(size_range)
        return size_range

    df["Trade_Size_USD_Midpoint"] = df["Trade_Size_USD"].apply(trade_size_to_midpoint)
    df["Trade_Size_USD_Midpoint"] = np.log1p(df["Trade_Size_USD_Midpoint"])

    # Scale filing_gap_days
    scaler = StandardScaler()
    df["filing_gap_days"] = scaler.fit_transform(df[["filing_gap_days"]])

    df.drop("Trade_Size_USD", axis=1, inplace=True)

    # One-hot encode categorical features, excluding BioGuideID
    categorical_cols = [
        "Chamber", "Party", "State", "Ticker", "TickerType", "Transaction"
    ]
    
    # Preserve transaction_id if it exists
    has_transaction_id = "transaction_id" in df.columns
    if has_transaction_id:
        transaction_id_col = df["transaction_id"].copy()
    
    # Dropping BioGuideID as it has too many unique values for one-hot encoding in a baseline model
    df.drop("BioGuideID", axis=1, inplace=True)

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_cols = pd.DataFrame(
        encoder.fit_transform(df[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index  # Preserve index for proper concatenation
    )
    df = pd.concat([df.drop(categorical_cols, axis=1), encoded_cols], axis=1)
    
    # Restore transaction_id if it was present
    if has_transaction_id:
        df["transaction_id"] = transaction_id_col

    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    input_path = get_data_path("processed", "ml_dataset_reduced_attributes.csv")
    output_path = get_data_path("processed", "ml_dataset_preprocessed.csv")
    preprocess_data(str(input_path), str(output_path))