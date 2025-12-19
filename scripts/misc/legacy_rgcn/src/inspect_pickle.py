import pickle
import pandas as pd

PRICE_PATH = "/data1/user_syeugene/fintech/apple/data/processed/all_tickers_historical_data.pkl"

try:
    with open(PRICE_PATH, 'rb') as f:
        data = pickle.load(f)
        
    print(f"Type of loaded data: {type(data)}")
    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"Number of keys: {len(keys)}")
        if keys:
            first_key = keys[0]
            first_val = data[first_key]
            print(f"Type of value for key '{first_key}': {type(first_val)}")
            print(f"Value sample (str representation): {str(first_val)[:200]}")
            
            if isinstance(first_val, pd.DataFrame):
                print("It IS a DataFrame.")
                print(first_val.head())
            elif isinstance(first_val, dict):
                print("It IS a dict.")
                print(f"Keys of inner dict: {list(first_val.keys())[:5]}")
except Exception as e:
    print(f"Error: {e}")
