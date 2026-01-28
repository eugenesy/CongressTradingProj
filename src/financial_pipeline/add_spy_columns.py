import pandas as pd
import numpy as np
import pickle
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from datetime import datetime

from src.financial_pipeline.utils import load_checkpoint, get_data_path

# Config - Input: clean, Output: v2
INPUT_CSV = get_data_path('raw', 'ml_dataset_clean.csv')
OUTPUT_CSV = get_data_path('processed', 'ml_dataset_v2.csv')
# UPDATED: Default now points to the big raw pickle
HIST_PKL = get_data_path('raw', 'all_tickers_historical_data.pkl')
CURRENT_DATE = datetime(2025, 5, 20)

def _get_spy_price(target_date, spy_data):
    if target_date > CURRENT_DATE:
        return np.nan
    date_str = target_date.strftime('%Y-%m-%d')
    if date_str in spy_data:
        return spy_data[date_str]['close']
    dates = sorted([d for d in spy_data.keys() if d <= date_str], reverse=True)
    return spy_data[dates[0]]['close'] if dates else np.nan

def add_spy_columns(
    input_csv=INPUT_CSV,
    output_csv=OUTPUT_CSV,
    hist_pkl=HIST_PKL, # Renamed argument to reflect source
    current_date=CURRENT_DATE
):
    print("Loading datasets for SPY Benchmark...")
    df = pd.read_csv(input_csv, parse_dates=['Filed'])
    
    # Load the full historical data
    hist_data = load_checkpoint(hist_pkl)
    if hist_data is None:
        raise FileNotFoundError(f"Historical data not found at {hist_pkl}")

    # Extract SPY specific data
    # We check common keys for S&P 500
    spy_keys = ['SPY', 'spy', 'IVV', 'VOO'] 
    spy_data = None
    for k in spy_keys:
        if k in hist_data:
            spy_data = hist_data[k]
            print(f"Found Benchmark data under key: '{k}'")
            break
            
    if spy_data is None:
        raise ValueError(f"Could not find SPY (or equivalent) in {hist_pkl}. Keys found: {list(hist_data.keys())[:5]}...")

    spy_cols = [
        'SPY_TradeDate', 'SPY_1Month', 'SPY_2Months', 'SPY_3Months', 
        'SPY_6Months', 'SPY_8Months', 'SPY_12Months', 'SPY_18Months', 'SPY_24Months'
    ]
    for col in spy_cols:
        df[col] = np.nan

    print("Adding SPY benchmark prices...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        trade_date = row['Filed']
        periods = {
            'TradeDate': trade_date + relativedelta(days=1),
            '1Month': trade_date + relativedelta(months=1),
            '2Months': trade_date + relativedelta(months=2),
            '3Months': trade_date + relativedelta(months=3),
            '6Months': trade_date + relativedelta(months=6),
            '8Months': trade_date + relativedelta(months=8),
            '12Months': trade_date + relativedelta(months=12),
            '18Months': trade_date + relativedelta(months=18),
            '24Months': trade_date + relativedelta(months=24)
        }

        for period, target_date in periods.items():
            col_name = f'SPY_{period}'
            if pd.isna(df.at[idx, col_name]):
                df.at[idx, col_name] = _get_spy_price(target_date, spy_data)

    df.to_csv(output_csv, index=False)
    print(f"Benchmarked dataset saved to {output_csv}")
    return df