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
SPY_PKL = get_data_path('processed', 'spy_historical_data.pkl')
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
    spy_pkl=SPY_PKL,
    current_date=CURRENT_DATE
):
    print("Loading datasets...")
    df = pd.read_csv(input_csv, parse_dates=['Filed'])
    spy_data = load_checkpoint(spy_pkl)

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