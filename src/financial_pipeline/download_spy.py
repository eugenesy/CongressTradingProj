import pandas as pd
import pickle
import os
from datetime import timedelta
from dateutil.relativedelta import relativedelta

from src.financial_pipeline.utils import load_checkpoint, save_checkpoint, get_data_path

# Config - updated to ml_dataset_clean.csv
INPUT_CSV = get_data_path('raw', 'ml_dataset_clean.csv')
SPY_PKL = get_data_path('processed', 'spy_historical_data.pkl')
SPY_PARQUET = get_data_path('parquet', 'SPY.parquet')

def download_spy_historical(
    input_csv=str(INPUT_CSV),
    spy_pkl=str(SPY_PKL),
    spy_parquet=str(SPY_PARQUET)
):
    print("Loading transaction data to determine date range...")
    df = pd.read_csv(input_csv, parse_dates=['Traded'])

    earliest_trade = df['Traded'].min() - timedelta(days=45)
    latest_trade = df['Traded'].max() + relativedelta(months=6)

    print(f"Date range needed: {earliest_trade.date()} to {latest_trade.date()}")

    spy_data = load_checkpoint(spy_pkl) or {}

    try:
        spy_df = pd.read_parquet(spy_parquet)
        spy_df_filtered = spy_df[(spy_df.index >= earliest_trade) & (spy_df.index <= latest_trade)]

        for date, row in spy_df_filtered.iterrows():
            date_str = date.strftime('%Y-%m-%d')
            if date_str not in spy_data:
                spy_data[date_str] = {
                    'open': row['open'], 'high': row['high'], 'low': row['low'],
                    'close': row['close'], 'volume': row['volume'],
                    'adjClose': row['adjClose'], 'unadjustedVolume': row['unadjustedVolume'],
                    'change': row['change'], 'changePercent': row['changePercent'],
                    'vwap': row['vwap'], 'label': row['label'],
                    'changeOverTime': row['changeOverTime']
                }

        save_checkpoint(spy_data, spy_pkl)
        print(f"SPY historical data saved to {spy_pkl}")
        return True

    except Exception as e:
        print(f"Error reading SPY data: {e}")
        return False