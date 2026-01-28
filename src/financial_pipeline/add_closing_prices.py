import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta

from src.financial_pipeline.utils import load_checkpoint, save_checkpoint, get_data_path

# Config - Input: v2, Output: v3
INPUT_CSV = get_data_path('processed', 'ml_dataset_v2.csv')
OUTPUT_CSV = get_data_path('processed', 'ml_dataset_v3.csv')
HIST_PKL = get_data_path('processed', 'all_tickers_historical_data.pkl')
CHECKPOINT_FILE = get_data_path('processed', 'closing_price_checkpoint.pkl')
CHECKPOINT_INTERVAL = 10000
CURRENT_DATE = datetime(2025, 5, 21)

def _get_close(ticker, target_date, hist_data):
    if pd.isna(ticker) or ticker not in hist_data or not hist_data[ticker]:
        return np.nan
    date_str = target_date.strftime('%Y-%m-%d')
    if date_str in hist_data[ticker]:
        return hist_data[ticker][date_str]['close']
    dates = sorted([d for d in hist_data[ticker] if d <= date_str], reverse=True)
    return hist_data[ticker][dates[0]]['close'] if dates else np.nan

def add_closing_prices_to_transactions(
    input_csv=INPUT_CSV,
    hist_pkl=HIST_PKL,
    output_csv=OUTPUT_CSV,
    checkpoint_file=CHECKPOINT_FILE,
    checkpoint_interval=CHECKPOINT_INTERVAL,
    current_date=CURRENT_DATE
):
    print("Loading data for closing prices...")
    df = pd.read_csv(input_csv, parse_dates=['Filed'])
    hist_data = load_checkpoint(hist_pkl)
    processed_indices = load_checkpoint(checkpoint_file) or set()

    price_cols = [
        'Close_TradeDate', 'Close_1Month', 'Close_2Months', 'Close_3Months', 
        'Close_6Months', 'Close_8Months', 'Close_12Months', 'Close_18Months', 'Close_24Months'
    ]
    for col in price_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Check correct ticker col
    ticker_col = 'Ticker' if 'Ticker' in df.columns else 'Appropriate_Ticker'

    print("Adding closing prices...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if idx in processed_indices: continue
        
        ticker = row[ticker_col]
        trade_date = row['Filed']
        
        if pd.isna(ticker) or pd.isna(trade_date) or pd.to_datetime(trade_date) > current_date:
            processed_indices.add(idx)
            continue
            
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
        for label, tdate in periods.items():
            col = f'Close_{label}'
            if pd.isna(df.at[idx, col]) and tdate <= current_date:
                df.at[idx, col] = _get_close(ticker, tdate, hist_data)
                
        processed_indices.add(idx)
        if len(processed_indices) % checkpoint_interval == 0:
            df.to_csv(output_csv, index=False)
            save_checkpoint(processed_indices, checkpoint_file)

    df.to_csv(output_csv, index=False)
    save_checkpoint(processed_indices, checkpoint_file)
    print(f"Final file saved as {output_csv}")
    return df