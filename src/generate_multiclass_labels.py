import pandas as pd
import numpy as np
import pickle
import os
import time
import warnings
from datetime import datetime
from dateutil.relativedelta import relativedelta
from multiprocessing import Pool, cpu_count, freeze_support, Manager
from tqdm import tqdm

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ================= CONFIGURATION =================
PICKLE_PATH = 'data/raw/all_tickers_historical_data.pkl'
CSV_PATH = 'data/raw/ml_dataset_reduced_attributes.csv'
OUTPUT_PATH = 'data/processed/ml_dataset_with_multiclass_labels.csv'

PERIODS = {
    'performance_1W': {'weeks': 1},
    'performance_2W': {'weeks': 2},
    'performance_1M': {'months': 1},
    'performance_2M': {'months': 2},
    'performance_3M': {'months': 3},
    'performance_6M': {'months': 6},
    'performance_8M': {'months': 8},
    'performance_1Y': {'years': 1}
}
# =================================================

# Global variables for workers
global_hist_data = None
global_spy_series = None
global_progress_queue = None

def init_worker(shared_data, spy_data, queue):
    """
    Initializes the worker with data and the progress queue.
    """
    global global_hist_data
    global global_spy_series
    global global_progress_queue
    
    global_hist_data = shared_data
    global_spy_series = spy_data
    global_progress_queue = queue

def get_ticker_series(ticker):
    if ticker not in global_hist_data:
        return None
    records = global_hist_data[ticker]
    try:
        if isinstance(records, pd.DataFrame):
            return records['adjClose']
        dates, prices = [], []
        for date_str, stats in records.items():
            val = stats.get('adjClose', stats.get('close'))
            if val:
                dates.append(date_str)
                prices.append(float(val))
        if not dates: return None
        return pd.Series(prices, index=pd.to_datetime(dates)).sort_index()
    except:
        return None

def get_price_at_date(price_series, target_date):
    idx = price_series.index.searchsorted(target_date)
    if idx >= len(price_series): return None, None
    actual_date = price_series.index[idx]
    if (actual_date - target_date).days > 10: return None, None
    return price_series.iloc[idx], actual_date

def process_row_logic(row):
    """
    The actual math logic for a single row.
    """
    ticker = row['Ticker']
    try:
        trade_date = pd.to_datetime(row['Traded'])
    except:
        return {k: 0.0 for k in PERIODS}

    t_series = get_ticker_series(ticker)
    if t_series is None or global_spy_series is None:
        return {k: 0.0 for k in PERIODS}

    start_stock, start_stock_dt = get_price_at_date(t_series, trade_date)
    start_spy, start_spy_dt = get_price_at_date(global_spy_series, trade_date)

    if start_stock is None or start_spy is None:
        return {k: 0.0 for k in PERIODS}

    results = {}
    for col_name, delta in PERIODS.items():
        try:
            target_end_date = trade_date + relativedelta(**delta)
            end_stock, end_stock_dt = get_price_at_date(t_series, target_end_date)
            end_spy, end_spy_dt = get_price_at_date(global_spy_series, target_end_date)
            
            if end_stock is None or end_spy is None:
                results[col_name] = 0.0
                continue

            s_days = (end_stock_dt - start_stock_dt).days
            s_cagr = (end_stock / start_stock) ** (1 / (s_days / 365.25)) - 1 if s_days > 0 else 0

            spy_days = (end_spy_dt - start_spy_dt).days
            spy_cagr = (end_spy / start_spy) ** (1 / (spy_days / 365.25)) - 1 if spy_days > 0 else 0
            
            results[col_name] = s_cagr - spy_cagr
        except:
            results[col_name] = 0.0
    return results

def process_chunk(df_chunk):
    """
    Iterates through the chunk manually to send progress updates.
    """
    results = []
    # Loop manually so we can ping the queue
    for index, row in df_chunk.iterrows():
        # Do the math
        res = process_row_logic(row)
        results.append(res)
        
        # SIGNAL PROGRESS: Tell the main process we finished 1 row
        if global_progress_queue:
            global_progress_queue.put(1)
            
    return pd.DataFrame(results, index=df_chunk.index)

def main():
    print(f"Loading data...")
    if not os.path.exists(PICKLE_PATH):
        print(f"ERROR: Pickle file not found at {PICKLE_PATH}")
        return

    with open(PICKLE_PATH, 'rb') as f:
        full_data = pickle.load(f)
    
    df = pd.read_csv(CSV_PATH)
    total_rows = len(df)
    
    # Identify SPY
    spy_key = next((k for k in ['SPY', '^GSPC', 'sp500'] if k in full_data), None)
    if not spy_key:
        for k in full_data.keys():
            if 'SPY' in str(k).upper(): spy_key = k; break
    
    print(f"SPY Ticker: {spy_key}")
    
    # Prep SPY Series
    spy_series = None
    if spy_key:
        records = full_data[spy_key]
        if isinstance(records, pd.DataFrame):
            spy_series = records['adjClose']
        else:
            d, p = [], []
            for k,v in records.items():
                val = v.get('adjClose', v.get('close'))
                if val: d.append(k); p.append(float(val))
            spy_series = pd.Series(p, index=pd.to_datetime(d)).sort_index()

    # Prep Multiprocessing
    cores = cpu_count()
    manager = Manager()
    queue = manager.Queue()
    
    print(f"Starting processing on {cores} cores for {total_rows} records...")
    
    # Split data
    df_chunks = np.array_split(df, cores * 2) 
    
    pool = Pool(processes=cores, initializer=init_worker, initargs=(full_data, spy_series, queue))
    
    async_result = pool.map_async(process_chunk, df_chunks)
    
    # --- THE PROGRESS BAR LOOP ---
    with tqdm(total=total_rows, unit="rows") as pbar:
        processed_count = 0
        while not async_result.ready():
            while not queue.empty():
                try:
                    _ = queue.get_nowait()
                    processed_count += 1
                    pbar.update(1)
                except:
                    break
            time.sleep(0.1)
            
        remaining = total_rows - processed_count
        if remaining > 0:
            pbar.update(remaining)

    results_list = async_result.get()
    pool.close()
    pool.join()
    
    print("Merging data...")
    new_cols = pd.concat(results_list)
    final_df = df.join(new_cols)
    
    # --- THE FIX IS HERE ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Done! Saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    freeze_support()
    main()