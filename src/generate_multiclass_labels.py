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
from scipy import stats

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ================= CONFIGURATION =================
PICKLE_PATH = 'data/raw/all_tickers_historical_data.pkl'
CSV_PATH = 'data/raw/ml_dataset_reduced_attributes.csv'
OUTPUT_PATH = 'data/processed/ml_dataset_with_multiclass_labels.csv'

PERIODS = {
    '1W': {'weeks': 1},
    '2W': {'weeks': 2},
    '1M': {'months': 1},
    '2M': {'months': 2},
    '3M': {'months': 3},
    '6M': {'months': 6},
    '8M': {'months': 8},
    '1Y': {'years': 1}
}

# Base POSITIVE thresholds
_BASE_THRESHOLDS = {
    '1W': [0, 2, 4, 6, 8],
    '2W': [0, 2.5, 5, 7.5, 10],
    '1M': [0, 4, 8, 12, 16],
    '2M': [0, 5, 10, 15, 20],
    '3M': [0, 6, 12, 18, 24],
    '6M': [0, 9, 18, 27, 36],
    '8M': [0, 11, 22, 33, 44],
    '1Y': [0, 15, 30, 45, 60]
}

# Generate Full Symmetric Thresholds (e.g., -8, -6, -4, -2, 0, 2, 4, 6, 8)
LABEL_THRESHOLDS = {}
for p, vals in _BASE_THRESHOLDS.items():
    negatives = sorted([-x for x in vals if x != 0])
    full_list = negatives + sorted(vals)
    LABEL_THRESHOLDS[p] = full_list

print("Configuration: Generated Symmetric Thresholds")
# =================================================

# Global variables for workers
global_hist_data = None
global_spy_series = None
global_progress_queue = None

def init_worker(pickle_path, spy_data, queue):
    global global_hist_data
    global global_spy_series
    global global_progress_queue
    
    with open(pickle_path, 'rb') as f:
        global_hist_data = pickle.load(f)
        
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

def get_categorical_label(value, thresholds):
    """
    Returns a string label based on which bin the value falls into.
    """
    if value < thresholds[0]:
        return f"Below {thresholds[0]*100:g}%"
    
    for i in range(len(thresholds) - 1):
        low = thresholds[i]
        high = thresholds[i+1]
        if low <= value < high:
            low_str = f"{low*100:g}" 
            high_str = f"{high*100:g}"
            return f"{low_str}% to {high_str}%"
            
    last_thresh = thresholds[-1]
    if value >= last_thresh:
        return f"{last_thresh*100:g}%+"
        
    return "Error"

def process_row_logic(row):
    ticker = row['Ticker']
    multiplier = -1 if row.get('Transaction') == 'Sale' else 1
    
    results = {}
    
    # 1. Pre-fill defaults
    for k in PERIODS:
        results[f'performance_{k}'] = 0.0
        results[f'log_performance_{k}'] = 0.0
        results[f'CAGR_{k}'] = 0.0
        results[f'{k}_Bin'] = "N/A" # Default for completely missing ticker data
        
        if k in LABEL_THRESHOLDS:
            for t in LABEL_THRESHOLDS[k]:
                t_str = f"minus{abs(t):g}" if t < 0 else f"{t:g}"
                col_name = f"{k}_{t_str}PC"
                results[col_name] = 0 # Default (will be overwritten to 0.5 if timeframe is undetermined)

    try:
        trade_date = pd.to_datetime(row['Traded'])
    except:
        return results

    t_series = get_ticker_series(ticker)
    if t_series is None or global_spy_series is None:
        return results

    start_stock, actual_start_date_stock = get_price_at_date(t_series, trade_date)
    start_spy, actual_start_date_spy = get_price_at_date(global_spy_series, trade_date)

    if start_stock is None or start_spy is None:
        return results
    
    for period_name, delta in PERIODS.items():
        perf_col = f'performance_{period_name}'
        log_col = f'log_performance_{period_name}'
        cagr_col = f'CAGR_{period_name}'
        bin_col = f'{period_name}_Bin'
        
        try:
            target_end_date = trade_date + relativedelta(**delta)
            
            end_stock, actual_end_date_stock = get_price_at_date(t_series, target_end_date)
            end_spy, actual_end_date_spy = get_price_at_date(global_spy_series, target_end_date)
            
            # === HANDLING UNDETERMINED / MISSING FUTURE DATA ===
            if end_stock is None or end_spy is None:
                results[perf_col] = 0.0
                results[log_col] = 0.0
                results[cagr_col] = 0.0
                results[bin_col] = "Undetermined"
                
                # Set all binary labels to 0.5
                if period_name in LABEL_THRESHOLDS:
                    for t_val in LABEL_THRESHOLDS[period_name]:
                        t_str = f"minus{abs(t_val):g}" if t_val < 0 else f"{t_val:g}"
                        col_name = f"{period_name}_{t_str}PC"
                        results[col_name] = 0.5
                continue

            # === Calculations (If Data Exists) ===
            stock_return = (end_stock / start_stock) - 1
            spy_return = (end_spy / start_spy) - 1
            abs_spread = (stock_return - spy_return) * multiplier
            
            stock_log = np.log(end_stock / start_stock)
            spy_log = np.log(end_spy / start_spy)
            log_spread = (stock_log - spy_log) * multiplier

            days_diff = (actual_end_date_stock - actual_start_date_stock).days
            if days_diff > 0:
                years = days_diff / 365.25
                stock_cagr = (end_stock / start_stock) ** (1 / years) - 1
                spy_cagr = (end_spy / start_spy) ** (1 / years) - 1
                cagr_spread = (stock_cagr - spy_cagr) * multiplier
            else:
                cagr_spread = 0.0
            
            results[perf_col] = abs_spread
            results[log_col] = log_spread
            results[cagr_col] = cagr_spread
            
            # === GENERATE LABELS ===
            if period_name in LABEL_THRESHOLDS:
                current_thresholds_pct = LABEL_THRESHOLDS[period_name]
                decimal_thresholds = [x / 100.0 for x in current_thresholds_pct]
                
                # 1. Categorical String Label
                results[bin_col] = get_categorical_label(abs_spread, decimal_thresholds)
                
                # 2. Binary Columns
                for i, t_val in enumerate(current_thresholds_pct):
                    thresh_decimal = decimal_thresholds[i]
                    
                    t_str = f"minus{abs(t_val):g}" if t_val < 0 else f"{t_val:g}"
                    col_name = f"{period_name}_{t_str}PC"
                    
                    if abs_spread >= thresh_decimal:
                        results[col_name] = 1
                    else:
                        results[col_name] = 0

        except Exception as e:
            pass

    return results

def process_chunk(df_chunk):
    results = []
    for index, row in df_chunk.iterrows():
        res = process_row_logic(row)
        results.append(res)
        if global_progress_queue:
            global_progress_queue.put(1)
            
    return pd.DataFrame(results, index=df_chunk.index)

def main():
    print(f"Loading data (Main Process)...")
    if not os.path.exists(PICKLE_PATH):
        print(f"ERROR: Pickle file not found at {PICKLE_PATH}")
        return

    with open(PICKLE_PATH, 'rb') as f:
        full_data = pickle.load(f)
    
    df = pd.read_csv(CSV_PATH)
    total_rows = len(df)
    
    spy_key = next((k for k in ['SPY', '^GSPC', 'sp500'] if k in full_data), None)
    if not spy_key:
        for k in full_data.keys():
            if 'SPY' in str(k).upper(): spy_key = k; break
    
    print(f"SPY Ticker: {spy_key}")
    
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
    
    del full_data 

    cores = cpu_count()
    manager = Manager()
    queue = manager.Queue()
    
    print(f"Starting processing on {cores} cores for {total_rows} records...")
    
    df_chunks = np.array_split(df, cores * 2) 
    
    pool = Pool(processes=cores, initializer=init_worker, initargs=(PICKLE_PATH, spy_series, queue))
    async_result = pool.map_async(process_chunk, df_chunks)
    
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
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Done! Saved to {OUTPUT_PATH}")

    # =========================================================================
    # STATISTICAL ANALYSIS
    # =========================================================================
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS: One-Sample T-Test on Log Excess Returns")
    print("-" * 60)
    print(f"{'Period':<10} | {'Count':<8} | {'Mean Log Return':<18} | {'T-Statistic':<12} | {'P-Value':<12}")
    print("-" * 60)

    for period_name in PERIODS:
        col_name = f'log_performance_{period_name}'
        
        if col_name in final_df.columns:
            data = final_df[final_df[col_name] != 0][col_name].dropna()
            
            if len(data) > 1:
                t_stat, p_val = stats.ttest_1samp(data, 0)
                mean_val = data.mean()
                significance = "*" if p_val < 0.05 else ""
                significance += "*" if p_val < 0.01 else ""
                print(f"{period_name:<10} | {len(data):<8} | {mean_val:.6f}           | {t_stat:6.2f}       | {p_val:.2e} {significance}")

    print("="*60 + "\n")

if __name__ == '__main__':
    freeze_support()
    main()