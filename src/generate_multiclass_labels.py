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
OUTPUT_PATH = 'data/raw/ml_dataset_with_multiclass_labels.csv'

# Updated PERIODS: 1Y renamed to 12M, new months added
PERIODS = {
    '1W': {'weeks': 1},
    '2W': {'weeks': 2},
    '1M': {'months': 1},
    '2M': {'months': 2},
    '3M': {'months': 3},
    '6M': {'months': 6},
    '7M': {'months': 7},
    '8M': {'months': 8},
    '9M': {'months': 9},
    '10M': {'months': 10},
    '11M': {'months': 11},
    '12M': {'months': 12}, # Renamed from 1Y
    '14M': {'months': 14},
    '16M': {'months': 16},
    '18M': {'months': 18},
    '20M': {'months': 20},
    '22M': {'months': 22},
    '24M': {'months': 24}
}

# Base POSITIVE thresholds
# Logic extrapolation: Roughly (3 + Month_Count) based on your original 1M-1Y curve
_BASE_THRESHOLDS = {
    '1W':  [0, 2, 4, 6, 8],
    '2W':  [0, 2.5, 5, 7.5, 10],
    '1M':  [0, 4, 8, 12, 16],
    '2M':  [0, 5, 10, 15, 20],
    '3M':  [0, 6, 12, 18, 24],
    '6M':  [0, 9, 18, 27, 36],
    '7M':  [0, 10, 20, 30, 40],   # New
    '8M':  [0, 11, 22, 33, 44],
    '9M':  [0, 12, 24, 36, 48],   # New
    '10M': [0, 13, 26, 39, 52],   # New
    '11M': [0, 14, 28, 42, 56],   # New
    '12M': [0, 15, 30, 45, 60],   # Renamed from 1Y
    '14M': [0, 17, 34, 51, 68],   # New (3+14)
    '16M': [0, 19, 38, 57, 76],   # New (3+16)
    '18M': [0, 21, 42, 63, 84],   # New (3+18)
    '20M': [0, 23, 46, 69, 92],   # New (3+20)
    '22M': [0, 25, 50, 75, 100],  # New (3+22)
    '24M': [0, 27, 54, 81, 108]   # New (3+24)
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
        
        # Labels defaults
        results[f'{k}_10bins'] = "N/A"
        results[f'{k}_6bins'] = "N/A"
        results[f'{k}_3bins'] = "N/A"
        
        if k in LABEL_THRESHOLDS:
            for t in LABEL_THRESHOLDS[k]:
                t_str = f"minus{abs(t):g}" if t < 0 else f"{t:g}"
                col_name = f"{k}_{t_str}PC"
                results[col_name] = 0 # Default

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
        
        try:
            target_end_date = trade_date + relativedelta(**delta)
            
            end_stock, actual_end_date_stock = get_price_at_date(t_series, target_end_date)
            end_spy, actual_end_date_spy = get_price_at_date(global_spy_series, target_end_date)
            
            # === HANDLING UNDETERMINED / MISSING FUTURE DATA ===
            if end_stock is None or end_spy is None:
                results[perf_col] = 0.0
                results[log_col] = 0.0
                results[cagr_col] = 0.0
                
                results[f'{period_name}_10bins'] = "Undetermined"
                results[f'{period_name}_6bins'] = "Undetermined"
                results[f'{period_name}_3bins'] = "Undetermined"
                
                if period_name in LABEL_THRESHOLDS:
                    for t_val in LABEL_THRESHOLDS[period_name]:
                        t_str = f"minus{abs(t_val):g}" if t_val < 0 else f"{t_val:g}"
                        col_name = f"{period_name}_{t_str}PC"
                        results[col_name] = -1.0
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
                # 1. 10 Bins (Symmetric: ~9 thresholds -> 10 bins)
                full_thresh = [x/100.0 for x in LABEL_THRESHOLDS[period_name]]
                results[f'{period_name}_10bins'] = get_categorical_label(abs_spread, full_thresh)
                
                # 2. 6 Bins (Combine 5 negs into 1 "Below 0%")
                base_thresh = [x/100.0 for x in _BASE_THRESHOLDS[period_name]]
                results[f'{period_name}_6bins'] = get_categorical_label(abs_spread, base_thresh)
                
                # 3. 3 Bins (Negs -> "Below 0%", Low Pos -> "0 to X", High Pos -> "X+")
                cutoff_pct = _BASE_THRESHOLDS[period_name][2]
                three_bin_thresh = [0.0, cutoff_pct/100.0]
                results[f'{period_name}_3bins'] = get_categorical_label(abs_spread, three_bin_thresh)
                
                # 4. Binary Columns (One-Hot style per threshold)
                current_thresholds_pct = LABEL_THRESHOLDS[period_name]
                decimal_thresholds = [x / 100.0 for x in current_thresholds_pct]
                
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

    # Sort periods for display
    def period_sort_key(k):
        if 'W' in k: return int(k.replace('W','')) * 0.25
        if 'M' in k: return int(k.replace('M',''))
        return 999

    sorted_periods = sorted(PERIODS.keys(), key=period_sort_key)

    for period_name in sorted_periods:
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