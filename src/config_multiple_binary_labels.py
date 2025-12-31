import os

# Project Root (Relative to this config file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data Directories
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

#Logging Directory
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Key Files
TX_FILENAME = "ml_dataset_with_multiclass_labels.csv"
TX_PATH = os.path.join(RAW_DATA_DIR, TX_FILENAME)

PRICE_FILENAME = "all_tickers_historical_data.pkl"
PRICE_PATH = os.path.join(RAW_DATA_DIR, PRICE_FILENAME)

# ==========================================
#         MODEL & LABEL CONFIGURATION
# ==========================================

# 1. Label Selection

TARGET_YEARS = [2022]
# Pick one: '1W', '2W', '1M', '2M', '3M', '6M', '8M', '1Y'
LABEL_TIMEFRAME = '3M'

# Dictionary defining the explicit thresholds for each timeframe
# Format: Tuple (List of Negatives [Worst->Best], List of Positives [Best->Worst])
# Note: Negatives should be sorted ascending (e.g., -16, -12, -8)
#       Positives should be sorted ascending (e.g., 4, 8, 12)
# 0 is ALWAYS implicitly included.

_THRESHOLD_MAP = {
    '1W': ([-8, -6, -4, -2], [0, 2, 4, 6, 8]),
    '2W': ([-10, -7.5, -5, -2.5], [0, 2.5, 5, 7.5, 10]),
    '1M': ([-16, -12, -8, -4], [0, 4, 8, 12, 16]),
    '2M': ([-20, -15, -10, -5], [0, 5, 10, 15, 20]),
    # '3M': ([-24, -18, -12, -6], [0, 6, 12, 18, 24]),
    '6M': ([-36, -27, -18, -9], [0, 9, 18, 27, 36]),
    '8M': ([-44, -33, -22, -11], [0, 11, 22, 33, 44]),
    '1Y': ([-60, -45, -30, -15], [0, 15, 30, 45, 60]),
    '3M': ([-6], [6]),
}

if LABEL_TIMEFRAME in _THRESHOLD_MAP:
    neg_vals, pos_vals = _THRESHOLD_MAP[LABEL_TIMEFRAME]
    
    # Generate Negative Strings: "minus16PC", "minus12PC"...
    negatives = [f"minus{str(abs(x)).replace('.0', '')}PC" for x in neg_vals]
    
    # Generate Positive Strings: "4PC", "8PC"...
    positives = [f"{str(x).replace('.0', '')}PC" for x in pos_vals]
    
    # Combine: [-16, ..., -4, 0PC, 4, ..., 16]
    LABEL_THRESHOLDS = negatives + ['0PC'] + positives
else:
    raise ValueError(f"Unknown timeframe: {LABEL_TIMEFRAME}")

# Auto-generate column names (e.g., "1M_minus16PC")
TARGET_COLUMNS = [f"{LABEL_TIMEFRAME}_{t}" for t in LABEL_THRESHOLDS]

# 2. Lookahead Logic (Auto-mapped)
LOOKAHEAD_MAP = {
    '1W': 7, '2W': 14, '1M': 30, '2M': 60, 
    '3M': 90, '6M': 180, '8M': 240, '1Y': 365
}
LABEL_LOOKAHEAD_DAYS = LOOKAHEAD_MAP.get(LABEL_TIMEFRAME, 30)

# 3. Filtering
MIN_TICKER_FREQ = 5

# Outputs
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", LABEL_TIMEFRAME)
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs", LABEL_TIMEFRAME)
VISUALIZATIONS_DIR = os.path.join(PROJECT_ROOT, "visualizations", LABEL_TIMEFRAME)

# Make sure local directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)