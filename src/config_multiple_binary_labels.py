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

COMMITTEE_FILENAME = "committee_assignments.csv"
COMMITTEE_PATH = os.path.join(RAW_DATA_DIR, COMMITTEE_FILENAME)

SIC_FILENAME = "company_sic_data.csv"
SIC_PATH = os.path.join(RAW_DATA_DIR, SIC_FILENAME)

FINANCIALS_FILENAME = "sec_quarterly_financials.csv"
FINANCIALS_PATH = os.path.join(RAW_DATA_DIR, FINANCIALS_FILENAME)

# ==========================================
#         MODEL & LABEL CONFIGURATION
# ==========================================

# 1. Label Selection

TARGET_YEARS = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
# Pick one: '1W', '2W', '1M', '2M', '3M', '4M', '5M', '6M', '7M', '8M', 
# '9M', '10M', '11M', '12M', '14M', '16M', 18M', '20M', '22M', '24M'
LABEL_TIMEFRAME = '12M'

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
    '3M': ([-24, -18, -12, -6], [0, 6, 12, 18, 24]),
    '6M': ([-36, -27, -18, -9], [0, 9, 18, 27, 36]),
    '7M': ([-40, -30, -20, -10], [0, 10, 20, 30, 40]),
    '8M': ([-44, -33, -22, -11], [0, 11, 22, 33, 44]),
    '9M': ([-48, -36, -24, -12], [0, 12, 24, 36, 48]),
    '10M': ([-52, -39, -26, -13], [0, 13, 26, 39, 52]),
    '11M': ([-56, -42, -28, -14], [0, 14, 28, 42, 56]),
    # '12M': ([-60, -45, -30, -15], [0, 15, 30, 45, 60]), # Renamed from 1Y
    '14M': ([-68, -51, -34, -17], [0, 17, 34, 51, 68]),
    '16M': ([-76, -57, -38, -19], [0, 19, 38, 57, 76]),
    '18M': ([-84, -63, -42, -21], [0, 21, 42, 63, 84]),
    '20M': ([-92, -69, -46, -23], [0, 23, 46, 69, 92]),
    '22M': ([-100, -75, -50, -25], [0, 25, 50, 75, 100]),
    '24M': ([-108, -81, -54, -27], [0, 27, 54, 81, 108]),
    '12M': ([-15], [15]),
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
    '1W': 7, 
    '2W': 14, 
    '1M': 30, 
    '2M': 60, 
    '3M': 90, 
    '6M': 180, 
    '7M': 210, 
    '8M': 240, 
    '9M': 270, 
    '10M': 300, 
    '11M': 330, 
    '12M': 365, # Renamed from 1Y
    '14M': 420, 
    '16M': 480, 
    '18M': 540, 
    '20M': 600, 
    '22M': 660, 
    '24M': 730
}

LABEL_LOOKAHEAD_DAYS = LOOKAHEAD_MAP.get(LABEL_TIMEFRAME, 30)

# 3. Filtering
MIN_TICKER_FREQ = 5

# 4. Feature Flags
INCLUDE_IDEOLOGY = True      # Set to False to exclude ideology scores
INCLUDE_DISTRICT_ECON = True # Set to False to exclude district economic data
INCLUDE_COMMITTEES = True
INCLUDE_COMPANY_SIC = True
INCLUDE_COMPANY_FINANCIALS = True

# Outputs
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", LABEL_TIMEFRAME)
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs", LABEL_TIMEFRAME)
VISUALIZATIONS_DIR = os.path.join(PROJECT_ROOT, "visualizations", LABEL_TIMEFRAME)

# Make sure local directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)