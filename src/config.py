import os

# Project Root (Relative to this config file)
# src/config.py -> Project Root is one level up (.. )
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data Directories
# --- Personal Config (Preserved) ---
# RAW_DATA_DIR = "/data1/user_syeugene/fintech/apple/data/processed"
# -----------------------------------

# Data Directories (Public Default)
# Expects data to be in 'data/raw' relative to project root
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Key Files
TX_FILENAME = "ml_dataset_with_multiclass_labels.csv"
TX_PATH = os.path.join(RAW_DATA_DIR, TX_FILENAME)

PRICE_FILENAME = "all_tickers_historical_data.pkl"
PRICE_PATH = os.path.join(RAW_DATA_DIR, PRICE_FILENAME)

# Outputs
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
VISUALIZATIONS_DIR = os.path.join(PROJECT_ROOT, "visualizations")

# Make sure local directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ==========================================
#         MODEL & LABEL CONFIGURATION
# ==========================================

# 1. Label Settings
# Column to use as the target
LABEL_COL = '1W_0PC'

# Defines the 'resolution_t' (when the label is revealed)
LABEL_LOOKAHEAD_DAYS = 7  

# 2. Filtering
# Minimum trades required to include a company node
MIN_TICKER_FREQ = 5