import os
from pathlib import Path

# Project Root (Relative to this config file)
# Works whether installed as package or run from source
try:
    # If installed as package
    import importlib.resources as pkg_resources
    PROJECT_ROOT = Path(pkg_resources.files("src").parent)
except (ImportError, AttributeError):
    # If running from source
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Allow override via environment variable
PROJECT_ROOT = Path(os.getenv("CHOCOLATE_PROJECT_ROOT", PROJECT_ROOT))

# Data Directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data"
PARQUET_DATA_DIR = PROJECT_ROOT / "data" / "parquet"

# Key Files
TX_FILENAME = "ml_dataset_reduced_attributes.csv"
TX_PATH = PROCESSED_DATA_DIR / "processed" / TX_FILENAME

PRICE_FILENAME = "all_tickers_historical_data.pkl"
PRICE_PATH = RAW_DATA_DIR / PRICE_FILENAME

# Outputs
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Make sure local directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Convert to strings for backward compatibility
PROJECT_ROOT = str(PROJECT_ROOT)
RAW_DATA_DIR = str(RAW_DATA_DIR)
PROCESSED_DATA_DIR = str(PROCESSED_DATA_DIR)
TX_PATH = str(TX_PATH)
PRICE_PATH = str(PRICE_PATH)
RESULTS_DIR = str(RESULTS_DIR)
LOGS_DIR = str(LOGS_DIR)
