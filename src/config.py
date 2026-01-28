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
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" # Fixed path structure
PARQUET_DATA_DIR = PROJECT_ROOT / "data" / "parquet"

# Feature Flags
INCLUDE_IDEOLOGY = True
INCLUDE_DISTRICT_ECON = True
INCLUDE_COMMITTEES = True
INCLUDE_COMPANY_SIC = True
INCLUDE_COMPANY_FINANCIALS = True

# Specific Data Paths
IDEOLOGY_PATH = RAW_DATA_DIR / "ideology_scores_quarterly.csv"
DISTRICT_ECON_DIR = RAW_DATA_DIR / "district_industries"
COMMITTEE_PATH = RAW_DATA_DIR / "committee_assignments.csv"
COMPANY_SIC_PATH = RAW_DATA_DIR / "company_sic_data.csv"
COMPANY_FIN_PATH = RAW_DATA_DIR / "sec_quarterly_financials_unzipped.csv"
CONGRESS_TERMS_PATH = RAW_DATA_DIR / "congress_terms_all_github.csv"

# Key Files
TX_FILENAME = "ml_dataset_clean.csv"
TX_PATH = PROCESSED_DATA_DIR / TX_FILENAME

# UPDATED: Price data now comes directly from raw pickle
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
IDEOLOGY_PATH = str(IDEOLOGY_PATH)
DISTRICT_ECON_DIR = str(DISTRICT_ECON_DIR)
COMMITTEE_PATH = str(COMMITTEE_PATH)
COMPANY_SIC_PATH = str(COMPANY_SIC_PATH)
COMPANY_FIN_PATH = str(COMPANY_FIN_PATH)
CONGRESS_TERMS_PATH = str(CONGRESS_TERMS_PATH)