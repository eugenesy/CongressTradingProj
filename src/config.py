import os
from pathlib import Path

# Project Root (Relative to this config file)
try:
    import importlib.resources as pkg_resources
    PROJECT_ROOT = Path(pkg_resources.files("src").parent)
except (ImportError, AttributeError):
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()

PROJECT_ROOT = Path(os.getenv("CHOCOLATE_PROJECT_ROOT", PROJECT_ROOT))

# Data Directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
LOBBYING_DIR = RAW_DATA_DIR / "lobbying_data_lobbyview"
CAMPAIGN_FINANCE_DIR = RAW_DATA_DIR / "campaign_finance_open_secrets"
DATA_527_DIR = RAW_DATA_DIR / "527_data_open_secrets"
INDUSTRY_CROSSWALK_DIR = RAW_DATA_DIR / "industry_codes_NAICS"

# --- Feature Flags (Updated for Stage 2) ---
INCLUDE_PERFORMANCE = True # NEW: Toggles the 3-dim trading performance stats
INCLUDE_POLITICIAN_BIO = True 
INCLUDE_IDEOLOGY = True
INCLUDE_COMMITTEES = True
INCLUDE_COMPANY_SIC = False
INCLUDE_DISTRICT_ECON = False
INCLUDE_COMPANY_FINANCIALS = False


# --- Specific Data Paths ---
IDEOLOGY_PATH = RAW_DATA_DIR / "ideology_scores_quarterly.csv"
DISTRICT_ECON_DIR = RAW_DATA_DIR / "district_industries"
COMMITTEE_PATH = RAW_DATA_DIR / "committee_assignments.csv"
COMPANY_SIC_PATH = RAW_DATA_DIR / "company_sic_data.csv"
COMPANY_FIN_PATH = RAW_DATA_DIR / "sec_quarterly_financials_unzipped.csv"
CONGRESS_TERMS_PATH = RAW_DATA_DIR / "congress_terms_all_github.csv"

# Make sure local directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Convert to strings for backward compatibility
PROJECT_ROOT = str(PROJECT_ROOT)
IDEOLOGY_PATH = str(IDEOLOGY_PATH)
DISTRICT_ECON_DIR = str(DISTRICT_ECON_DIR)
COMMITTEE_PATH = str(COMMITTEE_PATH)
COMPANY_SIC_PATH = str(COMPANY_SIC_PATH)
COMPANY_FIN_PATH = str(COMPANY_FIN_PATH)
CONGRESS_TERMS_PATH = str(CONGRESS_TERMS_PATH)