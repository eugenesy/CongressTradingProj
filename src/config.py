import os
from pathlib import Path

# Project Root (Relative to this config file)
# Works whether installed as package or run from source
try:
    import importlib.resources as pkg_resources
    PROJECT_ROOT = Path(pkg_resources.files("src").parent)
except (ImportError, AttributeError):
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Allow override via environment variable
PROJECT_ROOT = Path(os.getenv("CHOCOLATE_PROJECT_ROOT", PROJECT_ROOT))

# Data Directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PARQUET_DATA_DIR = PROJECT_ROOT / "data" / "parquet"
LOBBYING_DIR = RAW_DATA_DIR / "lobbying_data_lobbyview"
CAMPAIGN_FINANCE_DIR = RAW_DATA_DIR / "campaign_finance_open_secrets"
DATA_527_DIR = RAW_DATA_DIR / "527_data_open_secrets"
INDUSTRY_CROSSWALK_DIR = RAW_DATA_DIR / "industry_codes_NAICS"

# --- Feature Flags ---
INCLUDE_POLITICIAN_BIO = True 
INCLUDE_IDEOLOGY = True
INCLUDE_COMMITTEES = True
INCLUDE_COMPANY_SIC = True
INCLUDE_DISTRICT_ECON = False
INCLUDE_COMPANY_FINANCIALS = False
INCLUDE_LOBBYING_SPONSORSHIP = True
INCLUDE_LOBBYING_VOTING = True
INCLUDE_CAMPAIGN_FINANCE = True

# --- Specific Data Paths ---
IDEOLOGY_PATH = RAW_DATA_DIR / "ideology_scores_quarterly.csv"
DISTRICT_ECON_DIR = RAW_DATA_DIR / "district_industries"
COMMITTEE_PATH = RAW_DATA_DIR / "committee_assignments.csv"
COMPANY_SIC_PATH = RAW_DATA_DIR / "company_sic_data.csv"
COMPANY_FIN_PATH = RAW_DATA_DIR / "sec_quarterly_financials_unzipped.csv"
CONGRESS_TERMS_PATH = RAW_DATA_DIR / "congress_terms_all_github.csv"

# Lobbying & Votes
LOBBYING_BILLS_PATH = LOBBYING_DIR / "bills.csv"
LOBBYING_CLIENTS_PATH = LOBBYING_DIR / "clients.csv"
LOBBYING_REPORTS_PATH = LOBBYING_DIR / "reports.csv"
LOBBYING_ISSUES_PATH = LOBBYING_DIR / "issue_text.csv"

# VoteView Paths
VOTEVIEW_VOTES_PATH = RAW_DATA_DIR / "HSall_votes.csv"
VOTEVIEW_ROLLCALLS_PATH = RAW_DATA_DIR / "HSall_rollcalls.csv"

# Campaign Finance Paths
CAMPAIGN_CANDS_PATTERN = "cands*.csv"
CAMPAIGN_PACS_PATTERN = "pacs*.csv"
DATA_527_EXPENDITURES_PATH = DATA_527_DIR / "Expenditures.csv"
DATA_527_COMMITTEES_PATH = DATA_527_DIR / "Cmtes527.csv"

# Crosswalks
NAICS_TO_SIC_PATH = INDUSTRY_CROSSWALK_DIR / "2017-NAICS-to-SIC-Crosswalk.csv"
COMPANY_SIC_DATA_PATH = RAW_DATA_DIR / "company_sic_data.csv" 
LEGISLATORS_CROSSWALK_PATH = RAW_DATA_DIR / "congress_terms_all_github.csv"

# Key Files
TX_FILENAME = "ml_dataset_clean.csv"
TX_PATH = PROCESSED_DATA_DIR / TX_FILENAME

PRICE_FILENAME = "all_tickers_historical_data.pkl"
PRICE_PATH = RAW_DATA_DIR / PRICE_FILENAME

# Outputs
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
LOBBYING_EVENTS_PATH = PROCESSED_DATA_DIR / "events_lobbying.csv"
CAMPAIGN_FINANCE_EVENTS_PATH = PROCESSED_DATA_DIR / "events_campaign_finance.csv"

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

# Legacy Path String Exports
IDEOLOGY_PATH = str(IDEOLOGY_PATH)
DISTRICT_ECON_DIR = str(DISTRICT_ECON_DIR)
COMMITTEE_PATH = str(COMMITTEE_PATH)
COMPANY_SIC_PATH = str(COMPANY_SIC_PATH)
COMPANY_FIN_PATH = str(COMPANY_FIN_PATH)
CONGRESS_TERMS_PATH = str(CONGRESS_TERMS_PATH)