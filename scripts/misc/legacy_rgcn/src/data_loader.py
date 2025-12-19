import pandas as pd
import numpy as np
import torch
import pickle
from datetime import timedelta

class ChocolateDataLoader:
    def __init__(self, transaction_path, price_data_path):
        """
        Args:
            transaction_path (str): Path to v9_transactions.csv
            price_data_path (str): Path to all_tickers_historical_data.pkl
        """
        self.transaction_path = transaction_path
        self.price_data_path = price_data_path
        self.transactions = None
        self.price_data = None
        self.politician_features = {} # Map BioGuideID -> features
        self.company_features = {}    # Map Ticker -> features

    def load_data(self):
        print("Loading transactions...")
        self.transactions = pd.read_csv(self.transaction_path)
        
        # Date parsing
        date_cols = ['Traded', 'Filed']
        for col in date_cols:
            self.transactions[col] = pd.to_datetime(self.transactions[col], errors='coerce')
        
        # Sort by Filed date for temporal consistency
        self.transactions = self.transactions.sort_values('Filed').reset_index(drop=True)
        
        print(f"Loaded {len(self.transactions)} transactions.")

        print(f"Loaded {len(self.transactions)} transactions.")

        if self.price_data_path:
            print("Loading price data (optional)...")
            with open(self.price_data_path, 'rb') as f:
                self.price_data = pickle.load(f)
            print(f"Loaded price data for {len(self.price_data)} tickers.")
        else:
            print("Skipping price data load (not provided).")
            self.price_data = {}

    def preprocess_node_features(self):
        """
        Extracts static features for Politicians and Companies from the transaction data.
        Features:
            Politician: Party, State, Chamber (One-hot or Label encoded)
            Company: Simple ID for now (Placeholder for future embeddings)
        """
        if self.transactions is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # --- Politician Nodes ---
        # unique_pols = self.transactions[['BioGuideID', 'Name', 'Party', 'State', 'Chamber']].drop_duplicates('BioGuideID')
        
        # Mappings for categorical features
        # Note: In a real/large system, saved mappers would be needed for inference consistency.
        # For this pipeline, we build them from the data.
        
        # Party Map
        parties = self.transactions['Party'].unique()
        self.party_map = {p: i for i, p in enumerate(parties)}
        
        # State Map
        states = self.transactions['State'].unique()
        self.state_map = {s: i for i, s in enumerate(states)}
        
        # Chamber Map
        chambers = self.transactions['Chamber'].unique()
        self.chamber_map = {c: i for i, c in enumerate(chambers)}

        print(f"Politician Feature Maps Created: {len(self.party_map)} Parties, {len(self.state_map)} States, {len(self.chamber_map)} Chambers.")

    def get_price_at_filing(self, ticker, filing_date):
        """
        Retrieves the 1D price scalar (Close price, or similar) at the Filing Date.
        Returns a default value (e.g., 0.0) if missing or date not found.
        """
        if ticker not in self.price_data:
            return 0.0
        
        # Lazy conversion to DataFrame for efficient time indexing
        if not hasattr(self, 'price_dfs'):
            self.price_dfs = {}
            
        if ticker not in self.price_dfs:
            raw_data = self.price_data[ticker]
            if isinstance(raw_data, dict):
                # Convert dict of dicts to DataFrame
                df = pd.DataFrame.from_dict(raw_data, orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                self.price_dfs[ticker] = df
            elif isinstance(raw_data, pd.DataFrame):
                df = raw_data
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                self.price_dfs[ticker] = df
            else:
                return 0.0
        
        ticker_df = self.price_dfs[ticker]
            
        # Lookup
        try:
            # User said "simple price data will be enough". 
            # Use 'asof' for nearest past date to handle weekends.
            if filing_date in ticker_df.index:
                 row = ticker_df.loc[filing_date]
            else:
                 # asof lookback
                 idx = ticker_df.index.asof(filing_date)
                 if pd.isna(idx):
                     return 0.0
                 row = ticker_df.loc[idx]
            
            # Assuming 'Close' or 'Adj Close' is the target scalar.
            if 'Close' in row:
                return float(row['Close'])
            elif 'close' in row:
                return float(row['close'])
            elif 'adjClose' in row:
                return float(row['adjClose'])
            else:
                return 0.0
                
        except Exception as e:
            # print(f"Error lookup {ticker} {filing_date}: {e}")
            return 0.0

if __name__ == "__main__":
    # Quick Test
    TX_PATH = "/data1/user_syeugene/fintech/banana/data/v9_transactions.csv"
    PRICE_PATH = "/data1/user_syeugene/fintech/apple/data/processed/all_tickers_historical_data.pkl"
    
    loader = ChocolateDataLoader(TX_PATH, PRICE_PATH)
    loader.load_data()
    loader.preprocess_node_features()
    
    # Test Price Lookup
    test_tx = loader.transactions.iloc[0]
    p = loader.get_price_at_filing(test_tx['Ticker'], test_tx['Filed'])
    print(f"Test Price for {test_tx['Ticker']} on {test_tx['Filed']}: {p}")
