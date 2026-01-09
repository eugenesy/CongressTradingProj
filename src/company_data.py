import pandas as pd
import numpy as np
import os
import bisect

# Standard SIC Divisions (High-level grouping)
# We map 4-digit SIC codes to these ~11 broad sectors.
SIC_DIVISIONS = [
    (100, 999, 'Agriculture'),
    (1000, 1499, 'Mining'),
    (1500, 1799, 'Construction'),
    (2000, 3999, 'Manufacturing'),
    (4000, 4999, 'Transportation_Utilities'),
    (5000, 5199, 'Wholesale'),
    (5200, 5999, 'Retail'),
    (6000, 6799, 'Finance_Insurance'),
    (7000, 8999, 'Services'),
    (9100, 9729, 'Public_Admin'),
    (9900, 9999, 'NonClassifiable')
]

class CompanySICLookup:
    def __init__(self, sic_csv_path):
        self.sic_path = sic_csv_path
        self.ticker_to_vec = {}
        self.sector_names = [d[2] for d in SIC_DIVISIONS]
        self.output_dim = len(SIC_DIVISIONS)
        
        self._load_data()
        
    def _get_sector_index(self, sic_code):
        try:
            code = int(sic_code)
        except:
            return -1
            
        for i, (start, end, name) in enumerate(SIC_DIVISIONS):
            if start <= code <= end:
                return i
        return -1

    def _load_data(self):
        if not os.path.exists(self.sic_path):
            print(f"Warning: SIC file not found at {self.sic_path}")
            return
            
        df = pd.read_csv(self.sic_path)
        # Expect columns: ticker, sic
        
        count = 0
        for _, row in df.iterrows():
            ticker = row['ticker']
            sic = row['sic']
            
            idx = self._get_sector_index(sic)
            
            # Create One-Hot Vector
            vec = np.zeros(self.output_dim, dtype=np.float32)
            if idx >= 0:
                vec[idx] = 1.0
                
            self.ticker_to_vec[ticker] = vec
            count += 1
            
        print(f"Company Data: Loaded SIC sectors for {count} companies.")

    def get_sic_vector(self, ticker):
        """Returns 11-dim one-hot vector for the sector."""
        return self.ticker_to_vec.get(ticker, np.zeros(self.output_dim, dtype=np.float32))

    def get_feature_names(self):
        return [f"Sector_{n}" for n in self.sector_names]


class CompanyFinancialsLookup:
    def __init__(self, financials_csv_path):
        self.fin_path = financials_csv_path
        # Changed structure: {ticker: {'timestamps': [t1, t2], 'data': [v1, v2]}}
        self.history = {} 
        self.fact_vocab = []
        self.fact_map = {}
        self.output_dim = 0
        
        self._load_data()
        
    def _load_data(self):
        if not os.path.exists(self.fin_path):
            print(f"Warning: Financials file not found at {self.fin_path}")
            return

        print("Loading Quarterly Financials (this may take a moment)...")
        df = pd.read_csv(self.fin_path)
        
        # 1. Build Vocabulary of "Facts" (Metrics)
        self.fact_vocab = sorted(df['Fact'].dropna().unique().tolist())
        self.fact_map = {f: i for i, f in enumerate(self.fact_vocab)}
        self.output_dim = len(self.fact_vocab)
        
        print(f"Financials: Found {self.output_dim} unique financial metrics.")
        
        # 2. Process Timeline per Company
        df['FiledDate'] = pd.to_datetime(df['FiledDate'])
        df = df.sort_values('FiledDate')
        
        grouped = df.groupby('Ticker')
        
        for ticker, group in grouped:
            # Sort by date
            group = group.sort_values('FiledDate')
            
            ts_list = []
            vec_list = []
            
            current_vec = np.zeros(self.output_dim, dtype=np.float32)
            
            # Group by Date to handle multiple facts updated on the same day
            for date, day_group in group.groupby('FiledDate'):
                ts = date.timestamp()
                
                has_update = False
                for _, row in day_group.iterrows():
                    fact = row['Fact']
                    val = row['Value']
                    
                    if fact in self.fact_map:
                        idx = self.fact_map[fact]
                        # Normalize using arcsinh (handles negatives, log-like behavior)
                        norm_val = np.arcsinh(float(val))
                        current_vec[idx] = norm_val
                        has_update = True
                
                if has_update:
                    ts_list.append(ts)
                    vec_list.append(current_vec.copy())
            
            self.history[ticker] = {
                'timestamps': ts_list,
                'data': vec_list
            }
            
        print(f"Financials: Processed history for {len(self.history)} companies.")

    def get_financial_vector(self, ticker, timestamp):
        """
        Returns the most recently filed financial vector before timestamp.
        """
        if ticker not in self.history:
            return np.zeros(self.output_dim, dtype=np.float32)
            
        record = self.history[ticker]
        timestamps = record['timestamps']
        vectors = record['data']
        
        # Bisect only on the float timestamps
        # bisect_right returns insertion point after x. 
        idx = bisect.bisect_right(timestamps, timestamp)
        
        if idx == 0:
            return np.zeros(self.output_dim, dtype=np.float32)
            
        return vectors[idx-1]

    def get_feature_names(self):
        return [f"Fin_{f}" for f in self.fact_vocab]