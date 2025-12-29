import torch
from torch_geometric.data import TemporalData
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys

# ==========================================
#              IMPORTS & SETUP
# ==========================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config_multiple_binary_labels import (
    TX_PATH, 
    TARGET_COLUMNS,          # New Import
    LABEL_LOOKAHEAD_DAYS, 
    MIN_TICKER_FREQ
)

# ==========================================
#              CONFIGURATION
# ==========================================
CONFIG = {
    'TARGET_COLUMNS': TARGET_COLUMNS,
    'LABEL_LOOKAHEAD_DAYS': LABEL_LOOKAHEAD_DAYS,
    'MIN_TICKER_FREQ': MIN_TICKER_FREQ,
}

class TemporalGraphBuilder:
    def __init__(self, transactions_df, config):
        self.config = config
        self.transactions = transactions_df.copy()
        
        # Sort by time
        self.transactions['Traded_DT'] = pd.to_datetime(self.transactions['Traded'])
        self.transactions = self.transactions.sort_values('Traded').reset_index(drop=True)
        
        self.min_freq = self.config['MIN_TICKER_FREQ']
        
        # Mappings
        self.pol_id_map = {}
        self.company_id_map = {}
        self.party_map = {'Unknown': 0}
        self.state_map = {'Unknown': 0}
        
        self._build_mappings()
        
    def _build_mappings(self):
        # 1. Politicians
        pols = self.transactions['BioGuideID'].unique()
        self.pol_id_map = {pid: i for i, pid in enumerate(pols)}
        print(f"Mapped {len(self.pol_id_map)} Politicians")
        
        # 2. Companies (Filter rare tickers)
        ticker_counts = self.transactions['Ticker'].value_counts()
        valid_tickers = ticker_counts[ticker_counts >= self.min_freq].index
        self.company_id_map = {t: i for i, t in enumerate(valid_tickers)}
        print(f"Mapped {len(self.company_id_map)} Companies (min_freq={self.min_freq})")

        # 3. Static Mappers
        parties = self.transactions['Party'].fillna('Unknown').unique()
        for p in parties:
            if p not in self.party_map: self.party_map[p] = len(self.party_map)
            
        states = self.transactions['State'].fillna('Unknown').unique()
        for s in states:
            if s not in self.state_map: self.state_map[s] = len(self.state_map)
            
    def _parse_amount(self, amt_str):
        if pd.isna(amt_str): return 0.0
        clean = str(amt_str).replace('$','').replace(',','')
        try:
            return float(clean)
        except:
            return 0.0

    def process(self):
        src, dst, t, msg, y = [], [], [], [], []
        resolution_t = []
        
        # Static Features setup
        num_pols = len(self.pol_id_map)
        num_comps = len(self.company_id_map)
        total_nodes = num_pols + num_comps
        x_static = torch.zeros((total_nodes, 2), dtype=torch.long)
        
        # Pre-fill Static Features
        pol_meta = self.transactions.drop_duplicates('BioGuideID').set_index('BioGuideID')
        for pid, idx in self.pol_id_map.items():
            if pid in pol_meta.index:
                row = pol_meta.loc[pid]
                if isinstance(row, pd.DataFrame): row = row.iloc[0]
                p_code = self.party_map.get(row.get('Party', 'Unknown'), 0)
                s_code = self.state_map.get(row.get('State', 'Unknown'), 0)
                x_static[idx] = torch.tensor([p_code, s_code])

        # Base Timestamp
        if len(self.transactions) > 0:
            base_time = self.transactions['Traded_DT'].min().timestamp()
        else:
            base_time = 0
            
        skipped = 0
        self.transactions['Filed_DT'] = pd.to_datetime(self.transactions['Filed'])
        
        # Load Price Sequences
        if os.path.exists("data/price_sequences.pt"):
            print("Loading Price Sequences...")
            price_map = torch.load("data/price_sequences.pt")
        else:
            print("Warning: data/price_sequences.pt not found. Using zero sequences.")
            price_map = {}
            
        price_seqs = []
        target_cols = self.config['TARGET_COLUMNS']
        
        # Verify columns exist
        missing_cols = [c for c in target_cols if c not in self.transactions.columns]
        if missing_cols:
            raise ValueError(f"Missing target columns in CSV: {missing_cols}")

        lookahead_days = self.config['LABEL_LOOKAHEAD_DAYS']
        
        for _, row in tqdm(self.transactions.iterrows(), total=len(self.transactions), desc="Building Temporal Events"):
            pid = row['BioGuideID']
            ticker = row['Ticker']
            tid = row.get('transaction_id', -1)
            
            if pid not in self.pol_id_map or ticker not in self.company_id_map:
                skipped += 1
                continue
            
            p_idx = self.pol_id_map[pid]
            c_idx = self.company_id_map[ticker] + len(self.pol_id_map)
            
            src.append(p_idx)
            dst.append(c_idx)
            
            # Time
            if pd.isna(row['Traded_DT']): continue
            ts = row['Traded_DT'].timestamp() - base_time
            t.append(int(ts))
            
            # Features
            amt = np.log1p(self._parse_amount(row['Trade_Size_USD']))
            is_buy = 1.0 if 'Purchase' in str(row['Transaction']) else -1.0
            
            if pd.notnull(row['Filed_DT']):
                gap_days = max(0, (row['Filed_DT'] - row['Traded_DT']).days)
            else:
                gap_days = 30
            gap_feat = np.log1p(gap_days)
            msg.append([amt, is_buy, gap_feat])
            
            # Price Feature
            if tid in price_map:
                p_feat = price_map[tid]
            else:
                p_feat = torch.zeros((14,), dtype=torch.float32)
            price_seqs.append(p_feat)
            
            # --- STRUCTURED LABEL HANDLING ---
            # We treat the problem as classification into N+1 disjoint intervals.
            # We assume the CSV columns are binary flags (0.0 or 1.0).
            # If we sum them, we get the index of the interval.
            # Example: [>0, >4, >8]. Actual return 6%. Flags: [1, 1, 0]. Sum: 2.
            # Class 2 represents "Between 4 and 8".
            
            # Extract flags for this row
            # Infer objects to proper types first to avoid FutureWarning
            flags = row[target_cols].infer_objects(copy=False).fillna(0.0).values.astype(int)
            
            # The class index is simply the sum of flags.
            # Note: This relies on the columns being monotonic and sorted in config.
            label_idx = int(np.sum(flags))
            y.append(label_idx)
            
            # Resolution Time
            resolution_ts = (row['Traded_DT'] + pd.Timedelta(days=lookahead_days)).timestamp() - base_time
            resolution_t.append(int(resolution_ts))
            
        print(f"Skipped {skipped} transactions.")
        
        # Calculate Number of Classes for the Model
        # If we have N thresholds, we have N+1 possible intervals (0 to N).
        num_classes = len(target_cols) + 1
        
        data = TemporalData(
            src=torch.tensor(src, dtype=torch.long),
            dst=torch.tensor(dst, dtype=torch.long),
            t=torch.tensor(t, dtype=torch.long),
            msg=torch.tensor(msg, dtype=torch.float),
            y=torch.tensor(y, dtype=torch.long) # LongTensor for Class Index
        )
        
        if price_seqs:
            data.price_seq = torch.stack(price_seqs)
        else:
            data.price_seq = torch.zeros((len(src), 14))
        
        data.resolution_t = torch.tensor(resolution_t, dtype=torch.long)
        data.num_nodes = total_nodes
        data.x_static = x_static
        data.num_parties = len(self.party_map)
        data.num_states = len(self.state_map)
        
        # Save metadata so the model knows output size
        data.num_classes = num_classes
        
        return data

if __name__ == "__main__":
    df = pd.read_csv(TX_PATH)
    builder = TemporalGraphBuilder(df, CONFIG)
    data = builder.process()
    
    print(data)
    print(f"Task: Classification with {data.num_classes} intervals.")
        
    os.makedirs("data", exist_ok=True)
    torch.save(data, "data/temporal_data.pt")