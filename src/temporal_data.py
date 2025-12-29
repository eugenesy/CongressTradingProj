import torch
from torch_geometric.data import TemporalData
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
from collections import defaultdict

# ==========================================
#              IMPORTS & SETUP
# ==========================================
# Add parent directory to path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    TX_PATH, 
    LABEL_COL, 
    LABEL_LOOKAHEAD_DAYS, 
    MIN_TICKER_FREQ
)

# ==========================================
#              CONFIGURATION
# ==========================================
# We reconstruct the dictionary here so the class structure 
# (which relies on self.config['KEY']) remains unchanged.
CONFIG = {
    'LABEL_COL': LABEL_COL,
    'LABEL_LOOKAHEAD_DAYS': LABEL_LOOKAHEAD_DAYS,
    'MIN_TICKER_FREQ': MIN_TICKER_FREQ,
}
# ==========================================

class TemporalGraphBuilder:
    def __init__(self, transactions_df, config):
        """
        Builds a TemporalData object from transactions based on config.
        """
        self.config = config
        self.transactions = transactions_df.copy()
        
        # Sort by time to ensure chronological processing
        self.transactions['Traded_DT'] = pd.to_datetime(self.transactions['Traded'])
        self.transactions = self.transactions.sort_values('Traded').reset_index(drop=True)
        
        self.min_freq = self.config['MIN_TICKER_FREQ']
        
        # Mappings
        self.pol_id_map = {}
        self.company_id_map = {}
        self.party_map = {'Unknown': 0}
        self.state_map = {'Unknown': 0}
        self.label_map = {}  # Stores String -> Int mapping for categorical labels
        
        self._build_mappings()
        self._prepare_label_mapping()
        
    def _prepare_label_mapping(self):
        """
        Handles categorical labels (strings) by creating a mapping to integers.
        """
        label_col = self.config['LABEL_COL']
        
        # Check if the label column is numeric or categorical
        if not np.issubdtype(self.transactions[label_col].dtype, np.number):
            print(f"Detected categorical label: {label_col}. Building map...")
            
            # Explicit order for return bins (Worst -> Best)
            known_order = [
                "Below -16%", "-16% to -12%", "-12% to -8%", "-8% to -4%", "-4% to 0%",
                "0% to 4%", "4% to 8%", "8% to 12%", "12% to 16%", "16%+"
            ]
            
            unique_labels = self.transactions[label_col].dropna().unique()
            
            # Check if our known_order covers the data found
            if set(unique_labels).issubset(set(known_order)):
                self.label_map = {label: i for i, label in enumerate(known_order)}
            else:
                print("Warning: Unknown string categories. Using alphabetical sort.")
                self.label_map = {label: i for i, label in enumerate(sorted(unique_labels))}
            
            print(f"Label Mapping Created: {self.label_map}")

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

        # MAIN LOOP
        label_col = self.config['LABEL_COL']
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
            
            # --- LABEL HANDLING ---
            raw_lbl = row.get(label_col, None)
            
            if self.label_map:
                # If categorical, map string -> int
                if raw_lbl in self.label_map:
                    y.append(float(self.label_map[raw_lbl]))
                else:
                    y.append(-1.0) # Unknown class
            else:
                # If numeric, keep as float
                val = float(raw_lbl) if (pd.notnull(raw_lbl) and raw_lbl != '') else 0.0
                y.append(val)
            
            # Resolution Time (Masking Logic)
            # This is intrinsic to the label: if we predict 1M returns, 
            # we MUST mask for 30 days regardless of when we run the training.
            resolution_ts = (row['Traded_DT'] + pd.Timedelta(days=lookahead_days)).timestamp() - base_time
            resolution_t.append(int(resolution_ts))
            
        print(f"Skipped {skipped} transactions.")
        
        data = TemporalData(
            src=torch.tensor(src, dtype=torch.long),
            dst=torch.tensor(dst, dtype=torch.long),
            t=torch.tensor(t, dtype=torch.long),
            msg=torch.tensor(msg, dtype=torch.float),
            y=torch.tensor(y, dtype=torch.long if self.label_map else torch.float)
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
        
        # Save map for decoding later
        if self.label_map:
            data.label_map = self.label_map
        
        return data

if __name__ == "__main__":
    # Note: sys.path is already updated at the top of the file
    df = pd.read_csv(TX_PATH)
    
    builder = TemporalGraphBuilder(df, CONFIG)
    data = builder.process()
    
    print(data)
    if hasattr(data, 'label_map'):
        print(f"Classes: {len(data.label_map)}")
        
    os.makedirs("data", exist_ok=True)
    torch.save(data, "data/temporal_data.pt")