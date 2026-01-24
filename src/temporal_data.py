import torch
from torch_geometric.data import TemporalData
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle

from collections import defaultdict

class TemporalGraphBuilder:
    def __init__(self, transactions_df, min_freq=5):
        """
        Builds a TemporalData object from transactions.
        """
        # Sort by Filed Date
        self.transactions = transactions_df.sort_values('Filed').reset_index(drop=True)
        self.min_freq = min_freq
        
        # Mappings
        self.pol_id_map = {}
        self.company_id_map = {}
        # Mappers for static features
        self.party_map = {'Unknown': 0}
        self.state_map = {'Unknown': 0}
        
        # History Tracking for "Realized Win Rate" Feature
        # Key: BioGuideID -> List of (Resolution_Timestamp, Label)
        self.pol_history = defaultdict(list)
        
        self._build_mappings()
        
    def _build_mappings(self):
        # 1. Politicians
        pols = self.transactions['BioGuideID'].unique()
        self.pol_id_map = {pid: i for i, pid in enumerate(pols)}
        print(f"Mapped {len(self.pol_id_map)} Politicians")
        
        # 2. Companies (Filter rare tickers to reduce noise/memory)
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
        # Simple cleanup
        clean = str(amt_str).replace('$','').replace(',','')
        try:
            return float(clean)
        except:
            return 0.0

    def process(self):
        src = []
        dst = []
        t = []
        msg = [] # Edge Features
        y = []   # Labels
        resolution_t = []  # When each trade's label becomes known (Trade + 30d)
        
        # Static Features: [num_nodes, 2] -> (Party, State)
        # Initialize with 0
        num_pols = len(self.pol_id_map)
        num_comps = len(self.company_id_map)
        total_nodes = num_pols + num_comps
        
        x_static = torch.zeros((total_nodes, 2), dtype=torch.long)
        
        # Pre-fill Static Features for Politicians
        # (Companies remain 0,0)
        pol_meta = self.transactions.drop_duplicates('BioGuideID').set_index('BioGuideID')
        
        for pid, idx in self.pol_id_map.items():
            if pid in pol_meta.index:
                row = pol_meta.loc[pid]
                if isinstance(row, pd.DataFrame): row = row.iloc[0] # Handle duplicates
                
                party = row.get('Party', 'Unknown')
                state = row.get('State', 'Unknown')
                
                p_code = self.party_map.get(party, 0)
                s_code = self.state_map.get(state, 0)
                
                x_static[idx] = torch.tensor([p_code, s_code])

        
        # Base Timestamp (for normalization)
        if len(self.transactions) > 0:
            # Use Filed date for base time
            base_time = pd.to_datetime(self.transactions['Filed'].min()).timestamp()
        else:
            base_time = 0
            
        skipped = 0
        
        # Use datetime objects for Gap calculation
        self.transactions['Traded_DT'] = pd.to_datetime(self.transactions['Traded'])
        self.transactions['Filed_DT'] = pd.to_datetime(self.transactions['Filed'])
        
        if os.path.exists("data/price_sequences.pt"):
            print("Loading Price Sequences...")
            price_map = torch.load("data/price_sequences.pt")
        else:
            print("Warning: data/price_sequences.pt not found. Using zero sequences.")
            price_map = {}
            
        price_seqs = [] # New feature list

        for _, row in tqdm(self.transactions.iterrows(), total=len(self.transactions), desc="Building Temporal Events"):
            pid = row['BioGuideID']
            ticker = row['Ticker']
            tid = row.get('transaction_id', -1) # Ensure we access ID
            
            if pid not in self.pol_id_map or ticker not in self.company_id_map:
                skipped += 1
                continue
            
            p_idx = self.pol_id_map[pid]
            c_idx = self.company_id_map[ticker] + len(self.pol_id_map)
            
            src.append(p_idx)
            dst.append(c_idx)
            
            # Time (Seconds since start)
            # Use Filed Date as the event time
            if pd.isna(row['Filed_DT']):
                skipped += 1
                continue
            ts = row['Filed_DT'].timestamp() - base_time
            t.append(int(ts))
            
            # --- Edge Features ---
            # 1. Amount
            amt = self._parse_amount(row['Trade_Size_USD'])
            amt = np.log1p(amt)
            
            # 2. Buy/Sell
            is_buy = 1.0 if 'Purchase' in str(row['Transaction']) else -1.0
            
            # 3. Filing Gap (Days)
            # Gap is still Filed - Traded
            if pd.notnull(row['Filed_DT']) and pd.notnull(row['Traded_DT']):
                gap_days = (row['Filed_DT'] - row['Traded_DT']).days
                gap_days = max(0, gap_days)
            else:
                gap_days = 30
            gap_feat = np.log1p(gap_days)
            
            # Edge features: [Amount, Is_Buy, Filing_Gap]
            # Label will be added DYNAMICALLY during training (with temporal masking)
            msg.append([amt, is_buy, gap_feat])
            
            # --- Price Features (Engineered) ---
            # Get 14-dim feature vector (7 Stock + 7 SPY features)
            # Default to zeros if missing
            if tid in price_map:
                p_feat = price_map[tid] # Tensor (14,)
            else:
                p_feat = torch.zeros((14,), dtype=torch.float32)
            price_seqs.append(p_feat)
            
            # Label
            lbl = row.get('Label_1M', 0.0)
            if pd.isna(lbl): lbl = 0.0
            y.append(lbl)
            
            # Resolution Time (when this trade's label becomes "known")
            # Label_1M = 1-month forward return, so resolution = Trade + 30 days
            resolution_ts = (row['Traded_DT'] + pd.Timedelta(days=30)).timestamp() - base_time
            resolution_t.append(int(resolution_ts))
            
        print(f"Skipped {skipped} transactions due to rare tickers/missing IDs.")
        
        data = TemporalData(
            src=torch.tensor(src, dtype=torch.long),
            dst=torch.tensor(dst, dtype=torch.long),
            t=torch.tensor(t, dtype=torch.long),
            msg=torch.tensor(msg, dtype=torch.float),
            y=torch.tensor(y, dtype=torch.float)
        )
        
        # Add price features (engineered)
        if price_seqs:
            data.price_seq = torch.stack(price_seqs) # (Num_Events, 14)
        else:
            # Fallback for empty (shouldn't happen if skip logic holds)
            data.price_seq = torch.zeros((len(src), 14))
        
        # Add resolution_t as separate attribute (for dynamic label masking)
        data.resolution_t = torch.tensor(resolution_t, dtype=torch.long)
        
        # Meta info
        data.num_nodes = total_nodes
        data.x_static = x_static
        data.num_parties = len(self.party_map)
        data.num_states = len(self.state_map)
        
        return data

if __name__ == "__main__":
    # Test
    # from src.config import TX_PATH # Need to ensure src is in path or relative import
    # Hack for script execution
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.config import TX_PATH
    
    df = pd.read_csv(TX_PATH)
    builder = TemporalGraphBuilder(df)
    data = builder.process()
    print(data)
    os.makedirs("data", exist_ok=True)
    torch.save(data, "data/temporal_data.pt")
