import torch
from torch_geometric.data import TemporalData
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle
from collections import defaultdict

# Import Config and Lookups
import sys
# Ensure src is in path
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    INCLUDE_IDEOLOGY, INCLUDE_DISTRICT_ECON, INCLUDE_COMMITTEES,
    INCLUDE_COMPANY_SIC, INCLUDE_COMPANY_FINANCIALS,
    IDEOLOGY_PATH, DISTRICT_ECON_DIR, COMMITTEE_PATH,
    COMPANY_SIC_PATH, COMPANY_FIN_PATH, CONGRESS_TERMS_PATH
)
from src.data_processing.feature_lookups import (
    TermLookup, IdeologyLookup, DistrictEconLookup, CommitteeLookup,
    CompanySICLookup, CompanyFinancialsLookup
)

class TemporalGraphBuilder:
    def __init__(self, transactions_df, min_freq=1):
        """
        Builds a TemporalData object from transactions.
        """
        self.transactions = transactions_df.sort_values('Filed').reset_index(drop=True)
        self.min_freq = min_freq
        
        self.pol_id_map = {}
        self.company_id_map = {}
        self.party_map = {'Unknown': 0}
        self.state_map = {'Unknown': 0}
        
        self.pol_history = defaultdict(list)
        
        self._build_mappings()
        
        # --- Initialize Feature Lookups ---
        self.lookups = {}
        self._init_feature_lookups()
        
    def _build_mappings(self):
        pols = self.transactions['BioGuideID'].unique()
        self.pol_id_map = {pid: i for i, pid in enumerate(pols)}
        print(f"Mapped {len(self.pol_id_map)} Politicians")
        
        ticker_counts = self.transactions['Ticker'].value_counts()
        valid_tickers = ticker_counts[ticker_counts >= self.min_freq].index
        self.company_id_map = {t: i for i, t in enumerate(valid_tickers)}
        print(f"Mapped {len(self.company_id_map)} Companies (min_freq={self.min_freq})")

        parties = self.transactions['Party'].fillna('Unknown').unique()
        for p in parties:
            if p not in self.party_map: self.party_map[p] = len(self.party_map)
            
        states = self.transactions['State'].fillna('Unknown').unique()
        for s in states:
            if s not in self.state_map: self.state_map[s] = len(self.state_map)
            
    def _init_feature_lookups(self):
        """Initialize helper classes based on config flags."""
        print("Initializing Feature Lookups...")
        
        # Base Term Lookup (Needed for Geography-based features)
        if INCLUDE_IDEOLOGY or INCLUDE_DISTRICT_ECON or INCLUDE_COMMITTEES:
            self.term_lookup = TermLookup(CONGRESS_TERMS_PATH)
        
        if INCLUDE_IDEOLOGY:
            self.lookups['ideology'] = IdeologyLookup(IDEOLOGY_PATH, self.term_lookup)
            
        if INCLUDE_DISTRICT_ECON:
            self.lookups['econ'] = DistrictEconLookup(DISTRICT_ECON_DIR, self.term_lookup)
            
        if INCLUDE_COMMITTEES:
            self.lookups['committee'] = CommitteeLookup(COMMITTEE_PATH, self.term_lookup)
            
        if INCLUDE_COMPANY_SIC:
            self.lookups['sic'] = CompanySICLookup(COMPANY_SIC_PATH)
            
        if INCLUDE_COMPANY_FINANCIALS:
            self.lookups['financials'] = CompanyFinancialsLookup(COMPANY_FIN_PATH)

    def _parse_amount(self, amt_str):
        if pd.isna(amt_str): return 0.0
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
        resolution_t = []
        
        # Static Features
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
                party = row.get('Party', 'Unknown')
                state = row.get('State', 'Unknown')
                x_static[idx] = torch.tensor([self.party_map.get(party, 0), self.state_map.get(state, 0)])

        # Base Timestamp
        if len(self.transactions) > 0:
            base_time = pd.to_datetime(self.transactions['Filed'].min()).timestamp()
        else:
            base_time = 0
            
        skipped = 0
        self.transactions['Traded_DT'] = pd.to_datetime(self.transactions['Traded'])
        self.transactions['Filed_DT'] = pd.to_datetime(self.transactions['Filed'])
        
        # Load Price Sequences if available
        if os.path.exists("data/price_sequences.pt"):
            print("Loading Price Sequences...")
            price_map = torch.load("data/price_sequences.pt")
            valid_ids = set(price_map.keys())
            self.transactions['transaction_id'] = self.transactions['transaction_id'].fillna(-1).astype(int)
            mask = self.transactions['transaction_id'].isin(valid_ids)
            self.transactions = self.transactions[mask].reset_index(drop=True)
            print(f"Filtered: {len(self.transactions)}")
            self.transactions.to_csv("data/processed/ml_dataset_clean.csv", index=False)
        else:
            print("Warning: data/price_sequences.pt not found. Using zero sequences.")
            price_map = {}
            
        price_seqs = [] 

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
            
            if pd.isna(row['Filed_DT']):
                skipped += 1
                continue
                
            # Event Time
            event_ts = row['Filed_DT'].timestamp()
            ts_norm = event_ts - base_time
            t.append(int(ts_norm))
            
            # --- Base Edge Features ---
            amt = np.log1p(self._parse_amount(row['Trade_Size_USD']))
            is_buy = 1.0 if 'Purchase' in str(row['Transaction']) else -1.0
            gap_days = max(0, (row['Filed_DT'] - row['Traded_DT']).days) if pd.notnull(row['Traded_DT']) else 30
            gap_feat = np.log1p(gap_days)
            
            feat_vec = [amt, is_buy, gap_feat]
            
            # --- Append Configurable Features ---
            # IMPORTANT: We pass 'event_ts' (absolute timestamp) to lookups, not normalized 'ts_norm'
            
            # 1. Ideology (Politician)
            if INCLUDE_IDEOLOGY:
                vec = self.lookups['ideology'].get_vector(pid, event_ts)
                feat_vec.extend(vec.tolist())
                
            # 2. District Econ (Politician)
            if INCLUDE_DISTRICT_ECON:
                vec = self.lookups['econ'].get_vector(pid, event_ts)
                feat_vec.extend(vec.tolist())

            # 3. Committees (Politician)
            if INCLUDE_COMMITTEES:
                vec = self.lookups['committee'].get_vector(pid, event_ts)
                feat_vec.extend(vec.tolist())
                
            # 4. SIC (Company)
            if INCLUDE_COMPANY_SIC:
                vec = self.lookups['sic'].get_vector(ticker, event_ts)
                feat_vec.extend(vec.tolist())
                
            # 5. Financials (Company)
            if INCLUDE_COMPANY_FINANCIALS:
                vec = self.lookups['financials'].get_vector(ticker, event_ts)
                feat_vec.extend(vec.tolist())
            
            msg.append(feat_vec)
            
            # --- Price Features ---
            if tid in price_map:
                p_feat = price_map[tid]
            else:
                p_feat = torch.zeros((14,), dtype=torch.float32)
            price_seqs.append(p_feat)
            
            # --- Labels ---
            horizons = ['1M', '2M', '3M', '6M', '8M', '12M', '18M', '24M']
            labels_multi = []
            for h in horizons:
                labels_multi.append(row.get(f'Excess_Return_{h}', float('nan')))
            y.append(labels_multi)
            
            if pd.notnull(row['Traded_DT']):
                trade_ts = row['Traded_DT'].timestamp() - base_time
            else:
                trade_ts = ts_norm - (30*86400)
            resolution_t.append(int(trade_ts))
            
        print(f"Skipped {skipped} transactions.")
        
        # Convert msg to tensor
        msg_tensor = torch.tensor(msg, dtype=torch.float)
        print(f"Final Message Dimension: {msg_tensor.shape[1]}")
        
        data = TemporalData(
            src=torch.tensor(src, dtype=torch.long),
            dst=torch.tensor(dst, dtype=torch.long),
            t=torch.tensor(t, dtype=torch.long),
            msg=msg_tensor,
            y=torch.tensor(y, dtype=torch.float)
        )
        
        if price_seqs:
            data.price_seq = torch.stack(price_seqs)
        else:
            data.price_seq = torch.zeros((len(src), 14))
        
        data.trade_t = torch.tensor(resolution_t, dtype=torch.long)
        data.num_nodes = total_nodes
        data.x_static = x_static
        
        return data

if __name__ == "__main__":
    from src.config import TX_PATH
    if os.path.exists(TX_PATH):
        df = pd.read_csv(TX_PATH)
        builder = TemporalGraphBuilder(df)
        data = builder.process()
        print(data)
        os.makedirs("data", exist_ok=True)
        torch.save(data, "data/temporal_data.pt")
    else:
        print(f"Transaction file not found at {TX_PATH}")