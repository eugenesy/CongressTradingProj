import torch
from torch_geometric.data import TemporalData
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle
import sys
from collections import defaultdict

# Import Configs for Feature Flags
from src.config import (
    INCLUDE_POLITICIAN_BIO, INCLUDE_IDEOLOGY, INCLUDE_DISTRICT_ECON, INCLUDE_COMMITTEES,
    INCLUDE_COMPANY_SIC, INCLUDE_COMPANY_FINANCIALS,
    IDEOLOGY_PATH, DISTRICT_ECON_DIR, COMMITTEE_PATH,
    COMPANY_SIC_PATH, COMPANY_FIN_PATH, CONGRESS_TERMS_PATH
)

# Assuming these exist in your project structure
from src.data_processing.feature_lookups import (
    TermLookup, PoliticianBioLookup, IdeologyLookup, DistrictEconLookup, 
    CommitteeLookup, CompanySICLookup, CompanyFinancialsLookup
)

class TemporalGraphBuilder:
    def __init__(self, transactions_df, min_freq=1):
        """
        Builds a TemporalData object from transactions.
        """
        # Sort by Filed Date
        self.transactions = transactions_df.sort_values('Filed').reset_index(drop=True)
        self.min_freq = min_freq
        
        # Mappings
        self.pol_id_map = {}
        self.company_id_map = {}
        # Mappers for static features (Base Model requirement)
        self.party_map = {'Unknown': 0}
        self.state_map = {'Unknown': 0}
        
        # Lookups for Dynamic Features (Enhanced Model requirement)
        self.lookups = {}
        self.term_lookup = None 
        
        self._build_mappings()
        self._init_feature_lookups()
        
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

        # 3. Static Mappers (For Base ResearchTGN)
        parties = self.transactions['Party'].fillna('Unknown').unique()
        for p in parties:
            if p not in self.party_map: self.party_map[p] = len(self.party_map)
            
        states = self.transactions['State'].fillna('Unknown').unique()
        for s in states:
            if s not in self.state_map: self.state_map[s] = len(self.state_map)

    def _init_feature_lookups(self):
        print("Initializing Feature Lookups...")
        # Only init if needed
        if INCLUDE_POLITICIAN_BIO or INCLUDE_IDEOLOGY or INCLUDE_DISTRICT_ECON or INCLUDE_COMMITTEES:
            self.term_lookup = TermLookup(CONGRESS_TERMS_PATH)
        
        if INCLUDE_POLITICIAN_BIO:
            self.lookups['bio'] = PoliticianBioLookup(CONGRESS_TERMS_PATH, self.term_lookup)
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
        # Simple cleanup
        clean = str(amt_str).replace('$','').replace(',','')
        try:
            if '-' in clean:
                low, high = clean.split('-')[:2]
                return (float(low) + float(high)) / 2
            return float(clean)
        except:
            return 0.0

    def process(self, min_freq=1):
        src = []
        dst = []
        t = []
        msg = [] # Edge Features
        y = []   # Labels
        resolution_t = []  # When each trade's label becomes known (Trade + 30d)
        
        # --- STORAGE FOR FEATURES ---
        # 1. Static (Base Model): [num_nodes, 2] -> (Party, State)
        num_pols = len(self.pol_id_map)
        num_comps = len(self.company_id_map)
        total_nodes = num_pols + num_comps
        
        x_static = torch.zeros((total_nodes, 2), dtype=torch.long)
        
        # 2. Dynamic (Enhanced Model): List of vectors to be stacked later
        x_pol_list = []
        x_comp_list = []
        
        # Pre-fill Static Features for Politicians
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
            base_time = pd.to_datetime(self.transactions['Filed'].min()).timestamp()
        else:
            base_time = 0
            
        skipped = 0
        
        self.transactions['Traded_DT'] = pd.to_datetime(self.transactions['Traded'])
        self.transactions['Filed_DT'] = pd.to_datetime(self.transactions['Filed'])
        
        # --- PRICE SEQUENCE LOADING (New Logic) ---
        if os.path.exists("data/price_sequences.pt"):
            print("Loading Price Sequences...")
            price_map = torch.load("data/price_sequences.pt")
            
            # Filter DataFrame based on Price Map availability
            print(f"Filtering transactions... Original: {len(self.transactions)}")
            valid_ids = set(price_map.keys())
            
            self.transactions['transaction_id'] = self.transactions['transaction_id'].fillna(-1).astype(int)
            mask = self.transactions['transaction_id'].isin(valid_ids)
            self.transactions = self.transactions[mask].reset_index(drop=True)
            
            print(f"Filtered: {len(self.transactions)} (Removed {len(mask)-mask.sum()} missing price seqs)")
            
            clean_path = "data/processed/ml_dataset_clean.csv"
            self.transactions.to_csv(clean_path, index=False)
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
            
            # Time
            if pd.isna(row['Filed_DT']):
                skipped += 1
                continue
            
            # Use raw timestamp for lookups, normalized for TGN
            event_ts_raw = row['Filed_DT'].timestamp()
            ts_norm = int(event_ts_raw - base_time)
            t.append(ts_norm)
            
            # --- Edge Features ---
            amt = self._parse_amount(row['Trade_Size_USD'])
            amt_log = np.log1p(amt)
            is_buy = 1.0 if 'Purchase' in str(row['Transaction']) else -1.0
            
            if pd.notnull(row['Filed_DT']) and pd.notnull(row['Traded_DT']):
                gap_days = (row['Filed_DT'] - row['Traded_DT']).days
                gap_days = max(0, gap_days)
            else:
                gap_days = 30
            gap_feat = np.log1p(gap_days)
            
            msg.append([amt_log, is_buy, gap_feat])
            
            # --- Dynamic Node Features (Your Custom Logic) ---
            # 1. Politician
            pol_vec = []
            if INCLUDE_POLITICIAN_BIO: pol_vec.extend(self.lookups['bio'].get_vector(pid, event_ts_raw).tolist())
            if INCLUDE_IDEOLOGY: pol_vec.extend(self.lookups['ideology'].get_vector(pid, event_ts_raw).tolist())
            if INCLUDE_DISTRICT_ECON: pol_vec.extend(self.lookups['econ'].get_vector(pid, event_ts_raw).tolist())
            if INCLUDE_COMMITTEES: pol_vec.extend(self.lookups['committee'].get_vector(pid, event_ts_raw).tolist())
            if not pol_vec: pol_vec = [0.0]
            x_pol_list.append(pol_vec)

            # 2. Company
            comp_vec = []
            if INCLUDE_COMPANY_SIC: comp_vec.extend(self.lookups['sic'].get_vector(ticker, event_ts_raw).tolist())
            if INCLUDE_COMPANY_FINANCIALS: comp_vec.extend(self.lookups['financials'].get_vector(ticker, event_ts_raw).tolist())
            if not comp_vec: comp_vec = [0.0]
            x_comp_list.append(comp_vec)

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
                val = row.get(f'Excess_Return_{h}', float('nan'))
                labels_multi.append(val)
            y.append(labels_multi)
            
            # Trade Time for Resolution
            if pd.notnull(row['Traded_DT']):
                trade_ts = row['Traded_DT'].timestamp() - base_time
            else:
                trade_ts = ts_norm - (30*86400)
            resolution_t.append(int(trade_ts))
            
        print(f"Skipped {skipped} transactions due to rare tickers/missing IDs.")
        
        data = TemporalData(
            src=torch.tensor(src, dtype=torch.long),
            dst=torch.tensor(dst, dtype=torch.long),
            t=torch.tensor(t, dtype=torch.long),
            msg=torch.tensor(msg, dtype=torch.float),
            y=torch.tensor(y, dtype=torch.float)
        )
        
        # Add price features
        if price_seqs:
            data.price_seq = torch.stack(price_seqs)
        else:
            data.price_seq = torch.zeros((len(src), 14))
        
        data.trade_t = torch.tensor(resolution_t, dtype=torch.long)
        
        # Meta info
        data.num_nodes = total_nodes
        data.num_parties = len(self.party_map)
        data.num_states = len(self.state_map)
        
        # Attach Features: Both Static (Base) and Dynamic (Enhanced)
        data.x_static = x_static
        data.x_pol = torch.tensor(x_pol_list, dtype=torch.float)
        data.x_comp = torch.tensor(x_comp_list, dtype=torch.float)
        
        return data

if __name__ == "__main__":
    # Hack for script execution
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.config import TX_PATH
    
    if os.path.exists(TX_PATH):
        df = pd.read_csv(TX_PATH)
        builder = TemporalGraphBuilder(df)
        data = builder.process()
        print(data)
        os.makedirs("data", exist_ok=True)
        torch.save(data, "data/temporal_data.pt")