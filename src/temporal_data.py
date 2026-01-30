import torch
from torch_geometric.data import TemporalData
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    INCLUDE_POLITICIAN_BIO, INCLUDE_IDEOLOGY, INCLUDE_DISTRICT_ECON, INCLUDE_COMMITTEES,
    INCLUDE_COMPANY_SIC, INCLUDE_COMPANY_FINANCIALS,
    IDEOLOGY_PATH, DISTRICT_ECON_DIR, COMMITTEE_PATH,
    COMPANY_SIC_PATH, COMPANY_FIN_PATH, CONGRESS_TERMS_PATH,
    TX_PATH
)
from src.data_processing.feature_lookups import (
    TermLookup, PoliticianBioLookup, IdeologyLookup, DistrictEconLookup, 
    CommitteeLookup, CompanySICLookup, CompanyFinancialsLookup
)

class TemporalGraphBuilder:
    def __init__(self, transactions_df, min_freq=1):
        self.transactions = transactions_df.sort_values('Filed').reset_index(drop=True)
        self.min_freq = min_freq
        
        self.pol_id_map = {}
        self.company_id_map = {}
        
        self._build_mappings()
        
        # --- Initialize Feature Lookups ---
        self.lookups = {}
        self.term_lookup = None 
        self._init_feature_lookups()
        
    def _build_mappings(self):
        pols = self.transactions['BioGuideID'].unique()
        self.pol_id_map = {pid: i for i, pid in enumerate(pols)}
        print(f"Mapped {len(self.pol_id_map)} Politicians")
        
        ticker_counts = self.transactions['Ticker'].value_counts()
        valid_tickers = ticker_counts[ticker_counts >= self.min_freq].index
        self.company_id_map = {t: i for i, t in enumerate(valid_tickers)}
        print(f"Mapped {len(self.company_id_map)} Companies (min_freq={self.min_freq})")
            
    def _init_feature_lookups(self):
        print("Initializing Feature Lookups...")
        
        # Base Term Lookup
        if INCLUDE_POLITICIAN_BIO or INCLUDE_IDEOLOGY or INCLUDE_DISTRICT_ECON or INCLUDE_COMMITTEES:
            self.term_lookup = TermLookup(CONGRESS_TERMS_PATH)
        
        if INCLUDE_POLITICIAN_BIO:
            print("  - Loading Politician Bio Data...")
            self.lookups['bio'] = PoliticianBioLookup(CONGRESS_TERMS_PATH, self.term_lookup)

        if INCLUDE_IDEOLOGY:
            print("  - Loading Ideology Data...")
            self.lookups['ideology'] = IdeologyLookup(IDEOLOGY_PATH, self.term_lookup)
            
        if INCLUDE_DISTRICT_ECON:
            print("  - Loading District Econ Data...")
            self.lookups['econ'] = DistrictEconLookup(DISTRICT_ECON_DIR, self.term_lookup)
            
        if INCLUDE_COMMITTEES:
            print("  - Loading Committee Data...")
            self.lookups['committee'] = CommitteeLookup(COMMITTEE_PATH, self.term_lookup)
            
        if INCLUDE_COMPANY_SIC:
            print("  - Loading Company SIC Data...")
            self.lookups['sic'] = CompanySICLookup(COMPANY_SIC_PATH)
            
        if INCLUDE_COMPANY_FINANCIALS:
            print("  - Loading Company Financials...")
            self.lookups['financials'] = CompanyFinancialsLookup(COMPANY_FIN_PATH)

    def _parse_amount(self, amt_str):
        if pd.isna(amt_str): return 0.0
        clean = str(amt_str).replace('$','').replace(',','')
        try:
            if '-' in clean:
                low, high = clean.split('-')[:2]
                return (float(low) + float(high)) / 2
            return float(clean)
        except:
            return 0.0

    def process(self):
        src = []
        dst = []
        t = []
        
        # THREE separate feature lists
        msg_list = []      # Edge Features (Trade)
        x_pol_list = []    # Source Node Features (Politician)
        x_comp_list = []   # Dest Node Features (Company)
        
        y = []   
        resolution_t = []
        
        num_pols = len(self.pol_id_map)
        num_comps = len(self.company_id_map)
        total_nodes = num_pols + num_comps

        if len(self.transactions) > 0:
            base_time = pd.to_datetime(self.transactions['Filed'].min()).timestamp()
        else:
            base_time = 0
            
        skipped = 0
        self.transactions['Traded_DT'] = pd.to_datetime(self.transactions['Traded'])
        self.transactions['Filed_DT'] = pd.to_datetime(self.transactions['Filed'])
        
        # Load Price Sequences
        if os.path.exists("data/price_sequences.pt"):
            print("Loading Price Sequences...")
            price_map = torch.load("data/price_sequences.pt")
            if 'transaction_id' in self.transactions.columns:
                self.transactions['transaction_id'] = self.transactions['transaction_id'].fillna(-1).astype(int)
            else:
                price_map = {}
        else:
            price_map = {}
            
        price_seqs = [] 

        for _, row in tqdm(self.transactions.iterrows(), total=len(self.transactions), desc="Building Temporal Events"):
            pid = row['BioGuideID']
            ticker = row['Ticker']
            tid = row.get('transaction_id', -1)
            
            if pid not in self.pol_id_map or ticker not in self.company_id_map:
                skipped += 1
                continue
            
            if pd.isna(row['Filed_DT']):
                skipped += 1
                continue
                
            p_idx = self.pol_id_map[pid]
            c_idx = self.company_id_map[ticker] + len(self.pol_id_map)
            
            src.append(p_idx)
            dst.append(c_idx)
            
            event_ts = row['Filed_DT'].timestamp()
            ts_norm = event_ts - base_time
            t.append(int(ts_norm))
            
            # --- 1. EDGE FEATURES (Trade Only) ---
            amt = self._parse_amount(row['Trade_Size_USD'])
            amt_log = np.log1p(amt)
            
            tx_type = str(row.get('Transaction', '')).lower()
            is_buy = 0.0
            if 'purchase' in tx_type: is_buy = 1.0
            elif 'sale' in tx_type: is_buy = -1.0
            
            gap_days = max(0, (row['Filed_DT'] - row['Traded_DT']).days) if pd.notnull(row['Traded_DT']) else 30
            gap_feat = np.log1p(gap_days)
            
            # [Amount, Type, Gap] (Dim=3)
            msg_list.append([amt_log, is_buy, gap_feat])
            
            # --- 2. DYNAMIC NODE FEATURES (Politician) ---
            # Evaluated at event_ts (updates with Congress/Term changes)
            pol_vec = []
            if INCLUDE_POLITICIAN_BIO:
                pol_vec.extend(self.lookups['bio'].get_vector(pid, event_ts).tolist())
            if INCLUDE_IDEOLOGY:
                pol_vec.extend(self.lookups['ideology'].get_vector(pid, event_ts).tolist())
            if INCLUDE_DISTRICT_ECON:
                pol_vec.extend(self.lookups['econ'].get_vector(pid, event_ts).tolist())
            if INCLUDE_COMMITTEES:
                pol_vec.extend(self.lookups['committee'].get_vector(pid, event_ts).tolist())
            
            # If no features enabled, add dummy 1-dim
            if not pol_vec: pol_vec = [0.0]
            x_pol_list.append(pol_vec)

            # --- 3. DYNAMIC NODE FEATURES (Company) ---
            comp_vec = []
            if INCLUDE_COMPANY_SIC:
                comp_vec.extend(self.lookups['sic'].get_vector(ticker, event_ts).tolist())
            if INCLUDE_COMPANY_FINANCIALS:
                comp_vec.extend(self.lookups['financials'].get_vector(ticker, event_ts).tolist())
                
            if not comp_vec: comp_vec = [0.0]
            x_comp_list.append(comp_vec)
            
            # --- 4. Price & Labels ---
            if tid in price_map:
                p_feat = price_map[tid]
            else:
                p_feat = torch.zeros((14,), dtype=torch.float32)
            price_seqs.append(p_feat)
            
            horizons = ['1M', '2M', '3M', '6M', '8M', '12M', '18M', '24M']
            labels_multi = [row.get(f'Excess_Return_{h}', float('nan')) for h in horizons]
            y.append(labels_multi)
            
            if pd.notnull(row['Traded_DT']):
                trade_ts = row['Traded_DT'].timestamp() - base_time
            else:
                trade_ts = ts_norm - (30*86400)
            resolution_t.append(int(trade_ts))
            
        print(f"Skipped {skipped} transactions.")
        
        # Tensor Conversion
        msg_tensor = torch.tensor(msg_list, dtype=torch.float)
        x_pol_tensor = torch.tensor(x_pol_list, dtype=torch.float)
        x_comp_tensor = torch.tensor(x_comp_list, dtype=torch.float)
        
        print(f"Dimensions -> Msg: {msg_tensor.shape[1]}, Pol: {x_pol_tensor.shape[1]}, Comp: {x_comp_tensor.shape[1]}")
        
        data = TemporalData(
            src=torch.tensor(src, dtype=torch.long),
            dst=torch.tensor(dst, dtype=torch.long),
            t=torch.tensor(t, dtype=torch.long),
            msg=msg_tensor,
            y=torch.tensor(y, dtype=torch.float)
        )
        
        # Attach the Dynamic Node Features
        data.x_pol = x_pol_tensor
        data.x_comp = x_comp_tensor
        
        if price_seqs:
            data.price_seq = torch.stack(price_seqs)
        else:
            data.price_seq = torch.zeros((len(src), 14))
        
        data.trade_t = torch.tensor(resolution_t, dtype=torch.long)
        data.num_nodes = total_nodes
        data.num_parties = 0 # No longer needed for static lookup
        data.num_states = 0
        
        return data

if __name__ == "__main__":
    if os.path.exists(TX_PATH):
        df = pd.read_csv(TX_PATH)
        builder = TemporalGraphBuilder(df)
        data = builder.process()
        print(data)
        os.makedirs("data", exist_ok=True)
        torch.save(data, "data/temporal_data.pt")
    else:
        print(f"Transaction file not found at {TX_PATH}")