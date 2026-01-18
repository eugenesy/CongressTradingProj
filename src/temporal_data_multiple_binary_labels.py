import torch
from torch_geometric.data import TemporalData
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
import bisect

# ==========================================
#              IMPORTS & SETUP
# ==========================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config_multiple_binary_labels import (
    TX_PATH, 
    TARGET_COLUMNS,
    LABEL_LOOKAHEAD_DAYS, 
    MIN_TICKER_FREQ,
    RAW_DATA_DIR,
    # Flags
    INCLUDE_IDEOLOGY,
    INCLUDE_DISTRICT_ECON,
    INCLUDE_COMMITTEES,
    INCLUDE_COMPANY_SIC,        
    INCLUDE_COMPANY_FINANCIALS, 
    # Paths
    COMMITTEE_PATH,
    SIC_PATH,                   
    FINANCIALS_PATH             
)

# Import Lookups
from src.district_economic_data import DistrictEconomicLookup
from src.committee_data import CommitteeLookup
from src.company_data import CompanySICLookup, CompanyFinancialsLookup

# ==========================================
#              CONFIGURATION
# ==========================================
CONFIG = {
    'TX_PATH': TX_PATH,
    'IDEOLOGY_PATH': 'data/raw/ideology_scores_quarterly.csv',
    'MEMBERS_PATH': 'data/raw/HSall_members.csv',
    'RAW_DATA_DIR': RAW_DATA_DIR,
    'COMMITTEE_PATH': COMMITTEE_PATH,
    'SIC_PATH': SIC_PATH,                 
    'FINANCIALS_PATH': FINANCIALS_PATH,   
    
    'TARGET_COLUMNS': TARGET_COLUMNS,
    'LABEL_LOOKAHEAD_DAYS': LABEL_LOOKAHEAD_DAYS,
    'MIN_TICKER_FREQ': MIN_TICKER_FREQ,
    
    'INCLUDE_IDEOLOGY': INCLUDE_IDEOLOGY,
    'INCLUDE_DISTRICT_ECON': INCLUDE_DISTRICT_ECON,
    'INCLUDE_COMMITTEES': INCLUDE_COMMITTEES,
    'INCLUDE_COMPANY_SIC': INCLUDE_COMPANY_SIC,               
    'INCLUDE_COMPANY_FINANCIALS': INCLUDE_COMPANY_FINANCIALS, 
}

class IdeologyLookup:
    """
    Manages temporal lookups for politician ideology scores.
    """
    def __init__(self, ideology_path, members_path):
        self.ideology_path = ideology_path
        self.members_path = members_path
        self.bioguide_to_icpsr = {}
        self.history = {} # {icpsr: [(timestamp, [coord1, coord2]), ...]}
        
        self._load_mapping()
        self._load_scores()
        
    def _load_mapping(self):
        if not os.path.exists(self.members_path):
            print(f"Warning: Members file not found at {self.members_path}")
            return

        df = pd.read_csv(self.members_path)
        if 'bioguide_id' in df.columns and 'icpsr' in df.columns:
            subset = df[['bioguide_id', 'icpsr']].dropna().drop_duplicates()
            self.bioguide_to_icpsr = dict(zip(subset['bioguide_id'], subset['icpsr']))
        print(f"Loaded ID mapping for {len(self.bioguide_to_icpsr)} politicians.")

    def _load_scores(self):
        if not os.path.exists(self.ideology_path):
            print(f"Warning: Ideology file not found at {self.ideology_path}")
            return

        df = pd.read_csv(self.ideology_path)
        df['date_dt'] = pd.to_datetime(df['date_window_end'])
        df = df.sort_values('date_dt')
        
        count = 0
        for _, row in df.iterrows():
            icpsr = int(row['icpsr'])
            ts = row['date_dt'].timestamp()
            c1 = row.get('coord1D', 0.0)
            c2 = row.get('coord2D', 0.0)
            feats = [
                float(c1) if pd.notnull(c1) else 0.0,
                float(c2) if pd.notnull(c2) else 0.0
            ]
            if icpsr not in self.history: self.history[icpsr] = []
            self.history[icpsr].append((ts, feats))
            count += 1
        print(f"Loaded {count} ideology score records.")

    def get_score_at_time(self, bioguide_id, timestamp):
        icpsr = self.bioguide_to_icpsr.get(bioguide_id)
        if icpsr is None or icpsr not in self.history:
            return [0.0, 0.0]
        
        records = self.history[icpsr]
        idx = bisect.bisect_right(records, (timestamp, []))
        if idx == 0: return records[0][1]
        return records[idx - 1][1]
    
class TemporalGraphBuilder:
    def __init__(self, transactions_df, config):
        self.config = config
        self.transactions = transactions_df.copy()
        
        self.transactions['Traded_DT'] = pd.to_datetime(self.transactions['Traded'])
        self.transactions = self.transactions.sort_values('Traded').reset_index(drop=True)
        self.min_freq = self.config['MIN_TICKER_FREQ']
        
        self.pol_id_map = {}
        self.company_id_map = {}
        self.party_map = {'Unknown': 0}
        self.state_map = {'Unknown': 0}
        
        self._build_mappings()
        
        # 1. Ideology
        if self.config.get('INCLUDE_IDEOLOGY', True):
            print("Initializing Ideology Lookup...")
            self.ideology_lookup = IdeologyLookup(self.config['IDEOLOGY_PATH'], self.config['MEMBERS_PATH'])
        else: self.ideology_lookup = None
            
        # 2. District Econ
        if self.config.get('INCLUDE_DISTRICT_ECON', True):
            print("Initializing District Economic Data Lookup...")
            self.dist_econ_lookup = DistrictEconomicLookup(self.config['RAW_DATA_DIR'], self.config['MEMBERS_PATH'])
        else: self.dist_econ_lookup = None

        # 3. Committees
        if self.config.get('INCLUDE_COMMITTEES', True):
            print("Initializing Committee Data Lookup...")
            self.committee_lookup = CommitteeLookup(self.config['COMMITTEE_PATH'], self.config['MEMBERS_PATH'])
        else: self.committee_lookup = None

        # 4. Company SIC (NEW)
        if self.config.get('INCLUDE_COMPANY_SIC', True):
            print("Initializing Company SIC Lookup...")
            self.sic_lookup = CompanySICLookup(self.config['SIC_PATH'])
        else: self.sic_lookup = None

        # 5. Company Financials (NEW)
        if self.config.get('INCLUDE_COMPANY_FINANCIALS', True):
            print("Initializing Company Financials Lookup...")
            self.fin_lookup = CompanyFinancialsLookup(self.config['FINANCIALS_PATH'])
        else: self.fin_lookup = None
        
    def _build_mappings(self):
        pols = self.transactions['BioGuideID'].unique()
        self.pol_id_map = {pid: i for i, pid in enumerate(pols)}
        
        ticker_counts = self.transactions['Ticker'].value_counts()
        valid_tickers = ticker_counts[ticker_counts >= self.min_freq].index
        self.company_id_map = {t: i for i, t in enumerate(valid_tickers)}
        
        parties = self.transactions['Party'].fillna('Unknown').unique()
        for p in parties:
            if p not in self.party_map: self.party_map[p] = len(self.party_map)
            
        states = self.transactions['State'].fillna('Unknown').unique()
        for s in states:
            if s not in self.state_map: self.state_map[s] = len(self.state_map)
            
    def _parse_amount(self, amt_str):
        if pd.isna(amt_str): return 0.0
        clean = str(amt_str).replace('$','').replace(',','')
        try: return float(clean)
        except: return 0.0

    def _get_dynamic_features(self, row, 
                              ideology_scores, 
                              district_vector, 
                              committee_vector, comm_names,
                              sic_vector, sic_names,          
                              fin_vector, fin_names):         
        feats = []
        names = []

        # 1. Trade Amount
        amt = np.log1p(self._parse_amount(row['Trade_Size_USD']))
        feats.append(amt)
        names.append("log_amount")

        # 2. Direction
        is_buy = 1.0 if 'Purchase' in str(row['Transaction']) else -1.0
        feats.append(is_buy)
        names.append("is_buy")

        # 3. Gap
        if pd.notnull(row['Filed_DT']):
            gap_days = max(0, (row['Filed_DT'] - row['Traded_DT']).days)
        else:
            gap_days = 30
        gap_feat = np.log1p(gap_days)
        feats.append(gap_feat)
        names.append("log_gap_days")

        # 4. Ideology
        if ideology_scores:
            feats.extend(ideology_scores) 
            names.extend(["ideology_eco", "ideology_soc"])
            
        # 5. District Economics
        if district_vector is not None:
            feats.extend(district_vector.tolist())
            names.extend([f"econ_feat_{i}" for i in range(len(district_vector))])
            
        # 6. Committee Membership
        if committee_vector is not None:
            feats.extend(committee_vector.tolist())
            if comm_names: names.extend(comm_names)
            else: names.extend([f"comm_feat_{i}" for i in range(len(committee_vector))])

        # 7. Company SIC 
        if sic_vector is not None:
            feats.extend(sic_vector.tolist())
            if sic_names: names.extend(sic_names)
            else: names.extend([f"sic_feat_{i}" for i in range(len(sic_vector))])

        # 8. Company Financials 
        if fin_vector is not None:
            feats.extend(fin_vector.tolist())
            if fin_names: names.extend(fin_names)
            else: names.extend([f"fin_feat_{i}" for i in range(len(fin_vector))])

        return feats, names

    def process(self):
        src, dst, t, msg, y = [], [], [], [], []
        resolution_t = []
        
        # ... [Static Features Logic unchanged] ...
        
        num_pols = len(self.pol_id_map)
        num_comps = len(self.company_id_map)
        total_nodes = num_pols + num_comps
        x_static = torch.zeros((total_nodes, 2), dtype=torch.long)
        
        pol_meta = self.transactions.drop_duplicates('BioGuideID').set_index('BioGuideID')
        for pid, idx in self.pol_id_map.items():
            if pid in pol_meta.index:
                row = pol_meta.loc[pid]
                if isinstance(row, pd.DataFrame): row = row.iloc[0]
                p_code = self.party_map.get(row.get('Party', 'Unknown'), 0)
                s_code = self.state_map.get(row.get('State', 'Unknown'), 0)
                x_static[idx] = torch.tensor([p_code, s_code])

        if len(self.transactions) > 0:
            base_time = self.transactions['Traded_DT'].min().timestamp()
        else: base_time = 0
            
        skipped = 0
        self.transactions['Filed_DT'] = pd.to_datetime(self.transactions['Filed'])
        
        if os.path.exists("data/price_sequences.pt"):
            print("Loading Price Sequences...")
            price_map = torch.load("data/price_sequences.pt")
        else:
            print("Warning: data/price_sequences.pt not found. Using zero sequences.")
            price_map = {}
            
        price_seqs = []
        target_cols = self.config['TARGET_COLUMNS']
        lookahead_days = self.config['LABEL_LOOKAHEAD_DAYS']
        
        # Missing column check
        missing = [c for c in target_cols if c not in self.transactions.columns]
        if missing: raise ValueError(f"Missing columns: {missing}")

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
            
            if pd.isna(row['Traded_DT']): continue
            ts = row['Traded_DT'].timestamp()
            t_norm = ts - base_time
            t.append(int(t_norm))
            
            # --- CONTEXT LOOKUPS ---
            # 1-3. Political Features
            ideology_scores = []
            if self.ideology_lookup: ideology_scores = self.ideology_lookup.get_score_at_time(pid, ts)
                
            dist_vec = None
            if self.dist_econ_lookup: dist_vec = self.dist_econ_lookup.get_district_vector(pid, ts)
                
            comm_vec, comm_names = None, []
            if self.committee_lookup:
                comm_vec = self.committee_lookup.get_committee_vector(pid, ts)
                comm_names = self.committee_lookup.get_feature_names()
                
            # 4-5. Company Features 
            sic_vec, sic_names = None, []
            if self.sic_lookup:
                sic_vec = self.sic_lookup.get_sic_vector(ticker)
                sic_names = self.sic_lookup.get_feature_names()
                
            fin_vec, fin_names = None, []
            if self.fin_lookup:
                fin_vec = self.fin_lookup.get_financial_vector(ticker, ts)
                fin_names = self.fin_lookup.get_feature_names()
            
            # Construct Msg
            msg_vector, feature_names = self._get_dynamic_features(
                row, ideology_scores, dist_vec, comm_vec, comm_names,
                sic_vec, sic_names, fin_vec, fin_names 
            )
            msg.append(msg_vector)
            
            # Price
            if tid in price_map: p_feat = price_map[tid]
            else: p_feat = torch.zeros((14,), dtype=torch.float32)
            price_seqs.append(p_feat)
            
            # Label
            flags = row[target_cols].infer_objects(copy=False).fillna(0.0).values.astype(int)
            label_idx = int(np.sum(flags))
            y.append(label_idx)
            
            # Resolution
            resolution_ts = (row['Traded_DT'] + pd.Timedelta(days=lookahead_days)).timestamp() - base_time
            resolution_t.append(int(resolution_ts))
            
        print(f"Skipped {skipped} transactions.")
        
        num_classes = len(target_cols) + 1
        
        data = TemporalData(
            src=torch.tensor(src, dtype=torch.long),
            dst=torch.tensor(dst, dtype=torch.long),
            t=torch.tensor(t, dtype=torch.long),
            msg=torch.tensor(msg, dtype=torch.float), 
            y=torch.tensor(y, dtype=torch.long)
        )

        data.msg_feature_names = feature_names 
        data.msg_dim = len(feature_names)       
        data.price_dim = 14                     
        
        if price_seqs: data.price_seq = torch.stack(price_seqs)
        else: data.price_seq = torch.zeros((len(src), 14))
        
        data.resolution_t = torch.tensor(resolution_t, dtype=torch.long)
        data.num_nodes = total_nodes
        data.x_static = x_static
        data.num_parties = len(self.party_map)
        data.num_states = len(self.state_map)
        data.num_classes = num_classes
        
        # --- UPDATE: Store specific counts for later printing ---
        data.num_pols = num_pols
        data.num_comps = num_comps
        
        return data

if __name__ == "__main__":
    df = pd.read_csv(CONFIG['TX_PATH'])
    builder = TemporalGraphBuilder(df, CONFIG)
    data = builder.process()
    
    print("\n" + "="*40)
    print("       GRAPH CONSTRUCTION SUMMARY")
    print("="*40)
    print(f"Total Nodes:      {data.num_nodes}")
    print(f"  - Congress:     {data.num_pols}")
    print(f"  - Companies:    {data.num_comps}")
    print(f"Total Edges (Tx): {data.src.shape[0]}")
    print(f"Message Dim:      {data.msg.shape[1]}") 
    print(f"Classes:          {data.num_classes}")
    print("="*40 + "\n")
        
    os.makedirs("data", exist_ok=True)
    torch.save(data, "data/temporal_data.pt")