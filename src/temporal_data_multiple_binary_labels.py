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
    MIN_TICKER_FREQ
)

# ==========================================
#              CONFIGURATION
# ==========================================
CONFIG = {
    'TX_PATH': TX_PATH,
    'IDEOLOGY_PATH': 'data/raw/ideology_scores_quarterly.csv',
    'MEMBERS_PATH': 'data/raw/HSall_members.csv', # Needed for ID mapping
    'TARGET_COLUMNS': TARGET_COLUMNS,
    'LABEL_LOOKAHEAD_DAYS': LABEL_LOOKAHEAD_DAYS,
    'MIN_TICKER_FREQ': MIN_TICKER_FREQ,
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
        """Creates BioGuideID -> ICPSR mapping from Voteview members file."""
        if not os.path.exists(self.members_path):
            print(f"Warning: Members file not found at {self.members_path}. Ideology linking may fail.")
            return

        df = pd.read_csv(self.members_path)
        # Filter to relevant columns and drop duplicates
        if 'bioguide_id' in df.columns and 'icpsr' in df.columns:
            subset = df[['bioguide_id', 'icpsr']].dropna().drop_duplicates()
            self.bioguide_to_icpsr = dict(zip(subset['bioguide_id'], subset['icpsr']))
        print(f"Loaded ID mapping for {len(self.bioguide_to_icpsr)} politicians.")

    def _load_scores(self):
        """Loads temporal ideology scores into sorted lists."""
        if not os.path.exists(self.ideology_path):
            print(f"Warning: Ideology file not found at {self.ideology_path}")
            return

        df = pd.read_csv(self.ideology_path)
        df['date_dt'] = pd.to_datetime(df['date_window_end'])
        
        # Sort by date ensures our lists are monotonic
        df = df.sort_values('date_dt')
        
        count = 0
        for _, row in df.iterrows():
            icpsr = int(row['icpsr'])
            ts = row['date_dt'].timestamp()
            
            # Feature Vector: [coord1D, coord2D]
            # We can extend this to include delta from previous, errors, etc.
            # New Code:
            c1 = row.get('coord1D', 0.0)
            c2 = row.get('coord2D', 0.0)
            feats = [
                float(c1) if pd.notnull(c1) else 0.0,
                float(c2) if pd.notnull(c2) else 0.0
            ]
            
            if icpsr not in self.history:
                self.history[icpsr] = []
            
            self.history[icpsr].append((ts, feats))
            count += 1
            
        print(f"Loaded {count} ideology score records.")

    def get_score_at_time(self, bioguide_id, timestamp):
        """
        Returns the most recent ideology [coord1D, coord2D] before `timestamp`.
        Returns [0.0, 0.0] if no history found.
        """
        icpsr = self.bioguide_to_icpsr.get(bioguide_id)
        if icpsr is None or icpsr not in self.history:
            return [0.0, 0.0] # Default neutral
        
        records = self.history[icpsr]
        
        # bisect_right finds the insertion point to maintain order.
        # We search using (timestamp, [inf]) to ensure we find the right spot
        # comparison logic: it compares the first element of tuple (the timestamp).
        idx = bisect.bisect_right(records, (timestamp, []))
        
        if idx == 0:
            # Timestamp is before the first recorded score. 
            # Strategy: Return the first available score (backfill) OR neutral.
            # Let's return the first available to avoid zeros if they have data shortly after.
            return records[0][1]
        
        # Return the record immediately preceding the insertion point
        return records[idx - 1][1]


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
        
        # Initialize Ideology Helper
        print("Initializing Ideology Lookup...")
        self.ideology_lookup = IdeologyLookup(
            self.config['IDEOLOGY_PATH'], 
            self.config['MEMBERS_PATH']
        )
        
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
        
    # Inside TemporalGraphBuilder class

    def _get_dynamic_features(self, row, ideology_scores):
        """
        Central place to define dynamic edge features.
        Returns: List[float] and List[str] (feature names)
        """
        feats = []
        names = []

        # 1. Trade Amount (Log)
        amt = np.log1p(self._parse_amount(row['Trade_Size_USD']))
        feats.append(amt)
        names.append("log_amount")

        # 2. Direction
        is_buy = 1.0 if 'Purchase' in str(row['Transaction']) else -1.0
        feats.append(is_buy)
        names.append("is_buy")

        # 3. Reporting Gap
        if pd.notnull(row['Filed_DT']):
            gap_days = max(0, (row['Filed_DT'] - row['Traded_DT']).days)
        else:
            gap_days = 30
        gap_feat = np.log1p(gap_days)
        feats.append(gap_feat)
        names.append("log_gap_days")

        # 4. Ideology (External Lookup)
        # ideology_scores is passed in because it depends on the loop's timestamp
        feats.extend(ideology_scores) 
        names.extend(["ideology_eco", "ideology_soc"])

        # --- ADD NEW FEATURES HERE IN THE FUTURE ---
        # e.g., feats.append(row['New_Metric']); names.append('new_metric')

        return feats, names

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
            ts = row['Traded_DT'].timestamp()
            t_norm = ts - base_time
            t.append(int(t_norm))
            
            # --- FEATURE CONSTRUCTION (MSG) ---
            # 1. Transaction Features
            amt = np.log1p(self._parse_amount(row['Trade_Size_USD']))
            is_buy = 1.0 if 'Purchase' in str(row['Transaction']) else -1.0
            
            if pd.notnull(row['Filed_DT']):
                gap_days = max(0, (row['Filed_DT'] - row['Traded_DT']).days)
            else:
                gap_days = 30
            gap_feat = np.log1p(gap_days)
            
            # 2. Dynamic Ideology Features
            # Look up the score valid at the time of trade
            ideology_scores = self.ideology_lookup.get_score_at_time(pid, ts)
            
            # Combine into Message Vector: [Amount, IsBuy, Gap, Coord1D, Coord2D]
            # dynamic feature extraction
            msg_vector, feature_names = self._get_dynamic_features(row, ideology_scores)
            msg.append(msg_vector)
            
            # Price Feature
            if tid in price_map:
                p_feat = price_map[tid]
            else:
                p_feat = torch.zeros((14,), dtype=torch.float32)
            price_seqs.append(p_feat)
            
            # --- LABEL HANDLING ---
            flags = row[target_cols].infer_objects(copy=False).fillna(0.0).values.astype(int)
            label_idx = int(np.sum(flags))
            y.append(label_idx)
            
            # Resolution Time
            resolution_ts = (row['Traded_DT'] + pd.Timedelta(days=lookahead_days)).timestamp() - base_time
            resolution_t.append(int(resolution_ts))
            
        print(f"Skipped {skipped} transactions.")
        
        # Calculate Number of Classes
        num_classes = len(target_cols) + 1
        
        data = TemporalData(
            src=torch.tensor(src, dtype=torch.long),
            dst=torch.tensor(dst, dtype=torch.long),
            t=torch.tensor(t, dtype=torch.long),
            msg=torch.tensor(msg, dtype=torch.float), # Now 5 dims
            y=torch.tensor(y, dtype=torch.long)
        )

        data.msg_feature_names = feature_names  # Save names for debugging
        data.msg_dim = len(feature_names)       # Save explicit dimension
        data.price_dim = 14                     # Save explicit price dimension
        
        if price_seqs:
            data.price_seq = torch.stack(price_seqs)
        else:
            data.price_seq = torch.zeros((len(src), 14))
        
        data.resolution_t = torch.tensor(resolution_t, dtype=torch.long)
        data.num_nodes = total_nodes
        data.x_static = x_static
        data.num_parties = len(self.party_map)
        data.num_states = len(self.state_map)
        data.num_classes = num_classes
        
        return data

if __name__ == "__main__":
    df = pd.read_csv(CONFIG['TX_PATH'])
    builder = TemporalGraphBuilder(df, CONFIG)
    data = builder.process()
    
    print(data)
    print(f"Msg Dimension: {data.msg.shape[1]}") # Should be 5
    print(f"Task: Classification with {data.num_classes} intervals.")
        
    os.makedirs("data", exist_ok=True)
    torch.save(data, "data/temporal_data.pt")