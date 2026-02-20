import torch
from torch_geometric.data import TemporalData
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys

# Import Configs
import src.config as config
from src.config import (
    INCLUDE_POLITICIAN_BIO, INCLUDE_IDEOLOGY, INCLUDE_DISTRICT_ECON, INCLUDE_COMMITTEES,
    INCLUDE_COMPANY_SIC, INCLUDE_COMPANY_FINANCIALS,
    IDEOLOGY_PATH, DISTRICT_ECON_DIR, COMMITTEE_PATH,
    COMPANY_SIC_PATH, COMPANY_FIN_PATH, CONGRESS_TERMS_PATH
)

# Safely get Phase 1 Flags (Default True if missing from config)
INCLUDE_LOBBYING_SPONSORSHIP = getattr(config, 'INCLUDE_LOBBYING_SPONSORSHIP', True)
INCLUDE_LOBBYING_VOTING = getattr(config, 'INCLUDE_LOBBYING_VOTING', True)
INCLUDE_CAMPAIGN_FINANCE = getattr(config, 'INCLUDE_CAMPAIGN_FINANCE', True)

from src.data_processing.feature_lookups import (
    TermLookup, PoliticianBioLookup, IdeologyLookup, DistrictEconLookup, 
    CommitteeLookup, CompanySICLookup, CompanyFinancialsLookup
)

class TemporalGraphBuilder:
    def __init__(self, transactions_df, min_freq=1):
        self.min_freq = min_freq
        
        # Mappings
        self.pol_id_map = {}
        self.company_id_map = {}
        self.party_map = {'Unknown': 0}
        self.state_map = {'Unknown': 0}
        
        # 1. Base Transactions
        self.transactions = transactions_df.sort_values('Filed').reset_index(drop=True)
        self.transactions['Traded_DT'] = pd.to_datetime(self.transactions['Traded'])
        self.transactions['Filed_DT'] = pd.to_datetime(self.transactions['Filed'])
        
        # 2. Store Paths for Later Chunking (Do not load into memory here!)
        self.lobbying_path = getattr(config, 'LOBBYING_EVENTS_PATH', 'data/processed/events_lobbying.csv')
        self.campaign_path = getattr(config, 'CAMPAIGN_FINANCE_EVENTS_PATH', 'data/processed/events_campaign_finance.csv')

        # Campaign Crosswalk can be loaded now as it is very small
        if INCLUDE_CAMPAIGN_FINANCE and os.path.exists(self.campaign_path):
            self._load_campaign_crosswalk()

        self.lookups = {}
        self.term_lookup = None 
        
        self._build_mappings()
        self._init_feature_lookups()

    def _merge_events(self):
        print("Merging all event streams into a unified timeline...")
        all_events = []
        
        # 1. Trades (Edge Type 0)
        trades = self.transactions.copy()
        trades['event_date'] = trades['Filed_DT']
        trades['event_type'] = 0
        trades['bioguide_id'] = trades['BioGuideID']
        trades['ticker'] = trades['Ticker']
        
        keep_cols = ['event_date', 'bioguide_id', 'ticker', 'event_type', 'transaction_id', 'Trade_Size_USD', 'Transaction', 'Traded_DT']
        horizons = ['1M', '2M', '3M', '6M', '8M', '12M', '18M', '24M']
        keep_cols += [f'Excess_Return_{h}' for h in horizons if f'Excess_Return_{h}' in trades.columns]
        all_events.append(trades[keep_cols])
        
        # 2. Lobbying (Edge Type 1 & 2) - Streamed in Chunks
        if (INCLUDE_LOBBYING_SPONSORSHIP or INCLUDE_LOBBYING_VOTING) and os.path.exists(self.lobbying_path):
            print("Processing 211M Lobbying Events in chunks to save memory...")
            chunk_size = 5_000_000 # Process 5 million rows at a time
            
            # Count rows for tqdm progress bar (approximate or just let it spin)
            total_chunks = 215_000_000 // chunk_size 
            
            for chunk in tqdm(pd.read_csv(self.lobbying_path, chunksize=chunk_size, low_memory=False), total=total_chunks, desc="Filtering Lobbying Chunks"):
                # ADD .copy() TO THE END OF THIS LINE
                chunk = chunk[chunk['bioguide_id'].isin(self.pol_id_map) & chunk['ticker'].isin(self.company_id_map)].copy()
                
                if chunk.empty:
                    continue
                    
                chunk['event_date'] = pd.to_datetime(chunk['date'])
                chunk['event_type'] = chunk['event_type'].map({'LOBBY_STRONG': 1, 'LOBBY_WEAK': 2}).fillna(1).astype(int)
                chunk['Trade_Size_USD'] = chunk['weight']
                chunk['Transaction'] = 'Lobbying'
                chunk['Traded_DT'] = chunk['event_date']
                chunk['transaction_id'] = -1
                
                all_events.append(chunk[['event_date', 'bioguide_id', 'ticker', 'event_type', 'transaction_id', 'Trade_Size_USD', 'Transaction', 'Traded_DT']])
        
        # 3. Campaign Finance (Edge Type 3 - Broadcasted)
        if INCLUDE_CAMPAIGN_FINANCE and os.path.exists(self.campaign_path) and getattr(self, 'realcode_to_tickers', None):
            print("Loading Campaign Finance Events...")
            camp = pd.read_csv(self.campaign_path, low_memory=False)
            camp['event_date'] = pd.to_datetime(camp['date'])
            
            broadcast_records = []
            for _, row in tqdm(camp.iterrows(), total=len(camp), desc="Broadcasting Donations"):
                for t in self.realcode_to_tickers.get(row['industry_code'], []):
                    if t in self.company_id_map and row['bioguide_id'] in self.pol_id_map:
                        broadcast_records.append({
                            'event_date': row['event_date'], 'bioguide_id': row['bioguide_id'], 'ticker': t,
                            'event_type': 3, 'transaction_id': -1, 'Trade_Size_USD': row['weight'], 
                            'Transaction': 'Donation', 'Traded_DT': row['event_date']
                        })
            if broadcast_records:
                all_events.append(pd.DataFrame(broadcast_records))

        combined = pd.concat(all_events, ignore_index=True)
        print(f"Total Combined Edges (Trades + Filtered Influence): {len(combined)}")
        print("Sorting chronological timeline...")
        combined = combined.sort_values('event_date').reset_index(drop=True)
        return combined
    
    def _load_campaign_crosswalk(self):
        self.realcode_to_tickers = {}
        cw_path = getattr(config, 'CATCODE_CROSSWALK_PATH', 'data/raw/industry_codes_NAICS/2013-CAT_to_SIC_to_NAICS_mappings.csv')
        sic_path = getattr(config, 'COMPANY_SIC_PATH', 'data/raw/company_sic_data.csv')

        if os.path.exists(cw_path) and os.path.exists(sic_path):
            print("Loading Campaign Finance Crosswalks...")
            
            # 1. Load SIC to Ticker mapping
            sic_data = pd.read_csv(sic_path)
            sic_data['sic'] = sic_data['sic'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(4)
            sic_to_ticker_map = sic_data.groupby('sic')['ticker'].apply(list).to_dict()

            # 2. Load Catcode to SIC mapping
            cat_data = pd.read_csv(cw_path)
            cat_data['SICcode'] = cat_data['SICcode'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(4)
            
            # Extract both 5-char and 3-char versions to be safe against different OpenSecrets formats
            cat_data['FullCode'] = cat_data['OpenSecretsCatcode'].astype(str).str.upper().str.strip()
            cat_data['ShortCode'] = cat_data['FullCode'].str[:3]

            # 3. Build Industry -> Ticker Dictionary
            for idx, row in cat_data.iterrows():
                sic = row['SICcode']
                tickers = sic_to_ticker_map.get(sic, [])
                
                if not tickers:
                    continue
                    
                full_code = row['FullCode']
                short_code = row['ShortCode']

                # Map the 5-character code
                if full_code not in self.realcode_to_tickers:
                    self.realcode_to_tickers[full_code] = set()
                self.realcode_to_tickers[full_code].update(tickers)

                # Map the 3-character code
                if short_code not in self.realcode_to_tickers:
                    self.realcode_to_tickers[short_code] = set()
                self.realcode_to_tickers[short_code].update(tickers)

            # Convert sets back to lists for faster iteration later
            self.realcode_to_tickers = {k: list(v) for k, v in self.realcode_to_tickers.items()}
            print(f"Mapped {len(self.realcode_to_tickers)} unique Industry Codes (Catcodes) to Tickers.")
        else:
            print(f"WARNING: Crosswalk files missing at {cw_path} or {sic_path}. Campaign events cannot be broadcast.")

    def _build_mappings(self):
        pols = self.transactions['BioGuideID'].unique()
        self.pol_id_map = {pid: i for i, pid in enumerate(pols)}
        print(f"Mapped {len(self.pol_id_map)} Politicians")
        
        ticker_counts = self.transactions['Ticker'].value_counts()
        valid_tickers = ticker_counts[ticker_counts >= self.min_freq].index
        self.company_id_map = {t: i for i, t in enumerate(valid_tickers)}
        print(f"Mapped {len(self.company_id_map)} Companies")

        parties = self.transactions['Party'].fillna('Unknown').unique()
        for p in parties:
            if p not in self.party_map: self.party_map[p] = len(self.party_map)
            
        states = self.transactions['State'].fillna('Unknown').unique()
        for s in states:
            if s not in self.state_map: self.state_map[s] = len(self.state_map)

    def _init_feature_lookups(self):
        print("Initializing Feature Lookups...")
        if INCLUDE_POLITICIAN_BIO or INCLUDE_IDEOLOGY or INCLUDE_DISTRICT_ECON or INCLUDE_COMMITTEES:
            self.term_lookup = TermLookup(CONGRESS_TERMS_PATH)
        if INCLUDE_POLITICIAN_BIO: self.lookups['bio'] = PoliticianBioLookup(CONGRESS_TERMS_PATH, self.term_lookup)
        if INCLUDE_IDEOLOGY: self.lookups['ideology'] = IdeologyLookup(IDEOLOGY_PATH, self.term_lookup)
        if INCLUDE_DISTRICT_ECON: self.lookups['econ'] = DistrictEconLookup(DISTRICT_ECON_DIR, self.term_lookup)
        if INCLUDE_COMMITTEES: self.lookups['committee'] = CommitteeLookup(COMMITTEE_PATH, self.term_lookup)
        if INCLUDE_COMPANY_SIC: self.lookups['sic'] = CompanySICLookup(COMPANY_SIC_PATH)
        if INCLUDE_COMPANY_FINANCIALS: self.lookups['financials'] = CompanyFinancialsLookup(COMPANY_FIN_PATH)

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

    def process(self, min_freq=1):
        combined = self._merge_events()
        combined['event_ts_raw'] = combined['event_date'].astype('int64') // 10**9
        base_time = combined['event_ts_raw'].min()
        
        # --- Pre-allocate Static Tensors ---
        num_pols = len(self.pol_id_map)
        num_comps = len(self.company_id_map)
        total_nodes = num_pols + num_comps
        x_static = torch.zeros((total_nodes, 2), dtype=torch.long)
        
        pol_meta = self.transactions.drop_duplicates('BioGuideID').set_index('BioGuideID')
        for pid, idx in self.pol_id_map.items():
            if pid in pol_meta.index:
                row = pol_meta.loc[pid]
                if isinstance(row, pd.DataFrame): row = row.iloc[0]
                x_static[idx] = torch.tensor([self.party_map.get(row.get('Party'), 0), self.state_map.get(row.get('State'), 0)])

        # --- Price Seqs ---
        price_map = torch.load("data/price_sequences.pt") if os.path.exists("data/price_sequences.pt") else {}
        
        print("Vectorizing Tensor Construction (Preventing Memory Explosion)...")
        num_events = len(combined)
        
        # 1. Edge Index
        src_arr = combined['bioguide_id'].map(self.pol_id_map).values.astype(np.int64)
        dst_arr = combined['ticker'].map(self.company_id_map).values.astype(np.int64) + num_pols
        t_arr = (combined['event_ts_raw'] - base_time).values.astype(np.int64)
        
        # 2. Message Tensor [Amt/Weight, is_buy, gap, EDGE_TYPE]
        msg_arr = np.zeros((num_events, 4), dtype=np.float32)
        edge_types = combined['event_type'].values.astype(int)
        msg_arr[:, 3] = edge_types
        
        trade_mask = (edge_types == 0)
        non_trade_mask = (edge_types > 0)
        
        # Parse Trades
        if trade_mask.sum() > 0:
            trade_amts = combined.loc[trade_mask, 'Trade_Size_USD'].apply(self._parse_amount).values
            msg_arr[trade_mask, 0] = np.log1p(trade_amts)
            msg_arr[trade_mask, 1] = np.where(combined.loc[trade_mask, 'Transaction'].astype(str).str.contains('Purchase'), 1.0, -1.0)
            trade_gap = (combined.loc[trade_mask, 'event_date'] - combined.loc[trade_mask, 'Traded_DT']).dt.days.values
            msg_arr[trade_mask, 2] = np.log1p(np.clip(trade_gap, 0, None))
            
        # Parse Lobby/Donation
        if non_trade_mask.sum() > 0:
            msg_arr[non_trade_mask, 0] = combined.loc[non_trade_mask, 'Trade_Size_USD'].values.astype(np.float32)
            
        # 3. Target Masking (y)
        horizons = ['1M', '2M', '3M', '6M', '8M', '12M', '18M', '24M']
        y_arr = np.full((num_events, 8), np.nan, dtype=np.float32)
        for i, h in enumerate(horizons):
            if f'Excess_Return_{h}' in combined.columns:
                vals = combined.loc[trade_mask, f'Excess_Return_{h}'].values
                y_arr[trade_mask, i] = np.where(pd.notnull(vals), vals.astype(np.float32), np.nan)
                
        # 4. Resolution Time
        res_t_arr = t_arr.copy()
        has_trade_dt = combined['Traded_DT'].notnull() & trade_mask
        trade_ts = (combined.loc[has_trade_dt, 'Traded_DT'].astype('int64') // 10**9).values - int(base_time)
        res_t_arr[has_trade_dt] = trade_ts
        res_t_arr[trade_mask & ~has_trade_dt] = t_arr[trade_mask & ~has_trade_dt] - (30*86400)

        # 5. Price Sequences
        price_seq_arr = np.zeros((num_events, 14), dtype=np.float32)
        valid_tids = combined['transaction_id'].values
        for i, tid in enumerate(valid_tids):
            if tid in price_map:
                price_seq_arr[i] = price_map[tid].numpy()

        # 6. Dynamic Features (Optimized Cache to save Days of execution time)
        print("Vectorizing Dynamic Lookups...")
        pol_pairs = combined[['bioguide_id', 'event_ts_raw']].drop_duplicates()
        pol_feat_dict = {}
        for row in tqdm(pol_pairs.itertuples(index=False), total=len(pol_pairs), desc="Caching Pol Feats"):
            pid, ts = row.bioguide_id, row.event_ts_raw
            vec = []
            if INCLUDE_POLITICIAN_BIO: vec.extend(self.lookups['bio'].get_vector(pid, ts).tolist())
            if INCLUDE_IDEOLOGY: vec.extend(self.lookups['ideology'].get_vector(pid, ts).tolist())
            if INCLUDE_DISTRICT_ECON: vec.extend(self.lookups['econ'].get_vector(pid, ts).tolist())
            if INCLUDE_COMMITTEES: vec.extend(self.lookups['committee'].get_vector(pid, ts).tolist())
            pol_feat_dict[(pid, ts)] = vec if vec else [0.0]
            
        x_pol_list = [pol_feat_dict[(pid, ts)] for pid, ts in zip(combined['bioguide_id'], combined['event_ts_raw'])]
        
        comp_pairs = combined[['ticker', 'event_ts_raw']].drop_duplicates()
        comp_feat_dict = {}
        for row in tqdm(comp_pairs.itertuples(index=False), total=len(comp_pairs), desc="Caching Comp Feats"):
            t, ts = row.ticker, row.event_ts_raw
            vec = []
            if INCLUDE_COMPANY_SIC: vec.extend(self.lookups['sic'].get_vector(t, ts).tolist())
            if INCLUDE_COMPANY_FINANCIALS: vec.extend(self.lookups['financials'].get_vector(t, ts).tolist())
            comp_feat_dict[(t, ts)] = vec if vec else [0.0]
            
        x_comp_list = [comp_feat_dict[(t, ts)] for t, ts in zip(combined['ticker'], combined['event_ts_raw'])]

        # Construct Temporal Data
        data = TemporalData(
            src=torch.tensor(src_arr, dtype=torch.long),
            dst=torch.tensor(dst_arr, dtype=torch.long),
            t=torch.tensor(t_arr, dtype=torch.long),
            msg=torch.tensor(msg_arr, dtype=torch.float),
            y=torch.tensor(y_arr, dtype=torch.float)
        )
        data.price_seq = torch.tensor(price_seq_arr, dtype=torch.float)
        data.trade_t = torch.tensor(res_t_arr, dtype=torch.long)
        
        data.num_nodes = total_nodes
        data.num_parties = len(self.party_map)
        data.num_states = len(self.state_map)
        data.x_static = x_static
        data.x_pol = torch.tensor(x_pol_list, dtype=torch.float)
        data.x_comp = torch.tensor(x_comp_list, dtype=torch.float)
        
        return data

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.config import TX_PATH
    if os.path.exists(TX_PATH):
        df = pd.read_csv(TX_PATH)
        builder = TemporalGraphBuilder(df)
        data = builder.process()
        print(data)
        os.makedirs("data", exist_ok=True)
        torch.save(data, "data/temporal_data.pt")