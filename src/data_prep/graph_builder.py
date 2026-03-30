import pandas as pd
import numpy as np
import torch
import logging

# Import Config Flags
from src import config
from src.config import (
    INCLUDE_POLITICIAN_BIO, INCLUDE_IDEOLOGY, INCLUDE_COMMITTEES,
    INCLUDE_DISTRICT_ECON, INCLUDE_COMPANY_SIC, INCLUDE_COMPANY_FINANCIALS,
    IDEOLOGY_PATH, DISTRICT_ECON_DIR, COMMITTEE_PATH, COMPANY_SIC_PATH,
    COMPANY_FIN_PATH, CONGRESS_TERMS_PATH
)
# We add a custom flag for the historical performance stats
INCLUDE_PERFORMANCE = True 

from src.data_prep.feature_lookups import (
    TermLookup, PoliticianBioLookup, IdeologyLookup, DistrictEconLookup,
    CommitteeLookup, CompanySICLookup, CompanyFinancialsLookup
)
from src.models.stage2_gnn.graphsage import BipartiteSAGEExtended

log = logging.getLogger("stage2_builder")

class DynamicGraphBuilder:
    def __init__(self):
        """Initializes the active lookups based on src/config.py flags."""
        self.lookups = {}
        self.pol_dim = 0
        self.comp_dim = 0
        
        # 1. Initialize Term Lookup (Required if any Bio/Pol feature is active)
        if any([INCLUDE_POLITICIAN_BIO, INCLUDE_IDEOLOGY, INCLUDE_COMMITTEES, INCLUDE_DISTRICT_ECON]):
            self.term_lookup = TermLookup(CONGRESS_TERMS_PATH)
        else:
            self.term_lookup = None

        # 2. Politician Lookups
        if INCLUDE_PERFORMANCE:
            self.pol_dim += 3 # win_rate, log_count, buy_ratio
            
        if INCLUDE_POLITICIAN_BIO:
            self.lookups['bio'] = PoliticianBioLookup(CONGRESS_TERMS_PATH, self.term_lookup)
            # Note: We subtract state_dim (56) from bio.dim (62) because State is a learned embedding now!
            self.pol_dim += (self.lookups['bio'].dim - self.lookups['bio'].state_dim) 
            
        if INCLUDE_IDEOLOGY:
            self.lookups['ideology'] = IdeologyLookup(IDEOLOGY_PATH, self.term_lookup)
            self.pol_dim += self.lookups['ideology'].dim
            
        if INCLUDE_COMMITTEES:
            self.lookups['committee'] = CommitteeLookup(COMMITTEE_PATH, self.term_lookup)
            self.pol_dim += self.lookups['committee'].dim
            
        if INCLUDE_DISTRICT_ECON:
            self.lookups['econ'] = DistrictEconLookup(DISTRICT_ECON_DIR, self.term_lookup)
            self.pol_dim += self.lookups['econ'].dim

        # 3. Company Lookups
        if INCLUDE_COMPANY_SIC:
            self.lookups['sic'] = CompanySICLookup(COMPANY_SIC_PATH)
            self.comp_dim += self.lookups['sic'].dim
            
        if INCLUDE_COMPANY_FINANCIALS:
            self.lookups['financials'] = CompanyFinancialsLookup(COMPANY_FIN_PATH)
            self.comp_dim += self.lookups['financials'].dim

        # 4. New Edge Lookups
        if getattr(config, 'INCLUDE_LOBBYING_SPONSORSHIP', False) or getattr(config, 'INCLUDE_LOBBYING_VOTING', False):
            from src.data_prep.lookups.lobbying_lookups import LobbyingLookup
            self.lookups['lobbying'] = LobbyingLookup()
            
        if getattr(config, 'INCLUDE_CAMPAIGN_FINANCE', False):
            from src.data_prep.lookups.campaign_lookups import CampaignLookup
            self.lookups['campaign'] = CampaignLookup()

        log.info(f"Initialized Graph Builder: Pol_Dyn_Dim={self.pol_dim}, Comp_Dyn_Dim={self.comp_dim}")

    def build_pol_features(self, train_df, le_pol, n_pol, timestamp: pd.Timestamp, device, label_col="Excess_Return_6M"):
        """Builds the x_pol_dyn tensor for all politicians at the given timestamp."""
        feats = np.zeros((n_pol, self.pol_dim), dtype=np.float32)
        
        # Mapping index to bioguide_id
        idx_to_bio = {idx: bio for bio, idx in zip(train_df["BioGuideID"].fillna("UNK"), 
                                                   le_pol.transform(train_df["BioGuideID"].fillna("UNK")))}
        
        # Precompute performance stats if requested
        perf_dict = {}
        if INCLUDE_PERFORMANCE:
            t_df = train_df.copy()
            t_df["_bio"] = t_df["BioGuideID"].fillna("UNK")
            t_df["_lbl"] = (t_df[label_col] > 0).astype(float) 
            t_df["_buy"] = t_df["Transaction"].str.lower().str.contains("purchase", na=False).astype(float)
            
            for bio_id, grp in t_df.groupby("_bio"):
                perf_dict[bio_id] = [
                    grp["_lbl"].mean(),             # win_rate
                    np.log1p(len(grp)),             # log_count
                    grp["_buy"].mean()              # buy_ratio
                ]

        # Iterate over all known politicians and build their concatenated vector
        for idx in range(n_pol):
            bio_id = idx_to_bio.get(idx, "UNK")
            vec = []
            
            if INCLUDE_PERFORMANCE:
                vec.extend(perf_dict.get(bio_id, [0.5, 0.0, 0.5]))
                
            if INCLUDE_POLITICIAN_BIO:
                full_bio = self.lookups['bio'].get_vector(bio_id, timestamp).tolist()
                vec.extend(full_bio[:5] + full_bio[61:]) # Keep chamber, party, leadership
                
            if INCLUDE_IDEOLOGY:
                vec.extend(self.lookups['ideology'].get_vector(bio_id, timestamp).tolist())
            if INCLUDE_COMMITTEES:
                vec.extend(self.lookups['committee'].get_vector(bio_id, timestamp).tolist())
            if INCLUDE_DISTRICT_ECON:
                vec.extend(self.lookups['econ'].get_vector(bio_id, timestamp).tolist())
                
            if vec:
                feats[idx] = vec
                
        return torch.tensor(feats, dtype=torch.float32, device=device)
        
    def build_comp_features(self, train_df, le_tick, n_tick, timestamp: pd.Timestamp, device):
        """Builds the x_comp_dyn tensor for all companies at the given timestamp."""
        feats = np.zeros((n_tick, self.comp_dim), dtype=np.float32)
        
        # Mapping index to ticker
        idx_to_tick = {idx: tick for tick, idx in zip(train_df["Ticker"].fillna("UNK"), 
                                                      le_tick.transform(train_df["Ticker"].fillna("UNK")))}
        
        for idx in range(n_tick):
            ticker = idx_to_tick.get(idx, "UNK")
            vec = []
            
            if INCLUDE_COMPANY_SIC:
                vec.extend(self.lookups['sic'].get_vector(ticker, timestamp).tolist())
            if INCLUDE_COMPANY_FINANCIALS:
                vec.extend(self.lookups['financials'].get_vector(ticker, timestamp).tolist())
                
            if vec:
                feats[idx] = vec
                
        return torch.tensor(feats, dtype=torch.float32, device=device)

    def build_auxiliary_edges(self, train_df, le_pol, le_tick, timestamp: pd.Timestamp, device):
        """Builds heterogeneous edge tensors for Lobbying and Campaign Finance."""
        aux_edges = {}
        
        # --- Lobbying Edges ---
        if 'lobbying' in self.lookups:
            lobby_df = self.lookups['lobbying'].get_edges(timestamp)
            if not lobby_df.empty:
                # Filter to known politicians and tickers in this snapshot
                lobby_df = lobby_df[
                    lobby_df['bioguide_id'].isin(le_pol.classes_) & 
                    lobby_df['ticker'].isin(le_tick.classes_)
                ].copy()
                
                # Calculate recency (Days since interaction)
                lobby_df['days_since'] = (timestamp - lobby_df['latest_interaction_date']).dt.days
                
                if getattr(config, 'INCLUDE_LOBBYING_SPONSORSHIP', False):
                    strong_df = lobby_df[lobby_df['event_type'] == 'LOBBY_STRONG']
                    if not strong_df.empty:
                        src = le_pol.transform(strong_df['bioguide_id'])
                        dst = le_tick.transform(strong_df['ticker'])
                        feats = strong_df[['interaction_count', 'days_since']].values
                        aux_edges['lobby_strong'] = {
                            # FIXED PyTorch Warning by wrapping in np.array()
                            'edge_index': torch.tensor(np.array([src, dst]), dtype=torch.long, device=device),
                            'features': torch.tensor(feats, dtype=torch.float32, device=device)
                        }
                        
                if getattr(config, 'INCLUDE_LOBBYING_VOTING', False):
                    weak_df = lobby_df[lobby_df['event_type'] == 'LOBBY_WEAK']
                    if not weak_df.empty:
                        src = le_pol.transform(weak_df['bioguide_id'])
                        dst = le_tick.transform(weak_df['ticker'])
                        feats = weak_df[['interaction_count', 'days_since']].values
                        aux_edges['lobby_weak'] = {
                            # FIXED PyTorch Warning by wrapping in np.array()
                            'edge_index': torch.tensor(np.array([src, dst]), dtype=torch.long, device=device),
                            'features': torch.tensor(feats, dtype=torch.float32, device=device)
                        }
                        
        # --- Campaign Finance Edges ---
        if 'campaign' in self.lookups:
            camp_df = self.lookups['campaign'].get_edges(timestamp)
            if not camp_df.empty:
                camp_df = camp_df[camp_df['bioguide_id'].isin(le_pol.classes_)].copy()
                
                # Map Industry Code to Tickers present in the graph
                ind_to_tickers = train_df.dropna(subset=['Ticker', 'Industry']).groupby('Industry')['Ticker'].unique().to_dict()
                
                expanded_rows = []
                for _, row in camp_df.iterrows():
                    tickers = ind_to_tickers.get(row['industry_code'], [])
                    for t in tickers:
                        if t in le_tick.classes_:
                            expanded_rows.append({
                                'bioguide_id': row['bioguide_id'],
                                'ticker': t,
                                'total_donation_amount': row['total_donation_amount'],
                                'donation_count': row['donation_count'],
                                'latest_donation_date': row['latest_donation_date']
                            })
                
                if expanded_rows:
                    exp_camp_df = pd.DataFrame(expanded_rows)
                    exp_camp_df['days_since'] = (timestamp - exp_camp_df['latest_donation_date']).dt.days
                    
                    src = le_pol.transform(exp_camp_df['bioguide_id'])
                    dst = le_tick.transform(exp_camp_df['ticker'])
                    
                    log_amt = np.log1p(exp_camp_df['total_donation_amount'])
                    feats = np.column_stack((log_amt, exp_camp_df['donation_count'], exp_camp_df['days_since']))
                    
                    aux_edges['campaign'] = {
                        # FIXED PyTorch Warning by wrapping in np.array()
                        'edge_index': torch.tensor(np.array([src, dst]), dtype=torch.long, device=device),
                        'features': torch.tensor(feats, dtype=torch.float32, device=device)
                    }
                    
        return aux_edges
    
    def init_model(self, num_states, num_sectors, num_industries, out_dim=32, device='cpu'):
        """Initializes the updated BipartiteSAGEExtended with dynamic dimensions."""
        model = BipartiteSAGEExtended(
            pol_in_dim=self.pol_dim,
            comp_in_dim=self.comp_dim,
            hidden_dim=64,
            out_dim=out_dim,
            num_states=num_states,
            num_sectors=num_sectors,
            num_industries=num_industries
        ).to(device)
        return model