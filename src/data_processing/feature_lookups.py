import pandas as pd
import numpy as np
import torch
from pathlib import Path

class FeatureLookupBase:
    def get_vector(self, entity_id, timestamp):
        raise NotImplementedError

class TermLookup:
    """Helper to map a date to a specific Congress term ID."""
    def __init__(self, terms_csv_path):
        self.df = pd.read_csv(terms_csv_path, parse_dates=['start', 'end'], low_memory=False)
        self.df = self.df.sort_values('start')

    def get_term_data(self, bioguide_id, timestamp):
        """Find the row in terms csv active at timestamp."""
        date = pd.to_datetime(timestamp, unit='s')
        
        # Filter by ID
        subset = self.df[self.df['id_bioguide'] == bioguide_id]
        if subset.empty:
            return None
            
        # Filter by Date Range
        # We look for start <= date <= end
        # Handle cases where 'end' might be NaN (current term) - assume active
        mask = (subset['start'] <= date) & ((subset['end'] >= date) | subset['end'].isna())
        active = subset[mask]
        
        if not active.empty:
            return active.iloc[0]
        
        # Fallback: Find closest term if no exact overlap (e.g. gap between terms)
        # Using the last term that started before this date
        past = subset[subset['start'] <= date]
        if not past.empty:
            return past.iloc[-1]
            
        return None

class PoliticianBioLookup(FeatureLookupBase):
    """
    Dynamic lookup for Politician Bio Info (Chamber, Party, State, Leadership).
    """
    def __init__(self, terms_csv_path, term_lookup=None):
        if term_lookup:
            self.term_lookup = term_lookup
        else:
            self.term_lookup = TermLookup(terms_csv_path)
            
        # Encoding Mappings
        self.chamber_map = {'rep': 0, 'sen': 1} # 0=House, 1=Senate
        self.party_map = {'Democrat': 0, 'Republican': 1, 'Independent': 2, 'Libertarian': 3}
        
        # State Map (Alphabetical US States)
        states = [
            'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD',
            'MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC',
            'SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'
        ]
        self.state_map = {s: i for i, s in enumerate(states)}
        
    def get_vector(self, bioguide_id, timestamp):
        """
        Returns vector: [Chamber, Party_OneHot(4), State_OneHot(50), Is_Leader]
        Total Dim: 1 + 4 + 50 + 1 = 56
        """
        row = self.term_lookup.get_term_data(bioguide_id, timestamp)
        
        # Defaults
        chamber_val = 0 # Default to House
        party_vec = [0] * 4 # Default Unknown
        state_vec = [0] * 50
        is_leader = 0.0
        
        if row is not None:
            # 1. Chamber
            c_type = str(row.get('type', 'rep')).lower()
            chamber_val = self.chamber_map.get(c_type, 0)
            
            # 2. Party
            p_name = str(row.get('party', ''))
            p_idx = self.party_map.get(p_name, -1)
            if p_idx >= 0:
                party_vec[p_idx] = 1.0
                
            # 3. State
            s_name = str(row.get('state', ''))
            s_idx = self.state_map.get(s_name, -1)
            if s_idx >= 0:
                state_vec[s_idx] = 1.0
                
            # 4. Leadership
            # 'leadership_roles' column is often null if none. If present, it's a list/string.
            roles = row.get('leadership_roles', None)
            if pd.notnull(roles) and str(roles).strip() != '[]':
                is_leader = 1.0
                
        # Construct Tensor
        # Chamber (1) + Party (4) + State (50) + Leader (1)
        final = [float(chamber_val)] + party_vec + state_vec + [is_leader]
        return np.array(final, dtype=np.float32)

class IdeologyLookup(FeatureLookupBase):
    def __init__(self, ideology_csv_path, term_lookup):
        self.df = pd.read_csv(ideology_csv_path)
        self.term_lookup = term_lookup
        
    def get_vector(self, bioguide_id, timestamp):
        # Placeholder for actual ideology lookup logic
        # Assuming 2 dimensions: dim1, dim2
        return np.array([0.0, 0.0], dtype=np.float32)

class DistrictEconLookup(FeatureLookupBase):
    def __init__(self, district_dir, term_lookup):
        self.term_lookup = term_lookup
    
    def get_vector(self, bioguide_id, timestamp):
        # Placeholder: Census data (Median Income, Unemployment, etc.)
        # Returning dummy 5-dim vector
        return np.zeros(5, dtype=np.float32)

class CommitteeLookup(FeatureLookupBase):
    def __init__(self, committee_csv_path, term_lookup):
        self.term_lookup = term_lookup
        
    def get_vector(self, bioguide_id, timestamp):
        # Placeholder: Committee assignments (Finance, Intel, etc.)
        # Returning dummy 10-dim vector (one-hot top committees)
        return np.zeros(10, dtype=np.float32)

class CompanySICLookup(FeatureLookupBase):
    def __init__(self, sic_csv_path):
        pass
        
    def get_vector(self, ticker, timestamp):
        # Placeholder: Sector/Industry embedding
        return np.zeros(8, dtype=np.float32)

class CompanyFinancialsLookup(FeatureLookupBase):
    def __init__(self, financials_csv_path):
        pass
        
    def get_vector(self, ticker, timestamp):
        # Placeholder: Market Cap, PE Ratio, etc.
        return np.zeros(6, dtype=np.float32)