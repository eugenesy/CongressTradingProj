import pandas as pd
import numpy as np
from .base_lookups import FeatureLookupBase

class PoliticianBioLookup(FeatureLookupBase):
    def __init__(self, terms_csv_path, term_lookup):
        self.term_lookup = term_lookup
        self.chamber_map = {'rep': 0, 'sen': 1}
        self.party_map = {'Democrat': 0, 'Republican': 1, 'Independent': 2, 'Libertarian': 3}
        
        states = [
            'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD',
            'MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC',
            'SD','TN','TX','UT','VT','VA','WA','WV','WI','WY', 'DC', 'PR', 'VI', 'GU', 'AS', 'MP'
        ]
        self.state_map = {s: i for i, s in enumerate(states)}
        self.state_dim = len(states)
        self.dim = 1 + 4 + self.state_dim + 1 
        
    def get_vector(self, bioguide_id, timestamp: pd.Timestamp):
        row = self.term_lookup.get_term_data(bioguide_id, timestamp)
        
        chamber_val = 0
        party_vec = [0] * 4
        state_vec = [0] * self.state_dim
        is_leader = 0.0
        
        if row is not None:
            c_type = str(row.get('type', 'rep')).lower()
            chamber_val = self.chamber_map.get(c_type, 0)
            
            p_name = str(row.get('party', ''))
            p_idx = self.party_map.get(p_name, -1)
            if p_idx >= 0: party_vec[p_idx] = 1.0
                
            s_name = str(row.get('state', ''))
            s_idx = self.state_map.get(s_name, -1)
            if s_idx >= 0: state_vec[s_idx] = 1.0
                
            roles = row.get('leadership_roles', None)
            if pd.notnull(roles) and str(roles).strip() not in ['[]', '', 'nan']:
                is_leader = 1.0
                
        final = [float(chamber_val)] + party_vec + state_vec + [is_leader]
        return np.array(final, dtype=np.float32)

class IdeologyLookup(FeatureLookupBase):
    def __init__(self, ideology_csv_path, term_lookup):
        self.term_lookup = term_lookup 
        self.df = pd.read_csv(ideology_csv_path, low_memory=False)
        self.dim = 2
        
        if 'date_window_end' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date_window_end'])
        
        if 'icpsr' in self.df.columns:
            self.df['icpsr'] = pd.to_numeric(self.df['icpsr'], errors='coerce').fillna(0).astype(int).astype(str)

    def _safe_float(self, row, key):
        val = row.get(key)
        if pd.isna(val): return 0.0
        return float(val)

    def get_vector(self, bioguide_id, timestamp: pd.Timestamp):
        target_icpsr = self.term_lookup.get_icpsr(bioguide_id, timestamp)
        if not target_icpsr: return np.zeros(self.dim, dtype=np.float32)

        subset = self.df[self.df['icpsr'] == target_icpsr]
        if subset.empty: return np.zeros(self.dim, dtype=np.float32)
            
        row = None
        if 'date' in subset.columns:
            valid = subset[subset['date'] <= timestamp]
            if not valid.empty:
                row = valid.sort_values('date').iloc[-1]
        
        if row is None: row = subset.iloc[-1]

        return np.array([self._safe_float(row, 'coord1D'), self._safe_float(row, 'coord2D')], dtype=np.float32)

class CommitteeLookup(FeatureLookupBase):
    def __init__(self, committee_csv_path, term_lookup):
        self.term_lookup = term_lookup
        self.df = pd.read_csv(committee_csv_path)
        self.all_committees = sorted([
            'Aging', 'Agriculture', 'Appropriations', 'Armed Services', 'Banking', 
            'Budget', 'Commerce', 'Education', 'Energy', 'Environment', 'Ethics', 
            'Finance', 'Financial Services', 'Foreign Affairs', 'Foreign Relations', 
            'HELP', 'Homeland Security', 'House Administration', 'Indian Affairs', 
            'Intelligence', 'Joint Economic', 'Joint Taxation', 'Judiciary', 
            'Natural Resources', 'Oversight', 'Rules', 'Science', 'Small Business', 
            'Transportation', 'Veterans Affairs', 'Ways and Means'
        ])
        self.comm_map = {c.lower(): i for i, c in enumerate(self.all_committees)}
        self.dim = len(self.all_committees)
        
    def get_vector(self, bioguide_id, timestamp: pd.Timestamp):
        vec = np.zeros(self.dim, dtype=np.float32)
        name_str = self.term_lookup.get_name_for_committee(bioguide_id, timestamp)
        if not name_str: return vec
            
        subset = self.df[self.df['Congressperson'] == name_str]
        if subset.empty: return vec
            
        congress_num = int((timestamp.year - 1789) / 2) + 1
        active = subset[subset['Meeting'] == congress_num]
        
        for _, row in active.iterrows():
            comms = str(row.get('Committees', ''))
            for c in comms.split(';'):
                c_clean = c.strip().lower()
                if c_clean in self.comm_map:
                    vec[self.comm_map[c_clean]] = 1.0
                else:
                    for k, idx in self.comm_map.items():
                        if k in c_clean or c_clean in k:
                            vec[idx] = 1.0
        return vec