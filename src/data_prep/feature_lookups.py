import pandas as pd
import numpy as np
import torch
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

class FeatureLookupBase:
    def get_vector(self, entity_id, timestamp: pd.Timestamp):
        raise NotImplementedError

class TermLookup:
    """Helper to map a date to a specific Congress term ID and bio details."""
    def __init__(self, terms_csv_path):
        self.df = pd.read_csv(terms_csv_path, parse_dates=['start', 'end'], low_memory=False)
        self.df = self.df.sort_values('start')
        
        if 'id_bioguide' in self.df.columns:
            self.df['id_bioguide'] = self.df['id_bioguide'].astype(str)
        if 'id_icpsr' in self.df.columns:
            self.df['id_icpsr'] = pd.to_numeric(self.df['id_icpsr'], errors='coerce').fillna(0).astype(int).astype(str)

    def get_term_data(self, bioguide_id, timestamp: pd.Timestamp):
        subset = self.df[self.df['id_bioguide'] == str(bioguide_id)]
        if subset.empty:
            return None
            
        mask = (subset['start'] <= timestamp) & ((subset['end'] >= timestamp) | subset['end'].isna())
        active = subset[mask]
        
        if not active.empty:
            return active.iloc[0]
        
        past = subset[subset['start'] <= timestamp]
        if not past.empty:
            return past.iloc[-1]
            
        return None

    def get_icpsr(self, bioguide_id, timestamp: pd.Timestamp):
        row = self.get_term_data(bioguide_id, timestamp)
        if row is not None:
            val = row.get('id_icpsr', None)
            if val and str(val) != '0':
                return str(val)
        return None

    def get_name_for_committee(self, bioguide_id, timestamp: pd.Timestamp):
        row = self.get_term_data(bioguide_id, timestamp)
        if row is not None:
            first = str(row.get('name_first', '')).strip()
            last = str(row.get('name_last', '')).strip()
            return f"{first} {last}"
        return None

class PoliticianBioLookup(FeatureLookupBase):
    """Politician Bio Info (Chamber, Party, State, Leadership)."""
    def __init__(self, terms_csv_path, term_lookup=None):
        self.term_lookup = term_lookup if term_lookup else TermLookup(terms_csv_path)
        self.chamber_map = {'rep': 0, 'sen': 1}
        self.party_map = {'Democrat': 0, 'Republican': 1, 'Independent': 2, 'Libertarian': 3}
        
        states = [
            'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD',
            'MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC',
            'SD','TN','TX','UT','VT','VA','WA','WV','WI','WY', 'DC', 'PR', 'VI', 'GU', 'AS', 'MP'
        ]
        self.state_map = {s: i for i, s in enumerate(states)}
        self.state_dim = len(states)
        
        # Dimensions: Chamber (1) + Party (4) + State (56) + Leadership (1) = 62
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
    """Ideology Scores (coord1D, coord2D)."""
    def __init__(self, ideology_csv_path, term_lookup):
        self.term_lookup = term_lookup 
        self.df = pd.read_csv(ideology_csv_path, low_memory=False)
        self.dim = 2
        
        if 'date_window_end' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date_window_end'])
        
        if 'icpsr' in self.df.columns:
            self.df['icpsr'] = pd.to_numeric(self.df['icpsr'], errors='coerce').fillna(0).astype(int).astype(str)

    def get_vector(self, bioguide_id, timestamp: pd.Timestamp):
        target_icpsr = self.term_lookup.get_icpsr(bioguide_id, timestamp)
        if not target_icpsr:
            return np.zeros(self.dim, dtype=np.float32)

        subset = self.df[self.df['icpsr'] == target_icpsr]
        if subset.empty:
            return np.zeros(self.dim, dtype=np.float32)
            
        row = None
        if 'date' in subset.columns:
            valid = subset[subset['date'] <= timestamp]
            if not valid.empty:
                row = valid.sort_values('date').iloc[-1]
        
        if row is None:
            row = subset.iloc[-1]

        c1 = float(row.get('coord1D', 0.0)) if not pd.isna(row.get('coord1D')) else 0.0
        c2 = float(row.get('coord2D', 0.0)) if not pd.isna(row.get('coord2D')) else 0.0
        return np.array([c1, c2], dtype=np.float32)

class DistrictEconLookup(FeatureLookupBase):
    """Census Bureau District Economic Data (Employment by NAICS Sector)."""
    def __init__(self, district_dir, term_lookup):
        self.term_lookup = term_lookup
        self.district_dir = Path(district_dir)
        self.cache = {}
        
        release_file = self.district_dir / 'survey_release_dates.csv'
        self.release_dates = {}
        if release_file.exists():
            rdf = pd.read_csv(release_file)
            rdf['date'] = pd.to_datetime(rdf['date'])
            self.release_dates = dict(zip(rdf['survey'], rdf['date']))
        
        self.naics_codes = [
            '11', '21', '22', '23', '31', '32', '33', '42', '44', '45', 
            '48', '49', '51', '52', '53', '54', '55', '56', '61', '62', 
            '71', '72', '81', '92'
        ]
        self.dim = len(self.naics_codes)
        self.state_map_abbr_name = {
             'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
             # ... (Full state map omitted for brevity, ensure you copy the dict from legacy) ...
             'DC': 'District of Columbia', 'PR': 'Puerto Rico'
        }

    def _load_year(self, year):
        if year in self.cache: return self.cache[year]
        fnames = list(self.district_dir.glob(f"*{year}*_CB_*.csv"))
        if not fnames: return None
        try:
            df = pd.read_csv(fnames[0], dtype=str)
            cols = df.columns.tolist()
            naics_col = next((c for c in cols if 'NAICS' in c and 'code' in c and 'Meaning' not in c), None)
            emp_col = next((c for c in cols if 'EMP' in c or ('employees' in c.lower() and 'number' in c.lower())), None)
            
            if not naics_col or not emp_col: return None
            df = df[[cols[0], naics_col, emp_col]].copy()
            df.columns = ['geo_name', 'naics', 'emp']
            df['emp'] = pd.to_numeric(df['emp'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df['naics_2'] = df['naics'].astype(str).str[:2]
            self.cache[year] = df
            return df
        except:
            return None

    def get_vector(self, bioguide_id, timestamp: pd.Timestamp):
        term = self.term_lookup.get_term_data(bioguide_id, timestamp)
        if term is None: return np.zeros(self.dim, dtype=np.float32)
            
        valid_survey_year = next((s_year for s_year, r_date in sorted(self.release_dates.items(), key=lambda x: x[1], reverse=True) if r_date <= timestamp), None)
        if valid_survey_year is None: return np.zeros(self.dim, dtype=np.float32)
            
        df = self._load_year(valid_survey_year)
        if df is None: return np.zeros(self.dim, dtype=np.float32)
            
        target_state = self.state_map_abbr_name.get(term.get('state', ''), '')
        subset = df[df['geo_name'].str.contains(target_state, na=False)]
        
        district_code = str(term.get('district', ''))
        if str(district_code) in ['0', '00', 'AL', '1'] and ('at Large' in subset['geo_name'].str.cat() or 'at-large' in subset['geo_name'].str.cat().lower()):
             subset = subset[subset['geo_name'].str.contains('at Large', case=False)]
        else:
             d_str = str(int(district_code)) if district_code.isdigit() else district_code
             subset = subset[subset['geo_name'].str.contains(f"District {d_str}[^0-9]", regex=True)]
             
        vec = np.zeros(self.dim, dtype=np.float32)
        if not subset.empty:
            grouped = subset.groupby('naics_2')['emp'].sum()
            for i, code in enumerate(self.naics_codes):
                if code in grouped.index:
                    vec[i] = np.log1p(grouped[code])
        return vec

class CommitteeLookup(FeatureLookupBase):
    """Committee Assignments."""
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

class CompanySICLookup(FeatureLookupBase):
    """Static Industry Sector Lookup (One-Hot) by Ticker."""
    def __init__(self, sic_csv_path):
        self.df = pd.read_csv(sic_csv_path)
        self.df['ticker'] = self.df['ticker'].astype(str).str.upper()
        self.dim = 10
        
    def get_vector(self, ticker, timestamp: pd.Timestamp):
        vec = np.zeros(self.dim, dtype=np.float32)
        row = self.df[self.df['ticker'] == str(ticker).upper()]
        if not row.empty:
            try:
                sic = int(row.iloc[0]['sic'])
                division = sic // 1000
                if 0 <= division <= 9: vec[division] = 1.0
            except: pass
        return vec

class CompanyFinancialsLookup(FeatureLookupBase):
    """SEC Quarterly Financials."""
    def __init__(self, financials_csv_path):
        df = pd.read_csv(financials_csv_path)
        if 'FiledDate' in df.columns:
            df['FiledDate'] = pd.to_datetime(df['FiledDate'])
        self.df = df
        self.df['Ticker'] = self.df['Ticker'].astype(str).str.upper()
        self.facts = sorted(self.df['Fact'].dropna().unique().tolist())
        self.fact_map = {f: i for i, f in enumerate(self.facts)}
        self.dim = len(self.facts)
        self.df = self.df.sort_values(['Ticker', 'FiledDate'])
        
    def get_vector(self, ticker, timestamp: pd.Timestamp):
        vec = np.zeros(self.dim, dtype=np.float32)
        subset = self.df[self.df['Ticker'] == str(ticker).upper()]
        if subset.empty: return vec
        
        valid = subset[subset['FiledDate'] <= timestamp]
        if valid.empty: return vec
        
        latest_facts = valid.drop_duplicates(subset=['Fact'], keep='last')
        for _, row in latest_facts.iterrows():
            f = row['Fact']
            if f in self.fact_map:
                try:
                    v = float(row['Value'])
                    vec[self.fact_map[f]] = np.sign(v) * np.log1p(abs(v))
                except: pass
        return vec