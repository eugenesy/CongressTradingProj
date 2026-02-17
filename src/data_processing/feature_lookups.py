import pandas as pd
import numpy as np
import torch
from pathlib import Path
import os
import warnings

# Suppress DtypeWarnings
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

class FeatureLookupBase:
    def get_vector(self, entity_id, timestamp):
        """
        Returns a numpy float32 vector for the entity at the given timestamp (Unix epoch).
        """
        raise NotImplementedError

class TermLookup:
    """
    Helper to map a date to a specific Congress term ID and bio details.
    Source: data/raw/congress_all_terms_github.csv
    """
    def __init__(self, terms_csv_path):
        self.df = pd.read_csv(terms_csv_path, parse_dates=['start', 'end'], low_memory=False)
        self.df = self.df.sort_values('start')
        
        # Ensure ID columns are consistent strings for matching
        if 'id_bioguide' in self.df.columns:
            self.df['id_bioguide'] = self.df['id_bioguide'].astype(str)
        if 'id_icpsr' in self.df.columns:
            # Handle potential floats in CSV reading
            self.df['id_icpsr'] = pd.to_numeric(self.df['id_icpsr'], errors='coerce').fillna(0).astype(int).astype(str)

    def get_term_data(self, bioguide_id, timestamp):
        """Find the row in terms csv active at timestamp for the given bioguide_id."""
        date = pd.to_datetime(timestamp, unit='s')
        
        # Filter by ID
        subset = self.df[self.df['id_bioguide'] == str(bioguide_id)]
        if subset.empty:
            return None
            
        # Active term logic: start <= date <= end (or end is NaN/current)
        mask = (subset['start'] <= date) & ((subset['end'] >= date) | subset['end'].isna())
        active = subset[mask]
        
        if not active.empty:
            return active.iloc[0]
        
        # Fallback: Most recent past term
        past = subset[subset['start'] <= date]
        if not past.empty:
            return past.iloc[-1]
            
        return None

    def get_icpsr(self, bioguide_id, timestamp):
        """Retrieve ICPSR ID for Ideology lookup."""
        row = self.get_term_data(bioguide_id, timestamp)
        if row is not None:
            val = row.get('id_icpsr', None)
            if val and str(val) != '0':
                return str(val)
        return None

    def get_name_for_committee(self, bioguide_id, timestamp):
        """Construct 'First Last' name string for Committee lookup."""
        row = self.get_term_data(bioguide_id, timestamp)
        if row is not None:
            # Construct "First Last" to match committee_assignments.csv format
            first = str(row.get('name_first', '')).strip()
            last = str(row.get('name_last', '')).strip()
            return f"{first} {last}"
        return None

class PoliticianBioLookup(FeatureLookupBase):
    """
    Politician Bio Info (Chamber, Party, State, Leadership).
    """
    def __init__(self, terms_csv_path, term_lookup=None):
        self.term_lookup = term_lookup if term_lookup else TermLookup(terms_csv_path)
            
        self.chamber_map = {'rep': 0, 'sen': 1}
        self.party_map = {'Democrat': 0, 'Republican': 1, 'Independent': 2, 'Libertarian': 3}
        
        # Expanded State List including Territories
        states = [
            'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD',
            'MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC',
            'SD','TN','TX','UT','VT','VA','WA','WV','WI','WY',
            'DC', 'PR', 'VI', 'GU', 'AS', 'MP'
        ]
        self.state_map = {s: i for i, s in enumerate(states)}
        self.state_dim = len(states)
        
    def get_vector(self, bioguide_id, timestamp):
        row = self.term_lookup.get_term_data(bioguide_id, timestamp)
        
        chamber_val = 0
        party_vec = [0] * 4
        state_vec = [0] * self.state_dim
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
            roles = row.get('leadership_roles', None)
            if pd.notnull(roles) and str(roles).strip() not in ['[]', '', 'nan']:
                is_leader = 1.0
                
        final = [float(chamber_val)] + party_vec + state_vec + [is_leader]
        return np.array(final, dtype=np.float32)

class IdeologyLookup(FeatureLookupBase):
    """
    Ideology Scores (coord1D, coord2D).
    Links bioguide_id -> icpsr (via TermLookup) -> scores.
    """
    def __init__(self, ideology_csv_path, term_lookup):
        self.term_lookup = term_lookup 
        self.df = pd.read_csv(ideology_csv_path, low_memory=False)
        
        # Ensure timestamp column
        if 'date_window_end' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date_window_end'])
        
        # Ensure ICPSR is string for matching
        if 'icpsr' in self.df.columns:
            self.df['icpsr'] = pd.to_numeric(self.df['icpsr'], errors='coerce').fillna(0).astype(int).astype(str)

    def _safe_float(self, row, key):
        """Helper to extract float and replace NaN/None with 0.0"""
        val = row.get(key)
        # pd.isna handles None, np.nan, and pd.NA
        if pd.isna(val):
            return 0.0
        return float(val)
        
    def get_vector(self, bioguide_id, timestamp):
        date = pd.to_datetime(timestamp, unit='s')
        
        # 1. Get ICPSR from BioGuide
        target_icpsr = self.term_lookup.get_icpsr(bioguide_id, timestamp)
        if not target_icpsr:
            return np.array([0.0, 0.0], dtype=np.float32)

        # 2. Filter Ideology Data
        subset = self.df[self.df['icpsr'] == target_icpsr]
        if subset.empty:
            return np.array([0.0, 0.0], dtype=np.float32)
            
        # 3. Get latest score before or at timestamp
        row = None
        if 'date' in subset.columns:
            valid = subset[subset['date'] <= date]
            if not valid.empty:
                row = valid.sort_values('date').iloc[-1]
        
        # Fallback to last available row if no date match found
        if row is None:
            row = subset.iloc[-1]

        # Return sanitized vector
        return np.array([
            self._safe_float(row, 'coord1D'), 
            self._safe_float(row, 'coord2D')
        ], dtype=np.float32)
    
class DistrictEconLookup(FeatureLookupBase):
    """
    Census Bureau District Economic Data (Employment by NAICS Sector).
    """
    def __init__(self, district_dir, term_lookup):
        self.term_lookup = term_lookup
        self.district_dir = Path(district_dir)
        self.cache = {}
        
        # Load Release Dates: survey, date
        release_file = self.district_dir / 'survey_release_dates.csv'
        self.release_dates = {}
        if release_file.exists():
            rdf = pd.read_csv(release_file)
            # Parse dates like "June 26, 2025"
            rdf['date'] = pd.to_datetime(rdf['date'])
            # Map survey (year) -> release_date
            self.release_dates = dict(zip(rdf['survey'], rdf['date']))
        
        self.naics_codes = [
            '11', '21', '22', '23', '31', '32', '33', '42', '44', '45', 
            '48', '49', '51', '52', '53', '54', '55', '56', '61', '62', 
            '71', '72', '81', '92'
        ]
        self.naics_map = {code: i for i, code in enumerate(self.naics_codes)}
        self.vec_size = len(self.naics_codes)

        self.state_map_abbr_name = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
            'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
            'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
            'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
            'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
            'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
            'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
            'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
            'DC': 'District of Columbia', 'PR': 'Puerto Rico'
        }

    def _load_year(self, year):
        if year in self.cache: return self.cache[year]
        
        fnames = list(self.district_dir.glob(f"*{year}*_CB_*.csv"))
        if not fnames: return None
        fname = fnames[0]
        try:
            df = pd.read_csv(fname, dtype=str)
            cols = df.columns.tolist()
            
            # Identify columns dynamically
            naics_col = next((c for c in cols if 'NAICS' in c and 'code' in c and 'Meaning' not in c), None)
            emp_col = next((c for c in cols if 'EMP' in c or ('employees' in c.lower() and 'number' in c.lower())), None)
            
            if not naics_col or not emp_col: return None
                
            df = df[[cols[0], naics_col, emp_col]].copy()
            df.columns = ['geo_name', 'naics', 'emp']
            df['emp'] = pd.to_numeric(df['emp'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df['naics_2'] = df['naics'].astype(str).str[:2]
            
            self.cache[year] = df
            return df
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            return None

    def get_vector(self, bioguide_id, timestamp):
        term = self.term_lookup.get_term_data(bioguide_id, timestamp)
        if term is None: return np.zeros(self.vec_size, dtype=np.float32)
            
        state_abbr = term.get('state', '')
        district_code = str(term.get('district', ''))
        
        query_date = pd.to_datetime(timestamp, unit='s')
        valid_survey_year = None
        
        # Check release dates: release_date <= query_date
        # Keys are survey years (integers/strings)
        for s_year, r_date in sorted(self.release_dates.items(), key=lambda x: x[1], reverse=True):
            if r_date <= query_date:
                valid_survey_year = s_year
                break
        
        if valid_survey_year is None: return np.zeros(self.vec_size, dtype=np.float32)
            
        df = self._load_year(valid_survey_year)
        if df is None: return np.zeros(self.vec_size, dtype=np.float32)
            
        target_state = self.state_map_abbr_name.get(state_abbr, '')
        subset = df[df['geo_name'].str.contains(target_state, na=False)]
        
        # Handle District Logic
        if str(district_code) in ['0', '00', 'AL', '1'] and ('at Large' in subset['geo_name'].str.cat() or 'at-large' in subset['geo_name'].str.cat().lower()):
             mask = subset['geo_name'].str.contains('at Large', case=False)
             subset = subset[mask]
        else:
             d_str = str(int(district_code)) if district_code.isdigit() else district_code
             pat = f"District {d_str}[^0-9]" 
             subset = subset[subset['geo_name'].str.contains(pat, regex=True)]
             
        if subset.empty: return np.zeros(self.vec_size, dtype=np.float32)

        vec = np.zeros(self.vec_size, dtype=np.float32)
        grouped = subset.groupby('naics_2')['emp'].sum()
        for i, code in enumerate(self.naics_codes):
            if code in grouped.index:
                vec[i] = np.log1p(grouped[code])
        return vec

class CommitteeLookup(FeatureLookupBase):
    """
    Committee Assignments.
    Links bioguide_id -> Name (First Last) -> 'Congressperson' column.
    """
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
        self.vec_size = len(self.all_committees)
        
    def get_vector(self, bioguide_id, timestamp):
        vec = np.zeros(self.vec_size, dtype=np.float32)
        
        # 1. Get Name from BioGuide (First Last)
        name_str = self.term_lookup.get_name_for_committee(bioguide_id, timestamp)
        if not name_str:
            return vec
            
        # 2. Filter by Name (Exact match against Congressperson column)
        subset = self.df[self.df['Congressperson'] == name_str]
        
        if subset.empty:
            return vec
            
        # 3. Filter by Meeting (Congress Session)
        date = pd.to_datetime(timestamp, unit='s')
        congress_num = int((date.year - 1789) / 2) + 1
        
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
    """
    Static Industry Sector Lookup (One-Hot) by Ticker.
    Uses 'sic' column.
    """
    def __init__(self, sic_csv_path):
        self.df = pd.read_csv(sic_csv_path)
        self.df['ticker'] = self.df['ticker'].astype(str).str.upper()
        
    def get_vector(self, ticker, timestamp):
        vec = np.zeros(10, dtype=np.float32)
        row = self.df[self.df['ticker'] == str(ticker).upper()]
        if row.empty: return vec
            
        try:
            # Use 'sic' column as specified
            sic = int(row.iloc[0]['sic'])
            division = sic // 1000
            if 0 <= division <= 9: vec[division] = 1.0
        except: pass
        return vec

class CompanyFinancialsLookup(FeatureLookupBase):
    """
    SEC Quarterly Financials.
    Uses 'FiledDate' for temporal safety.
    """
    def __init__(self, financials_csv_path):
        df = pd.read_csv(financials_csv_path)
        if 'FiledDate' in df.columns:
            df['FiledDate'] = pd.to_datetime(df['FiledDate'])
        self.df = df
        self.df['Ticker'] = self.df['Ticker'].astype(str).str.upper()
        self.facts = sorted(self.df['Fact'].dropna().unique().tolist())
        self.fact_map = {f: i for i, f in enumerate(self.facts)}
        self.vec_size = len(self.facts)
        self.df = self.df.sort_values(['Ticker', 'FiledDate'])
        
    def get_vector(self, ticker, timestamp):
        vec = np.zeros(self.vec_size, dtype=np.float32)
        query_date = pd.to_datetime(timestamp, unit='s')
        
        subset = self.df[self.df['Ticker'] == str(ticker).upper()]
        if subset.empty: return vec
        
        # Strictly FiledDate <= query_date
        valid = subset[subset['FiledDate'] <= query_date]
        if valid.empty: return vec
        
        # Get latest value for each fact
        latest_facts = valid.drop_duplicates(subset=['Fact'], keep='last')
        
        for _, row in latest_facts.iterrows():
            f = row['Fact']
            if f in self.fact_map:
                try:
                    v = float(row['Value'])
                    vec[self.fact_map[f]] = np.sign(v) * np.log1p(abs(v))
                except: pass
        return vec