import pandas as pd
import numpy as np
from pathlib import Path
from .base_lookups import FeatureLookupBase

class DistrictEconLookup(FeatureLookupBase):
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
        except: return None

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