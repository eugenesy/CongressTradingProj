import pandas as pd
import numpy as np
import os
import re
import datetime

# Standard NAICS 2-digit sectors to ensure consistent vector positions
# 00 is Total. Others are specific sectors.
NAICS_ORDER = [
    '00', '11', '21', '22', '23', '31-33', '42', '44-45', '48-49', 
    '51', '52', '53', '54', '55', '56', '61', '62', '71', '72', '81', '92'
]

STATE_TO_ABBREV = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
    "Puerto Rico": "PR"
}

class DistrictEconomicLookup:
    def __init__(self, raw_data_dir, members_path):
        self.raw_data_dir = raw_data_dir
        self.members_path = members_path
        
        # 1. Load Release Dates
        self.release_schedule = self._load_release_dates()
        
        # 2. Load Member Mapping (BioGuideID + Congress -> State + District)
        self.member_map = self._load_member_mapping()
        
        # 3. Load Economic Data (Key: (State, District, Year) -> Vector)
        self.econ_data = self._load_all_surveys()
        
        # Calculate output dim
        self.output_dim = len(NAICS_ORDER) * 4 # 4 metrics per sector

    def _load_release_dates(self):
        """
        Loads survey release dates to prevent lookahead bias.
        Returns list of dicts sorted by date: [{'year': 2012, 'release': timestamp}, ...]
        """
        path = os.path.join(self.raw_data_dir, "survey_release_dates.csv")
        if not os.path.exists(path):
            print("Warning: survey_release_dates.csv not found. Assuming next-year availability.")
            return []
            
        df = pd.read_csv(path)
        schedule = []
        for _, row in df.iterrows():
            try:
                # Handle various date formats loosely
                dt = pd.to_datetime(row['date'])
                schedule.append({
                    'survey_year': int(row['survey']),
                    'release_date': dt
                })
            except:
                pass
        
        # Sort by release date ascending
        return sorted(schedule, key=lambda x: x['release_date'])

    def _load_member_mapping(self):
        """
        Builds map: (BioGuideID, Congress) -> (State_Abbrev, District_Code)
        """
        if not os.path.exists(self.members_path):
            print("Error: Members file not found.")
            return {}

        df = pd.read_csv(self.members_path)
        mapping = {}
        
        # Ensure columns exist
        req_cols = ['bioguide_id', 'congress', 'state_abbrev', 'district_code']
        if not all(c in df.columns for c in req_cols):
            return {}

        for _, row in df.iterrows():
            if pd.isna(row['bioguide_id']): continue
            
            key = (row['bioguide_id'], int(row['congress']))
            
            # Handle "At Large" districts which are often 0 in one file and 1 in another
            # We usually normalize to 0 or 1. Let's stick to int.
            try:
                dist = int(row['district_code'])
            except:
                dist = 0 # Fallback
                
            mapping[key] = (row['state_abbrev'], dist)
            
        print(f"Loaded mappings for {len(mapping)} member-congress terms.")
        return mapping

    def _parse_geo_name(self, name_str):
        """
        Parses: "Congressional District 1 (116th Congress), Alabama"
        Returns: (State_Abbrev, District_Int, Congress_Int)
        """
        # Regex to capture District Number, Congress Number, and State Name
        # Handles "District 1" and "District (at Large)"
        
        # 1. Extract State (Always after the last comma)
        if "," not in name_str: return None
        state_name = name_str.split(",")[-1].strip()
        state_abbr = STATE_TO_ABBREV.get(state_name)
        if not state_abbr: return None
        
        # 2. Extract Congress
        cong_match = re.search(r'\((\d{3})th Congress\)', name_str)
        congress = int(cong_match.group(1)) if cong_match else None
        
        # 3. Extract District
        if "at Large" in name_str or "at large" in name_str:
            dist = 0 # Standardize At-Large to 0
        else:
            dist_match = re.search(r'District (\d+)', name_str)
            dist = int(dist_match.group(1)) if dist_match else None
            
        if congress is None or dist is None:
            return None
            
        return (state_abbr, dist, congress)

    def _load_all_surveys(self):
        """
        Iterates all CSVs and builds the master data dictionary.
        Key: (State_Abbrev, District, Survey_Year)
        Value: np.array of shape (len(NAICS) * 4,)
        """
        data_store = {}
        
        files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('_CB_survey.csv') or f.endswith('_CB_estimates.csv')]
        
        for f in files:
            # Extract year from filename (e.g. 2013_CB_survey.csv)
            try:
                year = int(f.split('_')[0])
            except:
                continue
                
            path = os.path.join(self.raw_data_dir, f)
            df = pd.read_csv(path)
            
            # Normalize column names (strip whitespace, upper case)
            df.columns = [c.upper().strip() for c in df.columns]
            
            # Identify columns using loose matching
            col_map = {}
            for c in df.columns:
                if 'NAICS' in c and 'CODE' in c and 'LABEL' not in c: col_map['naics'] = c
                if 'ESTAB' in c: col_map['estab'] = c
                if 'EMP' in c and 'LABEL' not in c: col_map['emp'] = c
                if 'PAYANN' in c: col_map['payann'] = c
                if 'PAYQTR1' in c: col_map['payqtr'] = c
                if 'NAME' in c: col_map['geo'] = c
            
            if len(col_map) < 6:
                # Missing critical columns
                continue

            # Iterate rows
            current_geo_key = None # (State, Dist)
            current_congress = None
            
            # Temporary storage for the current district being processed
            # We need to gather all NAICS rows for one district before saving the vector
            # But the file format is usually sorted by District then NAICS.
            # Easiest way: GroupBy in Pandas
            
            # 1. Parse Geography for all rows first
            parsed_geos = df[col_map['geo']].apply(self._parse_geo_name)
            df['parsed_key'] = parsed_geos
            
            # Drop unparseable
            df = df.dropna(subset=['parsed_key'])
            
            # Group by District
            for (state, dist, congress), group in df.groupby('parsed_key'):
                
                # Init empty vector
                # 4 metrics * Num Sectors
                vec = np.zeros(len(NAICS_ORDER) * 4, dtype=np.float32)
                
                # Fill vector
                for _, row in group.iterrows():
                    code = str(row[col_map['naics']]).strip()
                    if code not in NAICS_ORDER:
                        continue
                        
                    idx = NAICS_ORDER.index(code)
                    base_idx = idx * 4
                    
                    # Parse values (remove commas, handle text)
                    try:
                        v_estab = float(str(row[col_map['estab']]).replace(',',''))
                        v_emp = float(str(row[col_map['emp']]).replace(',',''))
                        v_payann = float(str(row[col_map['payann']]).replace(',',''))
                        v_payqtr = float(str(row[col_map['payqtr']]).replace(',',''))
                        
                        # Apply Log Normalization immediately
                        # Log1p is safe for 0s
                        vec[base_idx] = np.log1p(v_estab)
                        vec[base_idx+1] = np.log1p(v_emp)
                        vec[base_idx+2] = np.log1p(v_payann)
                        vec[base_idx+3] = np.log1p(v_payqtr)
                    except:
                        pass
                
                # Store: Key is (State, Dist, Year)
                # Note: We don't store Congress in the key, because Survey Year determines data availability.
                # However, the FILE tells us which Congress definitions were used. 
                # When we look up later, we will look up by (State, Dist) derived from the politician's term.
                # Crucial: 2012 data uses 113th Congress boundaries. If a politician is in 112th (2011-2012), 
                # their district 1 might be physically different from district 1 in 2013.
                # BUT: The Census publishes data based on the *boundaries of that specific year*.
                # So we simply match State/Dist.
                
                data_store[(state, dist, year)] = vec
                
        print(f"District Econ: Loaded data for {len(data_store)} district-year combinations.")
        return data_store

    def get_district_vector(self, bioguide_id, trade_date_ts):
        """
        Main interface.
        1. Determine applicable survey year (based on trade date vs release dates).
        2. Determine Politician's State/District (based on trade date -> Congress).
        3. Look up vector.
        """
        trade_dt = pd.to_datetime(trade_date_ts, unit='s')
        
        # 1. Determine most recent available survey
        # Iterate backwards through release schedule
        target_survey_year = None
        for item in reversed(self.release_schedule):
            if trade_dt >= item['release_date']:
                target_survey_year = item['survey_year']
                break
        
        if target_survey_year is None:
            # No data released yet at time of trade
            return np.zeros(self.output_dim, dtype=np.float32)

        # 2. Determine Congress Number
        # Helper: Calculate Congress based on date
        # 112th: Jan 3 2011 - Jan 3 2013
        # Logic: (Year - 1789) / 2 + 1 (Approximate, simplified for speed)
        # Better: Strict date check
        year = trade_dt.year
        if trade_dt.month == 1 and trade_dt.day < 3:
            year -= 1
        congress_num = int((year - 1789) / 2) + 1
        
        # 3. Get State/District
        loc = self.member_map.get((bioguide_id, congress_num))
        if not loc:
            # Try previous congress if early in year (swearing in gap)?
            # Or just return zero
            return np.zeros(self.output_dim, dtype=np.float32)
            
        state, dist = loc
        
        # 4. Retrieve Data
        # Key: (State, Dist, SurveyYear)
        vec = self.econ_data.get((state, dist, target_survey_year))
        
        if vec is None:
            # Try District 1 if District 0 fails (At Large handling mismatch)
            if dist == 0:
                vec = self.econ_data.get((state, 1, target_survey_year))
            # Try District 0 if District 1 fails
            elif dist == 1:
                vec = self.econ_data.get((state, 0, target_survey_year))
                
        if vec is None:
            return np.zeros(self.output_dim, dtype=np.float32)
            
        return vec