import pandas as pd
import numpy as np
import os
import re

class CommitteeLookup:
    def __init__(self, committee_csv_path, members_csv_path):
        self.committee_path = committee_csv_path
        self.members_path = members_csv_path
        
        # 1. Build Vocabulary (Fixed list of all unique committees)
        self.committee_vocab = self._build_vocab()
        self.vocab_size = len(self.committee_vocab)
        self.vocab_map = {name: i for i, name in enumerate(self.committee_vocab)}
        
        print(f"Committee Data: Found {self.vocab_size} unique committees.")
        
        # 2. Build Member ID Bridge (State/Dist/Congress -> BioGuideID)
        self.loc_to_bioguide = self._build_member_mapping()
        
        # 3. Load Assignments (Key: (BioGuideID, Congress) -> Vector)
        self.assignments = self._load_assignments()

    def _build_vocab(self):
        """
        Scans the file to find all unique committee names to create a fixed feature vector.
        """
        if not os.path.exists(self.committee_path):
            print(f"Warning: Committee file not found at {self.committee_path}")
            return []
            
        df = pd.read_csv(self.committee_path)
        unique_committees = set()
        
        if 'Committees' not in df.columns:
            return []
            
        for val in df['Committees'].dropna():
            # Split by semi-colon
            items = [c.strip() for c in val.split(';') if c.strip()]
            unique_committees.update(items)
            
        return sorted(list(unique_committees))

    def _build_member_mapping(self):
        """
        Creates a map from (State, District, Congress) -> BioGuideID
        using the HSall_members.csv file. This is needed because committee_assignments.csv
        likely lacks BioGuideIDs.
        """
        if not os.path.exists(self.members_path):
            return {}

        df = pd.read_csv(self.members_path)
        mapping = {}
        
        # Ensure we have necessary columns
        # Voteview usually has: 'state_abbrev', 'district_code', 'congress', 'bioguide_id'
        
        for _, row in df.iterrows():
            if pd.isna(row['bioguide_id']): continue
            
            state = row['state_abbrev']
            try:
                congress = int(row['congress'])
                dist = int(row['district_code'])
            except:
                continue
            
            # Key: (State, District_Int, Congress_Int)
            key = (state, dist, congress)
            mapping[key] = row['bioguide_id']
            
        return mapping

    def _parse_district_str(self, dist_str):
        """
        Parses 'AL1' -> 1, 'AL2' -> 2. 
        Handles 'AL' (At Large) -> 0 usually, but we check digits first.
        """
        # Extract all digits
        digits = re.findall(r'\d+', str(dist_str))
        if digits:
            return int(digits[0])
        else:
            # If no digits found (e.g. "AL-At Large"), usually mapped to 0 in Voteview
            return 0

    def _load_assignments(self):
        """
        Loads the csv and converts rows to Multi-Hot vectors linked to BioGuideIDs.
        """
        data = {}
        if not os.path.exists(self.committee_path):
            return data

        df = pd.read_csv(self.committee_path)
        
        for _, row in df.iterrows():
            # Parse Location
            state = row['State']
            meeting = int(row['Meeting']) # Congress session
            dist_int = self._parse_district_str(row['District'])
            
            # 1. Resolve BioGuideID
            # Try Exact Match
            bioguide_id = self.loc_to_bioguide.get((state, dist_int, meeting))
            
            # Fallback: If dist 1 is missing, try 0 (At Large mismatch common issue)
            if not bioguide_id and dist_int == 1:
                bioguide_id = self.loc_to_bioguide.get((state, 0, meeting))
                
            if not bioguide_id:
                # Member not found in Voteview map (might be a vacancy filler or name mismatch)
                continue
                
            # 2. Build Vector
            vec = np.zeros(self.vocab_size, dtype=np.float32)
            raw_comms = str(row['Committees'])
            current_comms = [c.strip() for c in raw_comms.split(';') if c.strip()]
            
            for c in current_comms:
                if c in self.vocab_map:
                    idx = self.vocab_map[c]
                    vec[idx] = 1.0
            
            # Store
            data[(bioguide_id, meeting)] = vec
            
        print(f"Committee Data: Linked assignments for {len(data)} politician-congress periods.")
        return data

    def get_committee_vector(self, bioguide_id, timestamp):
        """
        Returns the multi-hot committee vector for a given politician at a specific time.
        """
        # Calculate Congress based on timestamp
        dt = pd.to_datetime(timestamp, unit='s')
        year = dt.year
        
        # Approximate Congress calculation (same as before)
        # 112th: 2011-2012. 
        # Formula: (Year - 1789) // 2 + 1
        # Adjustment: If Jan 1/2, it might still be previous congress, but keep simple for now.
        congress_num = int((year - 1789) / 2) + 1
        
        key = (bioguide_id, congress_num)
        
        if key in self.assignments:
            return self.assignments[key]
        
        return np.zeros(self.vocab_size, dtype=np.float32)

    def get_feature_names(self):
        return [f"Comm_{name.replace(' ', '_')}" for name in self.committee_vocab]