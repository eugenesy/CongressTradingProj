import pandas as pd

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