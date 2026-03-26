import pandas as pd
import numpy as np
from pathlib import Path
from .base_lookups import FeatureLookupBase

class CompanySICLookup(FeatureLookupBase):
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

class CompanyCategoricalLookup:
    """Dynamically resolves Sector and Industry strings using time-accurate crosswalks."""
    def __init__(self, sic_csv_path, crosswalk_dir):
        self.crosswalk_dir = Path(crosswalk_dir)
        
        self.sic_df = pd.read_csv(sic_csv_path)
        self.sic_df['ticker'] = self.sic_df['ticker'].astype(str).str.upper()
        self.sic_df['sic'] = self.sic_df['sic'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(4)
        
        self.ticker_to_sic = dict(zip(self.sic_df['ticker'], self.sic_df['sic']))
        self.ticker_to_desc = dict(zip(self.sic_df['ticker'], self.sic_df['sic_description']))
        
        cat_path = self.crosswalk_dir / "2013-CAT_to_SIC_to_NAICS_mappings.csv"
        self.cat_df = pd.read_csv(cat_path)
        self.cat_df['SICcode'] = self.cat_df['SICcode'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(4)
        self.sic_to_sector = dict(zip(self.cat_df['SICcode'], self.cat_df['OPEN SECRETS CATEGORY NAME']))
        self.sic3_to_sector = dict(zip(self.cat_df['SICcode'].str[:3], self.cat_df['OPEN SECRETS CATEGORY NAME']))

        dates_df = pd.read_csv(self.crosswalk_dir / "classification_effective_dates.csv")
        dates_df['Effective_Date'] = pd.to_datetime(dates_df['Effective_Date'])
        self.dates_df = dates_df[dates_df['System_Name'].str.contains('NAICS')].sort_values('Effective_Date', ascending=False)
        self.naics_cache = {}

    def _get_naics_industry(self, sic, timestamp: pd.Timestamp):
        active_standard = "2012"
        for _, row in self.dates_df.iterrows():
            if timestamp >= row['Effective_Date']:
                active_standard = row['System_Name'].split()[-1]
                break
                
        if active_standard not in self.naics_cache:
            cw_path = self.crosswalk_dir / f"{active_standard}-NAICS-to-SIC-Crosswalk.csv"
            if cw_path.exists():
                df = pd.read_csv(cw_path)
                sic_col = next(c for c in df.columns if 'SIC Code' in c or c == 'SIC')
                desc_col = next(c for c in df.columns if 'NAICS Title' in c or 'NAICS Description' in c)
                df[sic_col] = df[sic_col].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(4)
                self.naics_cache[active_standard] = dict(zip(df[sic_col], df[desc_col]))
            else:
                self.naics_cache[active_standard] = {}
                
        return self.naics_cache[active_standard].get(sic)

    def get_sector_industry(self, ticker, timestamp: pd.Timestamp):
        ticker = str(ticker).upper()
        sic = self.ticker_to_sic.get(ticker)
        if not sic: return "Unknown", "Unknown"
        
        sector = self.sic_to_sector.get(sic) or self.sic3_to_sector.get(sic[:3])
        industry = self._get_naics_industry(sic, timestamp)
        
        raw_desc = self.ticker_to_desc.get(ticker, "Unknown")
        sector = sector if pd.notna(sector) else raw_desc
        industry = industry if pd.notna(industry) else raw_desc
        
        return str(sector), str(industry)