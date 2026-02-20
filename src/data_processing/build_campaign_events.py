import pandas as pd
import glob
import os
from tqdm import tqdm
from src import config

def load_legislator_map():
    if os.path.exists(config.LEGISLATORS_CROSSWALK_PATH):
        # Added low_memory=False to suppress DtypeWarnings
        df = pd.read_csv(config.LEGISLATORS_CROSSWALK_PATH, low_memory=False)
        if 'id_opensecrets' in df.columns and 'id_bioguide' in df.columns:
            # Drop duplicates to prevent InvalidIndexError during mapping
            mapping_df = df[['id_opensecrets', 'id_bioguide']].dropna().drop_duplicates(subset=['id_opensecrets'])
            return mapping_df.set_index('id_opensecrets')['id_bioguide'].to_dict()
    print(f"WARNING: Legislator crosswalk not found at {config.LEGISLATORS_CROSSWALK_PATH}.")
    return {}

def build_campaign_events():
    print("Building Campaign Finance Event Stream (Using Filing Dates)...")
    
    cid_to_bioguide = load_legislator_map()
    all_events = []

    # --- Step 1: Corporate PACs ---
    pac_files = glob.glob(str(config.CAMPAIGN_FINANCE_DIR / config.CAMPAIGN_PACS_PATTERN))
    
    print(f"Found {len(pac_files)} PAC files.")
    for f in tqdm(pac_files, desc="Processing PAC Files"):
        df = pd.read_csv(f, on_bad_lines='skip', low_memory=False)
        
        if 'estimated_filing_date' not in df.columns:
            continue

        df = df.dropna(subset=['CID', 'RealCode', 'estimated_filing_date'])
        
        # Safely map to BioGuide IDs using the deduplicated dictionary
        df['bioguide_id'] = df['CID'].map(cid_to_bioguide)
        df = df.dropna(subset=['bioguide_id'])
        
        df = df[['estimated_filing_date', 'RealCode', 'bioguide_id', 'Amount']].copy()
        all_events.append(df)

    # --- Step 2: 527 Expenditures ---
    if os.path.exists(config.DATA_527_EXPENDITURES_PATH):
        print("\nProcessing 527 Expenditures...")
        exp_df = pd.read_csv(config.DATA_527_EXPENDITURES_PATH, on_bad_lines='skip', low_memory=False)
        cmtes_df = pd.read_csv(config.DATA_527_COMMITTEES_PATH, on_bad_lines='skip', low_memory=False)
        
        if 'estimated_filing_date' in exp_df.columns:
            ein_to_industry = cmtes_df.set_index('EIN')['PrimCode'].to_dict()
            
            exp_df['bioguide_id'] = exp_df['RecipID'].map(cid_to_bioguide)
            exp_df['RealCode'] = exp_df['EIN'].map(ein_to_industry)
            
            exp_df = exp_df.dropna(subset=['bioguide_id', 'RealCode', 'estimated_filing_date'])
            exp_df = exp_df[['estimated_filing_date', 'RealCode', 'bioguide_id', 'Amount']].copy()
            all_events.append(exp_df)
        else:
            print("Skipping 527 Expenditures: 'estimated_filing_date' missing.")

    # --- Step 3: Aggregation ---
    if all_events:
        print("\nConcatenating and Aggregating Events...")
        full_df = pd.concat(all_events, ignore_index=True)
        
        # Parse dates
        tqdm.pandas(desc="Parsing Dates")
        full_df['estimated_filing_date'] = pd.to_datetime(full_df['estimated_filing_date'], errors='coerce')
        full_df = full_df.dropna(subset=['estimated_filing_date'])
        
        # Aggregate to Weekly "Pulses"
        full_df = full_df.set_index('estimated_filing_date')
        
        # Group and sum
        agg_df = full_df.groupby([pd.Grouper(freq='W'), 'RealCode', 'bioguide_id'])['Amount'].sum().reset_index()
        
        agg_df.rename(columns={'estimated_filing_date': 'date', 'RealCode': 'industry_code', 'Amount': 'weight'}, inplace=True)
        agg_df['event_type'] = 'DONATION'
        
        agg_df.to_csv(config.CAMPAIGN_FINANCE_EVENTS_PATH, index=False)
        print(f"\nSUCCESS: Saved {len(agg_df)} Campaign Events to {config.CAMPAIGN_FINANCE_EVENTS_PATH}")
    else:
        print("\nNo Campaign Events found.")

if __name__ == "__main__":
    build_campaign_events()