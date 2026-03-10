import pandas as pd
import numpy as np
import os
import ast
import re
from tqdm import tqdm
from src import config

def load_crosswalks():
    """Loads NAICS->SIC, SIC->Ticker, and Legislator Mappings."""
    print("Loading Crosswalks...")
    
    naics_sic = pd.read_csv(config.NAICS_TO_SIC_PATH)
    naics_sic['NAICS'] = naics_sic['NAICS'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
    naics_sic['SIC'] = naics_sic['SIC'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(4)
    naics_to_sic_map = naics_sic.groupby('NAICS')['SIC'].apply(list).to_dict()

    if os.path.exists(config.COMPANY_SIC_DATA_PATH):
        sic_data = pd.read_csv(config.COMPANY_SIC_DATA_PATH)
        sic_data['sic'] = sic_data['sic'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(4)
        sic_to_ticker_map = sic_data.groupby('sic')['ticker'].apply(list).to_dict()
    else:
        print(f"WARNING: {config.COMPANY_SIC_DATA_PATH} not found.")
        sic_to_ticker_map = {}

    if os.path.exists(config.LEGISLATORS_CROSSWALK_PATH):
        leg_df = pd.read_csv(config.LEGISLATORS_CROSSWALK_PATH, low_memory=False)
        leg_map_df = leg_df[['id_icpsr', 'id_bioguide']].dropna().drop_duplicates()
        leg_map_df['id_icpsr'] = leg_map_df['id_icpsr'].astype(int)
        icpsr_to_bioguide = leg_map_df.set_index('id_icpsr')['id_bioguide'].to_dict()
    else:
        print(f"WARNING: Legislator crosswalk not found at {config.LEGISLATORS_CROSSWALK_PATH}. Voting edges will fail.")
        icpsr_to_bioguide = {}

    return naics_to_sic_map, sic_to_ticker_map, icpsr_to_bioguide

def parse_bill_ids(id_str):
    try:
        if pd.isna(id_str): return []
        matches = re.findall(r'[a-zA-Z0-9]+-[0-9]+', str(id_str))
        return [m.lower().strip() for m in matches]
    except:
        return []

def build_lobbying_events():
    print("Building Lobbying & Voting Event Stream (Using Filing Dates)...")
    
    bills_df = pd.read_csv(config.LOBBYING_BILLS_PATH)
    clients_df = pd.read_csv(config.LOBBYING_CLIENTS_PATH)
    reports_df = pd.read_csv(config.LOBBYING_REPORTS_PATH)
    issues_df = pd.read_csv(config.LOBBYING_ISSUES_PATH)
    
    has_votes = os.path.exists(config.VOTEVIEW_VOTES_PATH) and os.path.exists(config.VOTEVIEW_ROLLCALLS_PATH)
    if has_votes:
        print("Loading VoteView Data...")
        votes_df = pd.read_csv(config.VOTEVIEW_VOTES_PATH)
        rollcalls_df = pd.read_csv(config.VOTEVIEW_ROLLCALLS_PATH, low_memory=False)

    bills_df['bill_id'] = bills_df['bill_id'].astype(str).str.lower().str.strip()
    reports_df['report_uuid'] = reports_df['report_uuid'].astype(str)
    reports_df['lob_id'] = reports_df['lob_id'].astype(str)
    clients_df['lob_id'] = clients_df['lob_id'].astype(str)
    
    naics_map, sic_ticker_map, icpsr_map = load_crosswalks()

    # --- Step 1: Map Clients to Tickers ---
    clients_df['naics_str'] = clients_df['naics'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)
    client_ticker_records = []
    
    print("Mapping Clients to Tickers...")
    for idx, row in tqdm(clients_df.iterrows(), total=len(clients_df), desc="Mapping Clients"):
        target_tickers = []
        for sic in naics_map.get(row['naics_str'], []):
            target_tickers.extend(sic_ticker_map.get(sic, []))
        if target_tickers:
            for t in set(target_tickers):
                client_ticker_records.append({'lob_id': row['lob_id'], 'ticker': t})
                
    client_ticker_df = pd.DataFrame(client_ticker_records)
    print(f"Mapped {client_ticker_df['lob_id'].nunique()} clients to {client_ticker_df['ticker'].nunique()} unique tickers.")

    if client_ticker_df.empty:
        print("No clients mapped to tickers. Exiting.")
        return

    # --- Step 2: Link Reports to Bills ---
    print("Parsing Bill IDs from Issues...")
    tqdm.pandas(desc="Parsing Bill IDs")
    issues_with_bills = issues_df.dropna(subset=['bill_id_agg']).copy()
    issues_with_bills['bill_id_list'] = issues_with_bills['bill_id_agg'].progress_apply(parse_bill_ids)
    
    issues_exploded = issues_with_bills.explode('bill_id_list').rename(columns={'bill_id_list': 'bill_id'})
    issues_exploded['report_uuid'] = issues_exploded['report_uuid'].astype(str)
    issues_exploded['bill_id'] = issues_exploded['bill_id'].astype(str)

    print("Merging Issues with Reports...")
    bill_client_chain = pd.merge(
        issues_exploded[['bill_id', 'report_uuid']], 
        reports_df[['report_uuid', 'lob_id', 'estimated_filing_date']], 
        on='report_uuid',
        how='inner'
    )
    
    # NEW OPTIMIZATION: Drop duplicates BEFORE merging to save massive amounts of memory
    base_chain = bill_client_chain[['lob_id', 'bill_id', 'estimated_filing_date']].drop_duplicates()
    ticker_bill_df = pd.merge(base_chain, client_ticker_df, on='lob_id').drop(columns=['lob_id'])
    ticker_bill_df = ticker_bill_df.drop_duplicates()

    all_events_dfs = []

    # --- Step 3: Strong Edges (Sponsorship) ---
    if getattr(config, 'INCLUDE_LOBBYING_SPONSORSHIP', True):
        print("Generating Strong Edges (Sponsorship)...")
        strong_df = pd.merge(ticker_bill_df, bills_df[['bill_id', 'bioguide_id']], on='bill_id', how='inner')
        strong_df = strong_df.dropna(subset=['bioguide_id', 'estimated_filing_date'])
        
        # Vectorized Event Creation (Replaces the 9-minute loop)
        strong_df = strong_df[['estimated_filing_date', 'ticker', 'bioguide_id']].drop_duplicates()
        strong_df.rename(columns={'estimated_filing_date': 'date'}, inplace=True)
        strong_df['event_type'] = 'LOBBY_STRONG'
        strong_df['weight'] = 1.0
        
        print(f"Valid Sponsorship Connections Found: {len(strong_df)}")
        all_events_dfs.append(strong_df)
    else:
        print("Skipping Strong Edges (Sponsorship) per config.")

    # --- Step 4: Weak Edges (Voting) ---
    if getattr(config, 'INCLUDE_LOBBYING_VOTING', True) and has_votes:
        print("Generating Weak Edges (Voting)...")
        if 'bill_number' in bills_df.columns and 'bill_number' in rollcalls_df.columns:
            lobbied_bills = ticker_bill_df['bill_id'].unique()
            
            target_bills_df = bills_df[bills_df['bill_id'].isin(lobbied_bills)][['bill_id', 'bill_type', 'bill_number', 'congress_number']].copy()
            target_bills_df['clean_type'] = target_bills_df['bill_type'].astype(str).str.replace(r'[^a-zA-Z]', '', regex=True).str.upper()
            target_bills_df['clean_num'] = target_bills_df['bill_number'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            target_bills_df['vv_bill_number'] = target_bills_df['clean_type'] + target_bills_df['clean_num']
            target_bills_df['congress_number'] = target_bills_df['congress_number'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            
            rollcalls_merge = rollcalls_df[['congress', 'rollnumber', 'bill_number']].copy()
            rollcalls_merge['vv_bill_number'] = rollcalls_merge['bill_number'].astype(str).str.replace(r'[^a-zA-Z0-9]', '', regex=True).str.upper()
            rollcalls_merge['congress'] = rollcalls_merge['congress'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            rollcalls_merge['rollnumber'] = rollcalls_merge['rollnumber'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            
            bill_votes_map = pd.merge(
                target_bills_df[['bill_id', 'congress_number', 'vv_bill_number']], 
                rollcalls_merge[['congress', 'rollnumber', 'vv_bill_number']], 
                left_on=['congress_number', 'vv_bill_number'], 
                right_on=['congress', 'vv_bill_number']
            )
            
            bill_votes_map = bill_votes_map[~bill_votes_map['congress'].isin(['nan', 'None', ''])]
            bill_votes_map = bill_votes_map[~bill_votes_map['rollnumber'].isin(['nan', 'None', ''])]
            
            yea_votes = votes_df[votes_df['cast_code'].isin([1, 2, 3])].copy()
            yea_votes['bioguide_id'] = yea_votes['icpsr'].map(icpsr_map)
            yea_votes = yea_votes.dropna(subset=['bioguide_id', 'congress', 'rollnumber'])
            
            yea_votes['congress'] = yea_votes['congress'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            yea_votes['rollnumber'] = yea_votes['rollnumber'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            yea_votes = yea_votes[~yea_votes['congress'].isin(['nan', 'None', ''])]
            
            print("Mapping Legislator votes to Bills...")
            bill_to_legislator = pd.merge(
                bill_votes_map[['bill_id', 'congress', 'rollnumber']], 
                yea_votes[['congress', 'rollnumber', 'bioguide_id']], 
                on=['congress', 'rollnumber']
            )
            
            bill_to_legislator = bill_to_legislator[['bill_id', 'bioguide_id']].drop_duplicates()
            
            print("Merging Voting Records with Lobbying Clients (Chunked to prevent Memory Error)...")
            
            # NEW OPTIMIZATION: Process one Ticker at a time to prevent Cartesian explosion
            weak_events_list = []
            valid_bills_with_votes = bill_to_legislator['bill_id'].unique()
            tb_subset = ticker_bill_df[ticker_bill_df['bill_id'].isin(valid_bills_with_votes)]
            
            for ticker, group in tqdm(tb_subset.groupby('ticker'), desc="Building Weak Edges"):
                merged = pd.merge(group[['bill_id', 'estimated_filing_date']], bill_to_legislator, on='bill_id')
                unique_edges = merged[['estimated_filing_date', 'bioguide_id']].drop_duplicates()
                unique_edges['ticker'] = ticker
                weak_events_list.append(unique_edges)
                
            if weak_events_list:
                weak_df = pd.concat(weak_events_list, ignore_index=True)
                weak_df.rename(columns={'estimated_filing_date': 'date'}, inplace=True)
                weak_df['event_type'] = 'LOBBY_WEAK'
                weak_df['weight'] = 0.5
                print(f"Valid Voting Connections Found: {len(weak_df)}")
                all_events_dfs.append(weak_df)
            else:
                print("Valid Voting Connections Found: 0")
    else:
        print("Skipping Weak Edges (Voting) per config.")

    if all_events_dfs:
        final_events_df = pd.concat(all_events_dfs, ignore_index=True)
        print(f"\nSUCCESS: Generated {len(final_events_df)} total Lobbying/Voting Events.")
        final_events_df.to_csv(config.LOBBYING_EVENTS_PATH, index=False)
    else:
        print("\nGenerated 0 total events.")

if __name__ == "__main__":
    build_lobbying_events()