import pandas as pd
import numpy as np
import os
import gc
import sys
from pathlib import Path

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config

def compress_lobbying():
    csv_path = config.LOBBYING_EVENTS_PATH
    out_path = csv_path.replace('.csv', '_compressed.parquet')

    if not os.path.exists(csv_path):
        print(f"Warning: Lobbying CSV not found at {csv_path}")
        return

    print(f"\n--- Compressing Lobbying Events ---")
    chunk_aggs = []
    chunk_size = 2_000_000 # Small chunk size to keep RAM < 1GB

    # Map Phase: Read chunks, drop NaNs, aggregate identical events, store tiny summary
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, 
                                          usecols=['date', 'ticker', 'bioguide_id', 'event_type'], 
                                          dtype=str, on_bad_lines='skip')):
        
        chunk = chunk.dropna(subset=['bioguide_id', 'ticker', 'date', 'event_type'])
        chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce').dt.normalize()
        chunk = chunk.dropna(subset=['date'])

        # Aggregate interactions happening on the same exact day
        agg = chunk.groupby(['date', 'ticker', 'bioguide_id', 'event_type']).size().reset_index(name='count')
        chunk_aggs.append(agg)
        
        print(f"Processed chunk {i+1} (Rows: {i * chunk_size} to {(i+1) * chunk_size})...")
        
        # Explicit garbage collection to prevent memory leaks
        del chunk
        gc.collect()

    # Reduce Phase: Combine all summaries and aggregate one final time
    if chunk_aggs:
        print("Combining chunk summaries and writing Parquet...")
        full_df = pd.concat(chunk_aggs, ignore_index=True)
        final_agg = full_df.groupby(['date', 'ticker', 'bioguide_id', 'event_type'])['count'].sum().reset_index()

        # Downcast to make the file size extremely small
        final_agg['bioguide_id'] = final_agg['bioguide_id'].astype('category')
        final_agg['ticker'] = final_agg['ticker'].astype('category')
        final_agg['event_type'] = final_agg['event_type'].astype('category')
        final_agg['count'] = final_agg['count'].astype(np.uint32)

        final_agg.to_parquet(out_path, index=False)
        print(f"SUCCESS: Saved compressed Lobbying Parquet to {out_path}")
        del full_df, final_agg
        gc.collect()

def compress_campaign():
    csv_path = config.CAMPAIGN_FINANCE_EVENTS_PATH
    out_path = csv_path.replace('.csv', '_compressed.parquet')

    if not os.path.exists(csv_path):
        print(f"Warning: Campaign Finance CSV not found at {csv_path}")
        return

    print(f"\n--- Compressing Campaign Finance Events ---")
    chunk_aggs = []
    chunk_size = 2_000_000

    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, 
                                          usecols=['date', 'industry_code', 'bioguide_id', 'weight'], 
                                          dtype={'industry_code': str, 'bioguide_id': str, 'weight': float}, 
                                          on_bad_lines='skip')):
        
        chunk = chunk.dropna(subset=['bioguide_id', 'industry_code', 'date', 'weight'])
        chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce').dt.normalize()
        chunk = chunk.dropna(subset=['date'])

        agg = chunk.groupby(['date', 'industry_code', 'bioguide_id']).agg(
            weight=('weight', 'sum'),
            donation_count=('date', 'size')
        ).reset_index()
        
        chunk_aggs.append(agg)
        print(f"Processed chunk {i+1} (Rows: {i * chunk_size} to {(i+1) * chunk_size})...")
        
        del chunk
        gc.collect()

    if chunk_aggs:
        print("Combining chunk summaries and writing Parquet...")
        full_df = pd.concat(chunk_aggs, ignore_index=True)
        final_agg = full_df.groupby(['date', 'industry_code', 'bioguide_id']).agg(
            weight=('weight', 'sum'),
            donation_count=('donation_count', 'sum')
        ).reset_index()

        final_agg['bioguide_id'] = final_agg['bioguide_id'].astype('category')
        final_agg['industry_code'] = final_agg['industry_code'].astype('category')
        final_agg['weight'] = final_agg['weight'].astype(np.float32)
        final_agg['donation_count'] = final_agg['donation_count'].astype(np.uint32)

        final_agg.to_parquet(out_path, index=False)
        print(f"SUCCESS: Saved compressed Campaign Finance Parquet to {out_path}")

if __name__ == "__main__":
    compress_lobbying()
    compress_campaign()