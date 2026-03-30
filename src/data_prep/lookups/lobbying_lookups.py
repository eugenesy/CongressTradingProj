import pandas as pd
import numpy as np
import os
import pyarrow.parquet as pq
from src import config

class LobbyingLookup:
    def __init__(self, events_path=None):
        self.events_path = events_path or config.LOBBYING_EVENTS_PATH
        self.compressed_path = self.events_path.replace('.csv', '_compressed.parquet')
        
        if not os.path.exists(self.compressed_path):
            print(f"CRITICAL WARNING: Compressed parquet not found at {self.compressed_path}. Please run compression script.")

    def get_edges(self, timestamp: pd.Timestamp) -> pd.DataFrame:
        if not os.path.exists(self.compressed_path):
            return pd.DataFrame()
            
        chunk_aggs = []
        pf = pq.ParquetFile(self.compressed_path)
        
        # Read the file in memory-safe chunks
        for batch in pf.iter_batches(batch_size=5_000_000):
            df_chunk = batch.to_pandas()
            df_chunk['date'] = pd.to_datetime(df_chunk['date'])
            
            # Filter the chunk
            past_events = df_chunk[df_chunk['date'] <= timestamp]
            if past_events.empty:
                continue
                
            # Aggregate the valid rows in this chunk (observed=True fixes the warning)
            agg = past_events.groupby(['bioguide_id', 'ticker', 'event_type'], observed=True).agg(
                interaction_count=('count', 'sum'),
                latest_interaction_date=('date', 'max')
            ).reset_index()
            
            chunk_aggs.append(agg)
            
        if not chunk_aggs:
            return pd.DataFrame()
            
        # Combine the aggregated chunks and reduce one final time
        final_df = pd.concat(chunk_aggs, ignore_index=True)
        final_agg = final_df.groupby(['bioguide_id', 'ticker', 'event_type'], observed=True).agg(
            interaction_count=('interaction_count', 'sum'),
            latest_interaction_date=('latest_interaction_date', 'max')
        ).reset_index()
        
        return final_agg