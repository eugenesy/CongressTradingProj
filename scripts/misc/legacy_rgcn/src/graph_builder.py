import torch
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np

from tqdm import tqdm

class ChocolateGraphBuilder:
    def __init__(self, data_loader):
        """
        Args:
            data_loader (ChocolateDataLoader): Instance with loaded and processed data.
        """
        self.loader = data_loader

    def build_graph(self, transactions_df):
        """
        Constructs a HeteroData object from a specific subset of transactions.
        
        Args:
            transactions_df (pd.DataFrame): Subset of transactions for a specific window.
            
        Returns:
            HeteroData: The constructed heterogeneous graph.
        """
        data = HeteroData()

        # --- 1. Nodes ---
        # Politicians
        # extracting unique politicians from this window AND global mapping
        # Ideally, node IDs should be consistent across windows if we want to do temporal learning on nodes.
        # But for edge classification per snapshot, local IDs mapped to global features works, 
        # OR we use global indices. Let's use GLOBAL indices for simplicity and consistency.
        
        # We need a global mapping of BioGuideID -> Index
        if not hasattr(self, 'pol_id_map'):
             unique_pols = self.loader.transactions['BioGuideID'].unique()
             self.pol_id_map = {pid: i for i, pid in enumerate(unique_pols)}
             self.num_pols = len(unique_pols)
             
        # Companies
        if not hasattr(self, 'company_id_map'):
             unique_tickers = self.loader.transactions['Ticker'].unique()
             self.company_id_map = {tid: i for i, tid in enumerate(unique_tickers)}
             self.num_companies = len(unique_tickers)

        # Assign Node Features (Static)
        # For this prototype, we'll build simple feature tensors. 
        # In a full production, these would be pre-computed.
        
        # Politician Features: [Party_ID, State_ID, Chamber_ID]
        pol_features = torch.zeros((self.num_pols, 3), dtype=torch.float)
        
        # We need a lookup for static info. 
        # Let's create a quick reference DF from the main loader
        pol_ref = self.loader.transactions[['BioGuideID', 'Party', 'State', 'Chamber']].drop_duplicates('BioGuideID').set_index('BioGuideID')
        
        for pid, idx in self.pol_id_map.items():
            if pid in pol_ref.index:
                row = pol_ref.loc[pid]
                p_idx = self.loader.party_map.get(row['Party'], -1)
                s_idx = self.loader.state_map.get(row['State'], -1)
                c_idx = self.loader.chamber_map.get(row['Chamber'], -1)
                pol_features[idx] = torch.tensor([p_idx, s_idx, c_idx], dtype=torch.float)
        
        data['politician'].x = pol_features
        
        # Company Features: [Id_Placeholder]
        # User said "One-hot or simple ID; No industry embedding yet"
        # Let's just use a dummy feature or a simple ID for now. 
        # PyG often requires x, so let's give it the Tickder ID as a float? 
        # Or better, a 1-hot if small, but tickers are 3000+. 
        # Let's make it a generic '1' for now (structural only).
        data['company'].x = torch.ones((self.num_companies, 1), dtype=torch.float)

        # --- 2. Edges ---
        # Edges: (Politician, trades, Company)
        # Prepare storage for BUYS and SELLS
        buys_src, buys_dst, buys_attr, buys_y = [], [], [], []
        sells_src, sells_dst, sells_attr, sells_y = [], [], [], []

        for _, row in tqdm(transactions_df.iterrows(), total=len(transactions_df), desc="Building RGCN Edges", leave=False):
            pid = row['BioGuideID']
            ticker = row['Ticker']
            
            if pid not in self.pol_id_map or ticker not in self.company_id_map:
                continue 
                
            p_idx = self.pol_id_map[pid]
            c_idx = self.company_id_map[ticker]
            
            # Features
            # 1. Amount
            amt = self._parse_amount(row['Trade_Size_USD'])
            
            # 2. Filing Gap
            traded_date = row['Traded']
            filed_date = row['Filed']
            if pd.isna(traded_date) or pd.isna(filed_date):
                 gap_days = 0.0
            else:
                 gap_days = (filed_date - traded_date).days
            gap_feat = float(max(0, gap_days))

            # 3. Time Decay
            time_feat = 0.0 
            
            # New Feature Vector: [Amount, FilingGap, TimeDecay] (Size=3)
            # REMOVED: is_purchase (redundant due to edge type)
            feat_vec = [amt, gap_feat, time_feat]
            
            # Label
            lbl = row.get('Label_1M', 0.0)
            if pd.isna(lbl): lbl = 0.0
            
            # Route to correct edge type
            is_purchase = 'Purchase' in str(row['Transaction'])
            
            if is_purchase:
                buys_src.append(p_idx)
                buys_dst.append(c_idx)
                buys_attr.append(feat_vec)
                buys_y.append(lbl)
            else:
                sells_src.append(p_idx)
                sells_dst.append(c_idx)
                sells_attr.append(feat_vec)
                sells_y.append(lbl)

        # Helper to assign to graph
        def assign_edges(data, relation, src, dst, attr, y):
            if len(src) == 0:
                # Handle empty case safely
                data['politician', relation, 'company'].edge_index = torch.empty((2, 0), dtype=torch.long)
                data['politician', relation, 'company'].edge_attr = torch.empty((0, 3), dtype=torch.float)
                data['politician', relation, 'company'].y = torch.empty((0,), dtype=torch.float)
                return

            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_attr = torch.tensor(attr, dtype=torch.float)
            edge_labels = torch.tensor(y, dtype=torch.float)
            
            data['politician', relation, 'company'].edge_index = edge_index
            data['politician', relation, 'company'].edge_attr = edge_attr
            data['politician', relation, 'company'].y = edge_labels
            
        # Assign Buys and Sells
        assign_edges(data, 'buys', buys_src, buys_dst, buys_attr, buys_y)
        assign_edges(data, 'sells', sells_src, sells_dst, sells_attr, sells_y)
        
        # --- 3. Reverse Edges ---
        # We need reverse edges for BOTH types to allow info flow from Company -> Politician
        
        if len(buys_src) > 0:
            rev_buys_index = torch.stack([torch.tensor(buys_dst, dtype=torch.long), torch.tensor(buys_src, dtype=torch.long)], dim=0)
            data['company', 'rev_buys', 'politician'].edge_index = rev_buys_index
        else:
             data['company', 'rev_buys', 'politician'].edge_index = torch.empty((2, 0), dtype=torch.long)
            
        if len(sells_src) > 0:
            rev_sells_index = torch.stack([torch.tensor(sells_dst, dtype=torch.long), torch.tensor(sells_src, dtype=torch.long)], dim=0)
            data['company', 'rev_sells', 'politician'].edge_index = rev_sells_index
        else:
            data['company', 'rev_sells', 'politician'].edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # We don't necessarily need features on reverse edges for now, or we copy them.
        # Let's simple not assign attributes to rev_trades, or SAGEConv will ignore them if not configured.
        # But 'to_hetero' might expect consistent feature dims.
        # Let's just create the topology first.

        return data

    def _parse_amount(self, size_range):
        """
        Parses Trade_Size_USD string to numerical midpoint and applies log1p scaling.
        Logic adapted from apple/src/ml/preprocess.py.
        """
        try:
            if pd.isna(size_range):
                return 0.0
                
            val = 0.0
            if isinstance(size_range, str):
                size_range = size_range.replace(",", "")
                if " - " in size_range:
                    low, high = size_range.split(" - ")
                    # Cleanup "1,000" -> "1000" again if needed or strip
                    high = high.strip().rstrip('.')
                    val = (float(low) + float(high)) / 2
                else:
                    size_range = size_range.strip().rstrip('.')
                    # Handle cases like "50000000 +" if they exist, though apple logic handled specific strings
                    # If conversion fails, default to 0
                    val = float(size_range)
            elif isinstance(size_range, (int, float)):
                val = float(size_range)
                
            # Apple logic: np.log1p(midpoint)
            return float(np.log1p(val))
        except:
            return 0.0
