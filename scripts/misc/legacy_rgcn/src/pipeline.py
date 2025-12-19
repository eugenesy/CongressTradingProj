import pandas as pd
from datetime import timedelta
# from .data_loader import ChocolateDataLoader
# from .graph_builder import ChocolateGraphBuilder
try:
    from .data_loader import ChocolateDataLoader
    from .graph_builder import ChocolateGraphBuilder
except ImportError:
    from data_loader import ChocolateDataLoader
    from graph_builder import ChocolateGraphBuilder
import os
import torch
from dateutil.relativedelta import relativedelta

class ChocolatePipeline:
    def __init__(self, transaction_path, price_data_path):
        self.loader = ChocolateDataLoader(transaction_path, price_data_path)
        self.builder = None # Initialized after load
        
    def initialize(self):
        """Loads data and prepares the builder."""
        self.loader.load_data()
        self.loader.preprocess_node_features()
        self.builder = ChocolateGraphBuilder(self.loader)
        
    def generate_rolling_graphs(self, train_years=1.0, val_years=0.5, test_years=0.5, step_years=1.0):
        """
        Generates a sequence of (Train, Val, Test) HeteroData objects using a rolling window.
        
        Args:
            train_years (float): Duration of training window in years.
            val_years (float): Duration of validation window in years.
            test_years (float): Duration of test window in years.
            step_years (float): Sliding step in years.
            
        Yields:
            dict: {
                'window_id': int,
                'train_graph': HeteroData,
                'val_graph': HeteroData,
                'test_graph': HeteroData,
                'dates': {'train_start': ..., 'test_end': ...}
            }
        """
        if self.builder is None:
            self.initialize()
            
        # 1. Determine Date Range
        min_date = self.loader.transactions['Filed'].min()
        max_date = self.loader.transactions['Filed'].max()
        
        print(f"Dataset Range: {min_date} to {max_date}")
        
        # 2. Determine Gap (Embargo)
        # Search for 'Label_' columns to find lookahead
        # Use relativedelta for Month-based gaps to handle calendar logic better
        gap_delta = relativedelta(days=0)
        gap_desc = "0 days" 
        
        cols = self.loader.transactions.columns
        if 'Label_1M' in cols:
            gap_delta = relativedelta(months=1, days=7) # +7 days buffer for safety (filing delays etc)
            gap_desc = "~1 Month + buffer"
            print(f"Auto-detected Label_1M. Setting gap/embargo to {gap_desc}.")
        elif 'Label_3M' in cols:
             gap_delta = relativedelta(months=3, days=7)
             gap_desc = "~3 Months + buffer"
             print(f"Auto-detected Label_3M. Setting gap/embargo to {gap_desc}.")
             
        # Convert years to timedeltas (approximate)
        # Using relativedelta for cleaner month handling would be better per iteration
        
        current_start = min_date
        window_id = 0
        
        while True:
            # Calculate Split Boundaries
            # Train
            train_start = current_start
            train_end = train_start + relativedelta(days=int(train_years * 365))
            
            # Val (Start = Train End + Gap)
            val_start = train_end + gap_delta
            val_end = val_start + relativedelta(days=int(val_years * 365))
            
            # Test
            # Apply gap before test too to be safe/standard
            test_start = val_end + gap_delta
            test_end = test_start + relativedelta(days=int(test_years * 365))
            
            if test_end > max_date:
                break
                
            print(f"Window {window_id}: Train[{train_start.date()}:{train_end.date()}] -> Gap -> Val[{val_start.date()}:{val_end.date()}] -> Gap -> Test[{test_start.date()}:{test_end.date()}]")
            
            # Slice Data
            train_mask = (self.loader.transactions['Filed'] >= train_start) & (self.loader.transactions['Filed'] < train_end)
            val_mask = (self.loader.transactions['Filed'] >= val_start) & (self.loader.transactions['Filed'] < val_end)
            test_mask = (self.loader.transactions['Filed'] >= test_start) & (self.loader.transactions['Filed'] < test_end)
            
            # Build Graphs
            train_graph = self.builder.build_graph(self.loader.transactions[train_mask])
            val_graph = self.builder.build_graph(self.loader.transactions[val_mask])
            test_graph = self.builder.build_graph(self.loader.transactions[test_mask])
            
            yield {
                'window_id': window_id,
                'train_graph': train_graph,
                'val_graph': val_graph,
                'test_graph': test_graph,
                'dates': {
                    'train_start': train_start, 'train_end': train_end,
                    'val_start': val_start, 'val_end': val_end,
                    'test_start': test_start, 'test_end': test_end
                }
            }
            
            # Slide Window
            current_start += relativedelta(days=int(step_years * 365))
            window_id += 1

    def save_window(self, window_data, output_root="processed_graphs"):
        """
        Saves a generated window to disk as .pt files.
        Structure:
            output_root/
                window_0/
                    train.pt
                    val.pt
                    test.pt
                    metadata.pt (dates)
        """
        w_id = window_data['window_id']
        dir_path = os.path.join(output_root, f"window_{w_id}")
        os.makedirs(dir_path, exist_ok=True)
        
        torch.save(window_data['train_graph'], os.path.join(dir_path, "train.pt"))
        torch.save(window_data['val_graph'], os.path.join(dir_path, "val.pt"))
        torch.save(window_data['test_graph'], os.path.join(dir_path, "test.pt"))
        torch.save(window_data['dates'], os.path.join(dir_path, "metadata.pt"))
        
        print(f"Saved Window {w_id} to {dir_path}")

