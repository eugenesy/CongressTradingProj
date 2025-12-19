import os
import sys
import torch
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta

from src.pipeline import ChocolatePipeline
from src.visualize_graph import ChocolateVisualizer

def calculate_total_windows(min_date, max_date, train_years, val_years, test_years, step_years, gap_months=1):
    """
    Helper to estimate total windows for tqdm.
    """
    current_start = min_date
    count = 0
    
    # Gap approximation
    gap_delta = relativedelta(months=gap_months, days=7)
    
    while True:
        train_end = current_start + relativedelta(days=int(train_years * 365))
        val_start = train_end + gap_delta
        val_end = val_start + relativedelta(days=int(val_years * 365))
        test_start = val_end + gap_delta
        test_end = test_start + relativedelta(days=int(test_years * 365))
        
        if test_end > max_date:
            break
        count += 1
        current_start += relativedelta(days=int(step_years * 365))
        
    return count

def main():
    # --- Configuration ---
    # Paths (verified from previous runs)
    TX_PATH = "/data1/user_syeugene/fintech/apple/data/processed/ml_dataset_reduced_attributes.csv"
    # PRICE_PATH = "/data1/user_syeugene/fintech/apple/data/processed/all_tickers_historical_data.pkl"
    PRICE_PATH = None # User requested skip for speed
    
    # Window Config (2021-2024 Test Years)
    TRAIN_YEARS = 4.0
    VAL_YEARS = 1.0
    TEST_YEARS = 1.0
    STEP_YEARS = 1.0
    
    print(f"--- Chocolate Graph Generator ---")
    print(f"Config: Train={TRAIN_YEARS}y, Val={VAL_YEARS}y, Test={TEST_YEARS}y, Step={STEP_YEARS}y", flush=True)
    
    pipeline = ChocolatePipeline(TX_PATH, PRICE_PATH)
    print("Loading Data (this may take a minute)...", flush=True)
    pipeline.initialize()
    
    # Calculate Total Windows for TQDM
    min_date = pipeline.loader.transactions['Filed'].min()
    max_date = pipeline.loader.transactions['Filed'].max()
    
    # Gap check logic copied from pipeline roughly for estimation
    gap_months = 1 if 'Label_1M' in pipeline.loader.transactions.columns else 3
    
    total_windows = calculate_total_windows(min_date, max_date, TRAIN_YEARS, VAL_YEARS, TEST_YEARS, STEP_YEARS, gap_months)
    print(f"Estimated Total Windows: {total_windows}")
    
    output_root = "data/processed_graphs"
    viz_root = "visualizations"
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(viz_root, exist_ok=True)
    
    # Generator
    gen = pipeline.generate_rolling_graphs(
        train_years=TRAIN_YEARS, 
        val_years=VAL_YEARS, 
        test_years=TEST_YEARS, 
        step_years=STEP_YEARS
    )
    
    # Iterate with TQDM
    pbar = tqdm(gen, total=total_windows, desc="Generating Windows", unit="win")
    
    for window in pbar:
        w_id = window['window_id']
        dates = window['dates']
        test_year = dates['test_start'].year
        
        pbar.set_description(f"Gen Window {w_id} (Test '{test_year})")
        print(f"\n[Log] Processing Window {w_id}...")
        print(f"      Train: {dates['train_start'].date()} to {dates['train_end'].date()}")
        print(f"      Val:   {dates['val_start'].date()} to {dates['val_end'].date()}")
        print(f"      Test:  {dates['test_start'].date()} to {dates['test_end'].date()}")
        
        # 1. Save Data
        pipeline.save_window(window, output_root=output_root)
        
        # 2. Visualize (Train Graph Topology) - DISABLED for speed per user request
        # train_year_start = dates['train_start'].year
        # train_year_end = dates['train_end'].year
        # viz_name = f"train_graph_W{w_id}_{train_year_start}-{train_year_end}.png"
        # viz_path = os.path.abspath(os.path.join(viz_root, viz_name))
        
        # Simple Viz
        # viz = ChocolateVisualizer(pipeline.loader, pipeline.builder)
        # viz.visualize_full_graph(window['train_graph'], output_path=viz_path)
        # print(f"      Visualized graph to {viz_path}")
        
    print("\nGeneration Complete!")
    print(f"Data saved to: {os.path.abspath(output_root)}")
    print(f"Visualizations: {os.path.abspath(viz_root)}")

if __name__ == "__main__":
    main()
