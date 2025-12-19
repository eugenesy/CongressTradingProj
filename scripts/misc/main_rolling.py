import os
import torch
import pandas as pd
from src.pipeline import ChocolatePipeline
from src.inference import DailyRollingEvaluator
from src.models import Model
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Strict Daily Rolling Inference")
    parser.add_argument("--window_id", type=int, default=0, help="Window ID to evaluate (must typically match the one trained)")
    args = parser.parse_args()
    
    w_id = args.window_id
    data_root = "data/processed_graphs"
    window_path = os.path.join(data_root, f"window_{w_id}")
    
    print(f"--- Chocolate Strict Rolling Evaluator (Window {w_id}) ---")
    
    # 1. Load Data
    # We need the Base Graph (Train set used as history) and Test Set (Transactions)
    # The 'train.pt' saved by pipeline is the graph at the END of training.
    # This is exactly what we want as history for the start of testing.
    
    if not os.path.exists(window_path):
        print(f"Error: Path {window_path} not found.")
        return

    print("Loading Base Graph (Train)...")
    base_graph = torch.load(os.path.join(window_path, "train.pt"), weights_only=False)
    
    print("Loading Test Transactions...")
    # We need the raw DataFrame for the test set to sort by date.
    # We can reconstruct this via the pipeline loader logic OR simply load the full CSV and filter by date.
    # Since 'test.pt' assumes a built graph, we can't easily iterate edges by date unless we carry that metadata.
    # EASIEST: Load raw CSV and filter using the metadata dates from window_0.
    
    # Load metadata
    try:
        meta = torch.load(os.path.join(window_path, "metadata.pt"), weights_only=False)
        test_start = meta['test_start']
        test_end = meta['test_end']
        print(f"Test Period: {test_start.date()} to {test_end.date()}")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    # Load Raw Transactions
    pipeline = ChocolatePipeline(
        transaction_path="/data1/user_syeugene/fintech/apple/data/processed/ml_dataset_reduced_attributes.csv",
        price_data_path=None 
    )
    pipeline.initialize()
    
    # Filter Test Test
    df = pipeline.loader.transactions
    mask = (df['Filed'] >= test_start) & (df['Filed'] < test_end)
    test_df = df[mask].copy()
    print(f"Found {len(test_df)} test transactions.")
    
    # 2. Load Model
    print("Loading Trained Model...")
    model_path = os.path.join(window_path, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Model checkpoint {model_path} not found. Train first!")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(hidden_channels=64, metadata=base_graph.metadata())
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # 3. Run Inference
    evaluator = DailyRollingEvaluator(pipeline)
    results = evaluator.evaluate(model, base_graph, test_df, device=device)
    
    # Save
    import json
    save_path = f"results/strict_rolling_W{w_id}.json"
    with open(save_path, 'w') as f:
        json.dump(results, f)
    print(f"Strict results saved to {save_path}")

if __name__ == "__main__":
    main()
