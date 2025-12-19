import os
import glob
import re
from tqdm import tqdm
from src.train import train_window

def main():
    print("--- Chocolate Model Training Experiment ---")
    
    data_root = "data/processed_graphs"
    if not os.path.exists(data_root):
        print(f"Data root {data_root} not found. Run main_generate.py first.")
        return

    # Find all windows
    window_dirs = glob.glob(os.path.join(data_root, "window_*"))
    window_ids = []
    for d in window_dirs:
        match = re.search(r"window_(\d+)", d)
        if match:
            window_ids.append(int(match.group(1)))
            
    window_ids.sort()
    print(f"Found {len(window_ids)} windows: {window_ids}")
    
    results = []
    
    # Iterate with TQDM
    pbar = tqdm(window_ids, desc="Training Experiment")
    
    for w_id in pbar:
        pbar.set_description(f"Training Window {w_id}")
        
        # Train
        metrics = train_window(w_id, data_root=data_root, epochs=100, patience=15)
        
    # Save Best Model Logic (Placeholder if not implemented)
    # Ideally logic goes inside train_window.
    # Let's check src/train.py content first.
            
    print("\n=== Final Experiment Results ===")
    print(f"{'Window':<8} | {'AUC':<10} | {'F1':<10}")
    print("-" * 35)
    
    # Save results to file
    os.makedirs("results", exist_ok=True)
    import pandas as pd
    df_res = pd.DataFrame(results)
    df_res.to_csv("results/experiment_results.csv", index=False)
    
    for res in results:
        print(f"{res['window_id']:<8} | {res['auc']:.4f}     | {res['f1']:.4f}")
        
    print(f"\nResults saved to {os.path.abspath('results/experiment_results.csv')}")

if __name__ == "__main__":
    main()
