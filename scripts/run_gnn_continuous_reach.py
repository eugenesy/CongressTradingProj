"""
run_gnn_continuous_reach.py
===========================
Executable Pipeline Runners. Imports src.* modules to execute experiments.

Refactored/Audited: 2026-03-20
"""

import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


# Local duplicate vectorize_continuous_labels removed to use exact run_path_dependent_ml version.


def compute_precision_at_k(preds_df, k=0.10):
    """Calculate average monthly Precision@Top-k."""
    if preds_df.empty:
        return 0.0
    precisions = []
    # Group by month
    grouped = preds_df.groupby(['year', 'month'])
    for name, group in grouped:
         if len(group) < 5:
              continue
         group = group.sort_values(by='prob', ascending=False)
         n_top = max(1, int(len(group) * k))
         top_k = group.head(n_top)
         prec = top_k['label'].mean() # average rate of 1s is precision
         precisions.append(prec)
    return np.mean(precisions) if precisions else 0.0

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-year', type=int, default=2023, help='Test year')
    args = parser.parse_args()
    test_year = args.test_year

    df_raw = pd.read_csv("data/processed/ml_dataset_v2.csv")
    out_dir = Path("experiments/signal_isolation/results")
    out_dir.mkdir(exist_ok=True)
    final_stats = []

    # Process continuous labels
    try:
        labeled_df = pd.read_csv("data/processed/ml_dataset_continuous.csv")
    except Exception as e:
        print(f"Error loading continuous labels: {e}")
        return

    # Variants: Median vs Q3
    targets = {
        'Median': labeled_df['max_excess_6m'] > 0.088,
        'Q3': labeled_df['max_excess_6m'] > 0.178
    }

    configs = ['Combined', 'Structure-Only']

    print(f"\n--- Running GNN Continuous Ablation Study (Test Year {test_year}) ---")

    for name, mask in targets.items():
        for config in configs:
            print(f"\n" + "="*60)
            print(f"Running GNN-SAGE: {name} target | Configuration: {config}")
            print("="*60)
            
            temp_df = labeled_df.copy()
            temp_df[f'Excess_Return_6M'] = mask.astype(int)
            
            if config == 'Structure-Only':
                temp_df['Chamber'] = 'UNK'
                temp_df['Party'] = 'UNK'
                temp_df['State'] = 'UNK'
                
            temp_csv = f"data/processed/temp_gnn_study_{name.lower()}_{config.lower().replace('-','_')}_{test_year}.csv"
            temp_df.to_csv(temp_csv, index=False)
            
            cmd = [
                "/home/syeugene/miniconda3/envs/chocolate/bin/python",
                "experiments/signal_isolation/run_gnn_study.py",
                "--data", temp_csv,
                "--years", str(test_year),
                "--horizon", "6M",
                "--save-preds",
                "--epochs", "10",
                "--skip", "gat", "gnn_xgb", "gnn_gat"
            ]
            print(f"Command: {' '.join(cmd)}")
            subprocess.run(cmd, check=False)

            expected_pred = out_dir / f"preds_gnn_{test_year}_6M_seed42.csv"
            
            if expected_pred.exists():
                preds_df = pd.read_csv(expected_pred)
                test_preds = preds_df
                if not test_preds.empty:
                    try:
                        auc = roc_auc_score(test_preds['label'], test_preds['prob'])
                    except:
                        auc = 0.5
                    p_topk = compute_precision_at_k(test_preds, k=0.10)
                else:
                    auc, p_topk = 0.0, 0.0
                
                print(f"--> {name} ({config}) AUROC: {auc:.4f} | P@10%: {p_topk:.4f}")
                final_stats.append({
                    'test_year': test_year,
                    'Target': name,
                    'Config': config,
                    'AUROC': auc,
                    'P@10%': p_topk
                })
                saved_dest = out_dir / f"preds_gnn_{test_year}_6M_{name.lower()}_{config.lower().replace('-','_')}.csv"
                expected_pred.rename(saved_dest)
            else:
                print(f"Error: Output file not found for {name} ({config}) at {expected_pred}")

    res_df = pd.DataFrame(final_stats)
    res_df.to_csv(f"experiments/signal_isolation/results/gnn_res_{test_year}.csv", index=False)
    print("\n" + "#"*50)
    print(f"FINAL CONTINUOUS GNN ABLATION SUMMARY (Year {test_year})")
    print("#"*50)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()
