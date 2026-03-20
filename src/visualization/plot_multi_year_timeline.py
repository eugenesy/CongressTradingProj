"""
plot_multi_year_timeline.py
===========================
Visualization Utilities. Generates presentations plots and distributions.

Refactored/Audited: 2026-03-20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score

def compute_precision_at_k(df, k=0.10):
    if df.empty: return 0.0
    group = df.sort_values(by='prob', ascending=False)
    n_top = max(1, int(len(group) * k))
    return group.head(n_top)['label'].mean()

def main():
    out_dir = Path("experiments/signal_isolation/results")
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    print("📈 Generating Multi-Year Timeline Plots (3M Rolling)...")
    
    all_rows = []
    
    # 1. Load XGBoost Preds
    # Format: preds_xgb_{year}_{target}_{id|feat|all}.csv
    for p in out_dir.glob("preds_xgb_*_Label_*.csv"):
        name_parts = p.stem.split("_")
        years = name_parts[2]
        target = "Q3" if "Q3" in p.stem else "Median"
        config_raw = name_parts[-1]
        config = "Combined" if config_raw == "all" else ("Structure-Only" if config_raw == "id" else "Features-Only")
        
        df = pd.read_csv(p)
        df['model'] = 'XGBoost'
        df['target_label'] = target
        df['config'] = config
        all_rows.append(df)
        
    # 2. Load GNN Preds
    # Format: preds_gnn_{year}_6M_{median|q3}_{combined|structure_only}.csv
    for p in out_dir.glob("preds_gnn_*_6M_*.csv"):
        name_parts = p.stem.split("_")
        if len(name_parts) < 6: continue
        years = name_parts[2]
        target = "Median" if "median" in p.stem else "Q3"
        config = "Combined" if "combined" in p.stem else "Structure-Only"
        
        df = pd.read_csv(p)
        df['model'] = 'GNN'
        df['target_label'] = target
        df['config'] = config
        all_rows.append(df)

    if not all_rows:
        print("Error: No predictions found for plotting.")
        return
        
    master_df = pd.concat(all_rows)
    master_df['date_anchor'] = pd.to_datetime(master_df['year'].astype(str) + '-' + master_df['month'].astype(str) + '-01')

    # Group and Calculate Month-by-Month Metrics
    timeline_data = []
    grouped = master_df.groupby(['model', 'target_label', 'config', 'date_anchor'])
    
    for (model, target, config, date), group in grouped:
        if len(group) < 5: continue
        
        try:
            auc = roc_auc_score(group['label'], group['prob'])
        except:
            auc = np.nan
            
        p10 = compute_precision_at_k(group, k=0.10)
        
        timeline_data.append({
            'model': model, 'target': target, 'config': config, 'date': date,
            'auc': auc, 'p10': p10
        })

    time_df = pd.DataFrame(timeline_data)
    time_df = time_df.sort_values(by='date')

    # Plotting
    for target in ['Median', 'Q3']:
        for metric in ['auc', 'p10']:
            plt.figure(figsize=(12, 6))
            title_met = "AUROC" if metric == 'auc' else "Precision@10%"
            
            sub = time_df[time_df['target'] == target]
            
            # Group by model + config
            for (model, config), g in sub.groupby(['model', 'config']):
                if config == 'Features-Only' and model == 'GNN': continue # Skip as they don't exist
                
                label_str = f"{model} ({config})"
                
                # Pivot to date index, perform rolling
                series = g.set_index('date')[metric]
                rolling_series = series.rolling(window=3, min_periods=1).mean()
                
                if not rolling_series.dropna().empty:
                    plt.plot(rolling_series.index, rolling_series.values, label=label_str, linewidth=2)

            plt.title(f"Multi-Year {title_met} Timeline - {target} Continuous Target (3M Rolling Avg)")
            plt.xlabel("Test Date Split")
            plt.ylabel(title_met)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            f_name = f"timeline_{target.lower()}_{metric}.png"
            plt.savefig(plots_dir / f_name, dpi=150)
            print(f"-> Saved plot to {plots_dir / f_name}")
            plt.close()

if __name__ == "__main__":
    main()
