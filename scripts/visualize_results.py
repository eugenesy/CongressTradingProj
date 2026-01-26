
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_and_prep_data(tgn_path, baseline_path):
    # Load TGN
    try:
        df_tgn = pd.read_csv(tgn_path)
    except FileNotFoundError:
        print(f"Error: Could not find TGN file at {tgn_path}")
        return None

    try:
        df_base = pd.read_csv(baseline_path)
    except FileNotFoundError:
        print(f"Error: Could not find Baseline file at {baseline_path}")
        return None

    # TGN Preprocessing
    df_tgn['Model'] = 'TGN (Full)'
    df_tgn = df_tgn.drop_duplicates(subset=['Year', 'Month'], keep='last')
    
    # Create Date Column
    df_tgn['Date'] = pd.to_datetime(df_tgn[['Year', 'Month']].assign(day=1))
    df_base['Date'] = pd.to_datetime(df_base[['Year', 'Month']].assign(day=1))

    # Combine
    # Select all relevant columns including Directional ones
    cols = ['Date', 'Model', 'Accuracy', 'AUC', 'F1_Class1', 
            'Dir_Accuracy', 'Dir_AUC', 'Dir_F1']
    
    # Ensure columns exist (TGN might not have Dir columns if old run, but verified it does)
    for c in cols:
        if c not in df_tgn.columns: df_tgn[c] = float('nan')
        if c not in df_base.columns: df_base[c] = float('nan')

    df_all = pd.concat([
        df_tgn[cols],
        df_base[cols]
    ], ignore_index=True)
    
    return df_all

def plot_task(df, output_dir, task_name, metric_map, rolling_window=3):
    sns.set_theme(style="whitegrid")
    
    # Sort by date
    df = df.sort_values('Date')
    
    metrics = list(metric_map.keys())
    titles = list(metric_map.values())
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Compute Rolling Mean
        df_rolled = df.groupby('Model')[['Date', metric]].apply(
            lambda x: x.set_index('Date').rolling(window=rolling_window, min_periods=1).mean()
        ).reset_index()
        
        sns.lineplot(data=df_rolled, x='Date', y=metric, hue='Model', marker='o', ax=ax, linewidth=2)
        
        ax.set_title(f"{titles[i]} ({rolling_window}-Month Rolling Avg)", fontsize=14)
        ax.set_ylabel(metric, fontsize=12)
        
        if 'Accuracy' in metric or 'AUC' in metric:
            ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random Chance')
    
    plt.tight_layout()
    save_path = output_dir / f"comparison_1M_{task_name}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved {task_name} plot to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgn_csv', default='results/experiments/H_1M_A_0.0/summary_full.csv')
    parser.add_argument('--base_csv', default='results/baselines/H_1M_A_0.0/summary_all_models.csv')
    parser.add_argument('--out_dir', default='results/comparisons') # CHANGED
    args = parser.parse_args()
    
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df = load_and_prep_data(args.tgn_csv, args.base_csv)
    
    if df is not None:
        print(f"Data loaded. Models: {df['Model'].unique()}")
        
        # 1. Plot Win/Loss
        print("Plotting Win/Loss...")
        plot_task(df, output_dir, "WinLoss", {
            'Accuracy': 'Win/Loss Accuracy',
            'AUC': 'Win/Loss AUC',
            'F1_Class1': 'Win/Loss F1 Score'
        })
        
        # 2. Plot Directional
        print("Plotting Directional...")
        plot_task(df, output_dir, "Directional", {
            'Dir_Accuracy': 'Directional Accuracy',
            'Dir_AUC': 'Directional AUC',
            'Dir_F1': 'Directional F1 Score'
        })
        
        print("Done.")

if __name__ == "__main__":
    main()
