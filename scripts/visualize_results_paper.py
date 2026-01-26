
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score
import os
from pathlib import Path
import seaborn as sns

# Set aesthetics for the paper
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (10, 6)
})

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

HORIZONS = ['3M', '6M', '8M', '12M', '18M', '24M']

MODELS = {
    'TGN': 'GAP-TGN', 
    'XGB': 'XGBoost', 
    'MLP': 'MLP', 
    'LR': 'Logistic Reg'
}

def calculate_monthly_metrics(df):
    """Calculates Macro F1, F1-Up, and AUC for each month."""
    df['Date'] = pd.to_datetime(df['Date'])
    df['YearMonth'] = df['Date'].dt.to_period('M')
    
    results = []
    for ym, group in df.groupby('YearMonth'):
        y_true = group['True_Up'].values
        y_pred = group['Pred_Up'].values
        y_prob = group['Prob_Up'].values
        
        # Macro F1
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        # F1 Score for Class 1 (Up)
        f1_up = f1_score(y_true, y_pred, pos_label=1.0)
        
        # AUC (Handle single class case)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5
            
        results.append({
            'Date': ym.start_time,
            'F1_Macro': macro_f1,
            'F1_Up': f1_up,
            'AUC': auc,
            'Count': len(group)
        })
        
    return pd.DataFrame(results)

def main():
    print("Starting Visualization Pipeline...")
    
    agg_results = []
    rolling_data = {} # Key: (Horizon, Model) -> DataFrame

    for h in HORIZONS:
        print(f"Processing Horizon: {h}...")
        
        # 1. Load TGN Predictions
        tgn_path = RESULTS_DIR / f"predictions_tgn_{h}.csv"
        if tgn_path.exists():
            df_tgn = pd.read_csv(tgn_path)
            tgn_metrics = calculate_monthly_metrics(df_tgn)
            
            # Aggregate for Table
            agg_results.append({
                'Horizon': h,
                'Model': 'GAP-TGN',
                'F1_Macro': tgn_metrics['F1_Macro'].mean(),
                'F1_Up': tgn_metrics['F1_Up'].mean(),
                'AUC': tgn_metrics['AUC'].mean()
            })
            
            # Store for Plots
            rolling_data[(h, 'TGN')] = tgn_metrics.sort_values('Date') # Keep as DF with Date column

        # 2. Load Baseline Predictions
        base_path = RESULTS_DIR / f"predictions_baseline_{h}.csv"
        if base_path.exists():
            df_base = pd.read_csv(base_path)
            
            for model_code, model_name in [('XGB', 'XGBoost'), ('MLP', 'MLP'), ('LR', 'Logistic Reg')]:
                df_model = df_base[df_base['Model'] == model_code].copy()
                if not df_model.empty:
                    metrics = calculate_monthly_metrics(df_model)
                    
                    agg_results.append({
                        'Horizon': h,
                        'Model': f"{model_name}",
                        'F1_Macro': metrics['F1_Macro'].mean(),
                        'F1_Up': metrics['F1_Up'].mean(),
                        'AUC': metrics['AUC'].mean()
                    })
                    rolling_data[(h, model_code)] = metrics.sort_values('Date') # Keep as DF with Date column

    df_agg = pd.DataFrame(agg_results)
    
    # Ensure no duplicates for pivoting
    df_agg = df_agg.groupby(['Model', 'Horizon'], as_index=False).mean()
    
    # --- 3. Generate Classification Performance Summary (Table) ---
    # --- 3. Generate Classification Performance Summary (Table) ---
    print("\n--- MACRO F1 SCORES (For Table 1) ---")
    pivot_macro = df_agg.pivot(index='Model', columns='Horizon', values='F1_Macro')
    print(pivot_macro.to_string())

    print("\n--- F1-UP SCORES (For Analysis) ---")
    pivot_up = df_agg.pivot(index='Model', columns='Horizon', values='F1_Up')
    print(pivot_up.to_string())

    # Save to file
    with open(RESULTS_DIR / "performance_summary.txt", "w") as f:
        f.write("--- MACRO F1 SCORES ---\n")
        f.write(pivot_macro.to_string())
        f.write("\n\n--- F1-UP SCORES ---\n")
        f.write(pivot_up.to_string())
    print(f"Saved Performance Summary to {RESULTS_DIR / 'performance_summary.txt'}")
    
    # --- 4. Generate 2x6 Grid Plot (Figure 3) ---
    # Rows: Macro F1, F1-Up
    # Cols: Horizons 3M, 6M, 8M, 12M, 18M, 24M
    
    fig, axes = plt.subplots(2, 6, figsize=(24, 8), sharex=True, sharey='row')
    
    metrics_to_plot = [('F1_Macro', 'Macro F1'), ('F1_Up', 'F1-Score (Up/Class 1)')]
    
    # Define colors for models
    palette = {'GAP-TGN': 'red', 'XGBoost': 'blue', 'MLP': 'green', 'Logistic Reg': 'orange'}
    
    for row_idx, (metric_col, metric_label) in enumerate(metrics_to_plot):
        for col_idx, h in enumerate(HORIZONS):
            ax = axes[row_idx, col_idx]
            
            # Prepare data for Seaborn
            plot_dfs = []
            for model_code, model_name in MODELS.items():
                if (h, model_code) in rolling_data:
                    data = rolling_data[(h, model_code)].copy()
                    # Rolling 6-month average
                    data['Value'] = data[metric_col].rolling(window=6, min_periods=3).mean()
                    data['Model'] = model_name
                    plot_dfs.append(data[['Date', 'Value', 'Model']])
            
            if plot_dfs:
                combined_df = pd.concat(plot_dfs, ignore_index=True)
                sns.lineplot(data=combined_df, x='Date', y='Value', hue='Model', palette=palette, ax=ax, linewidth=1.5)
                
                # Highlight TGN with thicker line
                # Note: Seaborn doesn't easily allow varying linewidths by hue without a size mapping column.
                # We can overlay TGN again with thicker line or just rely on color.
                # Let's adding a size column for semantic mapping
                
            # Titles only on top row
            if row_idx == 0:
                ax.set_title(f"{h} Horizon", fontsize=14, fontweight='bold')
                
            # Y-labels only on left column
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=12)
            else:
                ax.set_ylabel('')
                
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # Remove individual legends
            ax.get_legend().remove()

    # Add a single global legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=12)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "multi_horizon_grid_2x6.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 2x6 Grid Plot to {PLOTS_DIR / 'multi_horizon_grid_2x6.png'}")

if __name__ == "__main__":
    main()
