
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_and_prep_data(tgn_path, baseline_path):
    # Load TGN
    try:
        df_tgn = pd.read_csv(tgn_path)
        df_tgn['Model'] = 'TGN (Directional)'
        df_tgn = df_tgn.drop_duplicates(subset=['Year', 'Month'], keep='last')
    except:
        df_tgn = pd.DataFrame()

    # Load Baselines
    try:
        df_base = pd.read_csv(baseline_path)
        df_base['Model'] = df_base['Model'].apply(lambda x: f"{x} (Directional)")
    except:
        df_base = pd.DataFrame()

    if df_tgn.empty and df_base.empty:
        return None

    # Date
    for df in [df_tgn, df_base]:
        if not df.empty:
            df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))

    # Combine
    cols = ['Date', 'Model', 'Accuracy', 'AUC', 'F1_Class1']
    df_all = pd.concat([df_tgn[cols] if not df_tgn.empty else pd.DataFrame(),
                        df_base[cols] if not df_base.empty else pd.DataFrame()], ignore_index=True)
    
    return df_all

def plot_metrics(df, output_dir, rolling_window=3):
    sns.set_theme(style="whitegrid")
    df = df.sort_values('Date')
    
    metrics = ['Accuracy', 'AUC', 'F1_Class1']
    titles = ['Directional Accuracy', 'Directional AUC', 'Directional F1 Score']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    for i, metric in enumerate(metrics):
        ax = axes[i]
        df_rolled = df.groupby('Model')[['Date', metric]].apply(
            lambda x: x.set_index('Date').rolling(window=rolling_window, min_periods=1).mean()
        ).reset_index()
        
        sns.lineplot(data=df_rolled, x='Date', y=metric, hue='Model', marker='o', ax=ax, linewidth=2)
        ax.set_title(f"{titles[i]} ({rolling_window}-Month Rolling Avg)", fontsize=14)
        if metric in ['Accuracy', 'AUC']:
            ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    
    plt.tight_layout()
    save_path = output_dir / "comparison_directional_rolling.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgn_csv', default='results/directional_tgn/H_1M_A_0.0/summary_full.csv')
    parser.add_argument('--base_csv', default='results/directional_baselines/H_1M_A_0.0/summary_all_models.csv')
    parser.add_argument('--out_dir', default='results/comparisons')
    args = parser.parse_args()
    
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_and_prep_data(args.tgn_csv, args.base_csv)
    if df is not None:
        plot_metrics(df, output_dir)

if __name__ == "__main__":
    main()
