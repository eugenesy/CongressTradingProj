import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import json
import numpy as np

def parse_horizon(dir_name):
    """Extract horizon (e.g., '1M') from directory name."""
    match = re.search(r'H_(\d+M)', dir_name)
    if match:
        return match.group(1)
    return None

def extract_metrics_from_json(json_path):
    """Load specific metrics from a JSON report."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract desired metrics
        metrics = {
            'Macro_F1': data.get('macro avg', {}).get('f1-score'),
            'Weighted_F1': data.get('weighted avg', {}).get('f1-score'),
            'Accuracy': data.get('accuracy'),
            'AUC': data.get('auc'),
            'PR_AUC': data.get('pr_auc')
        }
        return metrics
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return {}

def load_data(results_dir):
    """Load data from JSON reports for max granularity and consistency."""
    results_dir = Path(results_dir)
    experiments_dir = results_dir / "experiments"
    baselines_dir = results_dir / "baselines"
    
    records = []

    # Helper to process a directory of horizons
    def process_dirs(base_dir, model_type_mapper):
        for horizon_dir in base_dir.glob("H_*"):
            horizon = parse_horizon(horizon_dir.name)
            if not horizon:
                continue
            
            reports_dir = horizon_dir / "reports"
            if not reports_dir.exists():
                continue
                
            for report_file in reports_dir.glob("report_*.json"):
                # Determine standard vs directional
                is_directional = "_directional" in report_file.name
                
                # Parse filename for metadata: report_{model}_{year}_{month}[_directional].json
                # TGN: report_full_2019_01.json -> model=full
                # Baseline: report_xgboost_2019_01.json -> model=xgboost
                
                parts = report_file.stem.replace("_directional", "").split('_')
                # parts[0] = "report"
                # parts[-2] = year, parts[-1] = month
                year = int(parts[-2])
                month = int(parts[-1])
                model_name_raw = "_".join(parts[1:-2])
                
                model_pretty = model_type_mapper(model_name_raw)
                
                metrics = extract_metrics_from_json(report_file)
                if not metrics:
                    continue
                    
                record = {
                    'Model': model_pretty,
                    'Horizon': horizon,
                    'Date': pd.Timestamp(year=year, month=month, day=1),
                    'Type': 'Directional' if is_directional else 'Win/Loss',
                    **metrics
                }
                records.append(record)

    # 1. Process TGN (Experiments)
    process_dirs(experiments_dir, lambda m: "TGN (Full)" if m == "full" else m)
    
    # 2. Process Baselines
    process_dirs(baselines_dir, lambda m: m.replace("_", " ").title())

    df = pd.DataFrame(records)
    return df

def analyze_and_print_summary(df):
    """Generate and print statistical summary."""
    print("\n" + "="*80)
    print("COMPARISON ANALYSIS SUMMARY")
    print("="*80)
    
    # 1. Degenerate Predictions Analysis
    # A degenerate model often predicts only one class, leading to specific metric patterns 
    # (e.g., F1=0 for minority class, or exact 0.5 accuracy if balanced). 
    # Here we check for Macro F1 < 0.35 (random guess is ~0.5 usually, but bad models dip lower)
    # or Accuracy close to baseline?
    # Better: Count rows where Macro_F1 is surprisingly low.
    
    print("\n[Degenerate Prediction Check]")
    # Check for suspiciously low AUC (random = 0.5)
    near_random_auc = df[(df['AUC'] < 0.55) & (df['AUC'] > 0.45)]
    percent_random = (len(near_random_auc) / len(df)) * 100
    print(f"  - {percent_random:.1f}% of all monthly evaluations have near-random AUC (0.45-0.55).")
    
    # Check for "Flatliners" (Macro F1 near 0 or very low - usually implies one class is 0.0)
    flatliners = df[df['Macro_F1'] < 0.4] # Heuristic threshold
    if not flatliners.empty:
        print(f"  - Found {len(flatliners)} instances of low Macro F1 (< 0.40), indicating potential mode collapse.")
        print("    Top offending models:")
        print(flatliners['Model'].value_counts().head(3).to_string())
    
    # 2. Performance Leaderboard by Horizon and Type
    print("\n[Performance Leaderboard - Average AUC]")
    summary = df.groupby(['Type', 'Horizon', 'Model'])[['AUC', 'Macro_F1', 'PR_AUC']].mean().reset_index()
    
    # Sort horizons numerically
    summary['Horizon_Num'] = summary['Horizon'].apply(lambda x: int(x[:-1]))
    summary = summary.sort_values(['Type', 'Horizon_Num', 'AUC'], ascending=[True, True, False])
    
    for type_ in ['Win/Loss', 'Directional']:
        print(f"\n--- {type_} ---")
        for horizon in sorted(summary['Horizon'].unique(), key=lambda x: int(x[:-1])):
            print(f"  Horizon {horizon}:")
            subset = summary[(summary['Type'] == type_) & (summary['Horizon'] == horizon)]
            top_3 = subset.head(3)
            for _, row in top_3.iterrows():
                print(f"    1. {row['Model']:<15} | AUC: {row['AUC']:.3f} | Macro F1: {row['Macro_F1']:.3f}")

def plot_metrics(df, output_dir):
    """Generate plots organized by Type and Metric."""
    output_dir = Path(output_dir)
    
    # Define metrics to plot
    metric_map = {
        'Macro_F1': 'Macro F1 Score',
        'AUC': 'AUROC',
        'PR_AUC': 'AUPRC'
    }
    
    # Color palette
    models = sorted(df['Model'].unique())
    palette = sns.color_palette("tab10", n_colors=len(models))
    color_map = dict(zip(models, palette))
    if 'TGN (Full)' in color_map:
        color_map['TGN (Full)'] = 'red' # Highlight TGN
        
    unique_horizons = sorted(df['Horizon'].unique(), key=lambda x: int(x[:-1]))

    for plot_type in ['Win/Loss', 'Directional']:
        type_dir = output_dir / plot_type.lower().replace("/", "_")
        type_dir.mkdir(parents=True, exist_ok=True)
        
        subset_df = df[df['Type'] == plot_type]
        
        for metric, label in metric_map.items():
            if metric not in subset_df.columns:
                continue
                
            # Create subplots for horizons
            n_rows = len(unique_horizons)
            fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows), sharex=False)
            if n_rows == 1: axes = [axes]
            
            for i, horizon in enumerate(unique_horizons):
                ax = axes[i]
                data = subset_df[subset_df['Horizon'] == horizon].sort_values('Date')
                
                if data.empty:
                    continue
                
                # Rolling average
                rolled = (
                    data.set_index('Date')
                    .groupby('Model')[metric]
                    .rolling(window=3, min_periods=1)
                    .mean()
                    .reset_index()
                )
                
                sns.lineplot(
                    data=rolled, 
                    x='Date', 
                    y=metric, 
                    hue='Model', 
                    palette=color_map, 
                    ax=ax,
                    linewidth=2.0
                )
                
                # Mark potential "flatline" or degenerate points (e.g. F1 < 0.4)
                if metric == 'Macro_F1':
                    degenerate = data[data[metric] < 0.4]
                    if not degenerate.empty:
                        sns.scatterplot(
                            data=degenerate,
                            x='Date', 
                            y=metric, 
                            hue='Model', 
                            palette=color_map,
                            marker='X', 
                            s=100, 
                            legend=False, 
                            ax=ax, 
                            zorder=10
                        )

                ax.set_title(f"{label} - Horizon {horizon} ({plot_type})")
                ax.grid(True, alpha=0.3)
                ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
                
            plt.tight_layout()
            out_file = type_dir / f"comparison_{metric}.png"
            plt.savefig(out_file, dpi=150, bbox_inches='tight')
            print(f"Saved {out_file}")
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="results/plots")
    args = parser.parse_args()

    print("Loading data from JSON reports... (this may take a moment)")
    df = load_data(args.results_dir)
    print(f"Loaded {len(df)} records.")
    
    print("\nGenerating Analysis...")
    analyze_and_print_summary(df)
    
    print("\nGenerating Plots...")
    plot_metrics(df, args.output_dir)
    
    print("\nGenerating Probability Analysis...")
    plot_probability_distributions(args.results_dir, args.output_dir)
    print("\nDone.")

def load_probabilities(results_dir):
    """Load raw probabilities from JSON files."""
    results_dir = Path(results_dir)
    records = []
    
    # Helper for prob files
    def process_probs(base_dir, model_mapper):
        for horizon_dir in base_dir.glob("H_*"):
            horizon = parse_horizon(horizon_dir.name)
            if not horizon: continue
            
            probs_dir = horizon_dir / "probs"
            if not probs_dir.exists(): continue
            
            for prob_file in probs_dir.glob("probs_*.json"):
                # probs_{model}_{year}_{month}.json
                # TGN: probs_full_2019_01.json
                parts = prob_file.stem.split('_')
                # parts[1] = model, parts[-2]=year, parts[-1]=month
                
                year = int(parts[-2])
                month = int(parts[-1])
                model_raw = "_".join(parts[1:-2])
                model_pretty = model_mapper(model_raw)
                
                try:
                    with open(prob_file, 'r') as f:
                        data = json.load(f)
                    
                    y_prob = data.get('y_prob', [])
                    if not y_prob: continue
                    
                    # Store summary stats or full data?
                    # Storing full data might be heavy but accurate for boxplots
                    # Let's store individual records for dataframe
                    
                    # Optimization: Store deciles or random sample if too large?
                    # For boxplots, we need distribution. Let's store full list formatted for DataFrame explode
                    
                    records.append({
                        'Model': model_pretty,
                        'Horizon': horizon,
                        'Date': pd.Timestamp(year=year, month=month, day=1),
                        'Probabilities': y_prob
                    })
                    
                except Exception as e:
                    print(f"Error reading {prob_file}: {e}")

    # Process TGN
    process_probs(results_dir / "experiments", lambda m: "TGN (Full)" if m == "full" else m)
    # Process Baselines
    process_probs(results_dir / "baselines", lambda m: m.replace("_", " ").title())
    
    return pd.DataFrame(records)

def plot_probability_distributions(results_dir, output_dir):
    """Generate boxplots of predicted probabilities over time."""
    output_dir = Path(output_dir) / "probabilities"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_probs_list = load_probabilities(results_dir)
    if df_probs_list.empty:
        print("No probability data found.")
        return

    # Explode for plotting (careful with memory, maybe sample if massive)
    # A monthly set has ~500-1000 points. 5 years * 12 * 1000 = 60k points per model. Manageable.
    df_exploded = df_probs_list.explode('Probabilities')
    df_exploded['Probability'] = df_exploded['Probabilities'].astype(float)
    
    unique_horizons = df_exploded['Horizon'].unique()
    
    for horizon in unique_horizons:
        subset = df_exploded[df_exploded['Horizon'] == horizon].sort_values('Date')
        
        plt.figure(figsize=(16, 8))
        sns.boxplot(x='Date', y='Probability', hue='Model', data=subset)
        
        # Format X-axis to be readable
        dates = subset['Date'].dt.strftime('%Y-%m').unique()
        # Reduce tick density if needed
        plt.xticks(rotation=45)
        
        plt.title(f"Predicted Probability Distribution over Time - Horizon {horizon}")
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.ylim(-0.05, 1.05)
        plt.axhline(0.5, color='gray', linestyle='--')
        
        out_file = output_dir / f"probs_boxplot_H{horizon}.png"
        plt.tight_layout()
        plt.savefig(out_file, dpi=150)
        print(f"Saved {out_file}")
        plt.close()

        # Also plot Standard Deviation over time (Mode Collapse Indicator)
        std_df = df_probs_list[df_probs_list['Horizon'] == horizon].copy()
        std_df['StdDev'] = std_df['Probabilities'].apply(np.std)
        
        plt.figure(figsize=(14, 6))
        sns.lineplot(x='Date', y='StdDev', hue='Model', data=std_df, marker='o')
        plt.title(f"Probability Standard Deviation (Spread) over Time - Horizon {horizon}")
        plt.ylabel("Standard Deviation of Predictions")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        
        out_file_std = output_dir / f"probs_std_H{horizon}.png"
        plt.tight_layout()
        plt.savefig(out_file_std, dpi=150)
        print(f"Saved {out_file_std}")
        plt.close()
