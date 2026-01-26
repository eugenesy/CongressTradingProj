import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, classification_report
from pathlib import Path

def analyze_horizon(horizon):
    print(f"\nAnalyzing Horizon: {horizon}...")
    
    results_dir = Path("dev/results")
    tgn_path = results_dir / f"predictions_tgn_{horizon}.csv"
    baseline_path = results_dir / f"predictions_baseline_{horizon}.csv"
    
    # Validation
    if not tgn_path.exists():
        print(f"❌ Missing TGN file: {tgn_path}")
        return
    if not baseline_path.exists():
        print(f"❌ Missing Baseline file: {baseline_path}")
        return

    # Load & Standardize
    df_tgn = pd.read_csv(tgn_path)
    df_base = pd.read_csv(baseline_path)
    
    df_tgn['Date'] = pd.to_datetime(df_tgn['Date'])
    df_base['Date'] = pd.to_datetime(df_base['Date'])
    
    # Merge
    df_all = pd.concat([df_tgn, df_base])
    
    # Output Directories
    plot_dir = results_dir / "plots" / f"H_{horizon}"
    report_dir = results_dir / "reports"
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Monthly Metrics Calculation ---
    df_all['Month'] = df_all['Date'].dt.to_period('M')
    models = df_all['Model'].unique()
    monthly_metrics = []
    
    for model in models:
        model_df = df_all[df_all['Model'] == model]
        for month, group in model_df.groupby('Month'):
            if len(group) < 10: continue # Skip tiny months
            
            y_true = group['True_Up']
            y_prob = group['Prob_Up']
            y_pred = group['Pred_Up']
            
            try:
                auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
                prc = average_precision_score(y_true, y_prob)
                acc = accuracy_score(y_true, y_pred)
                f1_up = f1_score(y_true, y_pred, pos_label=1)
                
                monthly_metrics.append({
                    'Model': model,
                    'Month': month,
                    'Date': month.start_time,
                    'AUC': auc,
                    'PRC': prc,
                    'Accuracy': acc,
                    'F1_Up': f1_up,
                    'Count': len(group)
                })
            except Exception as e:
                print(f"Error for {model} {month}: {e}")

    metrics_df = pd.DataFrame(monthly_metrics)
    metrics_df = metrics_df.sort_values(['Model', 'Date'])
    
    # Calculate 3-Month Rolling Average
    metrics_df['Rolling_AUC'] = metrics_df.groupby('Model')['AUC'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    metrics_df['Rolling_PRC'] = metrics_df.groupby('Model')['PRC'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    metrics_df['Rolling_Accuracy'] = metrics_df.groupby('Model')['Accuracy'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    metrics_df['Rolling_F1_Up'] = metrics_df.groupby('Model')['F1_Up'].transform(lambda x: x.rolling(3, min_periods=1).mean())

    sns_styles = ['AUC', 'PRC', 'Accuracy', 'F1_Up']
    plt.figure(figsize=(20, 15))
    
    for i, metric in enumerate(sns_styles):
        plt.subplot(2, 2, i+1)
        for model in models:
            subset = metrics_df[metrics_df['Model'] == model]
            plt.plot(subset['Date'], subset[f'Rolling_{metric}'], label=model, marker='.')
        
        plt.title(f'3-Month Rolling {metric} (Horizon {horizon}, 2019-2024)')
        plt.xlabel('Date')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = plot_dir / f"rolling_metrics_{horizon}.png"
    plt.savefig(plot_path)
    print(f"✅ Saved Plot: {plot_path}")

    # --- 3. Final Report Generation ---
    report_path = report_dir / f"report_{horizon}.md"
    
    with open(report_path, "w") as f:
        f.write(f"# Analysis Report: Horizon {horizon}\n\n")
        f.write(f"**Period**: {df_all['Date'].min().date()} to {df_all['Date'].max().date()}\n\n")
        f.write(f"![Rolling Metrics]({plot_path.name})\n\n")
        
        f.write("## 1. Aggregate Leaderboard\n")
        summary_stats = []
        for model in models:
            sub = df_all[df_all['Model'] == model]
            auc = roc_auc_score(sub['True_Up'], sub['Prob_Up'])
            prc = average_precision_score(sub['True_Up'], sub['Prob_Up'])
            
            # Classification Report
            cr = classification_report(sub['True_Up'], sub['Pred_Up'], output_dict=True)
            # Handle float vs int keys in dict
            key_up = '1.0' if '1.0' in cr else 1.0
            recall_up = cr[key_up]['recall']
            prec_up = cr[key_up]['precision']
            
            summary_stats.append({
                'Model': model,
                'AUC': auc,
                'PRC': prc,
                'Recall (Up)': recall_up,
                'Precision (Up)': prec_up
            })

        summary_df = pd.DataFrame(summary_stats).sort_values('AUC', ascending=False)
        f.write("```\n" + summary_df.to_string(index=False, float_format="%.4f") + "\n```\n")
        f.write("\n\n")
        
        f.write("## 2. Detailed Classification Reports\n")
        for model in models:
            f.write(f"### Model: {model}\n")
            sub = df_all[df_all['Model'] == model]
            report_str = classification_report(sub['True_Up'], sub['Pred_Up'], target_names=['Down', 'Up'])
            f.write("```\n" + report_str + "\n```\n")
            
    print(f"✅ Saved Report: {report_path}")
    print("\n--- LEADERBOARD ---")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", required=True, help="Horizon to analyze (e.g., 1M, 6M)")
    args = parser.parse_args()
    
    analyze_horizon(args.horizon)
