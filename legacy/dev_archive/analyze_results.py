import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, classification_report
from pathlib import Path

def analyze_results():
    results_dir = Path("dev/results")
    tgn_path = results_dir / "predictions_tgn_6M.csv"
    baseline_path = results_dir / "predictions_baseline_6M.csv"
    
    if not tgn_path.exists() or not baseline_path.exists():
        print("Predictions files not found!")
        return

    df_tgn = pd.read_csv(tgn_path)
    df_base = pd.read_csv(baseline_path)
    
    # Standardize Dates
    df_tgn['Date'] = pd.to_datetime(df_tgn['Date'])
    df_base['Date'] = pd.to_datetime(df_base['Date'])
    
    # Merge Baselines into one DataFrame
    # df_base has columns: TransactionID, Date, True_Up, Model, Prob_Up, Pred_Up
    # df_tgn has columns: TransactionID, Date, Model, True_Up, Prob_Up, Pred_Up
    
    df_all = pd.concat([df_tgn, df_base])
    
    # Monthly Metrics Calculation
    models = df_all['Model'].unique()
    monthly_metrics = []
    
    # Group by Model and Month (Year-Month)
    df_all['Month'] = df_all['Date'].dt.to_period('M')
    
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
    rolling_df = metrics_df.groupby('Model')[['AUC', 'PRC', 'Accuracy', 'F1_Up']].rolling(window=3).mean().reset_index()
    
    # Merge Date back
    metrics_df['Rolling_AUC'] = metrics_df.groupby('Model')['AUC'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    metrics_df['Rolling_PRC'] = metrics_df.groupby('Model')['PRC'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    metrics_df['Rolling_Accuracy'] = metrics_df.groupby('Model')['Accuracy'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    metrics_df['Rolling_F1_Up'] = metrics_df.groupby('Model')['F1_Up'].transform(lambda x: x.rolling(3, min_periods=1).mean())

    # --- PLOTTING ---
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    sns_styles = ['AUC', 'PRC', 'Accuracy', 'F1_Up']
    
    plt.figure(figsize=(20, 15))
    
    for i, metric in enumerate(sns_styles):
        plt.subplot(2, 2, i+1)
        for model in models:
            subset = metrics_df[metrics_df['Model'] == model]
            plt.plot(subset['Date'], subset[f'Rolling_{metric}'], label=model, marker='.')
        
        plt.title(f'3-Month Rolling {metric} (2019-2024)')
        plt.xlabel('Date')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / "rolling_metrics_comparison.png")
    print(f"Saved plots to {plot_dir / 'rolling_metrics_comparison.png'}")
    
    # --- FINAL CLASSIFICATION REPORT ---
    print("\n" + "="*60)
    print("FINAL CLASSIFICATION REPORT (Total Aggregated 2019-2024)")
    print("="*60)
    
    summary_stats = []
    
    for model in models:
        model_df = df_all[df_all['Model'] == model]
        y_true = model_df['True_Up']
        y_pred = model_df['Pred_Up']
        y_prob = model_df['Prob_Up']
        
        print(f"\n--- Model: {model} ---")
        print(classification_report(y_true, y_pred, target_names=['Down', 'Up']))
        
        auc = roc_auc_score(y_true, y_prob)
        prc = average_precision_score(y_true, y_prob)
        
        print(f"Overall AUC: {auc:.4f}")
        print(f"Overall PRC: {prc:.4f}")
        
        summary_stats.append({
            'Model': model,
            'AUC': auc,
            'PRC': prc,
            'Count': len(model_df)
        })

    print("\n" + "="*60)
    print("SUMMARY LEADERBOARD")
    print("="*60)
    print(pd.DataFrame(summary_stats).sort_values('AUC', ascending=False).to_string(index=False))

if __name__ == "__main__":
    analyze_results()
