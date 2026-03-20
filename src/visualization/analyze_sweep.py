"""
analyze_sweep.py
================
Visualization Utilities. Generates presentations plots and distributions.

Refactored/Audited: 2026-03-20
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon, ttest_rel
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("research")

def load_preds(model, dz, results_dir="experiments/signal_isolation/results/raw_return"):
    results_dir = Path(results_dir)
    dz_tag = f"_DZ{int(dz*100)}pct" if dz > 0 else ""
    seed = 42
    horizon = "6M"
    label_type = "excess"
    level = 0

    if model == "gnn_sage":
        # For gnn_sage, try to stitch per-year files if combined doesn't exist
        years = [2019, 2020, 2021, 2022, 2023, 2024]
        dfs = []
        for year in years:
            fname = f"preds_{year}_{horizon}_{label_type}_L{level}_{model}{dz_tag}_seed{seed}.csv"
            fpath = results_dir / fname
            if not fpath.exists():
                # Try underscore-joined format
                fname_joined = f"preds_2019_2020_2021_2022_2023_2024_{horizon}_{label_type}_L{level}_{model}{dz_tag}_seed{seed}.csv"
                fpath = results_dir / fname_joined
                if not fpath.exists():
                    logger.warning("Missing GNN file: %s", fname)
                    continue
                df = pd.read_csv(fpath)
                df.columns = [c.lower() for c in df.columns]
                return df # already combined
            
            df = pd.read_csv(fpath)
            df.columns = [c.lower() for c in df.columns]
            # Add year if missing
            if 'year' not in df.columns and 'filed' in df.columns:
                df['year'] = pd.to_datetime(df['filed']).dt.year
            dfs.append(df)
        
        if not dfs:
             raise FileNotFoundError(f"No GNN preds found for dz={dz}")
        return pd.concat(dfs).sort_values('filed').reset_index(drop=True)
    
    else:
        # Flat models: range format (2019-2024)
        fname = f"preds_2019-2024_{horizon}_{label_type}_L{level}_{model}{dz_tag}_seed{seed}.csv"
        fpath = results_dir / fname
        if not fpath.exists():
             raise FileNotFoundError(f"Missing flat model file: {fpath}")
        
        df = pd.read_csv(fpath)
        df.columns = [c.lower() for c in df.columns]
        if 'year' not in df.columns and 'filed' in df.columns:
            df['year'] = pd.to_datetime(df['filed']).dt.year
        return df

def bootstrap_paired_delta(df0, df1, key_cols, score_col, label_col, n_boot=2000, seed=42):
    """
    Matched-pair bootstrap for delta AUROC.
    """
    # Join by keys to ensure alignment
    merged = pd.merge(df0[key_cols + [score_col, label_col]], 
                      df1[key_cols + [score_col]], 
                      on=key_cols, suffixes=('_0', '_1'))
    
    if len(merged) == 0:
        logger.error("No overlap between dataframes for matching!")
        return None

    y_true = merged[label_col].values
    y_pred0 = merged[score_col + '_0'].values
    y_pred1 = merged[score_col + '_1'].values
    
    rng = np.random.default_rng(seed)
    deltas = []
    
    # Pre-calculate individual results to speed up bootstrap
    auc0 = roc_auc_score(y_true, y_pred0)
    auc1 = roc_auc_score(y_true, y_pred1)
    observed_delta = auc1 - auc0

    indices = np.arange(len(merged))
    for _ in range(n_boot):
        boot_idx = rng.choice(indices, size=len(indices), replace=True)
        y_b = y_true[boot_idx]
        p0_b = y_pred0[boot_idx]
        p1_b = y_pred1[boot_idx]
        
        # Avoid errors if bootstrap sample has only one class
        if len(np.unique(y_b)) < 2:
            continue
            
        b_auc0 = roc_auc_score(y_b, p0_b)
        b_auc1 = roc_auc_score(y_b, p1_b)
        deltas.append(b_auc1 - b_auc0)
    
    deltas = np.array(deltas)
    return {
        'observed_delta': observed_delta,
        'median': np.median(deltas),
        'ci_lower': np.percentile(deltas, 2.5),
        'ci_upper': np.percentile(deltas, 97.5),
        'n_boot': len(deltas),
        'n_matched': len(merged)
    }

def per_year_auroc(df, score_col='y_prob', label_col='y_true'):
    years = sorted(df['year'].unique())
    results = {}
    for y in years:
        ydf = df[df['year'] == y]
        if len(np.unique(ydf[label_col])) < 2:
             continue
        results[y] = roc_auc_score(ydf[label_col], ydf[score_col])
    return results

def main():
    models = ['random', 'xgb', 'xgb_feat', 'logreg', 'rf', 'gnn_sage']
    dz_thresholds = [0.05, 0.10, 0.15, 0.20]
    
    viz_dir = Path("experiments/signal_isolation/results/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    summary_records = []
    per_year_records = []

    for model in models:
        logger.info("Analyzing model: %s", model)
        try:
            df_base = load_preds(model, 0.0)
            base_auc = roc_auc_score(df_base['label'], df_base['prob'])
            base_yearly = per_year_auroc(df_base, score_col='prob', label_col='label')
            
            # Record baseline stats
            summary_records.append({
                'model': model,
                'dz': 0.0,
                'pooled_auc': base_auc,
                'delta': 0.0,
                'ci_lower': 0.0,
                'ci_upper': 0.0,
                'p_wilcoxon': 1.0,
                'p_ttest': 1.0
            })
            
            for y, a in base_yearly.items():
                per_year_records.append({'model': model, 'dz': 0.0, 'year': y, 'auroc': a})

            for dz in dz_thresholds:
                try:
                    df_dz = load_preds(model, dz)
                    pooled_auc = roc_auc_score(df_dz['label'], df_dz['prob'])
                    dz_yearly = per_year_auroc(df_dz, score_col='prob', label_col='label')
                    
                    # Bootstrap delta
                    keys = ['transaction_id']
                    # Normalize keys to lowercase for merge
                    keys = [k.lower() for k in keys]
                    
                    boot = bootstrap_paired_delta(df_base, df_dz, keys, 'prob', 'label')
                    
                    # Tests across years
                    years = sorted(list(set(base_yearly.keys()) & set(dz_yearly.keys())))
                    v0 = [base_yearly[y] for y in years]
                    v1 = [dz_yearly[y] for y in years]
                    
                    w_stat, w_p = wilcoxon(v1, v0) if len(years) >= 5 else (np.nan, np.nan)
                    t_stat, t_p = ttest_rel(v1, v0) if len(years) >= 2 else (np.nan, np.nan)
                    
                    summary_records.append({
                        'model': model,
                        'dz': dz,
                        'pooled_auc': pooled_auc,
                        'delta': boot['observed_delta'] if boot else pooled_auc - base_auc,
                        'ci_lower': boot['ci_lower'] if boot else np.nan,
                        'ci_upper': boot['ci_upper'] if boot else np.nan,
                        'p_wilcoxon': w_p,
                        'p_ttest': t_p
                    })
                    
                    for y, a in dz_yearly.items():
                        per_year_records.append({'model': model, 'dz': dz, 'year': y, 'auroc': a})
                        
                except Exception as e:
                    logger.error("Error analyzing %s at DZ=%.2f: %s", model, dz, e)
                    
        except Exception as e:
            logger.error("Error loading baseline for %s: %s", model, e)

    # Save results
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(viz_dir / "sweep_stats_summary.csv", index=False)
    
    per_year_df = pd.DataFrame(per_year_records)
    per_year_df.to_csv(viz_dir / "sweep_per_year_auroc.csv", index=False)
    
    # Print formatted table
    print("\n" + "="*80)
    print(f"{'Model':<12} {'DZ':<6} {'Pooled AUC':<12} {'Delta':<10} {'95% CI':<18} {'p_W':<8} {'p_T':<8}")
    print("-" * 80)
    for _, r in summary_df.iterrows():
        ci = f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]" if not np.isnan(r['ci_lower']) else "N/A"
        print(f"{r['model']:<12} {r['dz']:<6.2f} {r['pooled_auc']:<12.4f} {r['delta']:<10.4f} {ci:<18} {r['p_wilcoxon']:<8.3f} {r['p_ttest']:<8.3f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
