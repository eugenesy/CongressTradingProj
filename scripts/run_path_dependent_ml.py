"""
run_path_dependent_ml.py
========================
Executable Pipeline Runners. Imports src.* modules to execute experiments.

Refactored/Audited: 2026-03-20
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import roc_auc_score

def vectorize_continuous_labels(df, parquet_dir, horizon_days=126):
    """
    Groups by ticker to load parquet once and quickly back-test Continuous Max Excess.
    """
    df = df.copy()
    df['Filed'] = pd.to_datetime(df['Filed'])
    
    # Load SPY
    spy_path = Path(parquet_dir) / "SPY.parquet"
    if not spy_path.exists():
        raise Exception("SPY.parquet not found")
    spy_df = pd.read_parquet(spy_path)
    spy_df.index = pd.to_datetime(spy_df.index)
    spy_df = spy_df.sort_index()

    print("Vectorizing max returns grouping by Ticker...")
    results = []
    
    # Group by ticker to optimize I/O
    grouped = df.dropna(subset=['Ticker', 'Filed']).groupby('Ticker')
    
    for ticker, group in tqdm(grouped, desc="Processing Tickers"):
        ticker_str = str(ticker).strip().upper()
        tick_path = Path(parquet_dir) / f"{ticker_str}.parquet"
        
        if not tick_path.exists():
            continue
            
        try:
            tick_df = pd.read_parquet(tick_path)
            tick_df.index = pd.to_datetime(tick_df.index)
            tick_df = tick_df.sort_index()
            
            for index, row in group.iterrows():
                trade_date = row['Filed']
                
                # Fetch windows
                stock_sub = tick_df.loc[trade_date:]
                spy_sub = spy_df.loc[trade_date:]
                
                if len(stock_sub) < horizon_days or len(spy_sub) < horizon_days:
                    continue
                    
                stock_w = stock_sub.head(horizon_days)
                spy_w = spy_sub.head(horizon_days)
                
                p0 = stock_w['close'].iloc[0]
                s0 = spy_w['close'].iloc[0]
                
                r_stock = stock_w['close'] / p0 - 1
                r_spy = spy_w['close'] / s0 - 1
                
                excess = (r_stock - r_spy).dropna()
                if not excess.empty:
                    max_excess = excess.max()
                    results.append({
                        'transaction_id': row['transaction_id'],
                        'max_excess_6m': max_excess
                    })
        except Exception:
            # Skip brittle files
            continue

    lbl_df = pd.DataFrame(results)
    if lbl_df.empty:
        raise Exception("No labels were successfully generated.")
        
    merged = df.merge(lbl_df, on='transaction_id', how='inner')
    print(f"Generated continuous max_excess_6m labels for {len(merged)} rows.")
    return merged

def compute_precision_at_k(y_true, y_prob, years, months, k=0.10):
    """Calculate average monthly Precision@Top-k to match GNN grouped averages."""
    preds_df = pd.DataFrame({
        'label': y_true,
        'prob': y_prob,
        'year': years,
        'month': months
    })
    
    precisions = []
    grouped = preds_df.groupby(['year', 'month'])
    for name, group in grouped:
         if len(group) < 5:
              continue
         group = group.sort_values(by='prob', ascending=False)
         # Pick top 10%
         n_top = max(1, int(len(group) * k))
         top_k = group.head(n_top)
         prec = top_k['label'].mean()
         precisions.append(prec)
    return np.mean(precisions) if precisions else 0.0

def run_ml_comparison(df):
    print("\n--- Running Machine Learning Comparison with P@10% ---")
    
    df = df.copy()
    df['Filed'] = pd.to_datetime(df['Filed'])
    df['year'] = df['Filed'].dt.year
    df['month'] = df['Filed'].dt.month
    
    # Standard 6M targets
    df['Label_Median'] = (df['max_excess_6m'] > 0.088).astype(int)
    df['Label_Q3'] = (df['max_excess_6m'] > 0.178).astype(int)
    
    print(f"Median positive rate: {df['Label_Median'].mean():.2%}")
    print(f"Q3 positive rate:     {df['Label_Q3'].mean():.2%}")

    # Encode IDs
    le_bio = LabelEncoder()
    le_tick = LabelEncoder()
    df['BioGuideID_enc'] = le_bio.fit_transform(df['BioGuideID'].astype(str))
    df['Ticker_enc'] = le_tick.fit_transform(df['Ticker'].astype(str))
    
    # Feature columns
    cat_feats = ['Chamber', 'Party', 'State', 'Transaction', 'Industry', 'Sector', 'Trade_Size_USD']
    num_feats = ['Filing_Gap', 'Late_Filing', 'In_SP100']
    
    for c in cat_feats:
        if c in df.columns:
            df[f"{c}_enc"] = LabelEncoder().fit_transform(df[c].astype(str))

    feats_all_enc = [c + "_enc" for c in cat_feats if c in df.columns] + num_feats
    
    # Splitting
    train = df[df['Filed'] < '2023-01-01']
    test = df[(df['Filed'] >= '2023-01-01') & (df['Filed'] < '2024-01-01')]
    
    if train.empty or test.empty:
         print("Error: Train or test splits empty.")
         return
         
    print(f"Train size: {len(train)}, Test size: {len(test)}")

    for target in ['Label_Median', 'Label_Q3']:
        print(f"\nTarget: {target} " + ("#"*20))
        
        # Test 1: ID Only
        X_train_id = train[['BioGuideID_enc', 'Ticker_enc']]
        X_test_id = test[['BioGuideID_enc', 'Ticker_enc']]
        y_train = train[target]
        y_test = test[target]
        
        clf_id = xgb.XGBClassifier(eval_metric='auc', random_state=42)
        clf_id.fit(X_train_id, y_train)
        preds_id = clf_id.predict_proba(X_test_id)[:, 1]
        auc_id = roc_auc_score(y_test, preds_id)
        p10_id = compute_precision_at_k(y_test, preds_id, test['year'], test['month'])
        
        # Test 2: Features Only
        X_train_feat = train[feats_all_enc]
        X_test_feat = test[feats_all_enc]
        
        clf_feat = xgb.XGBClassifier(eval_metric='auc', random_state=42)
        clf_feat.fit(X_train_feat, y_train)
        preds_feat = clf_feat.predict_proba(X_test_feat)[:, 1]
        auc_feat = roc_auc_score(y_test, preds_feat)
        p10_feat = compute_precision_at_k(y_test, preds_feat, test['year'], test['month'])
        
        # Test 3: Combined
        X_train_all = train[['BioGuideID_enc', 'Ticker_enc'] + feats_all_enc]
        X_test_all = test[['BioGuideID_enc', 'Ticker_enc'] + feats_all_enc]
        
        clf_all = xgb.XGBClassifier(eval_metric='auc', random_state=42)
        clf_all.fit(X_train_all, y_train)
        preds_all = clf_all.predict_proba(X_test_all)[:, 1]
        auc_all = roc_auc_score(y_test, preds_all)
        p10_all = compute_precision_at_k(y_test, preds_all, test['year'], test['month'])
        
        print(f"  AUROC (ID-only):   {auc_id:.4f} | P@10%: {p10_id:.4f}")
        print(f"  AUROC (Feat-only):  {auc_feat:.4f} | P@10%: {p10_feat:.4f}")
        print(f"  AUROC (Combined):   {auc_all:.4f} | P@10%: {p10_all:.4f}")

def run_ml_comparison_year(df, test_year):
    print(f"\n--- Running Machine Learning Comparison with P@10% for Test Year {test_year} ---")
    df = df.copy()
    df['Filed'] = pd.to_datetime(df['Filed'])
    df['year'] = df['Filed'].dt.year
    df['month'] = df['Filed'].dt.month
    
    df['Label_Median'] = (df['max_excess_6m'] > 0.088).astype(int)
    df['Label_Q3'] = (df['max_excess_6m'] > 0.178).astype(int)
    
    from sklearn.preprocessing import LabelEncoder
    df['BioGuideID_enc'] = LabelEncoder().fit_transform(df['BioGuideID'].astype(str))
    df['Ticker_enc'] = LabelEncoder().fit_transform(df['Ticker'].astype(str))
    
    cat_feats = ['Chamber', 'Party', 'State', 'Transaction', 'Industry', 'Sector', 'Trade_Size_USD']
    num_feats = ['Filing_Gap', 'Late_Filing', 'In_SP100']
    for c in cat_feats:
        df[f"{c}_enc"] = LabelEncoder().fit_transform(df[c].astype(str))
        
    # Splitting
    train = df[df['Filed'] < f'{test_year}-01-01']
    test = df[(df['Filed'] >= f'{test_year}-01-01') & (df['Filed'] < f'{test_year+1}-01-01')]
    
    if train.empty or test.empty:
        print(f"Error: Train or test splits empty for {test_year}.")
        return
        
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    results_list = []
    
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    for target in ['Label_Median', 'Label_Q3']:
        print(f"\nTarget: {target} (# test={len(test)})")
        X_train_id = train[['BioGuideID_enc', 'Ticker_enc']]
        X_test_id = test[['BioGuideID_enc', 'Ticker_enc']]
        y_train = train[target]
        y_test = test[target]
        
        # 1. ID only
        clf_id = xgb.XGBClassifier(eval_metric='auc', random_state=42, n_jobs=4)
        clf_id.fit(X_train_id, y_train)
        preds_id = clf_id.predict_proba(X_test_id)[:, 1]
        auc_id = roc_auc_score(y_test, preds_id)
        p10_id = compute_precision_at_k(y_test, preds_id, test['year'], test['month'])
        
        pd.DataFrame({
            'year': test['year'], 'month': test['month'], 'label': y_test, 'prob': preds_id
        }).to_csv(f"experiments/signal_isolation/results/preds_xgb_{test_year}_{target}_id.csv", index=False)

        # 2. Features only
        feats_all_enc = [c + "_enc" for c in cat_feats] + num_feats
        X_train_feat = train[feats_all_enc]
        X_test_feat = test[feats_all_enc]
        
        clf_feat = xgb.XGBClassifier(eval_metric='auc', random_state=42, n_jobs=4)
        clf_feat.fit(X_train_feat, y_train)
        preds_feat = clf_feat.predict_proba(X_test_feat)[:, 1]
        auc_feat = roc_auc_score(y_test, preds_feat)
        p10_feat = compute_precision_at_k(y_test, preds_feat, test['year'], test['month'])
        
        pd.DataFrame({
            'year': test['year'], 'month': test['month'], 'label': y_test, 'prob': preds_feat
        }).to_csv(f"experiments/signal_isolation/results/preds_xgb_{test_year}_{target}_feat.csv", index=False)

        # 3. Combined
        X_train_all = train[['BioGuideID_enc', 'Ticker_enc'] + feats_all_enc]
        X_test_all = test[['BioGuideID_enc', 'Ticker_enc'] + feats_all_enc]
        
        clf_all = xgb.XGBClassifier(eval_metric='auc', random_state=42, n_jobs=4)
        clf_all.fit(X_train_all, y_train)
        preds_all = clf_all.predict_proba(X_test_all)[:, 1]
        auc_all = roc_auc_score(y_test, preds_all)
        p10_all = compute_precision_at_k(y_test, preds_all, test['year'], test['month'])
        
        pd.DataFrame({
            'year': test['year'], 'month': test['month'], 'label': y_test, 'prob': preds_all
        }).to_csv(f"experiments/signal_isolation/results/preds_xgb_{test_year}_{target}_all.csv", index=False)

        print(f"  AUROC (ID-only):   {auc_id:.4f} | P@10%: {p10_id:.4f}")
        print(f"  AUROC (Feat-only):  {auc_feat:.4f} | P@10%: {p10_feat:.4f}")
        print(f"  AUROC (Combined):   {auc_all:.4f} | P@10%: {p10_all:.4f}")
        
        results_list.append({
            'test_year': test_year, 'target': target,
            'auc_id': auc_id, 'auc_feat': auc_feat, 'auc_all': auc_all,
            'p10_id': p10_id, 'p10_feat': p10_feat, 'p10_all': p10_all
        })
        
    res_df = pd.DataFrame(results_list)
    res_df.to_csv(f"experiments/signal_isolation/results/xgb_res_{test_year}.csv", index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-year', type=int, default=2023, help='Test year')
    args = parser.parse_args()
    
    try:
        labeled_df = pd.read_csv("data/processed/ml_dataset_continuous.csv")
        run_ml_comparison_year(labeled_df, args.test_year)
    except Exception as e:
         print(f"Error during execution: {e}")
