"""
Baseline Model Comparison Framework

Compares traditional ML models against TGN using identical features:
- Politician features: Party, State, BioGuideID, District (one-hot encoded)
- Transaction features: Log_Trade_Size, Is_Buy, Log_Gap
- Market features: 14-dim from price_sequences.pt

Evaluation: Monthly rolling window (same as TGN ablation)
Output: JSON reports matching TGN ablation format (standard + directional)

Usage:
    python baselines/run_baselines.py --horizon 1M --alpha 0.0 [--model xgboost]
    python baselines/run_baselines.py --horizon 1M --alpha 0.0 --all  # All models
"""

import pandas as pd
import numpy as np
import torch
import json
import os
import argparse
from datetime import datetime
from pathlib import Path

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, 
    roc_auc_score, precision_recall_curve, auc
)

# Model imports
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "baselines"


# Horizon mapping for label columns
HORIZON_DAYS = {
    '1M': 30, '2M': 60, '3M': 90, '6M': 180,
    '8M': 240, '12M': 365, '18M': 548, '24M': 730
}


def get_model(model_name: str):
    """Return a model instance by name with vanilla default parameters."""
    models = {
        'xgboost': xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        ),
        'lightgbm': lgb.LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'random_forest': RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ),
        'mlp': MLPClassifier(
            random_state=42
        ),
        'logistic_regression': LogisticRegression(
            random_state=42,
            n_jobs=-1,
            max_iter=1000  # Needed for convergence
        ),
        'knn': KNeighborsClassifier(
            n_jobs=-1
        )
    }
    return models.get(model_name.lower())


def load_data(horizon: str = '1M', alpha: float = 0.0):
    """Load and prepare data with exact TGN feature parity."""
    
    # Load ML dataset
    ml_path = DATA_DIR / "processed" / "ml_dataset_reduced_attributes.csv"
    print(f"Loading {ml_path}...")
    df = pd.read_csv(ml_path, low_memory=False)
    
    # Dates
    df['Traded'] = pd.to_datetime(df['Traded'])
    df['Filed'] = pd.to_datetime(df['Filed'])
    
    # Excess return column for this horizon
    er_col = f'Excess_Return_{horizon}'
    if er_col not in df.columns:
        raise ValueError(f"Column {er_col} not found. Available: {[c for c in df.columns if 'Excess' in c]}")
    
    df = df.dropna(subset=[er_col, 'Filed'])
    
    # Win/Loss Label (Training Target)
    # Alpha threshold: win if excess_return > alpha
    df['Target_WinLoss'] = (df[er_col] > alpha).astype(int)
    
    # Directional Label (for Directional Reporting)
    # Price went up = 1, down = 0 (regardless of transaction type)
    # We need Stock_Return column, derive from Excess + SPY
    # For simplicity: if Excess > 0 and transaction is Sale, or if Excess > 0 and Purchase, 
    # Actually TGN uses: For Sell, "success" = stock went DOWN
    # Let's compute from the raw returns if available
    stock_return_col = f'Stock_Return_{horizon}' if f'Stock_Return_{horizon}' in df.columns else None
    
    if stock_return_col and stock_return_col in df.columns:
        df['Target_Direction'] = (df[stock_return_col] > 0).astype(int)
    else:
        # Fallback: approximate from excess return (not perfect)
        df['Target_Direction'] = (df[er_col] > 0).astype(int)
    
    # --- Features (Matching TGN exactly) ---
    
    # 1. Parse Trade Size
    def parse_trade_size(val):
        try:
            val = str(val).replace('$', '').replace(',', '').strip()
            if ' - ' in val:
                low, high = val.split(' - ')
                return (float(low) + float(high)) / 2
            return float(val)
        except:
            return 0.0
    
    df['Trade_Size_USD_Parsed'] = df['Trade_Size_USD'].apply(parse_trade_size)
    df['Log_Trade_Size'] = np.log1p(df['Trade_Size_USD_Parsed'])
    
    # 2. Is Buy (-1 for Sell, +1 for Buy)
    df['Is_Buy'] = df['Transaction'].astype(str).apply(
        lambda x: 1.0 if 'Purchase' in x.title() else (-1.0 if 'Sale' in x.title() else 0.0)
    )
    df = df[df['Is_Buy'] != 0].copy()  # Filter Exchange
    
    # 3. Filing Gap
    def calc_gap(row):
        if pd.isnull(row['Filed']) or pd.isnull(row['Traded']):
            return 30
        gap = (row['Filed'] - row['Traded']).days
        return max(0, gap)
    
    df['Gap_Days'] = df.apply(calc_gap, axis=1)
    df['Log_Gap'] = np.log1p(df['Gap_Days'])
    
    # 4. Categoricals (One-Hot Encoded) - Match TGN static features
    for col in ['Party', 'State', 'BioGuideID', 'District']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
        else:
            df[col] = 'Unknown'
    
    # 5. Load 14-dim Market Features (from price_sequences.pt)
    price_path = DATA_DIR / "price_sequences.pt"
    if price_path.exists():
        print(f"Loading market features from {price_path}...")
        price_map = torch.load(price_path)
        
        # Create market feature columns
        mkt_cols = [f'Market_{i+1}' for i in range(14)]
        for col in mkt_cols:
            df[col] = 0.0
        
        # Map by transaction_id
        id_col = 'transaction_id' if 'transaction_id' in df.columns else 'ID'
        if id_col in df.columns:
            for idx, row in df.iterrows():
                tid = row[id_col]
                if tid in price_map:
                    feat = price_map[tid].numpy()
                    for i, val in enumerate(feat):
                        df.at[idx, mkt_cols[i]] = val
        print(f"Loaded market features for {(df[mkt_cols[0]] != 0).sum()} / {len(df)} transactions")
    else:
        print(f"Warning: {price_path} not found. Using zero market features.")
        mkt_cols = [f'Market_{i+1}' for i in range(14)]
        for col in mkt_cols:
            df[col] = 0.0
    
    return df


def compute_metrics(y_true, y_pred, y_prob):
    """Compute metrics matching TGN ablation format."""
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Add AUC
    try:
        report['auc'] = float(roc_auc_score(y_true, y_prob))
    except:
        report['auc'] = 0.5
    
    # Add PR-AUC
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        report['pr_auc'] = float(auc(recall, precision))
    except:
        report['pr_auc'] = 0.5
    
    return report


def run_monthly_evaluation(df, model_name: str, horizon: str, alpha: float):
    """Run monthly rolling evaluation (matching TGN ablation)."""
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name.upper()}")
    print(f"Horizon: {horizon}, Alpha: {alpha}")
    print(f"{'='*60}")
    
    # Feature columns
    cat_features = ['Party', 'State', 'BioGuideID', 'District']
    num_features = ['Log_Trade_Size', 'Is_Buy', 'Log_Gap'] + [f'Market_{i+1}' for i in range(14)]
    
    # Create output directory
    out_dir = RESULTS_DIR / f"H_{horizon}_A_{alpha}" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    start_year = 2019
    end_year = 2024
    
    all_results = []
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Test period
            test_start = pd.Timestamp(year=year, month=month, day=1)
            if month == 12:
                test_end = pd.Timestamp(year=year+1, month=1, day=1)
            else:
                test_end = pd.Timestamp(year=year, month=month+1, day=1)
            
            # Growing window: train on everything before test_start
            train_mask = df['Filed'] < test_start
            test_mask = (df['Filed'] >= test_start) & (df['Filed'] < test_end)
            
            train_df = df[train_mask]
            test_df = df[test_mask]
            
            if len(test_df) == 0 or len(train_df) < 100:
                continue
            
            print(f"  {year}-{month:02d} | Train: {len(train_df):5d} | Test: {len(test_df):4d}", end="")
            
            # Prepare features
            X_train = train_df[cat_features + num_features]
            y_train_winloss = train_df['Target_WinLoss']
            
            X_test = test_df[cat_features + num_features]
            y_test_winloss = test_df['Target_WinLoss']
            y_test_direction = test_df['Target_Direction']
            
            # Build pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), num_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
                ])
            
            model = get_model(model_name)
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # Train on Win/Loss
            pipeline.fit(X_train, y_train_winloss)
            
            # Predict
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            
            # Compute Standard Report (Win/Loss)
            report_standard = compute_metrics(y_test_winloss, y_pred, y_prob)
            
            # Compute Directional Report
            # For directional: prediction is same, but ground truth is direction
            report_directional = compute_metrics(y_test_direction, y_pred, y_prob)
            
            # Save JSON reports
            std_file = out_dir / f"report_{model_name}_{year}_{month:02d}.json"
            dir_file = out_dir / f"report_{model_name}_{year}_{month:02d}_directional.json"
            
            with open(std_file, 'w') as f:
                json.dump(report_standard, f, indent=4)
            with open(dir_file, 'w') as f:
                json.dump(report_directional, f, indent=4)
            
            print(f" | F1: {report_standard.get('1.0', {}).get('f1-score', 0):.3f} | AUC: {report_standard.get('auc', 0):.3f}")
            
            all_results.append({
                'Model': model_name,
                'Year': year,
                'Month': month,
                'Train_Size': len(train_df),
                'Test_Size': len(test_df),
                'Accuracy': report_standard.get('accuracy', 0),
                'F1_Class1': report_standard.get('1.0', {}).get('f1-score', 0),
                'AUC': report_standard.get('auc', 0),
                'PR_AUC': report_standard.get('pr_auc', 0),
                'Dir_Accuracy': report_directional.get('accuracy', 0),
                'Dir_F1': report_directional.get('1.0', {}).get('f1-score', 0),
                'Dir_AUC': report_directional.get('auc', 0)
            })
    
    # Save summary CSV
    summary_df = pd.DataFrame(all_results)
    summary_file = RESULTS_DIR / f"H_{horizon}_A_{alpha}" / f"summary_{model_name}.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\n{model_name.upper()} Summary:")
    print(f"  Avg F1 (Win/Loss): {summary_df['F1_Class1'].mean():.4f}")
    print(f"  Avg AUC: {summary_df['AUC'].mean():.4f}")
    print(f"  Avg Directional F1: {summary_df['Dir_F1'].mean():.4f}")
    print(f"  Results saved to: {out_dir}")
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Baseline Model Comparison Framework")
    parser.add_argument('--horizon', type=str, default='1M', 
                        choices=['1M', '2M', '3M', '6M', '8M', '12M', '18M', '24M'],
                        help='Prediction horizon')
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='Alpha threshold for win/loss classification')
    parser.add_argument('--model', type=str, default='xgboost',
                        choices=['xgboost', 'lightgbm', 'random_forest', 'mlp', 
                                 'logistic_regression', 'knn'],
                        help='Model to run')
    parser.add_argument('--all', action='store_true',
                        help='Run all models (sweep)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Baseline Model Comparison Framework")
    print("="*60)
    print(f"Horizon: {args.horizon}")
    print(f"Alpha: {args.alpha}")
    
    # Load data
    df = load_data(horizon=args.horizon, alpha=args.alpha)
    print(f"Loaded {len(df)} transactions")
    
    if args.all:
        models = ['xgboost', 'lightgbm', 'random_forest', 'mlp', 'logistic_regression', 'knn']
    else:
        models = [args.model]
    
    all_summaries = []
    for model_name in models:
        summary = run_monthly_evaluation(df, model_name, args.horizon, args.alpha)
        all_summaries.append(summary)
    
    # Combined summary
    if len(all_summaries) > 1:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined_file = RESULTS_DIR / f"H_{args.horizon}_A_{args.alpha}" / "summary_all_models.csv"
        combined.to_csv(combined_file, index=False)
        
        print("\n" + "="*60)
        print("ALL MODELS SUMMARY")
        print("="*60)
        print(combined.groupby('Model')[['F1_Class1', 'AUC', 'Dir_F1', 'Dir_AUC']].mean().round(4))
    
    print("\n✅ Baseline evaluation complete!")


if __name__ == "__main__":
    main()
