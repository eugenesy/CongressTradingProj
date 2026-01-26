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
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, 
    roc_auc_score, precision_recall_curve, auc
)

# Model imports
import xgboost as xgb
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
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



# Validating GPU availability for Torch MLP
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

class TorchMLPClassifier:
    """
    sklearn-compatible MLP Classifier using PyTorch for GPU acceleration.
    Replaces sklearn.neural_network.MLPClassifier.
    """
    def __init__(self, hidden_dim=128, lr=0.001, epochs=50, batch_size=1024, device=DEVICE):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self.classes_ = np.array([0, 1])
        
    def fit(self, X, y):
        # Convert to numpy if pandas
        if hasattr(X, 'values'): X = X.values
        if hasattr(y, 'values'): y = y.values
        
        input_dim = X.shape[1]
        
        # Simple MLP
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2), # Add dropout for regularization
            torch.nn.Linear(self.hidden_dim, 1),
            torch.nn.Sigmoid()
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.BCELoss()
        
        # Prepare Data
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                
        return self

    def predict_proba(self, X):
        if hasattr(X, 'values'): X = X.values
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()
            
        # Return [Prob_0, Prob_1]
        return np.hstack([1 - preds, preds])

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

def get_model(model_name: str):
    """Return a model instance by name with vanilla default parameters."""
    models = {
        'xgboost': xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        ),
        'random_forest': RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ),
        'mlp': TorchMLPClassifier(
            hidden_dim=100,
            epochs=50,
            lr=0.001
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
    
    if lgb is not None:
        models['lightgbm'] = lgb.LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    return models.get(model_name.lower())


def load_data(horizon: str = '1M', alpha: float = 0.0, **kwargs):
    """Load and prepare data with exact TGN feature parity."""
    
    # Load ML dataset
    ml_path = DATA_DIR / "processed" / "ml_dataset_clean.csv"
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
    
    # --- Step 1: Create Transaction Features FIRST ---
    
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
    
    # 2. Is Buy (-1 for Sell, +1 for Buy) - CREATE THIS FIRST BEFORE LABELS
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
    
    # --- Step 2: NOW Create Labels (using Is_Buy) ---
    # Win/Loss Label (Training Target) - TRANSACTION-AWARE
    # Matching TGN logic exactly:
    # - Buy: Win if excess_return > alpha (stock outperformed)
    # - Sell: Win if excess_return < alpha (stock underperformed)
    
    df['Target_WinLoss'] = 0  # Initialize
    
    # If Directional: Target is simply Excess Return > Alpha
    # (Regardless of Buy/Sell type)
    if 'directional' in kwargs and kwargs.get('directional'):
        print("Required: Directional Labels (Return > Alpha)")
        df.loc[df[er_col] > alpha, 'Target_WinLoss'] = 1
    else:
        # Standard Win/Loss (Transaction Aware)
        buy_mask = (df['Is_Buy'] == 1.0)
        sell_mask = (df['Is_Buy'] == -1.0)
        
        # Buy wins if return > alpha
        df.loc[buy_mask & (df[er_col] > alpha), 'Target_WinLoss'] = 1
        
        # Sell wins if return < alpha (stock went down relative to SPY)
        df.loc[sell_mask & (df[er_col] < alpha), 'Target_WinLoss'] = 1
    
    # Directional Label (for Directional Reporting)
    # This will be "flipped" for Sells during reporting (matching TGN)
    # For now, just use the same as win/loss - we'll flip during evaluation
    df['Target_Direction'] = df['Target_WinLoss'].copy()
    df['Transaction_Type'] = df['Is_Buy'].copy()  # Store for flipping later
    
    
    # 4. Categoricals (One-Hot Encoded) - Match TGN static features EXACTLY
    # TGN only uses Party and State (+ BioGuideID as node mapping)
    for col in ['Party', 'State', 'BioGuideID']:
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
    
    if 'directional' in kwargs:
        df.attrs['directional'] = kwargs['directional']
    
    return df


def compute_metrics(y_true, y_pred, y_prob):
    """Compute metrics matching TGN ablation format."""
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Add AUC
    try:
        report['auc'] = float(roc_auc_score(y_true, y_prob))
    except:
        report['auc'] = None
    
    # Add PR-AUC
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        report['pr_auc'] = float(auc(recall, precision))
    except:
        report['pr_auc'] = None
    
    return report


def run_monthly_evaluation(df, model_name: str, horizon: str, alpha: float, **kwargs):
    """Run monthly rolling evaluation (matching TGN ablation)."""
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name.upper()}")
    print(f"Horizon: {horizon}, Alpha: {alpha}")
    print(f"{'='*60}")
    
    # Feature columns (EXACT match with TGN)
    cat_features = ['Party', 'State', 'BioGuideID']
    num_features = ['Log_Trade_Size', 'Is_Buy', 'Log_Gap'] + [f'Market_{i+1}' for i in range(14)]
    
    # Print feature summary
    print(f"\n📊 Feature Summary:")
    print(f"  Categorical Features ({len(cat_features)}): {cat_features}")
    print(f"  Numerical Features ({len(num_features)}): {num_features[:3]} + 14 market features")
    print(f"  Total Features: {len(cat_features) + len(num_features)}")
    print(f"\n  Feature Alignment with TGN:")
    print(f"    ✓ Politician: Party, State, BioGuideID (one-hot)")
    print(f"    ✓ Transaction: Log_Trade_Size, Is_Buy, Log_Gap")
    print(f"    ✓ Market: 14-dim price features from price_sequences.pt")
    print(f"    ✓ Labels: Transaction-aware (Buy/Sell different win conditions)")
    print()
    
    # Create output directory
    if 'directional' in df.attrs and df.attrs['directional']:
        base_dir = PROJECT_ROOT / "results" / "directional_baselines"
        print("Saving to DIRECTIONAL output directory")
    else:
        base_dir = RESULTS_DIR
        
    out_dir = base_dir / f"H_{horizon}_A_{alpha}" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    start_year = 2019
    end_year = 2024
    
    all_results = []
    
    # Create list of all year-month combinations
    year_months = []
    # Use specified year if provided, else run full range
    target_year = kwargs.get('year', None)
    if target_year:
        eval_years = [target_year]
    else:
        eval_years = range(start_year, end_year + 1)
        
    for year in eval_years:
        for month in range(1, 13):
            year_months.append((year, month))
    
    # Progress bar for monthly evaluation
    for year, month in tqdm(year_months, desc=f"{model_name.upper()}", unit="month"):
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
            
            # Prepare features
            X_train = train_df[cat_features + num_features]
            y_train_winloss = train_df['Target_WinLoss']
            
            X_test = test_df[cat_features + num_features]
            y_test_winloss = test_df['Target_WinLoss']
            
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
            
            # Save raw probabilities (NEW)
            if df.attrs.get('directional', False):
                base_dir_probs = PROJECT_ROOT / "results" / "directional_baselines"
            else:
                base_dir_probs = RESULTS_DIR
                
            probs_dir = base_dir_probs / f"H_{horizon}_A_{alpha}" / "probs"
            probs_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to list for JSON serialization
            probs_output = {
                'y_true': y_test_winloss.tolist(),
                'y_pred': y_pred.tolist(),
                'y_prob': y_prob.tolist(),
                'dates': test_df['Filed'].astype(str).tolist()
            }
            probs_file = probs_dir / f"probs_{model_name}_{year}_{month:02d}.json"
            with open(probs_file, 'w') as f:
                json.dump(probs_output, f)

            # Compute Standard Report
            report_standard = compute_metrics(y_test_winloss, y_pred, y_prob)
            
            # Helper to safely get F1 Score
            def get_f1(report):
                if '1' in report: return report['1']['f1-score']
                if '1.0' in report: return report['1.0']['f1-score']
                if 1 in report: return report[1]['f1-score']
                return 0.0

            is_directional = df.attrs.get('directional', False)
            if is_directional:
                # In directional mode, the "standard" report is the directional one.
                report_directional = report_standard
                dir_f1_score_val = get_f1(report_directional)
                f1_score_val = dir_f1_score_val
            else:
                # Compute Directional Report (Legacy flipping)
                trans_types = test_df['Transaction_Type'].values
                sell_mask = (trans_types == -1.0)
                y_test_dir_flipped = y_test_winloss.copy()
                y_pred_flipped = y_pred.copy()
                y_prob_flipped = y_prob.copy()
                y_test_dir_flipped[sell_mask] = 1 - y_test_dir_flipped[sell_mask]
                y_pred_flipped[sell_mask] = 1 - y_pred_flipped[sell_mask]
                y_prob_flipped[sell_mask] = 1 - y_prob_flipped[sell_mask]
                report_directional = compute_metrics(y_test_dir_flipped, y_pred_flipped, y_prob_flipped)
                f1_score_val = get_f1(report_standard)
                dir_f1_score_val = get_f1(report_directional)

            # Save JSON reports
            std_file = out_dir / f"report_{model_name}_{year}_{month:02d}.json"
            with open(std_file, 'w') as f:
                json.dump(report_standard, f, indent=4)
            
            if not is_directional:
                 dir_file = out_dir / f"report_{model_name}_{year}_{month:02d}_directional.json"
                 with open(dir_file, 'w') as f:
                     json.dump(report_directional, f, indent=4)


            # Update progress bar with current metrics
            tqdm.write(f"  {year}-{month:02d} | Train: {len(train_df):5d} | Test: {len(test_df):4d} | F1: {f1_score_val:.3f} | AUC: {report_standard.get('auc', 0):.3f}")
            
            res = {
                'Model': model_name,
                'Year': year,
                'Month': month,
                'Train_Size': len(train_df),
                'Test_Size': len(test_df),
                'Accuracy': report_standard.get('accuracy', 0),
                'F1_Class1': f1_score_val,
                'AUC': report_standard.get('auc', 0),
                'PR_AUC': report_standard.get('pr_auc', 0),
            }
            if not is_directional:
                res.update({
                    'Dir_Accuracy': report_directional.get('accuracy', 0),
                    'Dir_F1': dir_f1_score_val,
                    'Dir_AUC': report_directional.get('auc', 0)
                })
            all_results.append(res)
    
    # Save summary CSV
    if 'directional' in df.attrs and df.attrs['directional']:
         base_dir = PROJECT_ROOT / "results" / "directional_baselines"
    else:
         base_dir = RESULTS_DIR

    summary_df = pd.DataFrame(all_results)
    summary_file = base_dir / f"H_{horizon}_A_{alpha}" / f"summary_{model_name}.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\n{model_name.upper()} Summary:")
    print(f"  Avg F1 (Standard): {summary_df['F1_Class1'].mean():.4f}")
    print(f"  Avg AUC: {summary_df['AUC'].mean():.4f}")
    
    if not is_directional:
        print(f"  Avg Directional F1: {summary_df['Dir_F1'].mean():.4f}")
        print(f"  Avg Directional AUC: {summary_df['Dir_AUC'].mean():.4f}")
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
    parser.add_argument('--directional', action='store_true',
                        help='Use Directional Target (Up/Down) instead of Win/Loss')
    parser.add_argument('--year', type=int, help='Specific year to run (e.g. 2023)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Baseline Model Comparison Framework")
    print("="*60)
    print(f"Horizon: {args.horizon}")
    print(f"Alpha: {args.alpha}")
    
    # Load data
    df = load_data(horizon=args.horizon, alpha=args.alpha, directional=args.directional)
    print(f"Loaded {len(df)} transactions")
    
    if args.all:
        # Filtered to main 3 models as requested
        models = ['xgboost', 'mlp', 'logistic_regression'] 
        print(f"Running models: {models}")
    else:
        models = [args.model]
    
    all_summaries = []
    for model_name in models:
        summary = run_monthly_evaluation(df, model_name, args.horizon, args.alpha, year=args.year)
        all_summaries.append(summary)
    
    # Combined summary
    if len(all_summaries) > 1:
        if args.directional:
             base_dir = PROJECT_ROOT / "results" / "directional_baselines"
        else:
             base_dir = RESULTS_DIR
             
        combined = pd.concat(all_summaries, ignore_index=True)
        combined_file = base_dir / f"H_{args.horizon}_A_{args.alpha}" / "summary_all_models.csv"
        combined.to_csv(combined_file, index=False)
        
        print("\n" + "="*60)
        print("ALL MODELS SUMMARY")
        print("="*60)
        
        summary_cols = ['F1_Class1', 'AUC']
        if not args.directional:
            summary_cols += ['Dir_F1', 'Dir_AUC']
            
        print(combined.groupby('Model')[summary_cols].mean().round(4))
    
    print("\n✅ Baseline evaluation complete!")


if __name__ == "__main__":
    main()
