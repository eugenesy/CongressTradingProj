import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("baselines")

def direction_targets(series, alpha):
    series = pd.Series(series)
    up = series > alpha
    down = series < alpha
    mask = series.notna() & (up | down)
    return up.astype(float), mask


class TorchMLPClassifier:
    def __init__(self, hidden_dim=128, lr=0.001, epochs=50, batch_size=512, device=None):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        input_dim = X.shape[1]

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.hidden_dim, 1),
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, device=self.device),
            torch.tensor(y, device=self.device),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, device=self.device))
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.hstack([1 - probs, probs])

def run_fair_baselines(horizon='6M', alpha=0.0, start_year=2023, end_year=2023):
    # 1. Load Data
    df = pd.read_csv("data/processed/ml_dataset_clean.csv")
    df['Filed'] = pd.to_datetime(df['Filed'])
    df['Traded'] = pd.to_datetime(df['Traded'])
    
    # 2. Features
    def clean_amt(x):
        if pd.isna(x): return 0.0
        if isinstance(x, (int, float)): return float(x)
        s = str(x).replace('$','').replace(',','').strip()
        if '-' in s:
            try:
                parts = s.split('-')
                low = float(parts[0].strip())
                high = float(parts[1].strip())
                return (low + high) / 2.0
            except:
                return 0.0
        try:
            return float(s)
        except:
            return 0.0

    df['amt_numeric'] = df['Trade_Size_USD'].apply(clean_amt)
    df['amt_enc'] = np.log1p(df['amt_numeric'])
    df['is_buy'] = df['Transaction'].apply(lambda x: 1.0 if 'Purchase' in str(x) else -1.0)
    df['filing_gap'] = (df['Filed'] - df['Traded']).dt.days.clip(0)
    df['gap_enc'] = np.log1p(df['filing_gap'])
    
    # Target
    target_col = f'Excess_Return_{horizon}'
    
    # Splitting logic
    h_days = {'1M':30, '2M':60, '3M':90, '6M':180, '8M':240, '12M':365, '18M':545, '24M':730}[horizon]
    
    results_xgb = []
    results_lr = []
    results_mlp = []
    
    # Store Row-Level Predictions
    # Schema: [TransactionID, Date, Model, Prob_Up, Pred_Up, True_Up]
    predictions_log = []

    # Features: Since XGBoost doesn't use the GNN emb, we give it the 14-dim price context + 3 core features.
    # Note: price_sequences.pt contains 14-dim vectors per transaction_id.
    price_map = torch.load("data/price_sequences.pt", weights_only=False)
    
    def get_features(indices):
        X = []
        for idx in indices:
            row = df.iloc[idx]
            tid = int(row['transaction_id'])
            p_feat = price_map[tid].numpy() if tid in price_map else np.zeros(14)
            core = [row['amt_enc'], row['is_buy'], row['gap_enc']]
            X.append(np.concatenate([core, p_feat]))
        return np.array(X)

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            logger.info(f"\n--- FAIR BASELINE: {year}-{month:02d} | Horizon: {horizon} ---")

            test_start = pd.Timestamp(year, month, 1)
            test_end = test_start + pd.offsets.MonthEnd(1)

            test_df = df[(df['Filed'] >= test_start) & (df['Filed'] <= test_end)]
            if len(test_df) == 0:
                continue

            # Training set: Filed BEFORE test_start
            train_df = df[df['Filed'] < test_start]

            # RESPECT THE GAP: Must be resolved BEFORE test_start
            # Resolution = Traded + Horizon
            train_df = train_df.copy()
            train_df['Resolution'] = train_df['Traded'] + pd.Timedelta(days=h_days)

            # Valid training rows: Resolution < test_start AND Label exists
            mask = (train_df['Resolution'] < test_start) & (train_df[target_col].notna())
            train_final = train_df.loc[mask]

            if len(train_final) < 100:
                logger.warning("Not enough training data, skipping.")
                continue

            X_train = get_features(train_final.index)
            y_train, train_mask = direction_targets(train_final[target_col], alpha)
            if train_mask.sum() == 0:
                logger.warning("No directional labels in training set, skipping.")
                continue
            train_mask_arr = train_mask.to_numpy()
            X_train = X_train[train_mask_arr]
            y_train = y_train[train_mask].astype(float).to_numpy()

            X_test = get_features(test_df.index)
            y_test, test_mask = direction_targets(test_df[target_col], alpha)
            if test_mask.sum() == 0:
                logger.warning("No directional labels in test set, skipping.")
                continue
            test_mask_arr = test_mask.to_numpy()
            X_test = X_test[test_mask_arr]
            y_test = y_test[test_mask].astype(float).to_numpy()

            # 1. XGBoost
            model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
            model_xgb.fit(X_train, y_train)
            preds_xgb = np.asarray(model_xgb.predict_proba(X_test))[:, 1]

            auc_xgb = roc_auc_score(y_test, preds_xgb) if len(set(y_test)) > 1 else 0.5
            results_xgb.append({'Year': year, 'Month': month, 'AUC': auc_xgb})

            # 2. LogReg
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model_lr = LogisticRegression(max_iter=1000)
            model_lr.fit(X_train_scaled, y_train)
            preds_lr = np.asarray(model_lr.predict_proba(X_test_scaled))[:, 1]

            auc_lr = roc_auc_score(y_test, preds_lr) if len(set(y_test)) > 1 else 0.5
            results_lr.append({'Year': year, 'Month': month, 'AUC': auc_lr})

            # 3. MLP (Torch)
            model_mlp = TorchMLPClassifier(
                hidden_dim=128,
                lr=0.001,
                epochs=50,
                batch_size=512,
            )
            model_mlp.fit(X_train_scaled, y_train)
            preds_mlp = np.asarray(model_mlp.predict_proba(X_test_scaled))[:, 1]

            auc_mlp = roc_auc_score(y_test, preds_mlp) if len(set(y_test)) > 1 else 0.5
            results_mlp.append({'Year': year, 'Month': month, 'AUC': auc_mlp})

            logger.info(
                f"  [RESULT {year}-{month:02}]: XGB_AUC={auc_xgb:.4f} | "
                f"LR_AUC={auc_lr:.4f} | MLP_AUC={auc_mlp:.4f}"
            )
            
            # --- DETAILED LOGGING ---
            # Re-fetch Transaction IDs for the test set (masked)
            test_ids = test_df.iloc[test_mask_arr]['transaction_id'].values
            test_dates = test_df.iloc[test_mask_arr]['Filed'].values
            
            for i in range(len(test_ids)):
                base_record = {
                    'TransactionID': test_ids[i],
                    'Date': test_dates[i],
                    'True_Up': y_test[i]
                }
                
                # XGB
                rec_xgb = base_record.copy()
                rec_xgb['Model'] = 'XGB'
                rec_xgb['Prob_Up'] = preds_xgb[i]
                rec_xgb['Pred_Up'] = 1.0 if preds_xgb[i] > 0.5 else 0.0
                predictions_log.append(rec_xgb)
                
                # LR
                rec_lr = base_record.copy()
                rec_lr['Model'] = 'LR'
                rec_lr['Prob_Up'] = preds_lr[i]
                rec_lr['Pred_Up'] = 1.0 if preds_lr[i] > 0.5 else 0.0
                predictions_log.append(rec_lr)
                
                # MLP
                rec_mlp = base_record.copy()
                rec_mlp['Model'] = 'MLP'
                rec_mlp['Prob_Up'] = preds_mlp[i]
                rec_mlp['Pred_Up'] = 1.0 if preds_mlp[i] > 0.5 else 0.0
                predictions_log.append(rec_mlp)

    # Summary
    res_xgb = pd.DataFrame(results_xgb)
    res_lr = pd.DataFrame(results_lr)
    res_mlp = pd.DataFrame(results_mlp)
    
    print("\n" + "="*40)
    print(f"FAIR BASELINES SUMMARY: Horizon={horizon}")
    print("="*40)
    print(f"XGB MEAN AUC: {res_xgb.AUC.mean():.4f}")
    print(f"LR  MEAN AUC: {res_lr.AUC.mean():.4f}")
    print(f"MLP MEAN AUC: {res_mlp.AUC.mean():.4f}")
    
    out_path = Path("dev/results")
    out_path.mkdir(exist_ok=True)
    res_xgb.to_csv(out_path / f"baseline_xgb_{horizon}.csv", index=False)
    res_lr.to_csv(out_path / f"baseline_lr_{horizon}.csv", index=False)
    res_mlp.to_csv(out_path / f"baseline_mlp_{horizon}.csv", index=False)
    
    # Save Detailed Logs
    pd.DataFrame(predictions_log).to_csv(out_path / f"predictions_baseline_{horizon}.csv", index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='6M')
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--start-year', type=int, default=2023)
    parser.add_argument('--end-year', type=int, default=2023)
    args = parser.parse_args()
    run_fair_baselines(
        horizon=args.horizon,
        alpha=args.alpha,
        start_year=args.start_year,
        end_year=args.end_year,
    )
