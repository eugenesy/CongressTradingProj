import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import os
import sys

# Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, '..'))
from src.config import TX_PATH

def run_xgb():
    print("="*50)
    print("STARTING XGBOOST BASELINE (2023)")
    print("="*50)

    # 1. Load Data
    print("Loading Data...")
    df = pd.read_csv(TX_PATH)
    
    # 2. Filter Time Windows
    df['Filed'] = pd.to_datetime(df['Filed'])
    
    # Train: Pre-2023
    train_mask = df['Filed'] < '2023-01-01'
    # Test: 2023
    test_mask = (df['Filed'] >= '2023-01-01') & (df['Filed'] <= '2023-12-31')
    
    # We will split AFTER feature engineering using these masks
    
    print(f"Total Data: {len(df)}")
    if len(df[train_mask]) == 0 or len(df[test_mask]) == 0: return

    # 3. Feature Engineering (Tabular Equivalent of TGN)
    # TGN used: BioGuideID (Emb), Ticker (Emb), Party, State, Trade_Size, Amount(Log)
    
    # Encode Categoricals
    le_pol = LabelEncoder()
    df['Pol_ID'] = le_pol.fit_transform(df['BioGuideID'])
    
    le_ticker = LabelEncoder()
    df['Ticker_ID'] = le_ticker.fit_transform(df['Ticker'])
    
    le_party = LabelEncoder()
    df['Party_ID'] = le_party.fit_transform(df['Party'].fillna('Unknown'))
    
    le_state = LabelEncoder()
    df['State_ID'] = le_state.fit_transform(df['State'].fillna('Unknown'))
    
    # Parse Amount
    def parse_amt(x):
        try: return float(str(x).replace('$','').replace(',',''))
        except: return 0.0
    df['Amount'] = df['Trade_Size_USD'].apply(parse_amt) # Restored
    df['Log_Amount'] = np.log1p(df['Amount'])
    
    # Transaction Type
    df['Is_Buy'] = df['Transaction'].apply(lambda x: 1 if 'Purchase' in str(x) else 0)
    
    # Features & Target
    features = ['Pol_ID', 'Ticker_ID', 'Party_ID', 'State_ID', 'Log_Amount', 'Is_Buy']
    target_col = 'Excess_Return_1M'
    threshold = 0.01

    # 4. Split
    # Now we apply the masks defined earlier
    X_train = df[train_mask]
    X_test = df[test_mask]

    X_train_feat = X_train[features]
    y_train = (X_train[target_col] > threshold).astype(int)
    
    X_test_feat = X_test[features]
    y_test = (X_test[target_col] > threshold).astype(int)
    
    print(f"Train (Pre-2023): {len(X_train)} | Test (2023): {len(X_test)}")
    print(f"Train Pos Ratio: {y_train.mean():.4f}")
    
    # 5. Train XGBoost
    # Using roughly same params: scale_pos_weight to match TGN pos_weight
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Imbalance Ratio: {ratio:.2f}")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=ratio, 
        random_state=42,
        eval_metric='auc'
    )
    
    model.fit(X_train_feat, y_train)
    
    # 6. Evaluate
    preds_proba = model.predict_proba(X_test_feat)[:, 1]
    preds_bin = model.predict(X_test_feat)
    
    auc = roc_auc_score(y_test, preds_proba)
    f1 = f1_score(y_test, preds_bin)
    
    print("-" * 30)
    print(f"XGBoost Result (2023):")
    print(f"AUC: {auc:.4f}")
    print(f"F1:  {f1:.4f}")
    print("-" * 30)
    print(classification_report(y_test, preds_bin))
    
    # Feature Importance
    print("\nFeature Importance:")
    fi = pd.DataFrame({'Feat': features, 'Imp': model.feature_importances_}).sort_values('Imp', ascending=False)
    print(fi)

if __name__ == "__main__":
    run_xgb()
