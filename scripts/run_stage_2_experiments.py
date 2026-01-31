import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from dateutil.relativedelta import relativedelta
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gap_tgn import GAPTGN
from src.temporal_data import TemporalGraphBuilder
from src.config import TX_PATH, RESULTS_DIR

CONFIG = {
    'epochs': 20,         # Reduced slightly for walk-forward speed
    'lr': 0.001,
    'seed': 42,
}

HORIZONS_LIST = ['3M', '6M', '8M', '12M', '18M', '24M']
TEST_START_DATE = pd.Timestamp('2019-01-01')
TEST_END_DATE = pd.Timestamp('2025-04-01')

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout), 
        logging.FileHandler("experiment_stage2_full.log")
    ]
)
logger = logging.getLogger(__name__)

def get_horizon_months(h_str):
    return int(h_str.replace('M', ''))

def align_data_with_graph(builder, graph_data):
    """Aligns the raw dataframe indices with the processed graph nodes."""
    df = builder.transactions.copy()
    valid_tickers = set(builder.company_id_map.keys())
    df = df[df['Ticker'].isin(valid_tickers)]
    valid_pols = set(builder.pol_id_map.keys())
    df = df[df['BioGuideID'].isin(valid_pols)]
    df = df[df['Filed_DT'].notna()]
    return df.reset_index(drop=True)

def run_experiment():
    logger.info("Initializing Data Pipeline...")
    if not os.path.exists(TX_PATH):
        logger.error(f"Data not found at {TX_PATH}")
        return

    df_raw = pd.read_csv(TX_PATH)
    # Build Graph Data
    builder = TemporalGraphBuilder(df_raw, min_freq=2)
    data = builder.process()
    
    meta_df = align_data_with_graph(builder, data)
    
    # Time setup for Walk-Forward
    base_time = pd.to_datetime(df_raw['Filed'].min()).timestamp()
    t_numpy = data.t.cpu().numpy()
    absolute_ts = t_numpy + base_time
    event_dates = pd.to_datetime(absolute_ts, unit='s')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running on device: {device}")
    
    # Move All Data to Device
    data.src = data.src.to(device)
    data.dst = data.dst.to(device)
    data.t = data.t.to(device)
    data.msg = data.msg.to(device)
    data.y = data.y.to(device)
    
    # --- CRITICAL: Dynamic Features to Device ---
    data.x_pol = data.x_pol.to(device)
    data.x_comp = data.x_comp.to(device)
    # ---------------------------------------------

    horizon_map = {'1M': 0, '2M': 1, '3M': 2, '6M': 3, '8M': 4, '12M': 5, '18M': 6, '24M': 7}

    for h_str in HORIZONS_LIST:
        logger.info(f"\n{'='*40}\nStarting Experiment: Horizon {h_str}\n{'='*40}")
        
        h_months = get_horizon_months(h_str)
        h_idx = horizon_map.get(h_str)
        
        # Prepare targets
        raw_targets = data.y[:, h_idx]
        valid_target_mask = ~torch.isnan(raw_targets)
        labels_binary = (raw_targets > 0).float().to(device)
        valid_target_mask = valid_target_mask.to(device)
        
        horizon_preds = []
        horizon_metrics = []
        
        # Walk-Forward Validation Loop
        test_dates = pd.date_range(start=TEST_START_DATE, end=TEST_END_DATE, freq='MS')
        
        for test_date in test_dates:
            test_month_end = test_date + relativedelta(months=1)
            # Training window: rolling or expanding? 
            # Using expanding window up to horizon gap
            train_cutoff = test_date - relativedelta(months=h_months)
            
            # Indices
            train_bool = (event_dates < train_cutoff)
            test_bool = (event_dates >= test_date) & (event_dates < test_month_end)
            
            train_idx = torch.where(torch.tensor(train_bool, device=device) & valid_target_mask)[0]
            test_idx = torch.where(torch.tensor(test_bool, device=device) & valid_target_mask)[0]
            
            if len(test_idx) == 0:
                continue
            
            # --- Instantiate Model (Re-init for every fold) ---
            torch.manual_seed(CONFIG['seed'])
            
            # Check dimensions from data
            pol_dim = data.x_pol.shape[1]
            comp_dim = data.x_comp.shape[1]
            
            model = GAPTGN(
                num_nodes=data.num_nodes,
                edge_feat_dim=data.msg.shape[1],
                pol_feat_dim=pol_dim,   # Dynamic Dimension
                comp_feat_dim=comp_dim, # Dynamic Dimension
                time_dim=100,
                memory_dim=100,
                embedding_dim=100,
                device=device
            ).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
            # Use BCELoss because model returns Sigmoid(logits)
            criterion = nn.BCELoss() 
            
            # --- Training Loop ---
            model.train()
            for epoch in range(CONFIG['epochs']):
                optimizer.zero_grad()
                model.reset_memory() # Reset memory state for new epoch
                
                # Forward Pass with Dynamic Features
                pred_prob, _ = model(
                    data.src, data.dst, data.t, data.msg,
                    data.x_pol, 
                    data.x_comp
                )
                
                # Compute Loss only on training indices
                loss = criterion(pred_prob[train_idx].squeeze(), labels_binary[train_idx])
                loss.backward()
                optimizer.step()
                
                # Detach memory to prevent backprop through epochs
                model.detach_memory()
            
            # --- Inference ---
            model.eval()
            with torch.no_grad():
                model.reset_memory()
                # Run full sequence to update memory states up to test time
                pred_prob, _ = model(
                    data.src, data.dst, data.t, data.msg,
                    data.x_pol, 
                    data.x_comp
                )
                
                # Extract Test Predictions
                probs = pred_prob[test_idx].squeeze().cpu().numpy()
                preds = (probs > 0.5).astype(int)
                actuals = labels_binary[test_idx].cpu().numpy()
                
                # Store Metadata
                batch_meta = meta_df.iloc[test_idx.cpu().numpy()].copy()
                batch_meta['horizon'] = h_str
                batch_meta['test_month'] = test_date
                batch_meta['prediction_prob'] = probs
                batch_meta['prediction_label'] = preds
                batch_meta['actual_label'] = actuals
                batch_meta['target_return'] = raw_targets[test_idx.cpu()].numpy()
                
                horizon_preds.append(batch_meta)
                
                # Metrics
                acc = accuracy_score(actuals, preds)
                f1 = f1_score(actuals, preds, zero_division=0)
                try: 
                    auc = roc_auc_score(actuals, probs)
                except: 
                    auc = 0.5
                    
                metrics_row = {
                    'month': test_date,
                    'horizon': h_str,
                    'n_train': len(train_idx),
                    'n_test': len(test_idx),
                    'accuracy': acc, 'auc': auc, 'f1': f1,
                    'pos_ratio': actuals.mean()
                }
                horizon_metrics.append(metrics_row)
                logger.info(f"{h_str} | {test_date.strftime('%Y-%m')} | Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

            # Clean up
            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save Results per Horizon
        save_dir = Path(RESULTS_DIR) / "experiments_detailed"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if horizon_preds:
            final_df = pd.concat(horizon_preds, axis=0)
            
            # Ensure nice column ordering
            cols = ['transaction_id', 'BioGuideID', 'Ticker', 'Filed', 'horizon', 'test_month', 
                    'prediction_prob', 'prediction_label', 'actual_label', 'target_return']
            cols = [c for c in cols if c in final_df.columns] + [c for c in final_df.columns if c not in cols]
            final_df = final_df[cols]
            
            csv_path = save_dir / f"results_detailed_{h_str}.csv"
            final_df.to_csv(csv_path, index=False)
            
            metrics_df = pd.DataFrame(horizon_metrics)
            metrics_df.to_csv(save_dir / f"metrics_log_{h_str}.csv", index=False)
            logger.info(f"Saved results for {h_str}")
        else:
            logger.warning(f"No results for {h_str}")

if __name__ == "__main__":
    run_experiment()