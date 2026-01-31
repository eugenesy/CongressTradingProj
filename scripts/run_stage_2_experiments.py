import logging
import os
import sys
import argparse
from pathlib import Path
from typing import Tuple
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    log_loss
)

sys.path.append(os.getcwd())
try:
    from src.gap_tgn import GAPTGN
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.gap_tgn import GAPTGN

from torch_geometric.loader import TemporalDataLoader
from torch_geometric.data import TemporalData
from torch_geometric.nn.models.tgn import LastNeighborLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("stage2_experiments")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def direction_targets(raw_return, alpha):
    has_label = ~torch.isnan(raw_return)
    up = raw_return > alpha
    down = raw_return < alpha
    mask = has_label & (up | down)
    return up.float(), mask

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_single_experiment(horizon, data, df, args):
    logger.info(f"STARTING EXPERIMENT: Horizon={horizon}")
    
    # --- SETUP OUTPUT DIRECTORY ---
    exp_dir = Path(args.out_dir) / f"horizon_{horizon}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    horizon_map = {'3M': 2, '6M': 3, '8M': 4, '12M': 5, '18M': 6, '24M': 7}
    h_idx = horizon_map[horizon]
    h_seconds = {'3M': 90, '6M': 180, '8M': 240, '12M': 365, '18M': 545, '24M': 730}[horizon] * 86400
    device = get_device()
    
    num_pols = len(df['BioGuideID'].unique()) 

    # --- STORAGE FOR RESULTS ---
    all_monthly_metrics = []
    all_detailed_preds = []

    def slice_data(indices):
        idx = torch.as_tensor(indices, dtype=torch.long, device=device)
        return TemporalData(
            src=data.src[idx], dst=data.dst[idx], t=data.t[idx], msg=data.msg[idx],
            y=data.y[idx], trade_t=data.trade_t[idx], price_seq=data.price_seq[idx]
        )

    for year in range(args.start_year, args.end_year + 1):
        for month in range(1, 13):
            test_start = pd.Timestamp(year, month, 1)
            if test_start < pd.Timestamp(2016, 1, 1): continue
            if test_start > pd.Timestamp(args.end_year, 12, 31): break
            
            next_month = test_start + pd.DateOffset(months=1)
            gap_start = test_start - pd.DateOffset(months=1)
            
            train_mask = df['Filed'] < gap_start
            gap_mask = (df['Filed'] >= gap_start) & (df['Filed'] < test_start)
            test_mask = (df['Filed'] >= test_start) & (df['Filed'] < next_month)
            
            # --- CRITICAL: Get slice of DF for metadata extraction ---
            # We reset index to match the 0-based batch iteration in test_loader
            test_df_slice = df[test_mask].reset_index(drop=True)
            
            train_data = slice_data(df[train_mask].index)
            gap_data = slice_data(df[gap_mask].index)
            test_data = slice_data(df[test_mask].index)
            
            if len(test_data.src) == 0: continue
            
            # Check labels
            raw_tr = train_data.y[:, h_idx]
            targets, lbl_mask = direction_targets(raw_tr, args.alpha)
            resolved = (train_data.trade_t + h_seconds) < train_data.t.max()
            valid_train = lbl_mask & resolved
            
            if valid_train.sum() == 0:
                logger.warning(f"  {year}-{month:02d}: SKIPPING (0 resolved labels)")
                continue
                
            pos = targets[valid_train].sum().item()
            neg = valid_train.sum().item() - pos
            pos_weight = neg / max(1, pos)
            
            logger.info(f"  {year}-{month:02d}: Train={len(train_data.src)}, Test={len(test_data.src)}")
            
            # --- MODEL SETUP ---
            model = GAPTGN(
                num_nodes=data.num_nodes,
                edge_feat_dim=data.msg.shape[1],
                pol_feat_dim=data.x_pol.shape[1],
                comp_feat_dim=data.x_comp.shape[1],
                device=device
            ).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
            
            # --- NEIGHBOR LOADER ---
            neighbor_loader = LastNeighborLoader(data.num_nodes, size=20, device=device)
            
            train_loader = TemporalDataLoader(train_data, batch_size=200, drop_last=True)
            gap_loader = TemporalDataLoader(gap_data, batch_size=200)
            test_loader = TemporalDataLoader(test_data, batch_size=200)
            
            # --- TRAIN ---
            model.train()
            for epoch in range(1, args.epochs + 1):
                model.memory.reset_state()
                neighbor_loader.reset_state()
                
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    
                    n_id = torch.cat([batch.src, batch.dst]).unique()
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                    assoc = {node.item(): i for i, node in enumerate(n_id)}
                    
                    hist_msg = data.msg[e_id]
                    rel_t = model.memory.last_update[n_id[edge_index[1]]] - data.t[e_id]
                    rel_t_enc = model.memory.time_enc(rel_t.float())
                    hist_price = model.get_price_embedding(data.price_seq[e_id])
                    edge_attr = torch.cat([rel_t_enc, hist_msg, hist_price], dim=-1)
                    
                    z = model(n_id, edge_index, edge_attr)
                    z_src = z[[assoc[i.item()] for i in batch.src]]
                    z_dst = z[[assoc[i.item()] for i in batch.dst]]
                    
                    d_src, d_dst = model.encode_dynamic(data.x_pol, data.x_comp, batch.src, batch.dst, num_pols)
                    p_context = model.get_price_embedding(batch.price_seq)
                    preds = model.predictor(z_src, z_dst, d_src, d_dst, p_context)
                    
                    raw_y = batch.y[:, h_idx]
                    targs, l_mask = direction_targets(raw_y, args.alpha)
                    res_mask = (batch.trade_t + h_seconds) < batch.t.max()
                    mask = l_mask & res_mask
                    
                    if mask.sum() > 0:
                        loss = criterion(preds[mask].squeeze(-1), targs[mask])
                        loss.backward()
                        optimizer.step()
                        
                    aug_msg = torch.cat([batch.msg, p_context], dim=1)
                    model.memory.update_state(batch.src, batch.dst, batch.t, aug_msg)
                    neighbor_loader.insert(batch.src, batch.dst)
                    model.memory.detach()
                    
            # --- GAP ---
            model.eval()
            if len(gap_data.src) > 0:
                with torch.no_grad():
                    for batch in gap_loader:
                        batch = batch.to(device)
                        p_context = model.get_price_embedding(batch.price_seq)
                        aug_msg = torch.cat([batch.msg, p_context], dim=1)
                        model.memory.update_state(batch.src, batch.dst, batch.t, aug_msg)
                        neighbor_loader.insert(batch.src, batch.dst)

            # --- TEST ---
            preds_probs = []
            preds_labels = []
            targets_true = []
            
            # Temporary storage for metadata to ensure alignment
            batch_metadata_list = []
            
            current_batch_idx = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    # 1. Grab metadata for this specific batch before processing
                    batch_size = batch.src.size(0)
                    batch_meta = test_df_slice.iloc[current_batch_idx : current_batch_idx + batch_size].copy()
                    current_batch_idx += batch_size
                    
                    batch = batch.to(device)
                    n_id = torch.cat([batch.src, batch.dst]).unique()
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                    assoc = {node.item(): i for i, node in enumerate(n_id)}
                    
                    hist_msg = data.msg[e_id]
                    rel_t = model.memory.last_update[n_id[edge_index[1]]] - data.t[e_id]
                    rel_t_enc = model.memory.time_enc(rel_t.float())
                    hist_price = model.get_price_embedding(data.price_seq[e_id])
                    edge_attr = torch.cat([rel_t_enc, hist_msg, hist_price], dim=-1)
                    
                    z = model(n_id, edge_index, edge_attr)
                    z_src = z[[assoc[i.item()] for i in batch.src]]
                    z_dst = z[[assoc[i.item()] for i in batch.dst]]
                    
                    d_src, d_dst = model.encode_dynamic(data.x_pol, data.x_comp, batch.src, batch.dst, num_pols)
                    p_context = model.get_price_embedding(batch.price_seq)
                    
                    # Probabilities
                    probs = model.predictor(z_src, z_dst, d_src, d_dst, p_context).sigmoid().cpu().squeeze(-1).numpy()
                    
                    # Targets
                    raw_y = batch.y[:, h_idx]
                    targs, l_mask = direction_targets(raw_y, args.alpha)
                    
                    # Move to CPU for list storage
                    batch_targs = targs.cpu().numpy()
                    batch_mask = l_mask.cpu().numpy()
                    
                    # 2. Attach predictions to metadata immediately
                    batch_meta['Prob_Pos'] = probs
                    batch_meta['Predicted_Label'] = (probs > 0.5).astype(int)
                    # We might have NaNs in targets (unresolved trades), store what we have
                    # But for metric calculation, we only use valid targets
                    batch_meta['True_Label_Raw'] = raw_y.cpu().numpy()
                    batch_meta['Is_Valid_Target'] = batch_mask
                    batch_meta['True_Label_Binary'] = batch_targs
                    
                    batch_metadata_list.append(batch_meta)
                    
                    # Collect valid data for Monthly Metrics
                    if np.any(batch_mask):
                        preds_probs.extend(probs[batch_mask])
                        preds_labels.extend((probs[batch_mask] > 0.5).astype(int))
                        targets_true.extend(batch_targs[batch_mask])
                    
                    # Update State
                    aug_msg = torch.cat([batch.msg, p_context], dim=1)
                    model.memory.update_state(batch.src, batch.dst, batch.t, aug_msg)
                    neighbor_loader.insert(batch.src, batch.dst)

            # --- CALCULATE MONTHLY STATISTICS ---
            if targets_true:
                y_true = np.array(targets_true)
                y_prob = np.array(preds_probs)
                y_pred = np.array(preds_labels)
                
                # Requested Metrics
                loss_val = log_loss(y_true, y_prob)
                acc = accuracy_score(y_true, y_pred)
                auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
                
                # Positive Class Metrics (Fraud/Up)
                prec_pos = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
                rec_pos = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
                f1_pos = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
                
                # Macro Metrics
                f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
                
                logger.info(f"    -> Loss: {loss_val:.3f} | F1(+): {f1_pos:.3f} | Acc: {acc:.3f}")
                
                all_monthly_metrics.append({
                    'Year': year, 
                    'Month': month, 
                    'Horizon': horizon,
                    'Loss': loss_val,
                    'Accuracy': acc,
                    'AUC': auc,
                    'Precision_Pos': prec_pos,
                    'Recall_Pos': rec_pos,
                    'F1_Pos': f1_pos,
                    'F1_Macro': f1_macro
                })
            
            # --- COLLECT DETAILED RESULTS ---
            if batch_metadata_list:
                month_detailed_df = pd.concat(batch_metadata_list)
                month_detailed_df['Test_Year'] = year
                month_detailed_df['Test_Month'] = month
                all_detailed_preds.append(month_detailed_df)

    # --- SAVE FINAL CSVs ---
    
    # 1. Monthly Performance Metrics
    if all_monthly_metrics:
        metrics_df = pd.DataFrame(all_monthly_metrics)
        metrics_path = exp_dir / f"monthly_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved monthly metrics to {metrics_path}")

    # 2. Detailed Predictions (with BioGuideID & Ticker)
    if all_detailed_preds:
        detailed_df = pd.concat(all_detailed_preds, ignore_index=True)
        
        # Reorder columns for readability if keys exist
        cols = ['Test_Year', 'Test_Month', 'BioGuideID', 'Ticker', 'Filed', 
                'True_Label_Binary', 'Predicted_Label', 'Prob_Pos']
        existing_cols = [c for c in cols if c in detailed_df.columns]
        remaining = [c for c in detailed_df.columns if c not in existing_cols]
        detailed_df = detailed_df[existing_cols + remaining]
        
        detailed_path = exp_dir / f"detailed_predictions.csv"
        detailed_df.to_csv(detailed_path, index=False)
        logger.info(f"Saved detailed predictions to {detailed_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--start-year', type=int, default=2019)
    parser.add_argument('--end-year', type=int, default=2024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', default='experiment_outputs') # Renamed for clarity
    parser.add_argument('--specific-horizon', type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    
    # Load Data
    try:
        data = torch.load("data/temporal_data.pt", map_location=device, weights_only=False)
    except:
        data = torch.load("data/temporal_data.pt", map_location=device)
        
    if not hasattr(data, 'x_pol'):
        logger.error("Data missing x_pol! Please run src/temporal_data.py")
        return

    df = pd.read_csv("data/processed/ml_dataset_clean.csv")
    df['Filed'] = pd.to_datetime(df['Filed'])
    df = df.sort_values('Filed').reset_index(drop=True)
    df = df.iloc[:len(data.src)]

    horizons = [args.specific_horizon] if args.specific_horizon else ['3M', '6M', '8M', '12M', '18M', '24M']
    
    for h in horizons:
        try:
            run_single_experiment(h, data, df, args)
        except Exception as e:
            logger.error(f"Experiment {h} failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()