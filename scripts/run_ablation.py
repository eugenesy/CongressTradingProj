
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
import datetime
import sys
import os
import logging
import json

sys.path.append(os.getcwd())

# Import from local directory if possible, or standard path
# Import from local directory if possible, or standard path
from src.models_tgn import TGN

from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

# --- Logging Setup ---
logger = logging.getLogger("ablation")
logger.setLevel(logging.INFO)
# Prevent duplicate handlers
if not logger.handlers:
    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler
    os.makedirs("ablation_study/logs", exist_ok=True)
    fh = logging.FileHandler("ablation_study/logs/ablation_log.txt")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_evaluate(data, df_filtered, args=None, target_years=[2023], num_nodes=None, num_parties=5, num_states=50,
                       max_epochs=20, patience=5, ablation_mode='full'):
    """
    ablation_mode: 'full', 'pol_only', 'mkt_only'
    """
    device = get_device()
    data = data.to(device)
    
    results = []
    
    logger.info(f"Starting Ablation Run: Mode={ablation_mode}, Years={target_years}")
    
    # We iterate chronologically through all target months
    for year in target_years:
        for month in range(1, 13):
            # Define Time Boundaries
            current_period_start = pd.Timestamp(year=year, month=month, day=1)
            
            # Next Month Start (for convenient filtering)
            if month == 12:
                next_period_start = pd.Timestamp(year=year+1, month=1, day=1)
            else:
                next_period_start = pd.Timestamp(year=year, month=month+1, day=1)
            
            train_end = current_period_start - pd.DateOffset(months=1)
            gap_start = train_end
            gap_end = current_period_start
            
            logger.info(f"\n=== RETRAINING For Window: {year}-{month:02d} | Mode: {ablation_mode} ===")
            
            # 1. Split Indices
            train_mask = df_filtered['Filed'] < gap_start
            gap_mask = (df_filtered['Filed'] >= gap_start) & (df_filtered['Filed'] < gap_end)
            test_mask = (df_filtered['Filed'] >= current_period_start) & (df_filtered['Filed'] < next_period_start)
            
            train_idx = np.where(train_mask)[0]
            gap_idx = np.where(gap_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            # Slice Bounds
            max_id = len(data.src)
            train_idx = [i for i in train_idx if i < max_id]
            gap_idx = [i for i in gap_idx if i < max_id]
            test_idx = [i for i in test_idx if i < max_id]
            
            if len(test_idx) == 0:
                logger.warning(f"No trades in {year}-{month:02d}, skipping.")
                continue
                
            train_data = data[train_idx]
            gap_data = data[gap_idx]
            test_data = data[test_idx]
            
            logger.info(f"Train: {len(train_data.src)} | Gap: {len(gap_data.src)} | Test: {len(test_data.src)}")
            
            # 2. INIT MODEL
            model = TGN(
                num_nodes=num_nodes, raw_msg_dim=3, memory_dim=100, time_dim=100, embedding_dim=100,
                num_parties=num_parties, num_states=num_states
            ).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            criterion = torch.nn.BCEWithLogitsLoss()
            neighbor_loader = LastNeighborLoader(num_nodes, size=10, device=device)
            
            # 3. TRAIN PHASE (With Backprop and Validation)
            train_size = int(len(train_data.src) * 0.9)
            val_split_idx = torch.arange(train_size, len(train_data.src))
            train_split_idx = torch.arange(0, train_size)
            
            actual_train_data = train_data[train_split_idx]
            val_data = train_data[val_split_idx]
            
            train_loader = TemporalDataLoader(actual_train_data, batch_size=200, drop_last=True)
            val_loader = TemporalDataLoader(val_data, batch_size=200, drop_last=True)
            
            model.train()
            
            min_val_loss = float('inf')
            epochs_without_improvement = 0
            best_epoch = None
            
            for epoch in range(1, max_epochs + 1):
                model.memory.reset_state()
                neighbor_loader.reset_state()
                model.memory.detach()
                
                epoch_losses = []
                epoch_preds_train = []
                epoch_targets_train = []
                
                # A. Train Loop
                for batch in tqdm(train_loader, desc=f"Train Ep {epoch}", leave=False):
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    
                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    
                    # --- DYNAMIC LABEL ---
                    horizon_map = {'1M':0, '2M':1, '3M':2, '6M':3, '8M':4, '12M':5, '18M':6, '24M':7}
                    h_idx = horizon_map.get(args.horizon, 0)
                    h_days = {'1M':30, '2M':60, '3M':90, '6M':180, '8M':240, '12M':365, '18M':545, '24M':730}.get(args.horizon, 30)
                    h_seconds = h_days * 86400
                    
                    # Ensure batch has valid attributes if slicing
                    # PyG might not slice stored multi-dim tensor attributes by default unless registered correctly or in data.
                    # But it worked in train_rolling.py.
                    
                    raw_return = batch.y[:, h_idx]
                    trade_t = batch.trade_t
                    resolution_t = trade_t + h_seconds
                    
                    batch_max_t = t.max()
                    
                    is_known = (resolution_t < batch_max_t)
                    has_label = ~torch.isnan(raw_return)
                    train_mask = is_known & has_label
                    
                    if train_mask.sum() == 0:
                         pass
                    
                    # Targets
                    alpha = args.alpha
                    is_buy = msg[:, 1]
                    batch_targets = torch.zeros_like(raw_return)
                    buy_mask = (is_buy == 1.0)
                    batch_targets[buy_mask & (raw_return > alpha)] = 1.0
                    sell_mask = (is_buy == -1.0)
                    batch_targets[sell_mask & (raw_return < alpha)] = 1.0

                    # --- ABLATION LOGIC: Price Seq ---
                    if hasattr(batch, 'price_seq'):
                         price_seq = batch.price_seq
                    else:
                         price_seq = torch.zeros((len(src), 14), device=device)
                    
                    if ablation_mode == 'pol_only':
                        price_seq = torch.zeros_like(price_seq)

                    price_emb = model.get_price_embedding(price_seq)
                    augmented_msg = torch.cat([msg, price_emb], dim=1)
                    
                    n_id = torch.cat([src, dst]).unique()
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                    assoc = dict(zip(n_id.tolist(), range(n_id.size(0))))
                    
                    hist_t = data.t[e_id]
                    hist_msg = data.msg[e_id]
                    
                    # --- ABLATION LOGIC: Hist Price Seq ---
                    hist_price_seq = data.price_seq[e_id]
                    if ablation_mode == 'pol_only':
                        hist_price_seq = torch.zeros_like(hist_price_seq)

                    hist_price_emb = model.get_price_embedding(hist_price_seq)
                    
                    # Dynamic Features (History Masking)
                    hist_raw_returns = data.y[e_id, h_idx]
                    hist_trade_t = data.trade_t[e_id]
                    hist_resolution = hist_trade_t + h_seconds
                    
                    hist_is_resolved = (hist_resolution < batch_max_t).float().unsqueeze(-1)
                    
                    hist_is_buy = hist_msg[:, 1]
                    hist_targets = torch.zeros_like(hist_raw_returns)
                    h_buy_mask = (hist_is_buy == 1.0)
                    hist_targets[h_buy_mask & (hist_raw_returns > alpha)] = 1.0
                    h_sell_mask = (hist_is_buy == -1.0)
                    hist_targets[h_sell_mask & (hist_raw_returns < alpha)] = 1.0
                    
                    isnan_mask = torch.isnan(hist_raw_returns)
                    hist_is_resolved[isnan_mask] = 0.0
                    hist_targets[isnan_mask] = 0.0
                    hist_targets = hist_targets.unsqueeze(-1)
                    
                    masked_label = hist_targets * hist_is_resolved + 0.5 * (1 - hist_is_resolved)
                    
                    age_seconds = (batch_max_t - hist_t).float()
                    age_days = age_seconds / 86400.0
                    age_feat = torch.log1p(age_days).unsqueeze(-1)
                    
                    target_nodes = n_id[edge_index[1]]
                    last_update = model.memory.last_update[target_nodes]
                    rel_t = last_update - hist_t
                    rel_t_enc = model.memory.time_enc(rel_t.to(torch.float))
                    
                    edge_attr = torch.cat([rel_t_enc, hist_msg, hist_price_emb, masked_label, age_feat], dim=-1)
                    
                    z = model(n_id, edge_index, edge_attr)
                    z_src = z[[assoc[i] for i in src.tolist()]]
                    z_dst = z[[assoc[i] for i in dst.tolist()]]
                    
                    s_src = model.encode_static(data.x_static[src])
                    s_dst = model.encode_static(data.x_static[dst])
                    
                    # --- ABLATION LOGIC: Static Embeddings ---
                    if ablation_mode == 'mkt_only':
                        s_src = torch.zeros_like(s_src)
                        s_dst = torch.zeros_like(s_dst)
                    
                    pred_pos = model.predictor(z_src, z_dst, s_src, s_dst, price_emb, price_emb)
                    
                    if train_mask.sum() > 0:
                        loss = criterion(pred_pos[train_mask].view(-1), batch_targets[train_mask].view(-1))
                        loss.backward()
                        optimizer.step()
                        epoch_losses.append(loss.item())
                        
                        with torch.no_grad():
                            epoch_preds_train.extend(pred_pos[train_mask].sigmoid().cpu().numpy().flatten())
                            epoch_targets_train.extend(batch_targets[train_mask].cpu().numpy())
                    else:
                        pass
                    
                    model.memory.update_state(src, dst, t, augmented_msg)
                    neighbor_loader.insert(src, dst)
                    model.memory.detach()
                    
                # B. Gap Phase
                gap_loader = TemporalDataLoader(gap_data, batch_size=200)
                model.eval()
                for batch in gap_loader:
                    batch = batch.to(device)
                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    
                    if hasattr(batch, 'price_seq'):
                         price_seq = batch.price_seq
                    else:
                         price_seq = torch.zeros((len(src), 14), device=device)
                         
                    if ablation_mode == 'pol_only':
                        price_seq = torch.zeros_like(price_seq)

                    price_emb = model.get_price_embedding(price_seq)
                    augmented_msg = torch.cat([msg, price_emb], dim=1)
                    
                    model.memory.update_state(src, dst, t, augmented_msg)
                    neighbor_loader.insert(src, dst)
                
                # C. Validation
                val_preds = []
                val_targets = []
                val_losses = []
                for batch in val_loader:
                    batch = batch.to(device)
                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    
                    # --- DYNAMIC LABEL ---
                    # Same logic
                    horizon_map = {'1M':0, '2M':1, '3M':2, '6M':3, '8M':4, '12M':5, '18M':6, '24M':7}
                    h_idx = horizon_map.get(args.horizon, 0)
                    h_days = {'1M':30, '2M':60, '3M':90, '6M':180, '8M':240, '12M':365, '18M':545, '24M':730}.get(args.horizon, 30)
                    h_seconds = h_days * 86400
                    
                    raw_return = batch.y[:, h_idx]
                    trade_t = batch.trade_t
                    resolution_t = trade_t + h_seconds
                    
                    batch_max_t = t.max()
                    
                    is_known = (resolution_t < batch_max_t)
                    has_label = ~torch.isnan(raw_return)
                    val_mask = is_known & has_label
                    
                    # Targets
                    alpha = args.alpha
                    is_buy = msg[:, 1]
                    batch_targets = torch.zeros_like(raw_return)
                    buy_mask = (is_buy == 1.0)
                    batch_targets[buy_mask & (raw_return > alpha)] = 1.0
                    sell_mask = (is_buy == -1.0)
                    batch_targets[sell_mask & (raw_return < alpha)] = 1.0
                    
                    if hasattr(batch, 'price_seq'):
                         price_seq = batch.price_seq
                    else:
                         price_seq = torch.zeros((len(src), 14), device=device)
                    if ablation_mode == 'pol_only':
                        price_seq = torch.zeros_like(price_seq)
                    
                    price_emb = model.get_price_embedding(price_seq)
                    augmented_msg = torch.cat([msg, price_emb], dim=1)
                    
                    n_id = torch.cat([src, dst]).unique()
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                    
                    if len(e_id) > 0:
                        hist_t = data.t[e_id]
                        hist_msg = data.msg[e_id]
                        
                        hist_price_seq = data.price_seq[e_id]
                        if ablation_mode == 'pol_only':
                             hist_price_seq = torch.zeros_like(hist_price_seq)
                        hist_price_emb = model.get_price_embedding(hist_price_seq)
                        
                        # Dynamic Masking
                        hist_raw_returns = data.y[e_id, h_idx]
                        hist_trade_t = data.trade_t[e_id]
                        hist_resolution = hist_trade_t + h_seconds
                        
                        hist_is_resolved = (hist_resolution < batch_max_t).float().unsqueeze(-1)
                        
                        hist_is_buy = hist_msg[:, 1]
                        hist_targets = torch.zeros_like(hist_raw_returns)
                        h_buy_mask = (hist_is_buy == 1.0)
                        hist_targets[h_buy_mask & (hist_raw_returns > alpha)] = 1.0
                        h_sell_mask = (hist_is_buy == -1.0)
                        hist_targets[h_sell_mask & (hist_raw_returns < alpha)] = 1.0
                        
                        isnan_mask = torch.isnan(hist_raw_returns)
                        hist_is_resolved[isnan_mask] = 0.0
                        hist_targets[isnan_mask] = 0.0
                        hist_targets = hist_targets.unsqueeze(-1)
                        
                        masked_label = hist_targets * hist_is_resolved + 0.5 * (1 - hist_is_resolved)
                        
                        age_seconds = (batch_max_t - hist_t).float()
                        age_days = age_seconds / 86400.0
                        age_feat = torch.log1p(age_days).unsqueeze(-1)
                        
                        assoc = dict(zip(n_id.tolist(), range(n_id.size(0))))
                        target_nodes = n_id[edge_index[1]]
                        last_update = model.memory.last_update[target_nodes]
                        rel_t = last_update - hist_t
                        rel_t_enc = model.memory.time_enc(rel_t.to(torch.float))
                        edge_attr = torch.cat([rel_t_enc, hist_msg, hist_price_emb, masked_label, age_feat], dim=-1)
                        
                        z = model(n_id, edge_index, edge_attr)
                        z_src = z[[assoc[i] for i in src.tolist()]]
                        z_dst = z[[assoc[i] for i in dst.tolist()]]
                        s_src = model.encode_static(data.x_static[src])
                        s_dst = model.encode_static(data.x_static[dst])
                        
                        if ablation_mode == 'mkt_only':
                            s_src = torch.zeros_like(s_src)
                            s_dst = torch.zeros_like(s_dst)
                        
                        with torch.no_grad():
                            pred_logits = model.predictor(z_src, z_dst, s_src, s_dst, price_emb, price_emb).squeeze()
                            
                            if val_mask.sum() > 0:
                                loss = criterion(pred_logits[val_mask].view(-1), batch_targets[val_mask].view(-1))
                                val_losses.append(loss.item())
                                
                                p = pred_logits[val_mask].sigmoid().cpu().numpy()
                                val_preds.extend(p.flatten())
                                val_targets.extend(batch_targets[val_mask].cpu().numpy())
                            else:
                                pass
                    
                    model.memory.update_state(src, dst, t, augmented_msg)
                    neighbor_loader.insert(src, dst)
                
                avg_loss = np.mean(epoch_losses) if epoch_losses else 0
                avg_val_loss = np.mean(val_losses) if val_losses else 0
                val_f1 = f1_score(val_targets, np.array(val_preds) > 0.5) if val_targets else 0
                
                logger.debug(f"  Ep {epoch}: Loss={avg_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val F1={val_f1:.4f}")
                
                if avg_val_loss < min_val_loss:
                    min_val_loss = avg_val_loss
                    epochs_without_improvement = 0
                    best_epoch = epoch
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        logger.info(f"  Early Stop at ep {epoch} (best Val Loss: {min_val_loss:.4f})")
                        break
                
                model.train()
            
            # --- PHASE 2: RETRAIN FULL ---
            if best_epoch is None: best_epoch = epoch
            logger.info(f"  Retraining Phase 2 for {best_epoch} epochs on FULL data")
            
            model = TGN(
                num_nodes=num_nodes, raw_msg_dim=3, memory_dim=100, time_dim=100, embedding_dim=100,
                num_parties=num_parties, num_states=num_states
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            neighbor_loader = LastNeighborLoader(num_nodes, size=10, device=device)
            full_train_loader = TemporalDataLoader(train_data, batch_size=200, drop_last=True)
            
            for epoch in range(1, best_epoch + 1):
                model.memory.reset_state()
                neighbor_loader.reset_state()
                model.memory.detach()
                model.train()
                for batch in full_train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    
                    # --- DYNAMIC LABEL ---
                    horizon_map = {'1M':0, '2M':1, '3M':2, '6M':3, '8M':4, '12M':5, '18M':6, '24M':7}
                    h_idx = horizon_map.get(args.horizon, 0)
                    h_days = {'1M':30, '2M':60, '3M':90, '6M':180, '8M':240, '12M':365, '18M':545, '24M':730}.get(args.horizon, 30)
                    h_seconds = h_days * 86400
                    
                    raw_return = batch.y[:, h_idx]
                    trade_t = batch.trade_t
                    resolution_t = trade_t + h_seconds
                    
                    batch_max_t = t.max()
                    
                    is_known = (resolution_t < batch_max_t)
                    has_label = ~torch.isnan(raw_return)
                    train_mask = is_known & has_label
                    
                    if train_mask.sum() == 0:
                         pass
                    
                    # Targets
                    alpha = args.alpha
                    is_buy = msg[:, 1]
                    batch_targets = torch.zeros_like(raw_return)
                    buy_mask = (is_buy == 1.0)
                    batch_targets[buy_mask & (raw_return > alpha)] = 1.0
                    sell_mask = (is_buy == -1.0)
                    batch_targets[sell_mask & (raw_return < alpha)] = 1.0
                    
                    if hasattr(batch, 'price_seq'): price_seq = batch.price_seq
                    else: price_seq = torch.zeros((len(src), 14), device=device)
                    if ablation_mode == 'pol_only': price_seq = torch.zeros_like(price_seq)
                    price_emb = model.get_price_embedding(price_seq)
                    augmented_msg = torch.cat([msg, price_emb], dim=1)
                    
                    n_id = torch.cat([src, dst]).unique()
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                    assoc = dict(zip(n_id.tolist(), range(n_id.size(0))))
                    hist_t = data.t[e_id]
                    hist_msg = data.msg[e_id]
                    hist_price_seq = data.price_seq[e_id]
                    if ablation_mode == 'pol_only': hist_price_seq = torch.zeros_like(hist_price_seq)
                    hist_price_emb = model.get_price_embedding(hist_price_seq)
                    
                    # Dynamic Masking (History)
                    hist_raw_returns = data.y[e_id, h_idx]
                    hist_trade_t = data.trade_t[e_id]
                    hist_resolution = hist_trade_t + h_seconds
                    
                    hist_is_resolved = (hist_resolution < batch_max_t).float().unsqueeze(-1)
                    
                    hist_is_buy = hist_msg[:, 1]
                    hist_targets = torch.zeros_like(hist_raw_returns)
                    h_buy_mask = (hist_is_buy == 1.0)
                    hist_targets[h_buy_mask & (hist_raw_returns > alpha)] = 1.0
                    h_sell_mask = (hist_is_buy == -1.0)
                    hist_targets[h_sell_mask & (hist_raw_returns < alpha)] = 1.0
                    
                    isnan_mask = torch.isnan(hist_raw_returns)
                    hist_is_resolved[isnan_mask] = 0.0
                    hist_targets[isnan_mask] = 0.0
                    hist_targets = hist_targets.unsqueeze(-1)
                    
                    masked_label = hist_targets * hist_is_resolved + 0.5 * (1 - hist_is_resolved)
                    
                    age_seconds = (batch_max_t - hist_t).float()
                    age_days = age_seconds / 86400.0
                    age_feat = torch.log1p(age_days).unsqueeze(-1)
                    
                    target_nodes = n_id[edge_index[1]]
                    last_update = model.memory.last_update[target_nodes]
                    rel_t = last_update - hist_t
                    rel_t_enc = model.memory.time_enc(rel_t.to(torch.float))
                    edge_attr = torch.cat([rel_t_enc, hist_msg, hist_price_emb, masked_label, age_feat], dim=-1)
                    
                    z = model(n_id, edge_index, edge_attr)
                    z_src = z[[assoc[i] for i in src.tolist()]]
                    z_dst = z[[assoc[i] for i in dst.tolist()]]
                    s_src = model.encode_static(data.x_static[src])
                    s_dst = model.encode_static(data.x_static[dst])
                    
                    if ablation_mode == 'mkt_only':
                        s_src = torch.zeros_like(s_src)
                        s_dst = torch.zeros_like(s_dst)
                    
                    pred_pos = model.predictor(z_src, z_dst, s_src, s_dst, price_emb, price_emb)
                    
                    if train_mask.sum() > 0:
                        loss = criterion(pred_pos[train_mask].view(-1), batch_targets[train_mask].view(-1))
                        loss.backward()
                        optimizer.step()
                    else:
                        pass
                    
                    model.memory.update_state(src, dst, t, augmented_msg)
                    neighbor_loader.insert(src, dst)
                    model.memory.detach()

                # Phase 2 Gap
                gap_loader = TemporalDataLoader(gap_data, batch_size=200)
                model.eval()
                for batch in gap_loader:
                    batch = batch.to(device)
                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    if hasattr(batch, 'price_seq'): price_seq = batch.price_seq
                    else: price_seq = torch.zeros((len(src), 14), device=device)
                    if ablation_mode == 'pol_only': price_seq = torch.zeros_like(price_seq)
                    price_emb = model.get_price_embedding(price_seq)
                    augmented_msg = torch.cat([msg, price_emb], dim=1)
                    model.memory.update_state(src, dst, t, augmented_msg)
                    neighbor_loader.insert(src, dst)

            # 4. TEST
            model.eval()
            test_loader = TemporalDataLoader(test_data, batch_size=200)
            preds = []
            targets = []
            transaction_types = []
            
            for batch in test_loader:
                batch = batch.to(device)
                src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                
                # --- DYNAMIC LABEL ---
                horizon_map = {'1M':0, '2M':1, '3M':2, '6M':3, '8M':4, '12M':5, '18M':6, '24M':7}
                h_idx = horizon_map.get(args.horizon, 0)
                h_days = {'1M':30, '2M':60, '3M':90, '6M':180, '8M':240, '12M':365, '18M':545, '24M':730}.get(args.horizon, 30)
                h_seconds = h_days * 86400
                
                raw_return = batch.y[:, h_idx]
                trade_t = batch.trade_t
                resolution_t = trade_t + h_seconds
                
                # For Test, we drop NaNs
                has_label = ~torch.isnan(raw_return)
                
                # Targets
                alpha = args.alpha
                is_buy = msg[:, 1]
                batch_targets = torch.zeros_like(raw_return)
                buy_mask = (is_buy == 1.0)
                batch_targets[buy_mask & (raw_return > alpha)] = 1.0
                sell_mask = (is_buy == -1.0)
                batch_targets[sell_mask & (raw_return < alpha)] = 1.0
                
                if hasattr(batch, 'price_seq'): price_seq = batch.price_seq
                else: price_seq = torch.zeros((len(src), 14), device=device)
                if ablation_mode == 'pol_only': price_seq = torch.zeros_like(price_seq)
                
                price_emb = model.get_price_embedding(price_seq)
                augmented_msg = torch.cat([msg, price_emb], dim=1)
                
                n_id = torch.cat([src, dst]).unique()
                n_id, edge_index, e_id = neighbor_loader(n_id)
                assoc = dict(zip(n_id.tolist(), range(n_id.size(0))))
                
                hist_t = data.t[e_id]
                hist_msg = data.msg[e_id]
                hist_price_seq = data.price_seq[e_id]
                if ablation_mode == 'pol_only': hist_price_seq = torch.zeros_like(hist_price_seq)
                hist_price_emb = model.get_price_embedding(hist_price_seq)
                
                # Dynamic Masking (History)
                hist_raw_returns = data.y[e_id, h_idx]
                hist_trade_t = data.trade_t[e_id]
                hist_resolution = hist_trade_t + h_seconds
                batch_max_t = t.max()
                
                hist_is_resolved = (hist_resolution < batch_max_t).float().unsqueeze(-1)
                
                hist_is_buy = hist_msg[:, 1]
                hist_targets = torch.zeros_like(hist_raw_returns)
                h_buy_mask = (hist_is_buy == 1.0)
                hist_targets[h_buy_mask & (hist_raw_returns > alpha)] = 1.0
                h_sell_mask = (hist_is_buy == -1.0)
                hist_targets[h_sell_mask & (hist_raw_returns < alpha)] = 1.0
                
                isnan_mask = torch.isnan(hist_raw_returns)
                hist_is_resolved[isnan_mask] = 0.0
                hist_targets[isnan_mask] = 0.0
                hist_targets = hist_targets.unsqueeze(-1)
                
                masked_label = hist_targets * hist_is_resolved + 0.5 * (1 - hist_is_resolved)
                
                age_seconds = (batch_max_t - hist_t).float()
                age_days = age_seconds / 86400.0
                age_feat = torch.log1p(age_days).unsqueeze(-1)
                
                target_nodes = n_id[edge_index[1]]
                last_update = model.memory.last_update[target_nodes]
                rel_t = last_update - hist_t
                rel_t_enc = model.memory.time_enc(rel_t.to(torch.float))
                edge_attr = torch.cat([rel_t_enc, hist_msg, hist_price_emb, masked_label, age_feat], dim=-1)
                
                z = model(n_id, edge_index, edge_attr)
                z_src = z[[assoc[i] for i in src.tolist()]]
                z_dst = z[[assoc[i] for i in dst.tolist()]]
                s_src = model.encode_static(data.x_static[src])
                s_dst = model.encode_static(data.x_static[dst])
                
                if ablation_mode == 'mkt_only':
                    s_src = torch.zeros_like(s_src)
                    s_dst = torch.zeros_like(s_dst)
                
                with torch.no_grad():
                    pred_logits = model.predictor(z_src, z_dst, s_src, s_dst, price_emb, price_emb).squeeze()
                    
                    if has_label.sum() > 0:
                        p = pred_logits[has_label].sigmoid().cpu().numpy()
                        preds.extend(p.flatten())
                        targets_batch = batch_targets[has_label].cpu().numpy()
                        targets.extend(targets_batch)
                        
                        is_buy_batch = is_buy[has_label].cpu().numpy()
                        transaction_types.extend(is_buy_batch)
                    else:
                        pass
                
                model.memory.update_state(src, dst, t, augmented_msg)
                neighbor_loader.insert(src, dst)
            
            # Metrics
            try:
                preds_arr = np.array(preds)
                targets_arr = np.array(targets)
                trans_types_arr = np.array(transaction_types)
                
                auc = roc_auc_score(targets_arr, preds_arr)
                pr_auc = average_precision_score(targets_arr, preds_arr)
                acc = accuracy_score(targets_arr, preds_arr > 0.5)
                f1 = f1_score(targets_arr, preds_arr > 0.5)
                macro_f1 = f1_score(targets_arr, preds_arr > 0.5, average='macro')
                count = len(preds)
                
                # Save to Dynamic Folder
                csv_path = f"{args.exp_dir}/ablation_monthly_breakdown.csv"
                with open(csv_path, 'a') as f:
                    if f.tell() == 0:
                        f.write("AblationMode,Year,Month,ROC_AUC,PR_AUC,ACC,F1,Macro_F1,Count\n")
                    f.write(f"{ablation_mode},{year},{month},{auc:.4f},{pr_auc:.4f},{acc:.4f},{f1:.4f},{macro_f1:.4f},{count}\n")
                
                logger.info(f"  [RESULT] Mode={ablation_mode} {year}-{month:02d}: AUC={auc:.4f} | F1={f1:.4f}")
                
                # Report 1: Standard
                report = classification_report(targets_arr, preds_arr > 0.5, output_dict=True)
                report['auc'] = auc
                report['pr_auc'] = pr_auc
                
                with open(f"{args.exp_dir}/reports/report_{ablation_mode}_{year}_{month:02d}.json", "w") as f:
                    json.dump(report, f, indent=4)
                
                logger.info(f"\n--- Classification Report for {year}-{month:02d} ---")
                logger.info("\n" + classification_report(targets_arr, preds_arr > 0.5, target_names=['Loss', 'Win']))
                    
                # Report 2: Adjusted (Flipped Sells)
                sell_mask = (trans_types_arr == -1.0)
                targets_flipped = targets_arr.copy()
                preds_flipped = preds_arr.copy()
                
                # Flip targets: 1 -> 0, 0 -> 1 for Sells
                targets_flipped[sell_mask] = 1 - targets_flipped[sell_mask]
                
                # Flip predictions: p -> 1-p for Sells
                preds_flipped[sell_mask] = 1 - preds_flipped[sell_mask]
                
                # Calculate metrics for flipped
                try:
                    auc_flipped = roc_auc_score(targets_flipped, preds_flipped)
                    pr_auc_flipped = average_precision_score(targets_flipped, preds_flipped)
                except:
                    auc_flipped = None
                    pr_auc_flipped = None
                
                report_flipped = classification_report(targets_flipped, preds_flipped > 0.5, output_dict=True)
                report_flipped['auc'] = auc_flipped
                report_flipped['pr_auc'] = pr_auc_flipped
                
                # Print Adjusted Report to log
                logger.info(f"\n--- Adjusted Classification Report (Stock Direction) for {year}-{month:02d} ---")
                logger.info("\n" + classification_report(targets_flipped, preds_flipped > 0.5, target_names=['Stock Down', 'Stock Up']))
                
                # Save Adjusted Report
                with open(f"{args.exp_dir}/reports/report_{ablation_mode}_{year}_{month:02d}_directional.json", "w") as f:
                    json.dump(report_flipped, f, indent=4)
                    
            except Exception as e:
                logger.error(f"Error in metrics: {e}")
                
    # Save
    logger.info("Ablation Run Complete.")

if __name__ == "__main__":
    # Load Data
    data_path = "data/temporal_data.pt"
    if not os.path.exists(data_path):
        logger.error(f"Data not found at {data_path}")
        sys.exit(1)
        
    data = torch.load(data_path, weights_only=False)
    
    # Metadata trick
    if hasattr(data, 'num_nodes'):
        num_nodes = data.num_nodes
        del data.num_nodes
    else: num_nodes = int(torch.cat([data.src, data.dst]).max()) + 1
        
    if hasattr(data, 'num_parties'): 
        num_parties = data.num_parties
        del data.num_parties
    else: num_parties = 5
        
    if hasattr(data, 'num_states'): 
        num_states = data.num_states
        del data.num_states
    else: num_states = 50
    
    # Load DF
    logger.info("Loading CSV...")
    
    # Path workaround for ablation study running from root
    from src.config import TX_PATH
    raw_df = pd.read_csv(TX_PATH)
    raw_df['Traded'] = pd.to_datetime(raw_df['Traded'])
    raw_df['Filed'] = pd.to_datetime(raw_df['Filed'])
    
    # Sort by Filed to align with temporal_data.py
    raw_df = raw_df.sort_values('Filed').reset_index(drop=True)
    
    # Filter
    ticker_counts = raw_df['Ticker'].value_counts()
    valid_tickers = ticker_counts[ticker_counts >= 1].index
    valid_set = set(valid_tickers)
    # Also filtered missing 'Filed'
    mask = raw_df['Ticker'].isin(valid_set) & raw_df['Filed'].notnull()
    df_filtered = raw_df[mask].reset_index(drop=True)
    
    # Parse Args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-run", action="store_true", help="Run full years")
    parser.add_argument("--full-only", action="store_true", help="Run ONLY 'full' mode (skip pol_only/mkt_only)")
    parser.add_argument("--horizon", type=str, default="1M", help="Return horizon (e.g., 1M, 6M)")
    parser.add_argument("--alpha", type=float, default=0.0, help="Excess return threshold")
    args_cli = parser.parse_args()

    target_years = [2019] 
    ablation_modes = ['pol_only', 'mkt_only', 'full']
    
    if args_cli.full_only:
        ablation_modes = ['full']
        logger.info("FULL ONLY MODE: Skipping ablation baselines.")
    
    if args_cli.full_run:
        target_years = [2019, 2020, 2021, 2022, 2023, 2024]
        logger.info("FULL RUN MODE ACTIVATED")
    
    # Create Result Directory based on Experiment Settings
    exp_dir = f"results/experiments/H_{args_cli.horizon}_A_{args_cli.alpha}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/reports", exist_ok=True)
    logger.info(f"Saving results to: {exp_dir}")
    
    # Store EXP_DIR in args so train_and_evaluate can use it
    args_cli.exp_dir = exp_dir

    for mode in ablation_modes:
        train_and_evaluate(data, df_filtered, target_years=target_years, args=args_cli,
                           num_nodes=num_nodes, num_parties=num_parties, num_states=num_states,
                           ablation_mode=mode, max_epochs=20 if args_cli.full_run else 1)

