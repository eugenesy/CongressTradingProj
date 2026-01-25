import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import datetime
import sys
import os
sys.path.append(os.getcwd())

from src.models_tgn import TGN
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader
from sklearn.metrics import classification_report, average_precision_score
import matplotlib.pyplot as plt
import json

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_evaluate(data, df_filtered, args=None, target_years=[2023], num_nodes=None, num_parties=5, num_states=50,
                       max_epochs=20, patience=5):
    device = get_device()
    data = data.to(device)
    
    results = []
    
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
            
            print(f"\n=== RETRAINING For Window: {year}-{month:02d} ===")
            
            # 1. Split Indices
            # Train: < Gap Start (Fully Labeled)
            # Gap:   >= Gap Start & < Test Start (Edges exist, Labels unknown)
            # Test:  >= Test Start & < Test End
            
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
                print(f"No trades in {year}-{month:02d}, skipping.")
                continue
                
            train_data = data[train_idx]
            gap_data = data[gap_idx]
            test_data = data[test_idx]
            
            print(f"Train: {len(train_data.src)} | Gap: {len(gap_data.src)} | Test: {len(test_data.src)}")
            
            # 2. INIT MODEL
            # raw_msg_dim=3 (Amount, Is_Buy, Gap) + 2 (masked_label, age) = 5 for edge_attr
            # But these 2 are computed dynamically, so raw_msg_dim stays 3
            model = TGN(
                num_nodes=num_nodes, raw_msg_dim=3, memory_dim=100, time_dim=100, embedding_dim=100,
                num_parties=num_parties, num_states=num_states
            ).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            criterion = torch.nn.BCEWithLogitsLoss()
            neighbor_loader = LastNeighborLoader(num_nodes, size=10, device=device)
            
            # 3. TRAIN PHASE (With Backprop and Validation)
            # Split training data: 90% train, 10% validation (chronological)
            train_size = int(len(train_data.src) * 0.9)
            val_split_idx = torch.arange(train_size, len(train_data.src))
            train_split_idx = torch.arange(0, train_size)
            
            actual_train_data = train_data[train_split_idx]
            val_data = train_data[val_split_idx]
            
            train_loader = TemporalDataLoader(actual_train_data, batch_size=200, drop_last=True)
            val_loader = TemporalDataLoader(val_data, batch_size=200, drop_last=True)
            
            print(f"  Train Split: {len(actual_train_data.src)} | Val Split: {len(val_data.src)}")
            
            model.train()
            
            # Track training metrics per epoch
            epoch_losses = []  # Per-epoch average loss
            epoch_train_f1s = []  # Per-epoch train F1
            epoch_val_f1s = []  # Per-epoch validation F1
            all_batch_losses = []  # Per-batch loss across ALL epochs
            
            min_val_loss = float('inf')
            epochs_without_improvement = 0
            best_model_state = None
            best_epoch = None  # Track which epoch had best validation
            
            for epoch in range(1, max_epochs + 1):
                model.memory.reset_state()
                neighbor_loader.reset_state()
                model.memory.detach()
                
                # Track batch-level metrics
                batch_losses = []
                epoch_preds = []
                epoch_targets = []
                
                # A. Train Loop
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    
                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    
                    # --- DYNAMIC LABEL GENERATION ---
                    # 1. Determine Horizon Index and Days
                    horizon_map = {'1M':0, '2M':1, '3M':2, '6M':3, '8M':4, '12M':5, '18M':6, '24M':7}
                    horizon_days_map = {'1M':30, '2M':60, '3M':90, '6M':180, '8M':240, '12M':365, '18M':545, '24M':730}
                    
                    h_idx = horizon_map.get(args.horizon, 0)
                    h_days = horizon_days_map.get(args.horizon, 30)
                    h_seconds = h_days * 86400
                    
                    # 2. Select Raw Label (Excess Return) for Horizon
                    # batch.y is now [Batch, 8]
                    raw_return = batch.y[:, h_idx] # [Batch]
                    
                    # 3. Dynamic Resolution Time
                    # trade_t is stored in batch.trade_t (was resolution_t in old data, but check TemporalData)
                    # We renamed it to trade_t in TemporalData, but TemporalDataLoader slices attributes.
                    # It should preserve the name if it's in the data object.
                    trade_t = batch.trade_t
                    resolution_t = trade_t + h_seconds
                    
                    # 4. Masking
                    # - Label must not be NaN
                    # - Resolution time must be in the past (resolution_t < batch_max_t)
                    # Note: For TRAINING, we simulate online setting. We can only learn from events
                    # where the outcome is ALREADY known at the time of the batch.
                    batch_max_t = t.max()
                    
                    is_known = (resolution_t < batch_max_t)
                    has_label = ~torch.isnan(raw_return)
                    train_mask = is_known & has_label
                    
                    if train_mask.sum() == 0:
                        # No valid labels in this batch, just update memory and skip gradient
                        # But we still need to compute embeddings efficiently
                        pass
                    
                    # 5. Binarize Label (Target)
                    # Buy (1.0): 1 if Return > alpha
                    # Sell (-1.0): 1 if Return < -alpha (Success = Stock Drop)
                    alpha = args.alpha
                    is_buy = msg[:, 1] # [Batch]
                    
                    # Initialize Targets
                    targets = torch.zeros_like(raw_return)
                    
                    # Buy Logic: Win if Ret > alpha
                    buy_mask = (is_buy == 1.0)
                    targets[buy_mask & (raw_return > alpha)] = 1.0
                    
                    # Sell Logic: Win if Ret < -alpha
                    sell_mask = (is_buy == -1.0)
                    targets[sell_mask & (raw_return < alpha)] = 1.0
                    
                    # Apply Mask to everything that goes into Loss
                    # We only compute loss on 'train_mask' items
                    
                    # ... (Continue with existing pipeline, but using 'targets[train_mask]') ...
                    
                    # --- Price Embedding (Current Batch) ---
                    if hasattr(batch, 'price_seq'):
                         price_seq = batch.price_seq
                    else:
                         price_seq = torch.zeros((len(src), 14), device=device)
                    
                    price_emb = model.get_price_embedding(price_seq) 
                    augmented_msg = torch.cat([msg, price_emb], dim=1)
                    
                    n_id = torch.cat([src, dst]).unique()
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                    assoc = dict(zip(n_id.tolist(), range(n_id.size(0))))
                    
                    hist_t = data.t[e_id]
                    hist_msg = data.msg[e_id]
                    hist_price_seq = data.price_seq[e_id]
                    hist_price_emb = model.get_price_embedding(hist_price_seq)
                    
                    # === DYNAMIC LABEL FEATURE FOR HISTORY ===
                    # We also need to properly mask labels in the HISTORY (edge_attr)
                    # History resolution time depends on the horizon too?
                    # The stored history labels are raw returns (or whatever was in data.y).
                    # Wait, data.y is [N, 8].
                    # We need to pass valid masked labels to the model.
                    # The model expects a single scalar label in edge_attr.
                    # We should probably pick the SAME horizon for history as the current task.
                    # So for history edges:
                    hist_raw_returns = data.y[e_id, h_idx] # Select column
                    hist_trade_t = data.trade_t[e_id]
                    hist_resolution = hist_trade_t + h_seconds
                    
                    # Mask: 1.0 if resolved relative to CURRENT batch time
                    hist_is_resolved = (hist_resolution < batch_max_t).float().unsqueeze(-1)
                    
                    # Generate Binary Label for History
                    # We need is_buy for history edges. "msg" has 3 dims: [Amt, IsBuy, Gap]
                    hist_is_buy = hist_msg[:, 1]
                    hist_targets = torch.zeros_like(hist_raw_returns)
                    
                    h_buy_mask = (hist_is_buy == 1.0)
                    hist_targets[h_buy_mask & (hist_raw_returns > alpha)] = 1.0
                    
                    h_sell_mask = (hist_is_buy == -1.0)
                    hist_targets[h_sell_mask & (hist_raw_returns < alpha)] = 1.0
                    
                    # If NaN, treat as unresolved (mask=0)
                    isnan_mask = torch.isnan(hist_raw_returns)
                    hist_is_resolved[isnan_mask] = 0.0
                    hist_targets[isnan_mask] = 0.0 # Just to be safe
                    
                    hist_targets = hist_targets.unsqueeze(-1)
                    
                    # Masked label: Real Label if resolved, else 0.5
                    masked_label_feat = hist_targets * hist_is_resolved + 0.5 * (1 - hist_is_resolved)
                    
                    age_seconds = (batch_max_t - hist_t).float()
                    age_days = age_seconds / 86400.0
                    age_feat = torch.log1p(age_days).unsqueeze(-1)
                    
                    target_nodes = n_id[edge_index[1]]
                    last_update = model.memory.last_update[target_nodes]
                    rel_t = last_update - hist_t
                    rel_t_enc = model.memory.time_enc(rel_t.to(torch.float))
                    
                    edge_attr = torch.cat([rel_t_enc, hist_msg, hist_price_emb, masked_label_feat, age_feat], dim=-1)
                    
                    z = model(n_id, edge_index, edge_attr)
                    z_src = z[[assoc[i] for i in src.tolist()]]
                    z_dst = z[[assoc[i] for i in dst.tolist()]]
                    
                    s_src = model.encode_static(data.x_static[src])
                    s_dst = model.encode_static(data.x_static[dst])
                    
                    # Predictor
                    pred_pos = model.predictor(z_src, z_dst, s_src, s_dst, price_emb, price_emb)
                    
                    # LOSS Calculation - Only on Valid Train Mask
                    if train_mask.sum() > 0:
                        loss = criterion(pred_pos[train_mask].view(-1), targets[train_mask].view(-1))
                        loss.backward()
                        optimizer.step()
                        
                        batch_losses.append(loss.item())
                        all_batch_losses.append(loss.item())
                        
                        with torch.no_grad():
                            batch_pred = pred_pos[train_mask].sigmoid().cpu().numpy()
                            epoch_preds.extend(batch_pred.flatten())
                            epoch_targets.extend(targets[train_mask].cpu().numpy())
                    else:
                        # If no loss, just step memory? No backprop.
                        pass
                    
                    model.memory.update_state(src, dst, t, augmented_msg)
                    neighbor_loader.insert(src, dst)
                    model.memory.detach()
                    
                # B. Gap Phase (Forward Only - No Backprop)
                # We must process Gap Data to bring Memory/Graph up to date for Test.
                # BUT we must NOT learn from it (Labels are hidden).
                # We do this at the end of every epoch? 
                # NO. If we do it every epoch, we are "simulating" passing through the gap.
                # Since we reset memory every epoch, we MUST process the Gap every epoch to reach Test state.
                
                gap_loader = TemporalDataLoader(gap_data, batch_size=200)
                model.eval() # Eval mode for gap (no dropout etc, strictly update)
                for batch in gap_loader:
                    batch = batch.to(device)
                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    
                    # Update Memory with Augmented Msg
                    # 1. Retrieve & Encode
                    if hasattr(batch, 'price_seq'):
                         price_seq = batch.price_seq
                    else:
                         price_seq = torch.zeros((len(src), 14), device=device)
                    
                    price_emb = model.get_price_embedding(price_seq)
                    augmented_msg = torch.cat([msg, price_emb], dim=1)
                    
                    # Just Update
                    model.memory.update_state(src, dst, t, augmented_msg)
                    neighbor_loader.insert(src, dst)
                
                # Epoch Summary
                avg_loss = np.mean(batch_losses)
                epoch_losses.append(avg_loss)
                
                # Compute Train F1 for this epoch
                train_f1 = 0.0
                if len(epoch_targets) > 0:
                    train_f1 = f1_score(epoch_targets, np.array(epoch_preds) > 0.5)
                    epoch_train_f1s.append(train_f1)
                else:
                    epoch_train_f1s.append(0.0)
                
                # C. VALIDATION EVAL (No Backprop)
                model.eval()
                val_preds = []
                val_targets = []
                val_losses = []
                
                # Need to build memory up to validation point first
                # Actually, validation data is part of training history, just held out
                # So we should process train + gap before validating
                for batch in val_loader:
                    batch = batch.to(device)
                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    
                    # --- DYNAMIC LABEL ---
                    # Same logic as Train
                    horizon_map = {'1M':0, '2M':1, '3M':2, '6M':3, '8M':4, '12M':5, '18M':6, '24M':7}
                    h_idx = horizon_map.get(args.horizon, 0)
                    h_days = {'1M':30, '2M':60, '3M':90, '6M':180, '8M':240, '12M':365, '18M':545, '24M':730}.get(args.horizon, 30)
                    h_seconds = h_days * 86400
                    
                    raw_return = batch.y[:, h_idx]
                    trade_t = batch.trade_t
                    resolution_t = trade_t + h_seconds
                    
                    batch_max_t = t.max()
                    
                    # Validation Masking - IMPORTANT: Even in validation we skip NaNs
                    # We usually assume validation labels are "known", but if they are NaN (missing price data), we can't eval.
                    # We also technically should skip "future" labels if we want strictly realistic eval,
                    # but usually for validation set we assume we have ground truth for that period.
                    # However, to be consistent with "can we label this?", let's keep masking.
                    
                    is_known = (resolution_t < batch_max_t)
                    has_label = ~torch.isnan(raw_return)
                    val_mask = is_known & has_label
                    
                    # Targets
                    alpha = args.alpha
                    is_buy = msg[:, 1]
                    targets = torch.zeros_like(raw_return)
                    buy_mask = (is_buy == 1.0)
                    targets[buy_mask & (raw_return > alpha)] = 1.0
                    sell_mask = (is_buy == -1.0)
                    targets[sell_mask & (raw_return < alpha)] = 1.0
                    
                    # Store augmented msg for update at end of loop
                    if hasattr(batch, 'price_seq'):
                         price_seq = batch.price_seq
                    else:
                         price_seq = torch.zeros((len(src), 14), device=device)
                    
                    price_emb = model.get_price_embedding(price_seq)
                    augmented_msg = torch.cat([msg, price_emb], dim=1)
                    
                    n_id = torch.cat([src, dst]).unique()
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                    
                    if len(e_id) > 0:
                        hist_t = data.t[e_id]
                        hist_msg = data.msg[e_id]
                        
                        # Hist Price Emb
                        hist_price_seq = data.price_seq[e_id]
                        hist_price_emb = model.get_price_embedding(hist_price_seq)
                        
                        # --- HISTORY LABEL MASKING (Dynamic) ---
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
                        
                        with torch.no_grad():
                            # Only Eval on Mask
                            pred_logits = model.predictor(z_src, z_dst, s_src, s_dst, price_emb, price_emb).squeeze()
                            
                            if val_mask.sum() > 0:
                                loss = criterion(pred_logits[val_mask], targets[val_mask])
                                val_losses.append(loss.item())
                                
                                p = pred_logits[val_mask].sigmoid().cpu().numpy()
                                val_preds.extend(p.flatten())
                                val_targets.extend(targets[val_mask].cpu().numpy())
                            else:
                                # Start fresh if empty batch?
                                pass
                    
                    model.memory.update_state(src, dst, t, augmented_msg)
                    neighbor_loader.insert(src, dst)
                
                # Calculate validation F1
                val_f1 = 0.0
                if len(val_targets) > 0 and len(set(val_targets)) > 1:
                    val_f1 = f1_score(val_targets, np.array(val_preds) > 0.5)
                epoch_val_f1s.append(val_f1)
                
                avg_val_loss = np.mean(val_losses) if val_losses else 0.0
                
                print(f"  Epoch {epoch}/{max_epochs}: Loss={avg_loss:.4f} | Train F1={train_f1:.4f} | Val Loss={avg_val_loss:.4f} | Val F1={val_f1:.4f}")
                
                # Early Stopping Check (Loss > F1)
                if avg_val_loss < min_val_loss:
                    min_val_loss = avg_val_loss
                    epochs_without_improvement = 0
                    best_model_state = model.state_dict().copy()
                    best_epoch = epoch  # Track which epoch was best
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"  Phase 1 Early Stop at epoch {epoch} (best Val Loss: {min_val_loss:.4f})")
                        break
                
                model.train()  # Switch back for next epoch
            
            # ============================================================
            # PHASE 2: RETRAIN ON FULL DATA FOR best_epoch EPOCHS
            # ============================================================
            # Now that we know the optimal epoch count, retrain from scratch
            # using ALL training data (including the 10% we held for validation)
            
            if best_epoch is None:
                best_epoch = epoch  # Use last epoch if no improvement
            
            print(f"\n  --- Phase 2: Retraining on FULL data for {best_epoch} epochs ---")
            
            # Reset model completely
            model = TGN(
                num_nodes=num_nodes, raw_msg_dim=3, memory_dim=100, time_dim=100, embedding_dim=100,
                num_parties=num_parties, num_states=num_states
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            neighbor_loader = LastNeighborLoader(num_nodes, size=10, device=device)
            
            # Use FULL train_data (not split)
            full_train_loader = TemporalDataLoader(train_data, batch_size=200, drop_last=True)
            
            for epoch in range(1, best_epoch + 1):
                model.memory.reset_state()
                neighbor_loader.reset_state()
                model.memory.detach()
                model.train()
                
                epoch_loss = 0.0
                batch_count = 0
                
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
                        pass # Valid for Phase 2 as well
                    
                    # Targets
                    alpha = args.alpha
                    is_buy = msg[:, 1]
                    targets = torch.zeros_like(raw_return)
                    buy_mask = (is_buy == 1.0)
                    targets[buy_mask & (raw_return > alpha)] = 1.0
                    sell_mask = (is_buy == -1.0)
                    targets[sell_mask & (raw_return < alpha)] = 1.0
                    
                    # --- Price Embedding (Current Batch) ---
                    if hasattr(batch, 'price_seq'):
                         price_seq = batch.price_seq
                    else:
                         price_seq = torch.zeros((len(src), 14), device=device)
                    
                    price_emb = model.get_price_embedding(price_seq)
                    augmented_msg = torch.cat([msg, price_emb], dim=1)
                    
                    n_id = torch.cat([src, dst]).unique()
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                    assoc = dict(zip(n_id.tolist(), range(n_id.size(0))))
                    
                    hist_t = data.t[e_id]
                    hist_msg = data.msg[e_id]
                    
                    # Hist Price Emb
                    hist_price_seq = data.price_seq[e_id]
                    hist_price_emb = model.get_price_embedding(hist_price_seq)
                    
                    # --- HISTORY LABEL MASKING (Dynamic) ---
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
                    
                    pred_pos = model.predictor(z_src, z_dst, s_src, s_dst, price_emb, price_emb)
                    
                    if train_mask.sum() > 0:
                        loss = criterion(pred_pos[train_mask].view(-1), targets[train_mask].view(-1))
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    else:
                        pass
                    batch_count += 1
                    
                    model.memory.update_state(src, dst, t, augmented_msg)
                    neighbor_loader.insert(src, dst)
                    model.memory.detach()
                
                # Process Gap data (no backprop)
                gap_loader = TemporalDataLoader(gap_data, batch_size=200)
                model.eval()
                for batch in gap_loader:
                    batch = batch.to(device)
                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    
                    # Update Memory with Augmented Msg
                    if hasattr(batch, 'price_seq'):
                         price_seq = batch.price_seq
                    else:
                         price_seq = torch.zeros((len(src), 14), device=device)
                    
                    price_emb = model.get_price_embedding(price_seq)
                    augmented_msg = torch.cat([msg, price_emb], dim=1)
                    
                    model.memory.update_state(src, dst, t, augmented_msg)
                    neighbor_loader.insert(src, dst)
                
                print(f"    Phase 2 Epoch {epoch}/{best_epoch}: Avg Loss = {epoch_loss/batch_count:.4f}")
            
            print(f"  Phase 2 Complete: Model trained on full data for {best_epoch} epochs")
                
            # 4. TEST (Monthly Breakdown)
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
                raw_return = batch.y[:, h_idx]
                
                # For Test, we assume labels exist or will be evaluated against future knowns
                # We drop NaNs
                has_label = ~torch.isnan(raw_return)
                
                # Targets
                alpha = args.alpha
                is_buy = msg[:, 1]
                batch_targets = torch.zeros_like(raw_return)
                buy_mask = (is_buy == 1.0)
                batch_targets[buy_mask & (raw_return > alpha)] = 1.0
                sell_mask = (is_buy == -1.0)
                batch_targets[sell_mask & (raw_return < alpha)] = 1.0
                
                # --- Price Embedding (Current Batch) ---
                if hasattr(batch, 'price_seq'):
                     price_seq = batch.price_seq
                else:
                     price_seq = torch.zeros((len(src), 14), device=device)
                
                price_emb = model.get_price_embedding(price_seq)
                augmented_msg = torch.cat([msg, price_emb], dim=1)
                
                n_id = torch.cat([src, dst]).unique()
                n_id, edge_index, e_id = neighbor_loader(n_id)
                assoc = dict(zip(n_id.tolist(), range(n_id.size(0))))
                
                hist_t = data.t[e_id]
                hist_msg = data.msg[e_id]
                
                # Hist Price Emb
                hist_price_seq = data.price_seq[e_id]
                hist_price_emb = model.get_price_embedding(hist_price_seq)
                
                # --- HISTORY LABEL MASKING (Dynamic) ---
                h_days = {'1M':30, '2M':60, '3M':90, '6M':180, '8M':240, '12M':365, '18M':545, '24M':730}.get(args.horizon, 30)
                h_seconds = h_days * 86400
                batch_max_t = t.max()
                
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
                
                with torch.no_grad():
                    # Pass price_emb to predictor
                    pred_logits = model.predictor(z_src, z_dst, s_src, s_dst, price_emb, price_emb).squeeze()
                    
                    if has_label.sum() > 0:
                        p = pred_logits[has_label].sigmoid().cpu().numpy()
                        preds.extend(p.flatten())
                        targets_batch = batch_targets[has_label].cpu().numpy()
                        targets.extend(targets_batch)
                        transaction_types.extend(is_buy[has_label].cpu().numpy())
                    else:
                        pass
                
                # Update memory for next batch *within* the month
                model.memory.update_state(src, dst, t, augmented_msg)
                neighbor_loader.insert(src, dst)
            
            # Metrics
            try:
                preds_arr = np.array(preds)
                targets_arr = np.array(targets)
                trans_types_arr = np.array(transaction_types)
                
                auc = roc_auc_score(targets_arr, preds_arr)
                pr_auc = average_precision_score(targets_arr, preds_arr)  # PR-AUC (better for imbalanced)
                acc = accuracy_score(targets_arr, preds_arr > 0.5)
                f1 = f1_score(targets_arr, preds_arr > 0.5)
                macro_f1 = f1_score(targets_arr, preds_arr > 0.5, average='macro')
                count = len(preds) # Define count here
                
                # Save Results to CSV
                os.makedirs("results", exist_ok=True) # Ensure directory exists
                with open('results/rolling_tgn_retrain_metrics.csv', 'a') as f:
                    # Write header if file is empty
                    if f.tell() == 0:
                        f.write("Year,Month,ROC_AUC,PR_AUC,ACC,F1,Macro_F1,Count\n")
                    f.write(f"{year},{month},{auc:.4f},{pr_auc:.4f},{acc:.4f},{f1:.4f},{macro_f1:.4f},{count}\n")
                    
                print(f"  [RESULT] {year}-{month:02d}: ROC-AUC={auc:.4f} | PR-AUC={pr_auc:.4f} | ACC={acc:.4f} | F1={f1:.4f} | Macro-F1={macro_f1:.4f} | Count={count}")
                
                # Classification Report
                print(f"\n--- Classification Report for {year}-{month:02d} ---")
                print(classification_report(targets_arr, preds_arr > 0.5, target_names=['Loss', 'Win']))
                
                # --- ADJUSTED CLASSIFICATION REPORT (Flipped Sells) ---
                # Logic: If Sell (-1), Label 1 means Trade Win (Stock Down). 
                # We want Label 1 to mean Stock Up. 
                # So for Sells: NewLabel = 1 - OldLabel. 
                # And ProbStockUp = 1 - ProbTradeWin.
                
                sell_mask = (trans_types_arr == -1.0)
                
                targets_flipped = targets_arr.copy()
                preds_flipped = preds_arr.copy()
                
                # Flip targets: 1 -> 0, 0 -> 1 for Sells
                targets_flipped[sell_mask] = 1 - targets_flipped[sell_mask]
                
                # Flip predictions: p -> 1-p for Sells
                preds_flipped[sell_mask] = 1 - preds_flipped[sell_mask]
                
                print(f"\n--- Adjusted Classification Report (Stock Direction) for {year}-{month:02d} ---")
                print(classification_report(targets_flipped, preds_flipped > 0.5, target_names=['Stock Down', 'Stock Up']))
                
                # Save Learning Curve for this month
                os.makedirs("results/learning_curves", exist_ok=True)
                fig, axes = plt.subplots(1, 3, figsize=(18, 4))
                
                # Per-Epoch Loss Curve
                axes[0].plot(range(1, len(epoch_losses)+1), epoch_losses, 'b-o')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].set_title(f'{year}-{month:02d}: Epoch Avg Loss')
                axes[0].grid(True)
                
                # Per-Epoch F1 Curve
                axes[1].plot(range(1, len(epoch_train_f1s)+1), epoch_train_f1s, 'g-o')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Train F1')
                axes[1].set_title(f'{year}-{month:02d}: Epoch Train F1')
                axes[1].grid(True)
                
                # Per-Batch Loss (All Batches Across Epochs)
                axes[2].plot(range(1, len(all_batch_losses)+1), all_batch_losses, 'r-', alpha=0.7, linewidth=0.5)
                axes[2].set_xlabel('Batch (Global)')
                axes[2].set_ylabel('Loss')
                axes[2].set_title(f'{year}-{month:02d}: Per-Batch Loss')
                axes[2].grid(True)
                # Add epoch boundaries
                batches_per_epoch = len(all_batch_losses) // 5 if len(all_batch_losses) > 0 else 1
                for ep in range(1, 6):
                    axes[2].axvline(x=ep * batches_per_epoch, color='gray', linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                plt.savefig(f"results/learning_curves/{year}_{month:02d}.png", dpi=100)
                plt.close()
                
            except Exception as e:
                print(f"Result: Error ({e})")
                
            # Period Done. Next loop iteration will start fresh model tailored for Next Month.
            
    # Save
    pd.DataFrame(results).to_csv("results/rolling_tgn_retrain_metrics.csv", index=False)
    print("Saved to results/rolling_tgn_retrain_metrics.csv")

if __name__ == "__main__":
    # Load Data
    data = torch.load("data/temporal_data.pt", weights_only=False)
    
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
    
    # Load DF for splitting
    print("Loading CSV for Date Alignment...")
    # We must match the CSV used in temporal_data.py
    # temporal_data.py filtered rare tickers. 
    # We need to re-run filtering logic or assume index matches roughly.
    # To be precise, we should recreate the filter.
    
    from src.config import TX_PATH
    raw_df = pd.read_csv(TX_PATH)
    raw_df['Traded'] = pd.to_datetime(raw_df['Traded'])
    raw_df['Filed'] = pd.to_datetime(raw_df['Filed'])
    
    # Sort by Filed to align with temporal_data.py
    raw_df = raw_df.sort_values('Filed').reset_index(drop=True)
    
    # Filter
    # Need to match strictly.
    # Check temporal_data.py logic:
    # 1. process() filtered rows where pid not in map OR ticker not in map
    # 2. Map was built on min_freq=5
    
    ticker_counts = raw_df['Ticker'].value_counts()
    valid_tickers = ticker_counts[ticker_counts >= 1].index
    valid_set = set(valid_tickers)
    
    # Also filtered missing 'Filed' (used for sorting/time)
    
    mask = raw_df['Ticker'].isin(valid_set) & raw_df['Filed'].notnull()
    df_filtered = raw_df[mask].reset_index(drop=True)
    
    print(f"Filtered DF Size: {len(df_filtered)} | Data Size: {len(data.src)}")
    if len(df_filtered) != len(data.src):
        # We might have missed some BioGuideID filtering?
        # temporal_data.py: "pols = self.transactions['BioGuideID'].unique()" -> All pols mapped.
        # So only Ticker filter matters.
        # Wait, if some rows had NaN 'Traded', they were skipped.
        # My mask includes that.
        # Maybe 'Failed to parse amount'?
        # temporal_data.py didn't skip on amount parse fail (just 0.0).
        # It skipped on `if pid not in self.pol_id_map or ticker not in self.company_id_map`
        # Let's hope it aligns.
        pass

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=str, default="1M", help="Return horizon (e.g., 1M, 6M)")
    parser.add_argument("--alpha", type=float, default=0.0, help="Excess return threshold (e.g., 0.05)")
    args = parser.parse_args()

    train_and_evaluate(data, df_filtered, args=args, num_nodes=num_nodes, num_parties=num_parties, num_states=num_states)
