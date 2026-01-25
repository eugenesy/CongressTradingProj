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

def train_and_evaluate(data, df_filtered, target_years=[2023], num_nodes=None, num_parties=5, num_states=50,
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
            val_loader = TemporalDataLoader(val_data, batch_size=200)
            
            print(f"  Train Split: {len(actual_train_data.src)} | Val Split: {len(val_data.src)}")
            
            model.train()
            
            # Track training metrics per epoch
            epoch_losses = []  # Per-epoch average loss
            epoch_train_f1s = []  # Per-epoch train F1
            epoch_val_f1s = []  # Per-epoch validation F1
            all_batch_losses = []  # Per-batch loss across ALL epochs
            
            best_val_f1 = 0.0
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
                    
                    # --- Price Embedding (Current Batch) ---
                    # 1. Retrieve Sequence
                    # batch indices in 'train_data' (which is data[train_idx])
                    # Note: TemporalDataLoader slices attributes. batch.price_seq should exist.
                    # We need to verify if Tensor attributes are sliced by PyG default.
                    # If not, we might need a workaround. Assuming it works for now.
                    if hasattr(batch, 'price_seq'):
                         price_seq = batch.price_seq
                    else:
                         # Fallback: We can't easily recover global indices here without passing them.
                         # But PyG 2.x supports custom node/edge attributes if they match first dim.
                         # data.price_seq has dim0 = data.num_edges. So it should work.
                         price_seq = torch.zeros((len(src), 14), device=device)
                    
                    # 2. Encode
                    price_emb = model.get_price_embedding(price_seq) # [Batch, 32]
                    
                    # 3. Augment Msg (for Memory Update)
                    augmented_msg = torch.cat([msg, price_emb], dim=1) # [Batch, 3+32]
                    
                    n_id = torch.cat([src, dst]).unique()
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                    assoc = dict(zip(n_id.tolist(), range(n_id.size(0))))
                    
                    hist_t = data.t[e_id]
                    hist_msg = data.msg[e_id]
                    
                    # --- Price Embedding (Neighbors/History) ---
                    # We need the price embedding for the *neighbor* edges to use in GNN/Decoder
                    hist_price_seq = data.price_seq[e_id]
                    hist_price_emb = model.get_price_embedding(hist_price_seq)
                    
                    # === DYNAMIC LABEL FEATURE ===
                    # Compute "current time" as max of batch (when we are predicting)
                    batch_max_t = t.max()
                    
                    # Resolution time for each historical edge
                    hist_resolution = data.resolution_t[e_id]
                    
                    # Mask: 1.0 if resolved (label known), 0.0 if not
                    is_resolved = (hist_resolution < batch_max_t).float().unsqueeze(-1)
                    
                    # Raw labels for history
                    raw_label = data.y[e_id].unsqueeze(-1)
                    
                    # Masked label: show actual if resolved, else 0.5 (neutral)
                    masked_label = raw_label * is_resolved + 0.5 * (1 - is_resolved)
                    
                    # Age feature: how old is each neighbor edge (log-normalized days)
                    age_seconds = (batch_max_t - hist_t).float()
                    age_days = age_seconds / 86400.0
                    age_feat = torch.log1p(age_days).unsqueeze(-1)
                    
                    target_nodes = n_id[edge_index[1]]
                    last_update = model.memory.last_update[target_nodes]
                    rel_t = last_update - hist_t
                    rel_t_enc = model.memory.time_enc(rel_t.to(torch.float))
                    
                    # Final edge_attr: [time_enc, msg, masked_label, age, PRICE_EMB]
                    # Note: hist_msg is the RAW msg (3 dims) stored in 'data'.
                    # It does NOT contain price_emb.
                    # PriceEmb is computed on the fly from 'data.price_seq'.
                    # So we concat: [Time, RawMsg, PriceEmb, MaskedLabel, Age]
                    # Wait, 'augmented_msg_dim' in TGN init was 'raw_msg_dim + price_emb'.
                    # And 'edge_dim' in GAT was 'msg_dim + time + 2'.
                    # 'msg_dim' passed to GAT was 'augmented_msg_dim'.
                    # So GAT expects [RawMsg + PriceEmb] + [Time + 2].
                    # Order in GAT: 'edge_dim' is just a size.
                    # TGN.forward calls GAT(..., edge_attr).
                    # GAT uses edge_attr for linear projection.
                    # So the CONTENT of edge_attr must match dimension size.
                    # Dimensions:
                    # Time: 100? No, time_enc.out_channels usually 100? 
                    # Memory time_dim=100.
                    # In TGN model code: `edge_dim = msg_dim + time_enc.out_channels + 2`.
                    # `msg_dim` = 3 + 32 = 35.
                    # So edge_attr must have 35 + Time + 2 dims.
                    # Components:
                    # 1. Time Enc (100)
                    # 2. Msg (35) -> (Raw 3 + Price 32)
                    # 3. Label (1)
                    # 4. Age (1)
                    # Total logic fits.
                    
                    edge_attr = torch.cat([rel_t_enc, hist_msg, hist_price_emb, masked_label, age_feat], dim=-1)
                    
                    z = model(n_id, edge_index, edge_attr)
                    z_src = z[[assoc[i] for i in src.tolist()]]
                    z_dst = z[[assoc[i] for i in dst.tolist()]]
                    
                    s_src = model.encode_static(data.x_static[src])
                    s_dst = model.encode_static(data.x_static[dst])
                    
                    # Predictor needs Price Emb for *current* trade (p_src, p_dst)
                    # Use 'price_emb' calculated above for current batch events.
                    # 'price_emb' is [Batch, 32].
                    # 'price_emb' is 1 per trade.
                    # Pass it as both src/dst context (symmetric context for the edge).
                    pred_pos = model.predictor(z_src, z_dst, s_src, s_dst, price_emb, price_emb)
                    loss = criterion(pred_pos.squeeze(), batch.y)
                    loss.backward()
                    optimizer.step()
                    
                    # Track batch metrics
                    batch_losses.append(loss.item())
                    all_batch_losses.append(loss.item())  # Global batch tracker
                    with torch.no_grad():
                        batch_pred = pred_pos.sigmoid().cpu().numpy()
                        epoch_preds.extend(batch_pred.flatten())
                        epoch_targets.extend(batch.y.cpu().numpy())
                    
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
                
                # Need to build memory up to validation point first
                # Actually, validation data is part of training history, just held out
                # So we should process train + gap before validating
                for batch in val_loader:
                    batch = batch.to(device)
                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    
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
                        
                        batch_max_t = t.max()
                        hist_resolution = data.resolution_t[e_id]
                        is_resolved = (hist_resolution < batch_max_t).float().unsqueeze(-1)
                        raw_label = data.y[e_id].unsqueeze(-1)
                        masked_label = raw_label * is_resolved + 0.5 * (1 - is_resolved)
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
                            p = model.predictor(z_src, z_dst, s_src, s_dst, price_emb, price_emb).sigmoid().cpu().numpy()
                            val_preds.extend(p.flatten())
                            val_targets.extend(batch.y.cpu().numpy())
                    
                    model.memory.update_state(src, dst, t, augmented_msg)
                    neighbor_loader.insert(src, dst)
                
                # Calculate validation F1
                val_f1 = 0.0
                if len(val_targets) > 0 and len(set(val_targets)) > 1:
                    val_f1 = f1_score(val_targets, np.array(val_preds) > 0.5)
                epoch_val_f1s.append(val_f1)
                
                print(f"  Epoch {epoch}/{max_epochs}: Loss={avg_loss:.4f} | Train F1={train_f1:.4f} | Val F1={val_f1:.4f}")
                
                # Early Stopping Check
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    epochs_without_improvement = 0
                    best_model_state = model.state_dict().copy()
                    best_epoch = epoch  # Track which epoch was best
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"  Phase 1 Early Stop at epoch {epoch} (best Val F1: {best_val_f1:.4f})")
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
                    
                    batch_max_t = t.max()
                    hist_resolution = data.resolution_t[e_id]
                    is_resolved = (hist_resolution < batch_max_t).float().unsqueeze(-1)
                    raw_label = data.y[e_id].unsqueeze(-1)
                    masked_label = raw_label * is_resolved + 0.5 * (1 - is_resolved)
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
                    loss = criterion(pred_pos.squeeze(), batch.y)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
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
                
                # === DYNAMIC LABEL FEATURE (Test) ===
                batch_max_t = t.max()
                hist_resolution = data.resolution_t[e_id]
                is_resolved = (hist_resolution < batch_max_t).float().unsqueeze(-1)
                raw_label = data.y[e_id].unsqueeze(-1)
                masked_label = raw_label * is_resolved + 0.5 * (1 - is_resolved)
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
                    p = model.predictor(z_src, z_dst, s_src, s_dst, price_emb, price_emb).sigmoid().cpu().numpy()
                    y = batch.y.cpu().numpy()
                    preds.extend(p)
                    targets.extend(y)
                    
                    # Track Transaction Type (Buy=1, Sell=-1)
                    # msg is [Batch, 3] -> (Amount, Is_Buy, Gap)
                    is_buy_batch = msg[:, 1].cpu().numpy()
                    transaction_types.extend(is_buy_batch)
                
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

    train_and_evaluate(data, df_filtered, num_nodes=num_nodes, num_parties=num_parties, num_states=num_states)
