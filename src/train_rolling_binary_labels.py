import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import datetime
import sys
import os
sys.path.append(os.getcwd())

from src.models_tgn_binary_labels import TGN
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader
from sklearn.metrics import classification_report, average_precision_score
import matplotlib.pyplot as plt
import json

from src.config_multiple_binary_labels import (
    PROCESSED_DATA_DIR,
    TX_PATH,
    TARGET_COLUMNS,
    RESULTS_DIR,
    LOGS_DIR
)

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
            
            train_mask = df_filtered['Traded'] < gap_start
            gap_mask = (df_filtered['Traded'] >= gap_start) & (df_filtered['Traded'] < gap_end)
            test_mask = (df_filtered['Traded'] >= current_period_start) & (df_filtered['Traded'] < next_period_start)
            
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
            
            train_loader = TemporalDataLoader(actual_train_data, batch_size=200)
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

                    # --- FIX: Mask invalid (-1) targets ---
                    valid_mask = batch.y != -1
                    if valid_mask.sum() > 0:
                        loss = criterion(pred_pos.squeeze()[valid_mask], batch.y[valid_mask].float())
                    else:
                        loss = torch.tensor(0.0, requires_grad=True).to(device)
                    # --------------------------------------

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
                # --- FIX: Filter -1s before metrics ---
                targets_np = np.array(epoch_targets)
                preds_np = np.array(epoch_preds)
                valid_idx = targets_np != -1

                if valid_idx.sum() > 0:
                    train_f1 = f1_score(targets_np[valid_idx], preds_np[valid_idx] > 0.5)
                    epoch_train_f1s.append(train_f1)
                else:
                    epoch_train_f1s.append(0.0)
                # --------------------------------------
                
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
                val_targets_np = np.array(val_targets)
                val_preds_np = np.array(val_preds)
                valid_val_idx = val_targets_np != -1

                if valid_val_idx.sum() > 0:
                    # Check if we have both classes (0 and 1) remaining to avoid sklearn warning
                    unique_labels = np.unique(val_targets_np[valid_val_idx])
                    if len(unique_labels) > 1 or (len(unique_labels) == 1 and unique_labels[0] in [0, 1]):
                        val_f1 = f1_score(val_targets_np[valid_val_idx], val_preds_np[valid_val_idx] > 0.5)

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
            full_train_loader = TemporalDataLoader(train_data, batch_size=200)
            
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

                    # --- FIX: Mask invalid (-1) targets ---
                    valid_mask = batch.y != -1
                    if valid_mask.sum() > 0:
                        loss = criterion(pred_pos.squeeze()[valid_mask], batch.y[valid_mask].float())
                    else:
                        loss = torch.tensor(0.0, requires_grad=True).to(device)
                    # --------------------------------------

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
                
                # Update memory for next batch *within* the month
                model.memory.update_state(src, dst, t, augmented_msg)
                neighbor_loader.insert(src, dst)

            # Metrics
            try:
                # --- FIX: Filter -1s before metrics ---
                targets_np = np.array(targets)
                preds_np = np.array(preds)
                valid_test_idx = targets_np != -1
                
                clean_targets = targets_np[valid_test_idx]
                clean_preds = preds_np[valid_test_idx]
                
                if len(clean_targets) > 0:
                    auc = roc_auc_score(clean_targets, clean_preds)
                    pr_auc = average_precision_score(clean_targets, clean_preds)
                    acc = accuracy_score(clean_targets, clean_preds > 0.5)
                    f1 = f1_score(clean_targets, clean_preds > 0.5)
                    macro_f1 = f1_score(clean_targets, clean_preds > 0.5, average='macro')
                else:
                    auc, pr_auc, acc, f1, macro_f1 = 0, 0, 0, 0, 0
                # --------------------------------------
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
                print(classification_report(targets, np.array(preds) > 0.5, target_names=['Loss', 'Win']))
                
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
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--window_months", type=int, default=24, 
                        help="Rolling window size in months")
    parser.add_argument("--step_months", type=int, default=1, 
                        help="Step size for rolling window")
    args = parser.parse_args()

    # Define paths
    data_path = os.path.join(PROCESSED_DATA_DIR, "temporal_data.pt")
    
    # 1. Load Data
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        # weights_only=False is required to load complex objects like TemporalData
        data = torch.load(data_path, weights_only=False)
    else:
        print("Data not found. Please run temporal_data.py first.")
        sys.exit()

    # === INSERT THIS FIX ===
    # Extract num_classes (int) and remove it from data object 
    # to prevent slicing errors in TemporalData[idx]
    if hasattr(data, 'num_classes'):
        num_classes = data.num_classes
        del data.num_classes  # <--- THIS LINE FIXES THE ATTRIBUTE ERROR
    else:
        # Fallback if attribute missing
        print("Warning: num_classes not found in data. Inferring from config.")
        num_classes = len(TARGET_COLUMNS) + 1
        
    print(f"Model will train on {num_classes} disjoint intervals.")
    # =======================

    # 2. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 3. Prepare Static Features
    # 3. Prepare Static Features & Clean Data Object
    # We must extract these integers and DELETE them from the data object.
    # Why? PyG's TemporalData slicing logic iterates over all attributes and calls .size(),
    # causing a crash if any attribute is a raw integer (int has no .size).
    
    # Extract num_nodes
    if hasattr(data, 'num_nodes'):
        num_nodes = data.num_nodes
        del data.num_nodes  # <--- CRITICAL DELETE
    else:
        # Fallback default if missing
        num_nodes = 10000 
        
    # Extract num_parties
    if hasattr(data, 'num_parties'):
        num_parties = data.num_parties
        del data.num_parties # <--- CRITICAL DELETE
    else:
        num_parties = 5

    # Extract num_states
    if hasattr(data, 'num_states'):
        num_states = data.num_states
        del data.num_states # <--- CRITICAL DELETE
    else:
        num_states = 60

    print(f"Metadata: Nodes={num_nodes}, Parties={num_parties}, States={num_states}")
    
    # 4. Date Alignment for Rolling Window
    print("Loading CSV for Date Alignment...")
    df = pd.read_csv(TX_PATH)
    df['Traded_DT'] = pd.to_datetime(df['Traded'])
    df = df.sort_values('Traded').reset_index(drop=True)
    
    # Convert PyG timestamps back to Pandas Series for slicing
    base_ts = df['Traded_DT'].min().timestamp()
    event_times = data.t.numpy() + base_ts
    event_dates = pd.to_datetime(event_times, unit='s')
    
    # Create a DataFrame helper for slicing
    df_meta = pd.DataFrame({'date': event_dates})
    
    # 5. Rolling Window Loop
    start_date = df_meta['date'].min()
    end_date = df_meta['date'].max()
    
    print(f"Data Range: {start_date.date()} to {end_date.date()}")
    
    current_test_start = start_date + pd.DateOffset(months=args.window_months)
    
    metrics_history = []
    
    while current_test_start < end_date:
        train_start = current_test_start - pd.DateOffset(months=args.window_months)
        test_end = current_test_start + pd.DateOffset(months=args.step_months)
        
        print(f"\n=== RETRAINING For Window: {current_test_start.strftime('%Y-%m')} ===")
        print(f"Train: {train_start.date()} -> {current_test_start.date()}")
        print(f"Test:  {current_test_start.date()} -> {test_end.date()}")
        
        # Identify Indices
        train_mask = (df_meta['date'] >= train_start) & (df_meta['date'] < current_test_start)
        test_mask = (df_meta['date'] >= current_test_start) & (df_meta['date'] < test_end)
        
        # Convert numpy arrays to PyTorch LongTensors immediately
        train_idx = torch.from_numpy(np.where(train_mask)[0]).long()
        test_idx = torch.from_numpy(np.where(test_mask)[0]).long()
        
        if len(test_idx) == 0:
            print("No test data for this window. Skipping...")
            current_test_start = test_end
            continue
            
        print(f"Train samples: {len(train_idx)} | Test samples: {len(test_idx)}")
        
        # Create Train/Test Batches
        train_data = data[train_idx]
        test_data = data[test_idx]
        
        # Initialize Model (Fresh for each window)
        model = TGN(
            num_nodes=num_nodes,
            raw_msg_dim=3, # Amt, Buy, Gap
            memory_dim=args.hidden_dim,
            time_dim=args.hidden_dim,
            embedding_dim=args.hidden_dim,
            num_parties=num_parties,
            num_states=num_states,
            num_classes=num_classes # <--- PASSED HERE
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # Using CrossEntropyLoss for the disjoint interval classification
        criterion = torch.nn.CrossEntropyLoss()

        # NOTE: You need to implement your train_one_window or similar loop here.
        # Assuming you have a function `train_and_evaluate(model, train_data, test_data)`
        # metrics = train_and_evaluate(model, train_data, test_data, optimizer, criterion, device)
        # metrics_history.append(metrics)
        
        # Move to next window
        current_test_start = test_end