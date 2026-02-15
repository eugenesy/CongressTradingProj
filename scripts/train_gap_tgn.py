import argparse
import logging
import torch
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gap_tgn import GAPTGN
from src.temporal_data import TemporalGraphBuilder
from src.config import TX_PATH, RESULTS_DIR

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_tgn_study(horizon='6M', epochs=50, hidden_dim=128, lr=0.001, seed=42):
    # Set Seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    logger.info("Loading Data...")
    
    if not os.path.exists(TX_PATH):
        logger.error(f"Dataset not found at {TX_PATH}. Run chocolate-build first.")
        return

    # Load original DF just to determine base time for alignment
    df_raw = pd.read_csv(TX_PATH)
    
    # Map horizon string to index
    horizons = ['3M', '6M', '8M', '12M', '18M', '24M']
    if horizon not in horizons:
        raise ValueError(f"Invalid horizon: {horizon}. Must be one of {horizons}")
    h_idx = horizons.index(horizon)
    
    # Build Graph Data
    logger.info("Constructing Temporal Graph...")
    builder = TemporalGraphBuilder(df_raw, min_freq=2)
    data = builder.process()
    
    # --- Date Reconstruction & Masking ---
    base_time = pd.to_datetime(df_raw['Filed'].min()).timestamp()
    t_numpy = data.t.cpu().numpy()
    absolute_ts = t_numpy + base_time
    event_dates = pd.to_datetime(absolute_ts, unit='s')
    
    # Standard Train/Test Split (Fixed Date)
    train_mask_np = (event_dates.year >= 2015) & (event_dates.year <= 2022)
    test_mask_np = (event_dates.year >= 2023)
    
    train_mask = torch.tensor(train_mask_np, dtype=torch.bool)
    test_mask = torch.tensor(test_mask_np, dtype=torch.bool)
    
    # --- Binarize Targets (Alpha > 0) ---
    raw_targets = data.y[:, h_idx]
    valid_label_mask = ~torch.isnan(raw_targets)
    labels_binary = (raw_targets > 0).float()
    
    # Compute Final Indices
    train_idx = torch.where(train_mask & valid_label_mask)[0]
    test_idx = torch.where(test_mask & valid_label_mask)[0]
    
    logger.info(f"Graph Events: {len(data.t)}")
    logger.info(f"Train Samples: {len(train_idx)} (Pos: {labels_binary[train_idx].sum().item():.0f})")
    logger.info(f"Test Samples: {len(test_idx)} (Pos: {labels_binary[test_idx].sum().item():.0f})")
    
    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Edge dim comes from msg
    edge_dim = data.msg.shape[1]
    # Node dims come from the new separated features
    pol_dim = data.x_pol.shape[1]
    comp_dim = data.x_comp.shape[1]
    
    # UPDATED INIT
    model = GAPTGN(
        edge_feat_dim=edge_dim,
        pol_dim=pol_dim,       # New
        comp_dim=comp_dim,     # New
        price_seq_dim=14,
        hidden_dim=hidden_dim,
        num_nodes=data.num_nodes
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Move data to device (UPDATED with new features)
    data.src = data.src.to(device)
    data.dst = data.dst.to(device)
    data.t = data.t.to(device)
    data.msg = data.msg.to(device)
    data.x_pol = data.x_pol.to(device)   # New
    data.x_comp = data.x_comp.to(device) # New
    data.price_seq = data.price_seq.to(device)
    data.trade_t = data.trade_t.to(device)
    labels_binary = labels_binary.to(device)
    
    # Training Loop
    logger.info("Starting Training...")
    
    for epoch in range(epochs):
        # 1. Reset Memory for the new epoch
        model.memory.reset_state()
        
        model.train()
        optimizer.zero_grad()
        
        # 2. Forward Pass (UPDATED argument list)
        out = model(
            data.src, data.dst, data.t, data.msg, 
            data.price_seq, data.trade_t, 
            data.x_pol, data.x_comp
        )
        
        for month in range(1, 13):
            logger.info(f"\n--- STUDY: {year}-{month:02d} | Horizon: {horizon} ---")

            test_start = pd.Timestamp(year, month, 1)
            next_month = test_start + pd.DateOffset(months=1)
            
            # Resolution-Based Splitting (Fair Baseline Comparison)
            # Resolution = Traded + Horizon (in days)
            df['Resolution'] = df['Traded'] + pd.to_timedelta(h_days, unit='D')
            
            # 1. Training Set: Filed before test_start AND Resolved before test_start
            train_mask_df = (df['Filed'] < test_start) & (df['Resolution'] < test_start)
            
            # 2. Gap Set: Filed before test_start AND NOT yet Resolved
            gap_mask_df = (df['Filed'] < test_start) & (df['Resolution'] >= test_start)
            
            # 3. Test Set: Filed within the test month
            test_mask_df = (df['Filed'] >= test_start) & (df['Filed'] < next_month)

            train_data = slice_data(df[train_mask_df].index)
            gap_data = slice_data(df[gap_mask_df].index)
            test_data = slice_data(df[test_mask_df].index)

            if len(test_data.src) == 0:
                logger.warning("No data for %s, skipping.", test_start.strftime('%Y-%m'))
                continue

            raw_train = train_data.y[:, h_idx]
            train_targets, train_label_mask = direction_targets(raw_train, alpha)
            train_resolved = (train_data.trade_t + h_seconds) < train_data.t.max()
            train_mask = train_label_mask & train_resolved

            if train_mask.sum() == 0:
                logger.warning("No resolved training labels for %s, skipping.", test_start.strftime('%Y-%m'))
                continue

            pos = train_targets[train_mask].sum().item()
            neg = train_mask.sum().item() - pos
            pos_weight = neg / max(1, pos)
            logger.info(
                "  Train=%s | Gap=%s | Test=%s | pos_weight=%.2f",
                len(train_data.src),
                len(gap_data.src),
                len(test_data.src),
                pos_weight,
            )

            model = ResearchTGN(
                num_nodes=num_nodes,
                raw_msg_dim=raw_msg_dim,
                memory_dim=100,
                time_dim=100,
                embedding_dim=100,
                num_parties=num_parties,
                num_states=num_states,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
            neighbor_loader = LastNeighborLoader(num_nodes, size=30, device=device)

            train_loader = TemporalDataLoader(train_data, batch_size=200, drop_last=True)
            gap_loader = TemporalDataLoader(gap_data, batch_size=200)

            for epoch in range(1, epochs + 1):
                model.train()
                model.memory.reset_state()
                neighbor_loader.reset_state()
                model.memory.detach()
                epoch_loss = []

                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()

                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    raw_y = batch.y[:, h_idx]
                    targets, label_mask = direction_targets(raw_y, alpha)
                    batch_max_t = t.max()
                    resolved_mask = (batch.trade_t + h_seconds) < batch_max_t
                    train_batch_mask = label_mask & resolved_mask

                    p_context = model.get_price_embedding(batch.price_seq)
                    aug_msg = torch.cat([msg, p_context], dim=1)

                    n_id = torch.cat([src, dst]).unique()
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                    assoc = {node.item(): i for i, node in enumerate(n_id)}

                    hist_y = data.y[e_id, h_idx]
                    hist_targets, hist_label_mask = direction_targets(hist_y, alpha)
                    hist_resolved = (data.trade_t[e_id] + h_seconds) < batch_max_t
                    hist_is_resolved = (hist_resolved & hist_label_mask).float().unsqueeze(-1)
                    hist_targets = hist_targets.unsqueeze(-1)
                    masked_label = hist_targets * hist_is_resolved + 0.5 * (1 - hist_is_resolved)

                    age_feat = torch.log1p((batch_max_t - data.t[e_id]).float() / 86400.0).unsqueeze(-1)
                    rel_t = model.memory.last_update[n_id[edge_index[1]]] - data.t[e_id]
                    rel_t_enc = model.memory.time_enc(rel_t.to(torch.float))
                    hist_price_emb = model.get_price_embedding(data.price_seq[e_id])
                    edge_attr = torch.cat(
                        [rel_t_enc, data.msg[e_id], hist_price_emb, masked_label, age_feat],
                        dim=-1,
                    )

                    z = model(n_id, edge_index, edge_attr)
                    z_src = z[[assoc[i.item()] for i in src]]
                    z_dst = z[[assoc[i.item()] for i in dst]]
                    s_src = model.encode_static(data.x_static[src])
                    s_dst = model.encode_static(data.x_static[dst])
                    preds = model.predictor(z_src, z_dst, s_src, s_dst, p_context)

                    if train_batch_mask.sum() > 0:
                        loss = criterion(preds[train_batch_mask].view(-1), targets[train_batch_mask].view(-1))
                        loss.backward()
                        optimizer.step()
                        epoch_loss.append(loss.item())

                    model.memory.update_state(src, dst, t, aug_msg)
                    neighbor_loader.insert(src, dst)
                    model.memory.detach()

                if len(gap_data.src) > 0:
                    model.eval()
                    with torch.no_grad():
                        for batch in gap_loader:
                            batch = batch.to(device)
                            src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                            p_context = model.get_price_embedding(batch.price_seq)
                            aug_msg = torch.cat([msg, p_context], dim=1)
                            model.memory.update_state(src, dst, t, aug_msg)
                            neighbor_loader.insert(src, dst)

                logger.info(
                    "    Epoch %s/%s | Loss: %.4f",
                    epoch,
                    epochs,
                    float(np.mean(epoch_loss)) if epoch_loss else 0.0,
                )

            model.eval()
            test_loader = TemporalDataLoader(test_data, batch_size=200)
            all_preds = []
            all_targets = []
            
            # For Logging: We need to map back to Transaction IDs
            # The test_loader yields batches. We need to track which *indices* of test_data these correspond to.
            # TemporalDataLoader yields samples sequentially from the TemporalData object.
            # So we can just iterate through our original dataframe slice in chunks of 200.
            test_df_slice = df[test_mask_df]
            current_log_idx = 0

            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    raw_y = batch.y[:, h_idx]
                    targets, label_mask = direction_targets(raw_y, alpha)
                    batch_max_t = t.max()

                    p_context = model.get_price_embedding(batch.price_seq)
                    aug_msg = torch.cat([msg, p_context], dim=1)

                    n_id = torch.cat([src, dst]).unique()
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                    assoc = {node.item(): i for i, node in enumerate(n_id)}

                    hist_y = data.y[e_id, h_idx]
                    hist_targets, hist_label_mask = direction_targets(hist_y, alpha)
                    hist_resolved = (data.trade_t[e_id] + h_seconds) < batch_max_t
                    hist_is_resolved = (hist_resolved & hist_label_mask).float().unsqueeze(-1)
                    hist_targets = hist_targets.unsqueeze(-1)
                    masked_label = hist_targets * hist_is_resolved + 0.5 * (1 - hist_is_resolved)

                    age_feat = torch.log1p((batch_max_t - data.t[e_id]).float() / 86400.0).unsqueeze(-1)
                    rel_t = model.memory.last_update[n_id[edge_index[1]]] - data.t[e_id]
                    rel_t_enc = model.memory.time_enc(rel_t.to(torch.float))
                    hist_price_emb = model.get_price_embedding(data.price_seq[e_id])
                    edge_attr = torch.cat(
                        [rel_t_enc, data.msg[e_id], hist_price_emb, masked_label, age_feat],
                        dim=-1,
                    )

                    z = model(n_id, edge_index, edge_attr)
                    z_src = z[[assoc[i.item()] for i in src]]
                    z_dst = z[[assoc[i.item()] for i in dst]]
                    s_src = model.encode_static(data.x_static[src])
                    s_dst = model.encode_static(data.x_static[dst])
                    preds = model.predictor(z_src, z_dst, s_src, s_dst, p_context).sigmoid()

                    if label_mask.sum() > 0:
                        batch_preds = preds[label_mask].cpu().numpy().flatten()
                        batch_targets = targets[label_mask].cpu().numpy().flatten()
                        all_preds.extend(batch_preds)
                        all_targets.extend(batch_targets)
                        
                        # --- DETAILED LOGGING ---
                        # Get the corresponding rows from the dataframe
                        # label_mask is a boolean mask for the current batch
                        # We need to find which rows in test_df_slice correspond to this batch
                        batch_size_actual = len(src)
                        batch_df_rows = test_df_slice.iloc[current_log_idx : current_log_idx + batch_size_actual]
                        
                        # Now apply the label_mask to these rows to get only the valid targets
                        # Note: label_mask is a tensor, we need to convert to numpy/bool
                        mask_np = label_mask.cpu().numpy().astype(bool)
                        valid_rows = batch_df_rows.iloc[mask_np]
                        
                        for i in range(len(valid_rows)):
                            row = valid_rows.iloc[i]
                            rec = {
                                'TransactionID': row['transaction_id'],
                                'Date': row['Filed'],
                                'Model': 'TGN',
                                'True_Up': batch_targets[i],
                                'Prob_Up': batch_preds[i],
                                'Pred_Up': 1.0 if batch_preds[i] > 0.5 else 0.0
                            }
                            predictions_log.append(rec)
                            
                        current_log_idx += batch_size_actual
                    else:
                        current_log_idx += len(src)

                    model.memory.update_state(src, dst, t, aug_msg)
                    neighbor_loader.insert(src, dst)

            if len(all_targets) == 0:
                logger.warning("No labeled targets for %s, skipping.", test_start.strftime('%Y-%m'))
                continue

            res_auc = roc_auc_score(all_targets, all_preds) if len(set(all_targets)) > 1 else 0.5
            res_acc = accuracy_score(all_targets, np.array(all_preds) > 0.5)
            res_f1 = f1_score(all_targets, np.array(all_preds) > 0.5)
            count = len(all_targets)

            logger.info(
                "  [RESULT %d-%02d]: AUC=%.4f | Acc=%.4f | F1=%.4f | Count=%s",
                year,
                month,
                res_auc,
                res_acc,
                res_f1,
                count,
            )
            results.append({'Year': year, 'Month': month, 'AUC': res_auc, 'Acc': res_acc, 'F1': res_f1, 'Count': count})
            
            logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Test ACC: {acc:.4f} | Test AUC: {auc:.4f}")

    # --- Save Results ---
    save_dir = Path(RESULTS_DIR) / "experiments"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = save_dir / f"gap_tgn_{horizon}.pt"
    torch.save(model.state_dict(), results_path)
    logger.info(f"Model saved to {results_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=str, default='6M', help='Prediction horizon (e.g., 6M)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    run_tgn_study(horizon=args.horizon, epochs=args.epochs, seed=args.seed)

if __name__ == "__main__":
    main()