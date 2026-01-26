
import os
import sys
import pandas as pd
import numpy as np
import torch
import logging
import json
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, average_precision_score
import argparse
from datetime import datetime
from pathlib import Path

sys.path.append(os.getcwd())
from src.models_tgn import TGN
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader

# --- Logging Setup ---
logger = logging.getLogger("directional")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_evaluate(data, df_filtered, args=None, target_years=[2023], num_nodes=None, num_parties=5, num_states=50,
                       max_epochs=20, patience=5, ablation_mode='full'):
    device = get_device()
    data = data.to(device)
    
    logger.info(f"Starting Directional Run: Mode={ablation_mode}, Years={target_years}")
    
    for year in target_years:
        for month in range(1, 13):
            current_period_start = pd.Timestamp(year=year, month=month, day=1)
            if month == 12: next_period_start = pd.Timestamp(year=year+1, month=1, day=1)
            else: next_period_start = pd.Timestamp(year=year, month=month+1, day=1)
            
            train_end = current_period_start - pd.DateOffset(months=1)
            gap_start = train_end
            gap_end = current_period_start
            
            logger.info(f"\n=== DIRECTIONAL RETRAINING: {year}-{month:02d} | Mode: {ablation_mode} ===")
            
            # Split
            train_mask = df_filtered['Filed'] < gap_start
            gap_mask = (df_filtered['Filed'] >= gap_start) & (df_filtered['Filed'] < gap_end)
            test_mask = (df_filtered['Filed'] >= current_period_start) & (df_filtered['Filed'] < next_period_start)
            
            train_idx = np.where(train_mask)[0]
            gap_idx = np.where(gap_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            max_id = len(data.src)
            train_idx = [i for i in train_idx if i < max_id]
            test_idx = [i for i in test_idx if i < max_id]
            
            if len(test_idx) == 0:
                logger.warning(f"No trades in {year}-{month:02d}, skipping.")
                continue
                
            train_data = data[train_idx]
            gap_data = data[[i for i in gap_idx if i < max_id]]
            test_data = data[test_idx]
            
            logger.info(f"Train: {len(train_data.src)} | Gap: {len(gap_data.src)} | Test: {len(test_data.src)}")
            
            # Init
            model = TGN(
                num_nodes=num_nodes, raw_msg_dim=4, memory_dim=100, time_dim=100, embedding_dim=100,
                num_parties=num_parties, num_states=num_states
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
            criterion = torch.nn.BCEWithLogitsLoss()
            neighbor_loader = LastNeighborLoader(num_nodes, size=30, device=device)
            
            # Sub-split for Early Stopping
            sub_train_size = int(len(train_data.src) * 0.9)
            train_split = train_data[:sub_train_size]
            val_split = train_data[sub_train_size:]
            
            train_loader = TemporalDataLoader(train_split, batch_size=200, drop_last=True)
            val_loader = TemporalDataLoader(val_split, batch_size=200, drop_last=True)
            
            min_val_loss = float('inf')
            best_epoch = 1
            patience_counter = 0
            
            for epoch in range(1, max_epochs + 1):
                model.memory.reset_state()
                neighbor_loader.reset_state()
                model.train()
                
                for batch in train_loader:
                    batch = batch.to(device)
                    model.memory.detach()
                    optimizer.zero_grad()
                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    
                    # Target: Direction (Return > Alpha)
                    horizon_map = {'1M':0, '2M':1, '3M':2, '6M':3, '8M':4, '12M':5, '18M':6, '24M':7}
                    h_idx = horizon_map.get(args.horizon, 0)
                    h_seconds = {'1M':30, '2M':60, '3M':90, '6M':180, '8M':240, '12M':365, '18M':545, '24M':730}.get(args.horizon, 30) * 86400
                    
                    raw_return = batch.y[:, h_idx]
                    batch_targets = (raw_return > args.alpha).float()
                    
                    # Embedding & Memory logic
                    price_seq = batch.price_seq if hasattr(batch, 'price_seq') else torch.zeros((len(src), 14), device=device)
                    if ablation_mode == 'pol_only': price_seq = torch.zeros_like(price_seq)
                    price_emb = model.get_price_embedding(price_seq)
                    augmented_msg = torch.cat([msg, price_emb], dim=1)
                    
                    n_id = torch.cat([src, dst]).unique()
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                    assoc = dict(zip(n_id.tolist(), range(n_id.size(0))))
                    
                    # History processing
                    hist_msg = data.msg[e_id]
                    hist_price_emb = model.get_price_embedding(data.price_seq[e_id] if ablation_mode != 'pol_only' else torch.zeros_like(data.price_seq[e_id]))
                    
                    hist_raw_returns = data.y[e_id, h_idx]
                    hist_is_resolved = (data.trade_t[e_id] + h_seconds < t.max()).float().unsqueeze(-1)
                    hist_targets = (hist_raw_returns > args.alpha).float()
                    
                    isnan_mask = torch.isnan(hist_raw_returns)
                    hist_is_resolved[isnan_mask] = 0.0
                    hist_targets[isnan_mask] = 0.0
                    masked_label = hist_targets.unsqueeze(-1) * hist_is_resolved + 0.5 * (1 - hist_is_resolved)
                    
                    age_feat = torch.log1p((t.max() - data.t[e_id]).float() / 86400.0).unsqueeze(-1)
                    rel_t_enc = model.memory.time_enc((model.memory.last_update[n_id[edge_index[1]]] - data.t[e_id]).float())
                    
                    edge_attr = torch.cat([rel_t_enc, hist_msg, hist_price_emb, masked_label, age_feat], dim=-1)
                    z = model(n_id, edge_index, edge_attr)
                    
                    s_src = model.encode_static(data.x_static[src]) if ablation_mode != 'mkt_only' else torch.zeros((len(src), 64), device=device)
                    s_dst = model.encode_static(data.x_static[dst]) if ablation_mode != 'mkt_only' else torch.zeros((len(dst), 64), device=device)
                    
                    logits = model.predictor(z[[assoc[i] for i in src.tolist()]], z[[assoc[i] for i in dst.tolist()]], s_src, s_dst, price_emb, price_emb).view(-1)
                    
                    mask = ~torch.isnan(raw_return) & (batch.trade_t + h_seconds < t.max())
                    if mask.sum() > 0:
                        loss = criterion(logits[mask], batch_targets[mask])
                        loss.backward()
                        optimizer.step()
                        
                    model.memory.update_state(src, dst, t, augmented_msg)
                    neighbor_loader.insert(src, dst)
                    model.memory.detach()

                # Validation Loop
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                        raw_return = batch.y[:, h_idx]
                        batch_targets = (raw_return > args.alpha).float()
                        mask = ~torch.isnan(raw_return) & (batch.trade_t + h_seconds < t.max())
                        
                        price_seq = batch.price_seq if hasattr(batch, 'price_seq') else torch.zeros((len(src), 14), device=device)
                        if ablation_mode == 'pol_only': price_seq = torch.zeros_like(price_seq)
                        price_emb = model.get_price_embedding(price_seq)
                        augmented_msg = torch.cat([msg, price_emb], dim=1)
                        
                        n_id = torch.cat([src, dst]).unique()
                        n_id, edge_index, e_id = neighbor_loader(n_id)
                        assoc = dict(zip(n_id.tolist(), range(n_id.size(0))))
                        
                        # (Minimal history logic for val)
                        hist_msg = data.msg[e_id]
                        hist_price_emb = model.get_price_embedding(data.price_seq[e_id] if ablation_mode != 'pol_only' else torch.zeros_like(data.price_seq[e_id]))
                        hist_targets = (data.y[e_id, h_idx] > args.alpha).float()
                        hist_is_resolved = (data.trade_t[e_id] + h_seconds < t.max()).float().unsqueeze(-1)
                        masked_label = hist_targets.unsqueeze(-1) * hist_is_resolved + 0.5 * (1 - hist_is_resolved)
                        age_feat = torch.log1p((t.max() - data.t[e_id]).float() / 86400.0).unsqueeze(-1)
                        rel_t_enc = model.memory.time_enc((model.memory.last_update[n_id[edge_index[1]]] - data.t[e_id]).float())
                        
                        edge_attr = torch.cat([rel_t_enc, hist_msg, hist_price_emb, masked_label, age_feat], dim=-1)
                        z = model(n_id, edge_index, edge_attr)
                        s_src = model.encode_static(data.x_static[src]) if ablation_mode != 'mkt_only' else torch.zeros((len(src), 64), device=device)
                        s_dst = model.encode_static(data.x_static[dst]) if ablation_mode != 'mkt_only' else torch.zeros((len(dst), 64), device=device)
                        logits = model.predictor(z[[assoc[i] for i in src.tolist()]], z[[assoc[i] for i in dst.tolist()]], s_src, s_dst, price_emb, price_emb).view(-1)
                        
                        if mask.sum() > 0:
                            v_loss = criterion(logits[mask], batch_targets[mask])
                            val_losses.append(v_loss.item())
                        model.memory.update_state(src, dst, t, augmented_msg)
                        neighbor_loader.insert(src, dst)
                
                cur_val_loss = np.mean(val_losses) if val_losses else 0
                if cur_val_loss < min_val_loss:
                    min_val_loss = cur_val_loss
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience: break
            
            # RETRAIN (Phase 2)
            logger.info(f"  Best Epoch: {best_epoch}. Retraining Phase 2...")
            model = TGN(num_nodes=num_nodes, raw_msg_dim=4, memory_dim=100, time_dim=100, embedding_dim=100,
                        num_parties=num_parties, num_states=num_states).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
            neighbor_loader = LastNeighborLoader(num_nodes, size=30, device=device)
            full_train_loader = TemporalDataLoader(train_data, batch_size=200, drop_last=True)
            
            for ep in range(1, best_epoch + 1):
                model.memory.reset_state()
                neighbor_loader.reset_state()
                model.train()
                for batch in full_train_loader:
                    batch = batch.to(device)
                    model.memory.detach()
                    optimizer.zero_grad()
                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    raw_return = batch.y[:, h_idx]
                    batch_targets = (raw_return > args.alpha).float()
                    price_seq = batch.price_seq if hasattr(batch, 'price_seq') else torch.zeros((len(src), 14), device=device)
                    if ablation_mode == 'pol_only': price_seq = torch.zeros_like(price_seq)
                    augmented_msg = torch.cat([msg, model.get_price_embedding(price_seq)], dim=1)
                    
                    n_id = torch.cat([src, dst]).unique()
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                    assoc = dict(zip(n_id.tolist(), range(n_id.size(0))))
                    hist_msg = data.msg[e_id]
                    hist_price_emb = model.get_price_embedding(data.price_seq[e_id] if ablation_mode != 'pol_only' else torch.zeros_like(data.price_seq[e_id]))
                    hist_targets = (data.y[e_id, h_idx] > args.alpha).float()
                    hist_is_resolved = (data.trade_t[e_id] + h_seconds < t.max()).float().unsqueeze(-1)
                    masked_label = hist_targets.unsqueeze(-1) * hist_is_resolved + 0.5 * (1 - hist_is_resolved)
                    age_feat = torch.log1p((t.max() - data.t[e_id]).float() / 86400.0).unsqueeze(-1)
                    rel_t_enc = model.memory.time_enc((model.memory.last_update[n_id[edge_index[1]]] - data.t[e_id]).float())
                    
                    edge_attr = torch.cat([rel_t_enc, hist_msg, hist_price_emb, masked_label, age_feat], dim=-1)
                    z = model(n_id, edge_index, edge_attr)
                    s_src = model.encode_static(data.x_static[src]) if ablation_mode != 'mkt_only' else torch.zeros((len(src), 64), device=device)
                    s_dst = model.encode_static(data.x_static[dst]) if ablation_mode != 'mkt_only' else torch.zeros((len(src), 64), device=device)
                    
                    logits = model.predictor(z[[assoc[i] for i in src.tolist()]], z[[assoc[i] for i in dst.tolist()]], s_src, s_dst, model.get_price_embedding(price_seq), model.get_price_embedding(price_seq)).view(-1)
                    mask = ~torch.isnan(raw_return) & (batch.trade_t + h_seconds < t.max())
                    if mask.sum() > 0:
                        criterion(logits[mask], batch_targets[mask]).backward()
                        optimizer.step()
                    model.memory.update_state(src, dst, t, augmented_msg)
                    neighbor_loader.insert(src, dst)
                    model.memory.detach()

            # GAP (Update Memory only)
            gap_loader = TemporalDataLoader(gap_data, batch_size=200)
            model.eval()
            for batch in gap_loader:
                batch = batch.to(device)
                src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                price_seq = batch.price_seq if hasattr(batch, 'price_seq') else torch.zeros((len(src), 14), device=device)
                if ablation_mode == 'pol_only': price_seq = torch.zeros_like(price_seq)
                augmented_msg = torch.cat([msg, model.get_price_embedding(price_seq)], dim=1)
                model.memory.update_state(src, dst, t, augmented_msg)
                neighbor_loader.insert(src, dst)

            # TEST
            test_loader = TemporalDataLoader(test_data, batch_size=200)
            all_preds, all_targets = [], []
            for batch in test_loader:
                batch = batch.to(device)
                src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                raw_return = batch.y[:, h_idx]
                mask = ~torch.isnan(raw_return)
                if mask.sum() == 0: continue
                
                price_seq = batch.price_seq if hasattr(batch, 'price_seq') else torch.zeros((len(src), 14), device=device)
                if ablation_mode == 'pol_only': price_seq = torch.zeros_like(price_seq)
                augmented_msg = torch.cat([msg, model.get_price_embedding(price_seq)], dim=1)
                
                n_id = torch.cat([src, dst]).unique()
                n_id, edge_index, e_id = neighbor_loader(n_id)
                assoc = dict(zip(n_id.tolist(), range(n_id.size(0))))
                
                hist_msg = data.msg[e_id]
                hist_price_emb = model.get_price_embedding(data.price_seq[e_id] if ablation_mode != 'pol_only' else torch.zeros_like(data.price_seq[e_id]))
                hist_targets = (data.y[e_id, h_idx] > args.alpha).float()
                hist_is_resolved = (data.trade_t[e_id] + h_seconds < t.max()).float().unsqueeze(-1)
                masked_label = hist_targets.unsqueeze(-1) * hist_is_resolved + 0.5 * (1 - hist_is_resolved)
                age_feat = torch.log1p((t.max() - data.t[e_id]).float() / 86400.0).unsqueeze(-1)
                rel_t_enc = model.memory.time_enc((model.memory.last_update[n_id[edge_index[1]]] - data.t[e_id]).float())
                
                edge_attr = torch.cat([rel_t_enc, hist_msg, hist_price_emb, masked_label, age_feat], dim=-1)
                z = model(n_id, edge_index, edge_attr)
                s_src = model.encode_static(data.x_static[src]) if ablation_mode != 'mkt_only' else torch.zeros((len(src), 64), device=device)
                s_dst = model.encode_static(data.x_static[dst]) if ablation_mode != 'mkt_only' else torch.zeros((len(dst), 64), device=device)
                
                with torch.no_grad():
                    logits = model.predictor(z[[assoc[i] for i in src.tolist()]], z[[assoc[i] for i in dst.tolist()]], s_src, s_dst, model.get_price_embedding(price_seq), model.get_price_embedding(price_seq)).view(-1)
                    p = logits[mask].sigmoid().cpu().numpy()
                    all_preds.extend(p.tolist())
                    all_targets.extend((raw_return[mask] > args.alpha).float().cpu().numpy().tolist())
                
                model.memory.update_state(src, dst, t, augmented_msg)
                neighbor_loader.insert(src, dst)

            # Metrics & Save
            if all_targets:
                y_true = np.array(all_targets)
                y_prob = np.array(all_preds)
                y_pred = (y_prob > 0.5).astype(int)
                
                auc = roc_auc_score(y_true, y_prob)
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                
                logger.info(f"  [RESULT] {year}-{month:02d}: AUC={auc:.4f} | Acc={acc:.4f}")
                
                # Probs
                probs_dir = f"{args.exp_dir}/probs"
                os.makedirs(probs_dir, exist_ok=True)
                with open(f"{probs_dir}/probs_{ablation_mode}_{year}_{month:02d}.json", 'w') as f:
                    json.dump({'y_true': y_true.tolist(), 'y_prob': y_prob.tolist()}, f)
                
                # Report
                report_dir = f"{args.exp_dir}/reports"
                os.makedirs(report_dir, exist_ok=True)
                rep_dict = classification_report(y_true, y_pred, output_dict=True)
                rep_dict['auc'] = auc
                with open(f"{report_dir}/report_{ablation_mode}_{year}_{month:02d}.json", 'w') as f:
                    json.dump(rep_dict, f, indent=4)
                
                # Summary CSV
                summary_path = f"{args.exp_dir}/summary_{ablation_mode}.csv"
                needs_header = not os.path.exists(summary_path)
                with open(summary_path, 'a') as f:
                    if needs_header: f.write("Model,Year,Month,Train_Size,Test_Size,Accuracy,F1_Class1,AUC\n")
                    f.write(f"{ablation_mode},{year},{month},{len(train_data.src)},{len(test_data.src)},{acc:.4f},{f1:.4f},{auc:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-run", action="store_true")
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--horizon", type=str, default="1M")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--year", type=int)
    args = parser.parse_args()
    
    args.exp_dir = f"results/directional_tgn/H_{args.horizon}_A_{args.alpha}"
    os.makedirs(args.exp_dir, exist_ok=True)
    
    data = torch.load("data/temporal_data.pt", weights_only=False)
    num_nodes = int(torch.cat([data.src, data.dst]).max()) + 1
    
    # Remove int attributes to avoid PyG slicing errors
    if hasattr(data, 'num_nodes'): del data.num_nodes
    if hasattr(data, 'num_parties'): del data.num_parties
    if hasattr(data, 'num_states'): del data.num_states
    
    from src.config import TX_PATH
    df = pd.read_csv(TX_PATH)
    df['Filed'] = pd.to_datetime(df['Filed'])
    df = df.sort_values('Filed').reset_index(drop=True)
    df = df[df['Ticker'].notnull() & df['Filed'].notnull()].reset_index(drop=True)
    
    target_years = [args.year] if args.year else [2019, 2020, 2021, 2022, 2023, 2024]
    modes = ['full'] if args.full_only else ['pol_only', 'mkt_only', 'full']
    
    for mode in modes:
        train_and_evaluate(data, df, args=args, target_years=target_years, num_nodes=num_nodes, ablation_mode=mode)
