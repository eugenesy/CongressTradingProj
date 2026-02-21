import logging
import os
import sys
from pathlib import Path
from typing import Tuple
from tqdm import tqdm

import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report

# Production imports
sys.path.append(os.getcwd())
try:
    from src.gap_tgn import ResearchTGN
except ImportError:
    # Fallback
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from src.gap_tgn import ResearchTGN

from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader
from torch_geometric.data import TemporalData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("research")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def direction_targets(raw_return: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    has_label = ~torch.isnan(raw_return)
    up = raw_return > alpha
    down = raw_return < alpha
    mask = has_label & (up | down)
    return up.float(), mask

def run_tgn_study(horizon='6M', alpha=0.0, epochs=5, start_year=2023, end_year=2023, seed=42, out_dir="results/experiments"):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Loading Data...")
    try:
        data = torch.load("data/temporal_data.pt", weights_only=False).to(device)
    except Exception:
        data = torch.load("data/temporal_data.pt").to(device)

    df = pd.read_csv("data/processed/ml_dataset_clean.csv")
    df['Filed'] = pd.to_datetime(df['Filed'])
    df['Traded'] = pd.to_datetime(df['Traded'])
    df = df.sort_values('Filed').reset_index(drop=True)

    # Note: df length corresponds to ONLY the trades, not the full graph (which has millions of influence edges)
    # The TemporalData loader handles the full graph, but we need to ensure the test_df aligns correctly.
    # Since our test loop indexes using the batch size, we only map the trade indices back to the df.

    # --- NEW: Calculate True Global Time Anchor ---
    global_t_base = df['Filed'].min().timestamp()
    if os.path.exists("data/processed/events_lobbying.csv"):
        lobby_dates = pd.read_csv("data/processed/events_lobbying.csv", usecols=['date'])
        global_t_base = min(global_t_base, pd.to_datetime(lobby_dates['date']).min().timestamp())
        
    if os.path.exists("data/processed/events_campaign_finance.csv"):
        camp_dates = pd.read_csv("data/processed/events_campaign_finance.csv", usecols=['date'])
        global_t_base = min(global_t_base, pd.to_datetime(camp_dates['date']).min().timestamp())

    trade_mask = (data.msg[:, 3] == 0)
    min_trade_t = data.t[trade_mask].min().item()
    min_trade_real = (df['Filed'].astype('int64') // 10**9).min()
    global_t_base = min_trade_real - min_trade_t
    # ----------------------------------------------

    print("\n--- SANITY CHECK DIAGNOSTICS ---")
    print(f"Total Edges in Graph: {data.t.size(0)}")
    print(f"Graph t min: {data.t.min().item()}")
    print(f"Graph t max: {data.t.max().item()}")
    print(f"Min Trade t in Graph: {min_trade_t}")
    print(f"Min Trade Real (Unix): {min_trade_real}")
    print(f"Calculated Global Base: {global_t_base}")
    
    test_ts = pd.Timestamp(start_year, 1, 1).timestamp()
    print(f"Test Start {start_year}-01 Real (Unix): {test_ts}")
    print(f"Test Start {start_year}-01 Graph t: {int(test_ts - global_t_base)}")
    print("--------------------------------\n")

    num_nodes = getattr(data, 'num_nodes', int(torch.cat([data.src, data.dst]).max().item()) + 1)
    raw_msg_dim = int(data.msg.size(-1))
    
    # Check for Static Features
    if hasattr(data, 'x_static'):
        num_parties = getattr(data, 'num_parties', int(data.x_static[:, 0].max().item()) + 1)
        num_states = getattr(data, 'num_states', int(data.x_static[:, 1].max().item()) + 1)
    else:
        num_parties = getattr(data, 'num_parties', 1)
        num_states = getattr(data, 'num_states', 1)

    # Check for Dynamic Features (Enhanced Mode)
    pol_dim = data.x_pol.size(1) if hasattr(data, 'x_pol') else 0
    comp_dim = data.x_comp.size(1) if hasattr(data, 'x_comp') else 0
    if pol_dim > 0:
        logger.info(f"Detected Dynamic Features: Pol_Dim={pol_dim}, Comp_Dim={comp_dim}")

    horizon_map = {'1M': 0, '2M': 1, '3M': 2, '6M': 3, '8M': 4, '12M': 5, '18M': 6, '24M': 7}
    if horizon not in horizon_map:
        raise ValueError(f"Unsupported horizon: {horizon}")
    h_idx = horizon_map[horizon]
    h_days = {'1M': 30, '2M': 60, '3M': 90, '6M': 180, '8M': 240, '12M': 365, '18M': 545, '24M': 730}[horizon]
    h_seconds = h_days * 86400

    results = []
    predictions_log = []
    
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    def slice_data_by_time(mask_start, mask_end, gap_mode=False):
        # We slice based on the timestamp of the edges
        t_sec = data.t.cpu().numpy() + data.t.min().item() # Approx true timestamp depending on normalization
        # Note: Since slicing millions of edges manually is complex, we rely on the temporal index.
        # For full graph, we must convert test_start and next_month to the normalized 't' scale.
        t_base = pd.to_datetime('2014-01-01').timestamp() # Fallback, assumes base_time
        
        start_t = (mask_start.timestamp() - t_base) if mask_start else 0
        end_t = (mask_end.timestamp() - t_base) if mask_end else float('inf')
        
        idx = torch.where((data.t >= start_t) & (data.t < end_t))[0]
        
        if idx.numel() == 0:
             return TemporalData(
                src=torch.empty((0,), dtype=torch.long, device=device),
                dst=torch.empty((0,), dtype=torch.long, device=device),
                t=torch.empty((0,), dtype=torch.long, device=device),
                msg=torch.empty((0, data.msg.size(-1)), dtype=torch.float, device=device),
                y=torch.empty((0, data.y.size(-1)), dtype=torch.float, device=device),
                price_seq=torch.empty((0, data.price_seq.size(-1)), dtype=torch.float, device=device),
                trade_t=torch.empty((0,), dtype=torch.long, device=device),
            )
             
        sliced = TemporalData(
            src=data.src[idx],
            dst=data.dst[idx],
            t=data.t[idx],
            msg=data.msg[idx],
            y=data.y[idx],
            price_seq=data.price_seq[idx],
            trade_t=data.trade_t[idx],
        )
        if hasattr(data, 'x_pol'): sliced.x_pol = data.x_pol[idx]
        if hasattr(data, 'x_comp'): sliced.x_comp = data.x_comp[idx]
        return sliced

    for year in range(start_year, end_year + 1):
        yearly_preds = []
        yearly_targets = []
        
        for month in range(1, 13):
            logger.info(f"\n--- STUDY: {year}-{month:02d} | Horizon: {horizon} ---")

            test_start = pd.Timestamp(year, month, 1)
            next_month = test_start + pd.DateOffset(months=1)
            
            # Use temporal index logic for the full graph
            t_base = df['Filed'].min().timestamp()
            test_start_t = int(test_start.timestamp() - global_t_base)
            next_month_t = int(next_month.timestamp() - global_t_base)
            
            # Slicing the full graph tensor directly
            train_idx = torch.where((data.t < test_start_t) & ((data.trade_t + h_seconds) < test_start_t))[0]
            gap_idx = torch.where((data.t < test_start_t) & ((data.trade_t + h_seconds) >= test_start_t))[0]
            test_idx = torch.where((data.t >= test_start_t) & (data.t < next_month_t))[0]

            def make_slice(idx):
                if idx.numel() == 0:
                    return TemporalData(src=torch.empty((0,), dtype=torch.long, device=device), dst=torch.empty((0,), dtype=torch.long, device=device), t=torch.empty((0,), dtype=torch.long, device=device), msg=torch.empty((0, data.msg.size(-1)), dtype=torch.float, device=device), y=torch.empty((0, data.y.size(-1)), dtype=torch.float, device=device), price_seq=torch.empty((0, data.price_seq.size(-1)), dtype=torch.float, device=device), trade_t=torch.empty((0,), dtype=torch.long, device=device))
                sliced = TemporalData(src=data.src[idx], dst=data.dst[idx], t=data.t[idx], msg=data.msg[idx], y=data.y[idx], price_seq=data.price_seq[idx], trade_t=data.trade_t[idx])
                if hasattr(data, 'x_pol'): sliced.x_pol = data.x_pol[idx]
                if hasattr(data, 'x_comp'): sliced.x_comp = data.x_comp[idx]
                return sliced

            train_data = make_slice(train_idx)
            gap_data = make_slice(gap_idx)
            test_data = make_slice(test_idx)

            if len(test_data.src) == 0:
                logger.warning("No data for %s, skipping.", test_start.strftime('%Y-%m'))
                continue

            raw_train = train_data.y[:, h_idx]
            train_targets, train_label_mask = direction_targets(raw_train, alpha)
            train_resolved = (train_data.trade_t + h_seconds) < train_data.t.max()
            
            # --- PHASE 3.3 MASKING: Only consider Trade Edges (edge_type == 0) ---
            train_is_trade = (train_data.msg[:, 3] == 0)
            train_mask = train_label_mask & train_resolved & train_is_trade

            if train_mask.sum() == 0:
                logger.warning("No resolved training labels for %s, skipping.", test_start.strftime('%Y-%m'))
                continue

            pos = train_targets[train_mask].sum().item()
            neg = train_mask.sum().item() - pos
            pos_weight = neg / max(1, pos)

            model = ResearchTGN(
                num_nodes=num_nodes,
                raw_msg_dim=raw_msg_dim,
                memory_dim=100,
                time_dim=100,
                embedding_dim=100,
                num_parties=num_parties,
                num_states=num_states,
                pol_dim=pol_dim,  
                comp_dim=comp_dim
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
            neighbor_loader = LastNeighborLoader(num_nodes, size=30, device=device)

            train_loader = TemporalDataLoader(train_data, batch_size=2048, drop_last=True)
            gap_loader = TemporalDataLoader(gap_data, batch_size=2048)

            for epoch in range(1, epochs + 1):
                model.train()
                model.memory.reset_state()
                neighbor_loader.reset_state()
                model.memory.detach()
                epoch_loss = []

                for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Training", leave=False):
                    batch = batch.to(device)
                    optimizer.zero_grad()

                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    raw_y = batch.y[:, h_idx]
                    targets, label_mask = direction_targets(raw_y, alpha)
                    batch_max_t = t.max()
                    resolved_mask = (batch.trade_t + h_seconds) < batch_max_t
                    
                    # --- MASK: Only calculate loss on Trades ---
                    is_trade = (msg[:, 3] == 0)
                    train_batch_mask = label_mask & resolved_mask & is_trade

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
                    
                    x_pol_batch = getattr(batch, 'x_pol', None)
                    x_comp_batch = getattr(batch, 'x_comp', None)
                    s_src = model.encode_node_features(data.x_static[src], x_dynamic=x_pol_batch, mode='pol')
                    s_dst = model.encode_node_features(data.x_static[dst], x_dynamic=x_comp_batch, mode='comp')
                    
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
                        for batch in tqdm(gap_loader, desc=f"Epoch {epoch}/{epochs} Gap Forward", leave=False):
                            batch = batch.to(device)
                            src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                            p_context = model.get_price_embedding(batch.price_seq)
                            aug_msg = torch.cat([msg, p_context], dim=1)
                            model.memory.update_state(src, dst, t, aug_msg)
                            neighbor_loader.insert(src, dst)

                logger.info("    Epoch %s/%s | Loss: %.4f", epoch, epochs, float(np.mean(epoch_loss)) if epoch_loss else 0.0)

            model.eval()
            test_loader = TemporalDataLoader(test_data, batch_size=2048)
            all_preds, all_targets = [], []

            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Epoch {epoch}/{epochs} Gap Forward", leave=False):
                    batch = batch.to(device)
                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    raw_y = batch.y[:, h_idx]
                    targets, label_mask = direction_targets(raw_y, alpha)
                    
                    # --- MASK: Only track Test metrics on Trades ---
                    is_trade = (msg[:, 3] == 0)
                    test_batch_mask = label_mask & is_trade
                    
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
                    
                    x_pol_batch = getattr(batch, 'x_pol', None)
                    x_comp_batch = getattr(batch, 'x_comp', None)
                    s_src = model.encode_node_features(data.x_static[src], x_dynamic=x_pol_batch, mode='pol')
                    s_dst = model.encode_node_features(data.x_static[dst], x_dynamic=x_comp_batch, mode='comp')
                    
                    preds = model.predictor(z_src, z_dst, s_src, s_dst, p_context).sigmoid()

                    if test_batch_mask.sum() > 0:
                        batch_preds = preds[test_batch_mask].cpu().numpy().flatten()
                        batch_targets = targets[test_batch_mask].cpu().numpy().flatten()
                        all_preds.extend(batch_preds)
                        all_targets.extend(batch_targets)
                        
                        # Store in log
                        for i, is_valid in enumerate(test_batch_mask.cpu().numpy()):
                            if is_valid:
                                rec = {
                                    'Date_Norm': t[i].item(),
                                    'Model': 'TGN',
                                    'True_Up': targets[i].item(),
                                    'Prob_Up': preds[i].item(),
                                    'Pred_Up': 1.0 if preds[i].item() > 0.5 else 0.0
                                }
                                predictions_log.append(rec)

                    model.memory.update_state(src, dst, t, aug_msg)
                    neighbor_loader.insert(src, dst)

            if len(all_targets) == 0:
                logger.warning("No labeled targets for %s, skipping.", test_start.strftime('%Y-%m'))
                continue

            res_auc = roc_auc_score(all_targets, all_preds) if len(set(all_targets)) > 1 else 0.5
            res_acc = accuracy_score(all_targets, np.array(all_preds) > 0.5)
            res_f1 = f1_score(all_targets, np.array(all_preds) > 0.5)
            count = len(all_targets)

            logger.info("  [RESULT %d-%02d]: AUC=%.4f | Acc=%.4f | F1=%.4f | Count=%s", 
                        year, month, res_auc, res_acc, res_f1, count)
            results.append({'Year': year, 'Month': month, 'AUC': res_auc, 'Acc': res_acc, 'F1': res_f1, 'Count': count})
            yearly_preds.extend(all_preds)
            yearly_targets.extend(all_targets)

        if yearly_targets:
            report = classification_report(yearly_targets, np.array(yearly_preds) > 0.5, target_names=['Down', 'Up'], zero_division=0)
            logger.info(f"\n--- Classification Report {year} ---\n{report}")
            with open(out_path / f"report_{year}_{horizon}.txt", "w") as f:
                f.write(report)

    res_df = pd.DataFrame(results)
    print("\n" + "=" * 40)
    print(f"RESEARCH SUMMARY: Horizon={horizon}")
    print("=" * 40)
    if not res_df.empty:
        print(res_df.to_string(index=False))
        print(f"\nMEAN AUC: {res_df.AUC.mean():.4f}")
    else:
        print("No results generated.")

    res_df.to_csv(out_path / f"study_{horizon}.csv", index=False)
    pd.DataFrame(predictions_log).to_csv(out_path / f"predictions_tgn_{horizon}.csv", index=False)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='6M')
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--start-year', type=int, default=2023)
    parser.add_argument('--end-year', type=int, default=2023)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', default='results/experiments')
    args = parser.parse_args()

    run_tgn_study(
        horizon=args.horizon,
        alpha=args.alpha,
        epochs=args.epochs,
        start_year=args.start_year,
        end_year=args.end_year,
        seed=args.seed,
        out_dir=args.out_dir,
    )

if __name__ == "__main__":
    main()