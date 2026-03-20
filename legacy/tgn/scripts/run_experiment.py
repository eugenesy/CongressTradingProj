import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, classification_report, precision_score, recall_score
import sys
import os
import logging
import datetime

# Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..')) # Access root for config if needed



from clean_tgn.src.models_basic import BasicTGN
from clean_tgn.src.temporal_data import TemporalGraphBuilder
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader

# Config
LOG_FILE = os.path.join(BASE_DIR, "experiment.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger()
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

from src.config import TX_PATH # Use shared data source

def train():
    logger.info("="*50)
    logger.info("STARTING NEW EXPERIMENT RUN: Phase 1 (Basic TGN)")
    logger.info("="*50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # DEBUG: Enable Anomaly Detection
    torch.autograd.set_detect_anomaly(True)

    # 1. Load Data (Filter for 2023 for initial test)
    logger.info("Loading Data...")
    df = pd.read_csv(TX_PATH)
    
    # Strict Date Handling
    df['Traded'] = pd.to_datetime(df['Traded'])
    df['Filed'] = pd.to_datetime(df['Filed'])
    df['Filed_DT'] = df['Filed'] # For builder

    # Filter: LOAD FULL HISTORY (2015-2025)
    # We need full history to calculate "Rolling Win Rate" correctly (Cold Start Fix)
    # df_2023 = df[mask].copy() # REMOVED
    
    # Just use the full df (sorted)
    df = df.sort_values('Filed').reset_index(drop=True)
    logger.info(f"Loaded Full History: {len(df)} rows")
    
    # 2. Build Graph (Full History)
    builder = TemporalGraphBuilder(df, min_freq=1)
    data, x_static, num_parties, num_states, num_chambers = builder.process(horizon="1M", threshold=0.01)
    
    # DEBUG
    print(f"Data Keys: {data.keys()}")
    
    # FIX: Extract scalars before slicing because PyG TemporalData slicing breaks on int attributes
    num_nodes = data.num_nodes
    # num_parties and num_states are now passed directly
    
    # Cleanup data object for slicing
    del data.num_nodes
    # del data.num_parties # Not attached
    # del data.num_states
    # del data.x_static
    
    data = data.to(device)
    x_static = x_static.to(device) # FIX: Move to device for slicing

    # 3. Model
    model = BasicTGN(
        num_nodes=num_nodes,
        raw_msg_dim=4, # UPDATED: [Amt, Buy, Gap, WinRate]
        memory_dim=100,
        time_dim=100,
        embedding_dim=100,
        num_parties=num_parties,
        num_states=num_states,
        num_chambers=num_chambers
    ).to(device)
    
    # Loss: Standard BCE - classes are balanced enough (41%/59%)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # 4. ROLLING EVALUATION LOOP (2023)
    # We want to test each month of 2023 separately, retraining on all prior data (Expanding Window)
    
    # Define Months
    months = [
        ('2023-01-01', '2023-02-01'),
        ('2023-02-01', '2023-03-01'),
        ('2023-03-01', '2023-04-01'),
        ('2023-04-01', '2023-05-01'),
        ('2023-05-01', '2023-06-01'),
        ('2023-06-01', '2023-07-01'),
        ('2023-07-01', '2023-08-01'),
        ('2023-08-01', '2023-09-01'),
        ('2023-09-01', '2023-10-01'),
        ('2023-10-01', '2023-11-01'),
        ('2023-11-01', '2023-12-01'),
        ('2023-12-01', '2024-01-01')
    ]
    
    # Determine base timestamps for cutting
    base_t = pd.to_datetime(df['Filed'].min()).timestamp() - 86400
    
    overall_preds = []
    overall_targets = []
    
    # Initialize neighbor_loader once outside the loop, reset its state inside
    neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)

    for start_str, end_str in months:
        logger.info(f"--- Processing Month: {start_str} to {end_str} ---")
        
        # Calculate timestamps
        t_start = pd.to_datetime(start_str).timestamp() - base_t
        t_end = pd.to_datetime(end_str).timestamp() - base_t
        
        # Masks
        train_mask = data.t < t_start
        test_mask = (data.t >= t_start) & (data.t < t_end)
        
        if test_mask.sum() == 0:
            logger.info("No events in test month. Skipping.")
            continue
            
        train_data = data[train_mask]
        test_data = data[test_mask]
        
        logger.info(f"Train History: {len(train_data)} | Test Events: {len(test_data)}")
        
        # Reset Model/Optimizer for Retraining (Concept Drift handling: start fresh or finetune?)
        # User said "retrain model monthly". Let's reset weights to avoid over-fitting to old history?
        # Actually, iterating from scratch is safer for "Retraining".
        model.apply(lambda m: hasattr(m, 'reset_parameters') and m.reset_parameters())
        # Re-init optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        train_loader = TemporalDataLoader(train_data, batch_size=200)
        test_loader = TemporalDataLoader(test_data, batch_size=200)
        neighbor_loader.reset_state() # Reset graph state
    
        # Train Loop (Short Epochs since we Retrain? Or full?)
        # Full training ensures we capture the new history.
        epochs = 10  # Increased from 5 to allow larger model to converge
        
        for epoch in range(1, epochs + 1):
            model.train()
            model.memory.reset_state()
            neighbor_loader.reset_state()
            
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                n_id = torch.cat([src, dst]).unique()
                n_id, edge_index, e_id = neighbor_loader(n_id)
                assoc = dict(zip(n_id.tolist(), range(n_id.size(0))))
                
                hist_t = data.t[e_id]
                hist_msg = data.msg[e_id]
                hist_msg = data.msg[e_id]
                target_nodes = n_id[edge_index[1]]
                # CLONE and DETACH last_update to prevent inplace modification error
                last_update = model.memory.last_update[target_nodes].clone().detach()
                rel_t = last_update - hist_t
                rel_t_enc = model.memory.time_enc(rel_t.to(torch.float))
                masked_label = torch.zeros((len(e_id), 1), device=device)
                edge_attr = torch.cat([rel_t_enc, hist_msg, masked_label], dim=-1)
                
                # Extract price sequences for current batch edges
                hist_price_seq = data.price_seq[e_id]
                z = model(n_id, edge_index, edge_attr, price_seq=hist_price_seq)
                z_src = z[[assoc[i] for i in src.tolist()]]
                z_dst = z[[assoc[i] for i in dst.tolist()]]
                s_src = model.encode_static(x_static[src])
                s_dst = model.encode_static(x_static[dst])
                
                # Get price embeddings for this batch
                batch_price_seq = batch.price_seq
                price_emb = model.price_encoder(batch_price_seq)
                
                # Node Identity REMOVED (Inductive)
                # id_src = model.encode_id(src)
                # id_dst = model.encode_id(dst)
                
                # Predictor: pass price_emb directly (like old working model)
                pred = model.predictor(z_src, z_dst, s_src, s_dst, price_emb, price_emb)
                loss = criterion(pred.squeeze(), batch.y.float())
                loss.backward()
                optimizer.step()
                
                # Detach memory update from graph
                with torch.no_grad():
                    # Augment msg with price embeddings for memory
                    if model.use_price:
                        batch_price_seq = batch.price_seq
                        price_emb = model.price_encoder(batch_price_seq)
                        msg_augmented = torch.cat([msg, price_emb], dim=-1)
                    else:
                        msg_augmented = msg
                    model.memory.update_state(src, dst, t, msg_augmented)
                    neighbor_loader.insert(src, dst)
                    model.memory.detach()
                
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                 logger.info(f"Ep {epoch} Loss: {total_loss / len(train_loader):.4f}")

        # Test
        model.eval()
        # Important: Test phase must continue memory update ONLY valid at test time?
        # Standard TGN eval: We update memory AFTER prediction to carry state forward?
        # No, for pure backtesting:
        # We predict t_test. We rely on memory built during training (up to t_train_end).
        # But within the test month, transactions happen sequentially.
        # Should we update memory during test? Yes, transductive.
        
        # But simpler: Just predict.
        # We need to populate memory up to start of test set first!
        # The training loop leaves memory at end of training set. So we are good.
        
        month_preds = []
        month_targets = []
        
        for batch in tqdm(test_loader, desc=f"Testing {start_str}"):
            batch = batch.to(device)
            src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc = dict(zip(n_id.tolist(), range(n_id.size(0))))
            
            hist_t = data.t[e_id]
            hist_msg = data.msg[e_id]
            target_nodes = n_id[edge_index[1]]
            last_update = model.memory.last_update[target_nodes]
            rel_t = last_update - hist_t
            rel_t_enc = model.memory.time_enc(rel_t.to(torch.float))
            masked_label = torch.zeros((len(e_id), 1), device=device)
            edge_attr = torch.cat([rel_t_enc, hist_msg, masked_label], dim=-1)
            
            # Extract price sequences for test batch edges
            hist_price_seq = data.price_seq[e_id]
            z = model(n_id, edge_index, edge_attr, price_seq=hist_price_seq)
            z_src = z[[assoc[i] for i in src.tolist()]]
            z_dst = z[[assoc[i] for i in dst.tolist()]]
            s_src = model.encode_static(x_static[src])
            s_dst = model.encode_static(x_static[dst])
            
            # Get price embeddings for test batch
            batch_price_seq = batch.price_seq
            price_emb = model.price_encoder(batch_price_seq)
            
            # id_src = model.encode_id(src)
            # id_dst = model.encode_id(dst)
            
            # Predictor: pass price_emb directly (like old working model)
            pred = model.predictor(z_src, z_dst, s_src, s_dst, price_emb, price_emb).sigmoid()
            
            batch_preds = pred.cpu().detach().numpy()
            batch_targets = batch.y.cpu().numpy()
            
            month_preds.extend(batch_preds)
            month_targets.extend(batch_targets)
            overall_preds.extend(batch_preds)
            overall_targets.extend(batch_targets)
            
            # Update Memory with Test events (transductive testing)
            # Augment msg with price embeddings before updating memory
            if model.use_price:
                batch_price_seq = batch.price_seq
                price_emb = model.price_encoder(batch_price_seq)
                msg_augmented = torch.cat([msg, price_emb], dim=-1)
            else:
                msg_augmented = msg
            model.memory.update_state(src, dst, t, msg_augmented)
            neighbor_loader.insert(src, dst)

        # Log Monthly
        if len(month_targets) > 0:
            mean_pred = np.mean(month_preds)
            mean_target = np.mean(month_targets)
            std_pred = np.std(month_preds)
            std_target = np.std(month_targets)
            
            try:
                m_auc = roc_auc_score(month_targets, month_preds)
                m_prec = precision_score(month_targets, np.array(month_preds) > 0.5, zero_division=0)
                m_rec = recall_score(month_targets, np.array(month_preds) > 0.5, zero_division=0)
                m_f1_macro = f1_score(month_targets, np.array(month_preds) > 0.5, average='macro', zero_division=0)
                
                logger.info(f"Result {start_str}: AUC={m_auc:.4f} | Pr={m_prec:.2f} Rc={m_rec:.2f} F1(macro)={m_f1_macro:.2f} | AvgPred={mean_pred:.2f}±{std_pred:.2f} vs AvgTgt={mean_target:.2f}±{std_target:.2f} (N={len(month_targets)})")
            except Exception as e:
                logger.info(f"Result {start_str}: AUC=Error ({e}) | AvgPred={mean_pred:.2f}±{std_pred:.2f} vs AvgTgt={mean_target:.2f}±{std_target:.2f}")

    # Final Overall
    params = (np.array(overall_preds) > 0.5).astype(int)
    f1 = f1_score(overall_targets, params)
    f1_macro = f1_score(overall_targets, params, average='macro')
    auc = roc_auc_score(overall_targets, overall_preds)
    logger.info(f"FINAL Rolling 2023 Result: F1={f1:.4f} | F1(macro)={f1_macro:.4f} | AUC={auc:.4f}")
    
    # Report
    print(classification_report(overall_targets, params))

if __name__ == "__main__":
    train()
