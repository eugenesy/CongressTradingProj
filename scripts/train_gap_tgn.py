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
    horizons = ['1M', '2M', '3M', '6M', '8M', '12M', '18M', '24M']
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
        
        # 3. Compute Loss
        loss = criterion(out[train_idx], labels_binary[train_idx])
        loss.backward()
        optimizer.step()

        # 4. Update Memory for next step
        with torch.no_grad():
            model.update_memory(data.src, data.dst, data.t, data.msg)
        
        # 5. Detach memory
        model.memory.detach()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            if len(test_idx) > 0:
                # Need to re-run forward pass or just use 'out' if strict causality is maintained
                # For safety/correctness in TGN evaluation, usually we just check the 'out' 
                # we already computed, because the TGN is causal by definition.
                logits = out[test_idx]
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                y_true = labels_binary[test_idx].cpu().numpy()
                
                acc = accuracy_score(y_true, preds)
                try:
                    auc = roc_auc_score(y_true, probs)
                except:
                    auc = 0.5
            else:
                acc, auc = 0.0, 0.0
            
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