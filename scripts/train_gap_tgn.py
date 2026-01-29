import argparse
import logging
import torch
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Note: Ensure src/gap_tgn.py contains class GAPTGN. 
# If you renamed it to ResearchTGN, update the import below.
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

    df = pd.read_csv(TX_PATH)
    
    # Map horizon string to index
    horizons = ['1M', '2M', '3M', '6M', '8M', '12M', '18M', '24M']
    if horizon not in horizons:
        raise ValueError(f"Invalid horizon: {horizon}. Must be one of {horizons}")
    h_idx = horizons.index(horizon)
    
    # Build Graph Data
    logger.info("Constructing Temporal Graph...")
    builder = TemporalGraphBuilder(df, min_freq=2)
    data = builder.process()
    
    # --- FIX: ALIGN MASKS WITH GRAPH DATA ---
    # The builder may skip transactions (e.g. missing IDs). 
    # We must rebuild dates from the graph's internal timestamps (data.t) to ensure size matches.
    
    # 1. Get Base Time from original DF (used by builder for normalization)
    base_time = pd.to_datetime(df['Filed'].min()).timestamp()
    
    # 2. Reconstruct Absolute Dates for the events in the graph
    # data.t contains 'seconds since base_time'
    t_numpy = data.t.numpy() 
    absolute_ts = t_numpy + base_time
    event_dates = pd.to_datetime(absolute_ts, unit='s')
    
    # 3. Create Masks based on aligned dates
    # Train: 2015-2022
    # Test: 2023-2024+
    train_mask_np = (event_dates.year >= 2015) & (event_dates.year <= 2022)
    test_mask_np = (event_dates.year >= 2023)
    
    train_mask = torch.tensor(train_mask_np)
    test_mask = torch.tensor(test_mask_np)
    
    # Extract Labels
    labels_all = data.y[:, h_idx]
    
    # Filter valid labels (not NaN)
    valid_label_mask = ~torch.isnan(labels_all)
    
    # Intersection of Date Split & Valid Labels
    train_idx = torch.where(train_mask & valid_label_mask)[0]
    test_idx = torch.where(test_mask & valid_label_mask)[0]
    
    logger.info(f"Graph Events: {len(data.t)}")
    logger.info(f"Train Samples: {len(train_idx)}, Test Samples: {len(test_idx)}")
    
    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    edge_dim = data.msg.shape[1]
    
    model = GAPTGN(
        edge_feat_dim=edge_dim,
        price_seq_dim=14,
        hidden_dim=hidden_dim,
        num_nodes=data.num_nodes,
        num_parties=data.num_parties,
        num_states=data.num_states
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Move data to device
    data.src = data.src.to(device)
    data.dst = data.dst.to(device)
    data.t = data.t.to(device)
    data.msg = data.msg.to(device)
    data.price_seq = data.price_seq.to(device)
    data.trade_t = data.trade_t.to(device)
    data.x_static = data.x_static.to(device)
    labels_all = labels_all.to(device)
    
    # Training Loop
    logger.info("Starting Training...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward Pass
        out = model(
            data.src, data.dst, data.t, data.msg, 
            data.price_seq, data.trade_t, data.x_static
        )
        
        # Loss only on Train
        loss = criterion(out[train_idx], labels_all[train_idx])
        loss.backward()
        optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            # Test metrics
            if len(test_idx) > 0:
                probs = torch.sigmoid(out[test_idx]).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                y_true = labels_all[test_idx].cpu().numpy()
                
                acc = accuracy_score(y_true, preds)
                auc = roc_auc_score(y_true, probs)
            else:
                acc, auc = 0.0, 0.0
            
            logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Test ACC: {acc:.4f} | Test AUC: {auc:.4f}")

    # Save Results
    results_path = Path(RESULTS_DIR) / f"gap_tgn_{horizon}.pt"
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