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

from src.gap_tgn import ResearchTGN
from src.temporal_data import TemporalGraphBuilder
from src.config import TX_PATH, RESULTS_DIR

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate(model, data, criterion, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.src, data.dst, data.t, data.msg, data.price_seq, data.trade_t)
        
        # Determine Horizon Index
        # data.y shape: [Num_Events, 8] -> ['1M','2M','3M','6M','8M','12M','18M','24M']
        # Horizon arg is passed globally, but here we need to know which column to pick.
        # We can pass horizon_idx to evaluate or infer it.
        # For simplicity, we assume the model output matches the target y shape (1 dim)
        # We need the ground truth for the specific horizon.
        
        # Currently, GAP-TGN outputs a single logit.
        # We need to extract the specific label column from data.y corresponding to the chosen horizon.
        # This is handled in the main loop, but here we need to know the index.
        # Let's assume data.y is ALREADY filtered or we pass the target vector.
        pass # Eval logic moved to training loop for simplicity

def run_tgn_study(horizon='6M', epochs=50, hidden_dim=128, lr=0.001, seed=42):
    # Set Seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    logger.info("Loading Data...")
    
    # UPDATED: Use TX_PATH from config instead of hardcoded string
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
    
    # Extract Labels for specific horizon
    # y shape: [N, 8]
    # We want column h_idx
    # Also handle NaNs (unresolved trades)
    
    labels_all = data.y[:, h_idx]
    
    # Create Masks
    # Train: 2019-2022
    # Test: 2023-2024
    # We need to map timestamps to dates.
    # builder.transactions has the info, but 'data' object has 't' (normalized).
    # Reconstruct dates? Or use builder.transactions['Filed']
    
    # Simple Split based on index (assuming sorted by time)
    # Better: Use years
    df_sorted = builder.transactions
    df_sorted['Filed_Date'] = pd.to_datetime(df_sorted['Filed'])
    
    train_mask = (df_sorted['Filed_Date'].dt.year >= 2015) & (df_sorted['Filed_Date'].dt.year <= 2022)
    test_mask = (df_sorted['Filed_Date'].dt.year >= 2023)
    
    # Filter valid labels (not NaN)
    valid_label_mask = ~torch.isnan(labels_all)
    
    train_idx = torch.where(torch.tensor(train_mask.values) & valid_label_mask)[0]
    test_idx = torch.where(torch.tensor(test_mask.values) & valid_label_mask)[0]
    
    logger.info(f"Train Samples: {len(train_idx)}, Test Samples: {len(test_idx)}")
    
    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Input dims
    # msg dim is builder.msg.shape[1]
    # x_static dim is 2 (party, state) -> Embedding needed
    edge_dim = data.msg.shape[1]
    
    model = ResearchTGN(
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
        # Note: GAP-TGN processes ALL events to update memory, 
        # but we only compute loss on TRAIN events.
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
            probs = torch.sigmoid(out[test_idx]).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            y_true = labels_all[test_idx].cpu().numpy()
            
            acc = accuracy_score(y_true, preds)
            auc = roc_auc_score(y_true, probs)
            
            if epoch % 1 == 0:
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