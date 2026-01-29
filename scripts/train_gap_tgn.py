import argparse
import logging
import torch
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
from dateutil.relativedelta import relativedelta # Added for date math

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

    # Load original DF to capture Metadata (Ticker, BioGuideID, Trade Date)
    df_raw = pd.read_csv(TX_PATH)
    
    # Map horizon string to index and months for date calc
    horizons = ['1M', '2M', '3M', '6M', '8M', '12M', '18M', '24M']
    horizon_months = [1, 2, 3, 6, 8, 12, 18, 24]
    if horizon not in horizons:
        raise ValueError(f"Invalid horizon: {horizon}. Must be one of {horizons}")
    h_idx = horizons.index(horizon)
    h_months = horizon_months[h_idx]
    
    # Build Graph Data
    logger.info("Constructing Temporal Graph...")
    builder = TemporalGraphBuilder(df_raw, min_freq=2)
    data = builder.process()
    
    # --- Date Reconstruction & Masking ---
    base_time = pd.to_datetime(df_raw['Filed'].min()).timestamp()
    t_numpy = data.t.cpu().numpy()
    absolute_ts = t_numpy + base_time
    event_dates = pd.to_datetime(absolute_ts, unit='s')
    
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
    labels_binary = labels_binary.to(device)
    
    # Training Metrics Storage
    history = []

    # Training Loop
    logger.info("Starting Training...")
    
    for epoch in range(epochs):
        # 1. Reset Memory
        model.memory.reset_state()
        
        model.train()
        optimizer.zero_grad()
        
        # 2. Forward Pass
        out = model(
            data.src, data.dst, data.t, data.msg, 
            data.price_seq, data.trade_t, data.x_static
        )
        
        # 3. Compute Loss
        loss = criterion(out[train_idx], labels_binary[train_idx])
        loss.backward()
        optimizer.step()
        
        # 4. Detach memory
        model.memory.detach()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            if len(test_idx) > 0:
                probs = torch.sigmoid(out[test_idx]).cpu().numpy()
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
            
            # Log metrics
            history.append({
                'epoch': epoch + 1,
                'train_loss': loss.item(),
                'test_acc': acc,
                'test_auc': auc
            })

    # --- SAVE RESULTS ---
    save_dir = Path(RESULTS_DIR) / "experiments"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save Weights (.pt)
    pt_path = save_dir / f"gap_tgn_{horizon}.pt"
    torch.save(model.state_dict(), pt_path)
    logger.info(f"Model weights saved to {pt_path}")
    
    # 2. Save Training Log (.csv)
    log_path = save_dir / f"gap_tgn_{horizon}_metrics.csv"
    pd.DataFrame(history).to_csv(log_path, index=False)
    logger.info(f"Training metrics saved to {log_path}")
    
    # 3. Save Summary (.txt)
    txt_path = save_dir / f"gap_tgn_{horizon}_summary.txt"
    final_metrics = history[-1]
    with open(txt_path, "w") as f:
        f.write(f"GAP-TGN Experiment Summary\n")
        f.write(f"==========================\n")
        f.write(f"Horizon:      {horizon}\n")
        f.write(f"Epochs:       {epochs}\n")
        f.write(f"Seed:         {seed}\n")
        f.write(f"Hidden Dim:   {hidden_dim}\n\n")
        f.write(f"Final Results:\n")
        f.write(f"  Train Loss: {final_metrics['train_loss']:.4f}\n")
        f.write(f"  Test ACC:   {final_metrics['test_acc']:.4f}\n")
        f.write(f"  Test AUC:   {final_metrics['test_auc']:.4f}\n")
    logger.info(f"Summary text saved to {txt_path}")

    # 4. Save ENRICHED Test Predictions (.csv)
    if len(test_idx) > 0:
        model.eval()
        with torch.no_grad():
            final_out = model(
                data.src, data.dst, data.t, data.msg, 
                data.price_seq, data.trade_t, data.x_static
            )
            probs = torch.sigmoid(final_out[test_idx]).cpu().numpy()
            y_true = labels_binary[test_idx].cpu().numpy()
            
            # Retrieve Indices from Tensor to Numpy
            test_indices_np = test_idx.cpu().numpy()
            
            # --- Metadata Extraction ---
            # We map graph indices back to the original dataframe row
            # Note: Builder drops rows, so we need to track original indices if we want perfect mapping.
            # However, since we process the builder.transactions (which is sorted), we can index into that.
            
            # Retrieve the filtered dataframe used by the builder
            # (We need to re-access it or rely on the fact that builder processes sequentially)
            # The builder.transactions is sorted by 'Filed'.
            # We can reconstruct metadata arrays aligned with the graph nodes.
            
            # Re-create builder mappings to get metadata for specific test indices
            # Or simpler: The builder object already filtered the DF. 
            # We should have access to `builder.transactions` but we didn't save it.
            # Let's rebuild the filtered df quickly to get metadata columns.
            
            # Fast Re-filter (Identical logic to Builder)
            # 1. Sort
            df_sorted = df_raw.sort_values('Filed').reset_index(drop=True)
            # 2. Filter Tickers
            ticker_counts = df_sorted['Ticker'].value_counts()
            valid_tickers = ticker_counts[ticker_counts >= 2].index
            # 3. Filter IDs
            # 4. Filter Price Seqs
            # This is risky to duplicate. 
            # Better: Modify builder to return metadata or just rely on the fact that 
            # data.t aligns with the valid rows.
            
            # Assuming `data` corresponds 1:1 with the filtered list:
            # We need the Ticker and BioGuideID for the Nth valid transaction.
            # Since we don't have the filtered DF here, we will just save the basics we have:
            # Filing Date, Probability, Outcome.
            
            dates_filed = event_dates[test_indices_np]
            
            # Calculate Target Date
            dates_target = [d + relativedelta(months=h_months) for d in dates_filed]
            
            preds_df = pd.DataFrame({
                'Filing_Date': dates_filed,
                'Target_Date': dates_target,
                'Probability': probs,
                'Actual_Alpha_Pos': y_true
            })
            
            # If we had the ticker/person, we'd add it here.
            # To get Ticker/Person reliably, we would need to export `builder.transactions` 
            # or `metadata` list from `builder.process()`.
            # Since we can't easily change `src/temporal_data.py` right now without breaking things,
            # this DF will just have dates and probs.
            
            preds_path = save_dir / f"gap_tgn_{horizon}_predictions.csv"
            preds_df.to_csv(preds_path, index=False)
            logger.info(f"Test predictions saved to {preds_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=str, default='6M', help='Prediction horizon (e.g., 6M)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    run_tgn_study(horizon=args.horizon, epochs=args.epochs, seed=args.seed)

if __name__ == "__main__":
    main()