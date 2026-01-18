import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
import sys
import os
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import datetime
import argparse

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ==========================================
#              IMPORTS & SETUP
# ==========================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Config Variables
from src.config_multiple_binary_labels import (
    PROCESSED_DATA_DIR,
    TX_PATH,
    TARGET_COLUMNS,
    TARGET_YEARS,
    RESULTS_DIR,
    LOGS_DIR,
    MIN_TICKER_FREQ
)

# Import the Binary/Structured Model
from src.models_tgn_binary_labels import TGN
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_evaluate(data, df_filtered, num_nodes, num_parties, num_states, num_classes,
                       msg_dim, price_dim,  # <--- NEW: Dynamic Dimensions
                       target_names=None,
                       target_years=TARGET_YEARS, max_epochs=20, patience=5):
    
    device = get_device()
    data = data.to(device)
    results = []
    
    # === HYPERPARAMETERS ===
    # Using passed dimensions instead of hardcoded constants
    TIME_DIM = 100
    NODE_EMBEDDING_DIM = 100  # Output dimension of the GNN (z)
    
    # Note: TGN class might internally map price_dim -> 32 (via PriceEncoder).
    # We assume the PriceEncoder output is 32 for the concatenation below.
    # If you change the PriceEncoder hidden size, update this constant.
    PRICE_EMB_OUTPUT_DIM = 32        
    
    print(f"DEBUG: Initializing training with Msg Dim={msg_dim}, Price Input Dim={price_dim}")

    # ---------------------------------------------------------
    # Helper: Decode "Interval Class" -> "Cumulative Binary Flags"
    # ---------------------------------------------------------
    def decode_to_binary(class_indices, num_thresholds):
        """
        Converts class index (0..N) back to N binary flags.
        Example: If Class=2 (Passed 2 thresholds), Output=[1, 1, 0, 0...]
        Logic: Flag[i] is True if Class > i
        """
        if isinstance(class_indices, np.ndarray):
            class_indices = torch.from_numpy(class_indices).to(device)
            
        # Create a range [0, 1, ... N-1] for thresholds
        threshold_indices = torch.arange(num_thresholds, device=device).unsqueeze(0) # [1, N_thresh]
        
        # Expand class_indices to [Batch, 1]
        class_indices = class_indices.unsqueeze(1) # [Batch, 1]
        
        # Broadcasting: Is the predicted class greater than the threshold index?
        return (class_indices > threshold_indices).float()
    
    # Feature Construction Helper
    def get_edge_attr(n_id, edge_index, e_id, batch_t):
        # FIX: Calculate dimension dynamically based on data
        # Edge Feature = Time(100) + Msg(msg_dim) + Price(32) + Label(1) + Age(1)
        expected_dim = TIME_DIM + msg_dim + PRICE_EMB_OUTPUT_DIM + 2
        
        if len(e_id) == 0:
            return torch.zeros((0, expected_dim), device=device)
        
        hist_t = data.t[e_id]
        hist_msg = data.msg[e_id]
        
        # This returns [N, 32] because PriceEncoder is hardcoded to 32 hidden units
        hist_price_emb = model.get_price_embedding(data.price_seq[e_id])
        
        target_nodes = n_id[edge_index[1]]
        last_update = model.memory.last_update[target_nodes]
        rel_t_enc = model.memory.time_enc((last_update - hist_t).float())
        
        batch_max_t = batch_t.max()
        hist_resolution = data.resolution_t[e_id]
        is_resolved = (hist_resolution < batch_max_t).float().unsqueeze(-1)
        
        # Masked Label Feature
        raw_label = data.y[e_id].float().unsqueeze(-1)
        raw_label[raw_label < 0] = -1.0 
        masked_label = raw_label * is_resolved + -1.0 * (1 - is_resolved)
        
        age_feat = torch.log1p((batch_max_t - hist_t).float() / 86400.0).unsqueeze(-1)
        
        return torch.cat([rel_t_enc, hist_msg, hist_price_emb, masked_label, age_feat], dim=-1)

    # ==========================================
    #          YEAR / MONTH LOOP
    # ==========================================
    for year in target_years:
        for month in range(1, 13):
            # Define Time Boundaries
            current_period_start = pd.Timestamp(year=year, month=month, day=1)
            
            # Next Month Start (for convenient filtering)
            if month == 12:
                next_period_start = pd.Timestamp(year=year+1, month=1, day=1)
            else:
                next_period_start = pd.Timestamp(year=year, month=month+1, day=1)
            
            # Define Gap and Train End
            train_end = current_period_start - pd.DateOffset(months=1)
            gap_start = train_end
            gap_end = current_period_start
            
            print(f"\n=== RETRAINING For Window: {year}-{month:02d} ===")
            
            # 1. SPLIT INDICES
            train_mask = (df_filtered['Traded'] < gap_start)
            gap_mask = (df_filtered['Traded'] >= gap_start) & (df_filtered['Traded'] < gap_end)
            test_mask = (df_filtered['Traded'] >= current_period_start) & (df_filtered['Traded'] < next_period_start)
            
            train_idx_np = np.where(train_mask)[0]
            gap_idx_np = np.where(gap_mask)[0]
            test_idx_np = np.where(test_mask)[0]
            
            # Safety Slicing
            max_id = len(data.src)
            train_idx = torch.tensor([i for i in train_idx_np if i < max_id], dtype=torch.long)
            gap_idx = torch.tensor([i for i in gap_idx_np if i < max_id], dtype=torch.long)
            test_idx = torch.tensor([i for i in test_idx_np if i < max_id], dtype=torch.long)
            
            if len(test_idx) == 0:
                print(f"No trades in {year}-{month:02d}, skipping.")
                continue
                
            train_data = data[train_idx]
            gap_data = data[gap_idx]
            test_data = data[test_idx]
            
            print(f"Train: {len(train_data.src)} | Gap: {len(gap_data.src)} | Test: {len(test_data.src)}")
            
            # 2. INIT MODEL
            model = TGN(
                num_nodes=num_nodes, 
                raw_msg_dim=msg_dim,  # <--- PASSED DYNAMICALLY
                memory_dim=100, 
                time_dim=TIME_DIM, 
                embedding_dim=NODE_EMBEDDING_DIM, # 100
                num_parties=num_parties, 
                num_states=num_states,
                num_classes=num_classes 
            ).to(device)

            # --- DYNAMIC CLASS WEIGHTING ---
            # 1. Calculate class counts in the current training set
            # Extract valid labels (ignore -100)
            y_train = train_data.y.long()
            y_train = y_train[y_train >= 0]
            
            # Count occurrences of each class (0, 1, 2, 3...)
            # We assume num_classes is correct (e.g., 4)
            class_counts = torch.bincount(y_train, minlength=num_classes).float()
            
            # 2. Compute Weights: Inverse frequency
            # Add small epsilon to avoid division by zero
            weights = 1.0 / (class_counts + 1.0)
            
            # 3. Normalize weights so they sum to num_classes (keeps learning rate stable)
            weights = weights * (num_classes / weights.sum())
            
            print(f"  Class Counts: {class_counts.tolist()} -> Weights: {weights.tolist()}")
            
            # 4. Pass to Loss Function
            criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
            
            # ... proceed to define optimizer ...
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            neighbor_loader = LastNeighborLoader(num_nodes, size=10, device=device)
            
            # 3. PHASE 1: TRAIN WITH VALIDATION
            train_size = int(len(train_data.src) * 0.9)
            actual_train_data = train_data[:train_size]
            val_data = train_data[train_size:]
            
            train_loader = TemporalDataLoader(actual_train_data, batch_size=200)
            val_loader = TemporalDataLoader(val_data, batch_size=200)
            
            print(f"  Train Split: {len(actual_train_data.src)} | Val Split: {len(val_data.src)}")
            
            best_val_f1 = 0.0
            epochs_without_improvement = 0
            best_epoch = 1
            
            for epoch in range(1, max_epochs + 1):
                model.memory.reset_state()
                neighbor_loader.reset_state()
                model.memory.detach()
                model.train()
                
                epoch_loss = 0
                batch_count = 0
                
                # A. Train
                for batch in train_loader:
                    if batch.src.size(0) <= 1:
                        continue
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    
                    price_seq = batch.price_seq if hasattr(batch, 'price_seq') else torch.zeros((len(src), price_dim), device=device)
                    price_emb = model.get_price_embedding(price_seq)
                    augmented_msg = torch.cat([msg, price_emb], dim=1)
                    
                    n_id = torch.cat([src, dst]).unique()
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                    assoc = dict(zip(n_id.tolist(), range(n_id.size(0))))
                    edge_attr = get_edge_attr(n_id, edge_index, e_id, t)
                    
                    z = model(n_id, edge_index, edge_attr)
                    z_src = z[[assoc[i] for i in src.tolist()]]
                    z_dst = z[[assoc[i] for i in dst.tolist()]]
                    s_src = model.encode_static(data.x_static[src])
                    s_dst = model.encode_static(data.x_static[dst])
                    
                    logits = model.predictor(z_src, z_dst, s_src, s_dst, price_emb, price_emb)
                    loss = criterion(logits, batch.y.long())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                    model.memory.update_state(src, dst, t, augmented_msg)
                    neighbor_loader.insert(src, dst)
                    model.memory.detach()
                
                # B. Gap
                gap_loader = TemporalDataLoader(gap_data, batch_size=200)
                model.eval()
                for batch in gap_loader:
                    batch = batch.to(device)
                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    price_seq = batch.price_seq if hasattr(batch, 'price_seq') else torch.zeros((len(src), price_dim), device=device)
                    price_emb = model.get_price_embedding(price_seq)
                    augmented_msg = torch.cat([msg, price_emb], dim=1)
                    model.memory.update_state(src, dst, t, augmented_msg)
                    neighbor_loader.insert(src, dst)

                # C. Validation
                val_preds_binary = []
                val_targets_binary = []
                
                for batch in val_loader:
                    batch = batch.to(device)
                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    
                    price_seq = batch.price_seq if hasattr(batch, 'price_seq') else torch.zeros((len(src), price_dim), device=device)
                    price_emb = model.get_price_embedding(price_seq)
                    augmented_msg = torch.cat([msg, price_emb], dim=1)
                    
                    n_id = torch.cat([src, dst]).unique()
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                    assoc = dict(zip(n_id.tolist(), range(n_id.size(0))))
                    edge_attr = get_edge_attr(n_id, edge_index, e_id, t)

                    z = model(n_id, edge_index, edge_attr)
                    z_src = z[[assoc[i] for i in src.tolist()]]
                    z_dst = z[[assoc[i] for i in dst.tolist()]]
                    s_src = model.encode_static(data.x_static[src])
                    s_dst = model.encode_static(data.x_static[dst])
                    
                    with torch.no_grad():
                        logits = model.predictor(z_src, z_dst, s_src, s_dst, price_emb, price_emb)
                        # --- DECODE TO BINARY FOR EVALUATION ---
                        pred_cls = logits.argmax(dim=1)
                        # Convert to [Batch, Num_Thresholds] binary matrix
                        pred_bin = decode_to_binary(pred_cls, num_classes-1)
                        
                        # Convert Targets (ignore -100s for conversion, handle later)
                        y_clean = batch.y.clone()
                        y_clean[y_clean < 0] = 0 # Dummy for conversion
                        target_bin = decode_to_binary(y_clean, num_classes-1)
                        
                        # Store valid rows only
                        valid_mask = (batch.y >= 0).cpu().numpy()
                        if valid_mask.any():
                            val_preds_binary.append(pred_bin[valid_mask].cpu())
                            val_targets_binary.append(target_bin[valid_mask].cpu())
                    
                    model.memory.update_state(src, dst, t, augmented_msg)
                    neighbor_loader.insert(src, dst)
                
                # Compute Binary Metrics
                if len(val_targets_binary) > 0:
                    val_t = torch.cat(val_targets_binary).numpy()
                    val_p = torch.cat(val_preds_binary).numpy()
                    val_f1 = f1_score(val_t, val_p, average='weighted') # Weighted across all binary labels
                else:
                    val_f1 = 0.0
                    
                avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
                print(f"  Epoch {epoch}/{max_epochs}: Loss={avg_loss:.4f} | Val F1 (Binary)={val_f1:.4f}")
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    epochs_without_improvement = 0
                    best_epoch = epoch
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"  Early Stop at epoch {epoch} (Best: {best_epoch})")
                        break
            
            # 4. PHASE 2: RETRAIN ON FULL DATA
            if best_epoch is None: best_epoch = 1
            print(f"  --- Phase 2: Retraining on FULL data for {best_epoch} epochs ---")
            
            model = TGN(
                num_nodes=num_nodes, 
                raw_msg_dim=msg_dim,  # <--- PASSED DYNAMICALLY
                memory_dim=100, 
                time_dim=TIME_DIM, 
                embedding_dim=NODE_EMBEDDING_DIM, # 100
                num_parties=num_parties, 
                num_states=num_states,
                num_classes=num_classes 
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            neighbor_loader.reset_state()
            
            full_train_loader = TemporalDataLoader(train_data, batch_size=200)
            
            for epoch in range(1, best_epoch + 1):
                model.memory.reset_state()
                neighbor_loader.reset_state()
                model.memory.detach()
                model.train()
                
                for batch in full_train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    
                    price_seq = batch.price_seq if hasattr(batch, 'price_seq') else torch.zeros((len(src), price_dim), device=device)
                    price_emb = model.get_price_embedding(price_seq)
                    augmented_msg = torch.cat([msg, price_emb], dim=1)
                    
                    n_id = torch.cat([src, dst]).unique()
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                    assoc = dict(zip(n_id.tolist(), range(n_id.size(0))))
                    edge_attr = get_edge_attr(n_id, edge_index, e_id, t)
                    
                    z = model(n_id, edge_index, edge_attr)
                    z_src = z[[assoc[i] for i in src.tolist()]]
                    z_dst = z[[assoc[i] for i in dst.tolist()]]
                    s_src = model.encode_static(data.x_static[src])
                    s_dst = model.encode_static(data.x_static[dst])
                    
                    logits = model.predictor(z_src, z_dst, s_src, s_dst, price_emb, price_emb)
                    loss = criterion(logits, batch.y.long())
                    loss.backward()
                    optimizer.step()
                    
                    model.memory.update_state(src, dst, t, augmented_msg)
                    neighbor_loader.insert(src, dst)
                    model.memory.detach()
                
                # Gap Loop
                gap_loader = TemporalDataLoader(gap_data, batch_size=200)
                model.eval()
                for batch in gap_loader:
                    batch = batch.to(device)
                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    price_seq = batch.price_seq if hasattr(batch, 'price_seq') else torch.zeros((len(src), price_dim), device=device)
                    price_emb = model.get_price_embedding(price_seq)
                    augmented_msg = torch.cat([msg, price_emb], dim=1)
                    model.memory.update_state(src, dst, t, augmented_msg)
                    neighbor_loader.insert(src, dst)
            
            # 5. TEST PHASE
            print(f"  Testing model...")
            model.eval()
            test_loader = TemporalDataLoader(test_data, batch_size=200)
            preds_binary_all = []
            targets_binary_all = []
            
            for batch in test_loader:
                batch = batch.to(device)
                src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                
                price_seq = batch.price_seq if hasattr(batch, 'price_seq') else torch.zeros((len(src), price_dim), device=device)
                price_emb = model.get_price_embedding(price_seq)
                augmented_msg = torch.cat([msg, price_emb], dim=1)
                
                n_id = torch.cat([src, dst]).unique()
                n_id, edge_index, e_id = neighbor_loader(n_id)
                assoc = dict(zip(n_id.tolist(), range(n_id.size(0))))
                edge_attr = get_edge_attr(n_id, edge_index, e_id, t)

                z = model(n_id, edge_index, edge_attr)
                z_src = z[[assoc[i] for i in src.tolist()]]
                z_dst = z[[assoc[i] for i in dst.tolist()]]
                s_src = model.encode_static(data.x_static[src])
                s_dst = model.encode_static(data.x_static[dst])
                
                with torch.no_grad():
                    logits = model.predictor(z_src, z_dst, s_src, s_dst, price_emb, price_emb)
                    
                    # --- DECODE TO BINARY FOR FINAL RESULT ---
                    pred_cls = logits.argmax(dim=1)
                    pred_bin = decode_to_binary(pred_cls, num_classes-1)
                    
                    # Targets
                    y_clean = batch.y.clone()
                    y_clean[y_clean < 0] = 0
                    target_bin = decode_to_binary(y_clean, num_classes-1)
                    
                    # Filter valids
                    valid_mask = (batch.y >= 0).cpu().numpy()
                    if valid_mask.any():
                        preds_binary_all.append(pred_bin[valid_mask].cpu())
                        targets_binary_all.append(target_bin[valid_mask].cpu())
                
                model.memory.update_state(src, dst, t, augmented_msg)
                neighbor_loader.insert(src, dst)
                
            # Compile Metrics
            if len(targets_binary_all) > 0:
                final_t = torch.cat(targets_binary_all).numpy()
                final_p = torch.cat(preds_binary_all).numpy()
                
                # subset accuracy: row must match exactly (strict)
                acc = accuracy_score(final_t, final_p)
                f1 = f1_score(final_t, final_p, average='weighted')
                count = len(final_p)
                
                print(f"  [RESULT] {year}-{month:02d}: BinarySubsetAcc={acc:.4f} | WeightedF1={f1:.4f} | Count={count}")
                print("-" * 50)
                print(classification_report(final_t, final_p, zero_division=0, target_names=target_names))
                
                results.append({
                    "Year": year,
                    "Month": month,
                    "Accuracy": acc,
                    "F1": f1,
                    "Count": count
                })
                
                df_results = pd.DataFrame(results)
                os.makedirs(RESULTS_DIR, exist_ok=True)
                df_results.to_csv(os.path.join(RESULTS_DIR, "rolling_binary_metrics.csv"), index=False)
            else:
                print("  [RESULT] No valid test samples.")
        

# ==========================================
#              MAIN SCRIPT
# ==========================================

if __name__ == "__main__":
    # 1. SETUP LOGGING
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"multilabel_binary_{timestamp}.txt")
    sys.stdout = Logger(log_path)
    print(f"Starting script... Output is being saved to: {log_path}")

    # 2. PARSE ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=TARGET_YEARS, help="Years to train on")
    args = parser.parse_args()

    # 3. LOAD DATA
    data_path = os.path.join(PROCESSED_DATA_DIR, "temporal_data.pt")
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        data = torch.load(data_path, weights_only=False)
    else:
        print(f"ERROR: Data file not found at {data_path}")
        sys.exit(1)

    # 4. === METADATA EXTRACTION & CLEANUP ===
    # CRITICAL: We must extract these values and then DELETE them from the object.
    # If we leave them attached, data[idx] will crash trying to slice them.

    # -- Msg Dim --
    if hasattr(data, 'msg_dim'):
        msg_dim = int(data.msg_dim) # Cast to int
        del data.msg_dim            # <--- DELETE
    else:
        print("Warning: 'msg_dim' not found. Inferring from tensor.")
        msg_dim = data.msg.shape[1]

    # -- Price Dim --
    if hasattr(data, 'price_dim'):
        price_dim = int(data.price_dim)
        del data.price_dim          # <--- DELETE
    else:
        if hasattr(data, 'price_seq'):
             price_dim = data.price_seq.shape[1]
        else:
             price_dim = 14 
    
    # -- Feature Names (List) --
    if hasattr(data, 'msg_feature_names'):
        # We don't need this for training, just clean it up so it doesn't crash slicing
        del data.msg_feature_names  # <--- DELETE

    print(f"Detected Dynamic Dimensions -> Msg: {msg_dim} | Price: {price_dim}")

    # 5. DATA SANITIZATION
    if data.y.min() < 0:
        bad_count = (data.y < 0).sum().item()
        print(f"Sanitizing Data: Found {bad_count} invalid targets. Mapping to -100 (ignore_index).")
        data.y[data.y < 0] = -100

    # 6. EXTRACT OTHER SCALARS
    # We must also extract and delete these if they were attached
    
    if hasattr(data, 'num_classes'):
        num_classes = int(data.num_classes)
        del data.num_classes
    else:
        print("Warning: num_classes not found. Inferring from config.")
        from src.config_multiple_binary_labels import TARGET_COLUMNS
        num_classes = len(TARGET_COLUMNS) + 1
        
    if hasattr(data, 'num_nodes'):
        num_nodes = int(data.num_nodes)
        del data.num_nodes
    else: 
        num_nodes = int(torch.cat([data.src, data.dst]).max()) + 1

    if hasattr(data, 'num_parties'):
        num_parties = int(data.num_parties)
        del data.num_parties
    else: 
        num_parties = 5

    if hasattr(data, 'num_states'):
        num_states = int(data.num_states)
        del data.num_states
    else: 
        num_states = 60

    # === NEW: Convert remaining ints to Tensors so they persist correctly ===
    # This fixes the AttributeError while keeping the data on the object
    if hasattr(data, 'num_pols'):
        # Convert to a 1-element tensor. 
        # PyG will see size(0)=1 != num_events, so it will copy it to train_data/test_data safely.
        data.num_pols = torch.tensor([data.num_pols])
        
    if hasattr(data, 'num_comps'):
        data.num_comps = torch.tensor([data.num_comps])
    # =======================================================================

    print(f"Model will train on {num_classes} intervals (Decoding to {num_classes-1} binary labels).")

    # 7. LOAD DATAFRAME FOR ALIGNMENT
    print("Loading CSV for Date Alignment...")
    df = pd.read_csv(TX_PATH)
    df['Traded'] = pd.to_datetime(df['Traded'])
    df = df.sort_values('Traded').reset_index(drop=True)
    
    ticker_counts = df['Ticker'].value_counts()
    valid_tickers = ticker_counts[ticker_counts >= MIN_TICKER_FREQ].index
    mask = df['Ticker'].isin(valid_tickers) & df['Traded'].notna()
    df = df[mask].reset_index(drop=True)
    
    print(f"Aligned DF Size: {len(df)} | PyG Data Size: {data.num_events}")
    
    # 8. START TRAINING
    train_and_evaluate(data, df, num_nodes, num_parties, num_states, num_classes, 
                       msg_dim, price_dim, 
                       target_names=TARGET_COLUMNS,
                       target_years=args.years)
    
    print("Experiment completed. Results saved to logs.")