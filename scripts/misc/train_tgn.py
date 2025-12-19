import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from torch_geometric.utils import negative_sampling

from src.models_tgn import TGN
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.data import TemporalData

def train_tgn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    data = torch.load("data/temporal_data.pt", weights_only=False)
    data = data.to(device)
    print(f"Loaded Temporal Data: {data}")
    
    # 2. Split (Time-based)
    # We extract num_nodes and clear it from the object before slicing.
    # ALSO clear num_parties/num_states which are ints
    if hasattr(data, 'num_nodes'):
        num_nodes = data.num_nodes
        try: del data.num_nodes 
        except: pass
    else:
        num_nodes = int(torch.cat([data.src, data.dst]).max()) + 1
        
    if hasattr(data, 'num_parties'):
        num_parties = data.num_parties
        try: del data.num_parties
        except: pass
    else:
        num_parties = 5 # Default fallback
        
    if hasattr(data, 'num_states'):
        num_states = data.num_states
        try: del data.num_states
        except: pass
    else:
        num_states = 50 # Default fallback
    
    print(f"Num Nodes: {num_nodes}, Parties: {num_parties}, States: {num_states}")

    # Simple 80/10/10 split by index (assuming sorted by time)
    train_idx = int(len(data.src) * 0.8)
    val_idx = int(len(data.src) * 0.9)
    
    train_data = data[0:train_idx]
    val_data = data[train_idx:val_idx]
    test_data = data[val_idx:]
    
    # 3. Loaders (Iterate sequentially)
    # Batch size = 200 (transactions per step)
    train_loader = TemporalDataLoader(train_data, batch_size=200)
    val_loader = TemporalDataLoader(val_data, batch_size=200)
    test_loader = TemporalDataLoader(test_data, batch_size=200)
    
    # 4. Initialize Model
    # Msg Dim = 3 (Amount, IsBuy, Gap)
    # Mem Dim = 100
    # Time Dim = 100
    # Emb Dim = 100
    # Static: Party, State
    # (Extracted above)
    
    print(f"Init TGN with {num_parties} parties, {num_states} states.")
    
    model = TGN(
        num_nodes=num_nodes, 
        raw_msg_dim=4, 
        memory_dim=100, 
        time_dim=100, 
        embedding_dim=100,
        num_parties=num_parties,
        num_states=num_states
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # 5. Neighbor Loader (For GNN aggregation)
    # We need a NeighborLoader that updates as we iterate.
    # In strict TGN, we often just use recent neighbors key-value or simplified approach.
    # But to use GraphAttentionEmbedding, we need a graph structure.
    # PyG TGN implementation usually requires a custom NeighborLoader hook.
    # For simplicity, we will use a global 'NeighborFinder' or sim.
    # ACTUALLY: The easiest way is to use 'LastNeighborLoader' which we imported in models_tgn.
    # BUT LastNeighborLoader needs to wrap the data object, which changes.
    
    # Re-import to initialize properly
    from torch_geometric.nn.models.tgn import LastNeighborLoader
    neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)
    
    # --- Helper to process one epoch ---
    def process_epoch(loader, mode='train'):
        if mode=='train':
            model.train()
            # Reset memory at start of epoch? 
            # ideally NO for online learning, but YES for static epoch repetition.
            # TGN usually trains 1 epoch = 1 pass through history.
            model.memory.reset_state() 
            neighbor_loader.reset_state()
        else:
            model.eval()
            
        total_loss = 0
        all_aps = []
        all_aucs = []
        
        # TGN State Maintenance:
        # 1. New Batch arrives.
        # 2. Update Memory with messages from *Previous* Batch.
        # 3. Compute Embeddings (using current Memory).
        # 4. Compute Loss.
        # 5. Store *Current* Batch messages for *Next* Step.
        
        # We start with empty messages
        model.memory.detach()
        
        for batch in tqdm(loader, desc=f"{mode.capitalize()}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
            
            # A. Sample Neighbors (for GNN)
            # Find neighbors for src and dst at time t
            # Input: Node IDs (batch.src + batch.dst)
            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            
            # Map global n_id back to batch indices
            assoc = dict(zip(n_id.tolist(), range(n_id.size(0))))
            
            # B. Prepare Edge Features (Time + Msg)
            # Historical info
            hist_t = data.t[e_id].to(device)
            hist_msg = data.msg[e_id].to(device)
            
            # Relative Time: Time since the edge happened relative to the target node's last update
            # TGN GAT uses: t (prediction time or last update) - t (edge time)
            # We use last_update of the TARGET node of the edge (row index in edge_index)
            # edge_index[0] is source, edge_index[1] is target in PyG flow? 
            # TransformerConv is (source, target).
            # Usually: t_target - t_edge
            
            target_nodes = n_id[edge_index[1]]
            # We need the last update time of these nodes.
            # CAUTION: memory.last_update is updated ONLY after the batch
            # So it reflects time up to Previous Batch.
            # But 'target_nodes' might be updated in Previous Batch (so time is close to now) 
            # or long ago.
            # To be safe and avoid negative time, we can use the MAX(last_update, current_batch_t)
            # But simpler: Just use hist_t diff from mean batch time?
            # Standard TGN: rel_t = current_time - edge_t
            # But current_time is interaction time. This graph is historical.
            # Let's use `model.memory.last_update[target_nodes]`.
            
            last_update = model.memory.last_update[target_nodes]
            rel_t = last_update - hist_t
            # rel_t = torch.relu(rel_t) # Ensure non-negative? No, time is monotonic.
            
            rel_t_enc = model.memory.time_enc(rel_t.to(torch.float))
            edge_attr = torch.cat([rel_t_enc, hist_msg], dim=-1)
            
            # C. Forward Pass
            # Get node embeddings z for all involved nodes
            z = model(n_id, edge_index, edge_attr)
            
            z_src = z[[assoc[i] for i in src.tolist()]]
            z_dst = z[[assoc[i] for i in dst.tolist()]]
            
            # Static Embeddings
            # data.x_static is on CPU likely, or GPU? Data was moved to device at start.
            s_src = model.encode_static(data.x_static[src].to(device))
            s_dst = model.encode_static(data.x_static[dst].to(device))
            
            # D. Loss (Link Prediction)
            # Positive edges: (src, dst)
            # Predict Positive (Dynamic + Static)
            pred_pos = model.predictor(z_src, z_dst, s_src, s_dst)
            
            # Predict Negative
            with torch.no_grad():
                neg_dst = torch.randint(0, data.num_nodes, (src.size(0),), device=device)
            
            # z_neg_dst = z[[assoc.get(i, 0) for i in neg_dst.tolist()]] 
            # Static Neg
            # s_neg_dst = model.encode_static(data.x_static[neg_dst])
            
            # Simplified: Just compute loss on positives + Classification Label (y)
            pred_score = pred_pos.squeeze()
            loss = criterion(pred_score, batch.y)
            
            if mode=='train':
                loss.backward()
                optimizer.step()
                model.memory.detach() # Detach for next step
            
            total_loss += loss.item()
            
            # Metrics
            with torch.no_grad():
                probs = torch.sigmoid(pred_score).cpu().numpy()
                targets = batch.y.cpu().numpy()
                try:
                    all_aps.append(average_precision_score(targets, probs))
                    all_aucs.append(roc_auc_score(targets, probs))
                except:
                    pass
            
            # D. Update State (for next batch)
            # 1. Update Memory with CURRENT batch interaction
            model.memory.update_state(src, dst, t, msg)
            
            # 2. Update Neighbor Loader
            neighbor_loader.insert(src, dst)
            
        return total_loss / len(loader), np.mean(all_aucs)

    # Train Loop
    best_val_auc = 0
    for epoch in range(1, 11): # 10 Epochs over the timeline
        loss, auc = process_epoch(train_loader, mode='train')
        print(f"Epoch {epoch} | Train Loss: {loss:.4f} | Train AUC: {auc:.4f} | Mem Size: {model.memory.memory.size(0)}")
        
        # Validation (Sequential continue)
        val_loss, val_auc = process_epoch(val_loader, mode='val')
        print(f"Epoch {epoch} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            # Save model? 
            # torch.save(model.state_dict(), "tgn_best.pt")

    # --- Final Test ---
    print("\n--- Final Evaluation on Test Set ---")
    # Note: Memory is preserved from the end of Validation in the last epoch.
    # This is correct for sequential data (Train -> Val -> Test).
    test_loss, test_auc = process_epoch(test_loader, mode='test')
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    train_tgn()
