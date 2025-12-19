import torch
import torch.nn.functional as F
import os
import copy
from .models import Model
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from tqdm import tqdm

def train_window(window_id, data_root="data/processed_graphs", hidden_channels=64, epochs=200, patience=15):
    """
    Trains a model on a specific window with Early Stopping.
    """
    print(f"\n=== Training on Window {window_id} ===")
    
    # 1. Load Data
    path = os.path.join(data_root, f"window_{window_id}")
    if not os.path.exists(path):
        print(f"Window {path} does not exist.")
        return None

    train_data = torch.load(os.path.join(path, "train.pt"), weights_only=False)
    val_data = torch.load(os.path.join(path, "val.pt"), weights_only=False)
    test_data = torch.load(os.path.join(path, "test.pt"), weights_only=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print(f"Loaded Graph Metadata: {train_data.metadata()}")
    
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    
    # 2. Initialize Model
    model = Model(hidden_channels=hidden_channels, metadata=train_data.metadata())
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Early Stopping Tracking
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Enable TQDM for training monitoring
    pbar = tqdm(range(1, epochs + 1), desc=f"Win {window_id}", leave=False)
    
    for epoch in pbar:
        # --- Train ---
        model.train()
        optimizer.zero_grad()
        
        # Forward Pass
        # Forward Pass (Encode)
        # Gets node embeddings based on ALL edges (buys and sells)
        z_dict = model(train_data.x_dict, train_data.edge_index_dict)
        
        # Compute Loss on BUYS
        loss = 0.0
        
        # BUYS
        if ('politician', 'buys', 'company') in train_data.edge_index_dict:
             buys_edge_index = train_data['politician', 'buys', 'company'].edge_index
             buys_edge_attr = train_data['politician', 'buys', 'company'].edge_attr
             buys_y = train_data['politician', 'buys', 'company'].y
             
             pred_buys = model.decode(z_dict, buys_edge_index, buys_edge_attr)
             loss += F.binary_cross_entropy_with_logits(pred_buys, buys_y)
             
        # SELLS
        if ('politician', 'sells', 'company') in train_data.edge_index_dict:
             sells_edge_index = train_data['politician', 'sells', 'company'].edge_index
             sells_edge_attr = train_data['politician', 'sells', 'company'].edge_attr
             sells_y = train_data['politician', 'sells', 'company'].y
             
             pred_sells = model.decode(z_dict, sells_edge_index, sells_edge_attr)
             loss += F.binary_cross_entropy_with_logits(pred_sells, sells_y)
        
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item()
        
        # --- Validation ---
        model.eval()
        with torch.no_grad():
            z_dict_val = model(val_data.x_dict, val_data.edge_index_dict)
            val_loss = 0.0
            
            all_val_probs = []
            all_val_targets = []
            
            # Val BUYS
            if ('politician', 'buys', 'company') in val_data.edge_index_dict:
                v_b_idx = val_data['politician', 'buys', 'company'].edge_index
                v_b_attr = val_data['politician', 'buys', 'company'].edge_attr
                v_b_y = val_data['politician', 'buys', 'company'].y
                
                p_b = model.decode(z_dict_val, v_b_idx, v_b_attr)
                val_loss += F.binary_cross_entropy_with_logits(p_b, v_b_y)
                
                all_val_probs.append(torch.sigmoid(p_b))
                all_val_targets.append(v_b_y)

            # Val SELLS
            if ('politician', 'sells', 'company') in val_data.edge_index_dict:
                v_s_idx = val_data['politician', 'sells', 'company'].edge_index
                v_s_attr = val_data['politician', 'sells', 'company'].edge_attr
                v_s_y = val_data['politician', 'sells', 'company'].y
                
                p_s = model.decode(z_dict_val, v_s_idx, v_s_attr)
                val_loss += F.binary_cross_entropy_with_logits(p_s, v_s_y)

                all_val_probs.append(torch.sigmoid(p_s))
                all_val_targets.append(v_s_y)
                
            # Metrics (Aggregate)
            if len(all_val_probs) > 0:
                val_probs_cat = torch.cat(all_val_probs).cpu().numpy()
                val_y_cat = torch.cat(all_val_targets).cpu().numpy()
                try:
                    val_auc = roc_auc_score(val_y_cat, val_probs_cat)
                except:
                    val_auc = 0.5
            else:
                 val_auc = 0.5 
        
        # Update Progress Bar
        pbar.set_postfix({'TL': f"{train_loss:.3f}", 'VL': f"{val_loss:.3f}", 'VAUC': f"{val_auc:.3f}"})
            
        # --- Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save Checkpoint
            torch.save(best_model_state, os.path.join(path, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # print(f"Early stop at epoch {epoch}")
                break
    
    pbar.close()
                
    # --- Final Test ---

    print("Restoring best model for Testing...")
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        z_dict_test = model(test_data.x_dict, test_data.edge_index_dict)
        all_test_probs = []
        all_test_targets = []
        
        # Test BUYS
        if ('politician', 'buys', 'company') in test_data.edge_index_dict:
             t_b_idx = test_data['politician', 'buys', 'company'].edge_index
             t_b_attr = test_data['politician', 'buys', 'company'].edge_attr
             t_b_y = test_data['politician', 'buys', 'company'].y
             
             p_b = model.decode(z_dict_test, t_b_idx, t_b_attr)
             all_test_probs.append(torch.sigmoid(p_b))
             all_test_targets.append(t_b_y)
             
        # Test SELLS
        if ('politician', 'sells', 'company') in test_data.edge_index_dict:
             t_s_idx = test_data['politician', 'sells', 'company'].edge_index
             t_s_attr = test_data['politician', 'sells', 'company'].edge_attr
             t_s_y = test_data['politician', 'sells', 'company'].y
             
             p_s = model.decode(z_dict_test, t_s_idx, t_s_attr)
             all_test_probs.append(torch.sigmoid(p_s))
             all_test_targets.append(t_s_y)
             
        if len(all_test_probs) > 0:
            test_probs = torch.cat(all_test_probs).cpu().numpy()
            test_target = torch.cat(all_test_targets).cpu().numpy()
            test_preds_binary = (test_probs > 0.5).astype(int)
        else:
            test_probs = []
            test_target = []
            test_preds_binary = []

        try:
            if len(test_target) > 0:
                test_auc = roc_auc_score(test_target, test_probs)
                test_f1 = f1_score(test_target, test_preds_binary)
                print("\nClassification Report (Test):")
                print(classification_report(test_target, test_preds_binary, target_names=['Fail', 'Success']))
            else: 
                test_auc = 0.5
                test_f1 = 0.0
                print("\nNo test samples found.")
        except:
            test_auc = 0.5
            test_f1 = 0.0
            
    print(f"Window {window_id} Results -> Test AUC: {test_auc:.4f}, Test F1: {test_f1:.4f}")
    return {'auc': test_auc, 'f1': test_f1}
