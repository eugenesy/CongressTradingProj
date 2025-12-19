import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import copy

class DailyRollingEvaluator:
    def __init__(self, pipeline):
        """
        Args:
            pipeline (ChocolatePipeline): pipeline instance with loader/builder.
        """
        self.pipeline = pipeline
        self.loader = pipeline.loader
        self.builder = pipeline.builder
        
    @torch.no_grad()
    def evaluate(self, model, base_graph, test_transactions, device='cuda'):
        """
        Performs strict daily rolling evaluation.
        
        Args:
            model (torch.nn.Module): Trained GNN model (weights frozen).
            base_graph (HeteroData): Graph containing all history up to start of test period.
            test_transactions (pd.DataFrame): The test set transactions to predict/add.
            device (str): Device to run on.
            
        Returns:
            dict: {auc, f1, report}
        """
        model.eval()
        model.to(device)
        
        # 1. Prepare Dynamic Graph
        # We perform a shallow copy to modify edges without affecting the original object locally if needed,
        # but here we will just modify 'current_graph' cumulatively.
        current_graph = base_graph.clone().to(device)
        
        # Ensure reverse edges exist in base (should be there from builder)
        
        # 2. Group Test Data by Date
        # 'Filed' is the timestamp we simulate "waking up" on.
        daily_groups = test_transactions.groupby('Filed')
        sorted_dates = sorted(daily_groups.groups.keys())
        
        all_preds = []
        all_targets = []
        
        print(f"Starting Rolling Inference over {len(sorted_dates)} trading days...")
        
        # 3. Rolling Loop
        pbar = tqdm(sorted_dates, desc="Daily Rolling")
        for date in pbar:
            # A. Get today's batch
            batch_df = daily_groups.get_group(date)
            
            # B. Build "Target" sub-graph for today's edges
            batch_graph = self.builder.build_graph(batch_df)
            batch_graph = batch_graph.to(device)
            
            # C. Encode History
            z_dict = model.encode(current_graph.x_dict, current_graph.edge_index_dict)
            
            # D. Predict & Update (Handle BUYS and SELLS)
            
            # --- BUYS ---
            if ('politician', 'buys', 'company') in batch_graph.edge_index_dict:
                q_b_idx = batch_graph['politician', 'buys', 'company'].edge_index
                q_b_attr = batch_graph['politician', 'buys', 'company'].edge_attr
                q_b_y = batch_graph['politician', 'buys', 'company'].y
                
                # Predict
                logits = model.decode(z_dict, q_b_idx, q_b_attr)
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu())
                all_targets.append(q_b_y.cpu())
                
                # Update Graph: Add Edges
                if ('politician', 'buys', 'company') not in current_graph.edge_index_dict:
                     # Initialize if empty (shouldn't happen if initialized properly, but safety first)
                     current_graph['politician', 'buys', 'company'].edge_index = q_b_idx
                     current_graph['politician', 'buys', 'company'].edge_attr = q_b_attr
                     current_graph['politician', 'buys', 'company'].y = q_b_y
                     # Rev
                     rev_b = torch.stack([q_b_idx[1], q_b_idx[0]], dim=0)
                     current_graph['company', 'rev_buys', 'politician'].edge_index = rev_b
                else:
                    current_graph['politician', 'buys', 'company'].edge_index = torch.cat(
                        [current_graph['politician', 'buys', 'company'].edge_index, q_b_idx], dim=1)
                    current_graph['politician', 'buys', 'company'].edge_attr = torch.cat(
                        [current_graph['politician', 'buys', 'company'].edge_attr, q_b_attr], dim=0)
                    current_graph['politician', 'buys', 'company'].y = torch.cat(
                        [current_graph['politician', 'buys', 'company'].y, q_b_y], dim=0)
                    
                    rev_b = torch.stack([q_b_idx[1], q_b_idx[0]], dim=0)
                    current_graph['company', 'rev_buys', 'politician'].edge_index = torch.cat(
                         [current_graph['company', 'rev_buys', 'politician'].edge_index, rev_b], dim=1)

            # --- SELLS ---
            if ('politician', 'sells', 'company') in batch_graph.edge_index_dict:
                q_s_idx = batch_graph['politician', 'sells', 'company'].edge_index
                q_s_attr = batch_graph['politician', 'sells', 'company'].edge_attr
                q_s_y = batch_graph['politician', 'sells', 'company'].y
                
                # Predict
                logits = model.decode(z_dict, q_s_idx, q_s_attr)
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu())
                all_targets.append(q_s_y.cpu())
                
                 # Update Graph: Add Edges
                if ('politician', 'sells', 'company') not in current_graph.edge_index_dict:
                     current_graph['politician', 'sells', 'company'].edge_index = q_s_idx
                     current_graph['politician', 'sells', 'company'].edge_attr = q_s_attr
                     current_graph['politician', 'sells', 'company'].y = q_s_y
                     rev_s = torch.stack([q_s_idx[1], q_s_idx[0]], dim=0)
                     current_graph['company', 'rev_sells', 'politician'].edge_index = rev_s
                else:
                    current_graph['politician', 'sells', 'company'].edge_index = torch.cat(
                        [current_graph['politician', 'sells', 'company'].edge_index, q_s_idx], dim=1)
                    current_graph['politician', 'sells', 'company'].edge_attr = torch.cat(
                        [current_graph['politician', 'sells', 'company'].edge_attr, q_s_attr], dim=0)
                    current_graph['politician', 'sells', 'company'].y = torch.cat(
                        [current_graph['politician', 'sells', 'company'].y, q_s_y], dim=0)

                    rev_s = torch.stack([q_s_idx[1], q_s_idx[0]], dim=0)
                    current_graph['company', 'rev_sells', 'politician'].edge_index = torch.cat(
                         [current_graph['company', 'rev_sells', 'politician'].edge_index, rev_s], dim=1)
            
            # E. Aging (Time Decay update)
            # Feature idx 3 was removed. Now Feature idx 2 is Time Decay (Amount=0, Gap=1, Time=2)
            # Update Buys
            if ('politician', 'buys', 'company') in current_graph.edge_index_dict:
                 current_graph['politician', 'buys', 'company'].edge_attr[:, 2] += 1.0
            # Update Sells
            if ('politician', 'sells', 'company') in current_graph.edge_index_dict:
                 current_graph['politician', 'sells', 'company'].edge_attr[:, 2] += 1.0
            
        # 4. Final Evaluation
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        binary_preds = (all_preds > 0.5).astype(int)
        
        auc = roc_auc_score(all_targets, all_preds)
        f1 = f1_score(all_targets, binary_preds)
        
        print("\n=== Strict Rolling Evaluation Results ===")
        print(f"Total Transactions Predicted: {len(all_targets)}")
        print(f"AUC: {auc:.4f}")
        print(f"F1:  {f1:.4f}")
        print(classification_report(all_targets, binary_preds, target_names=['Fail', 'Success']))
        
        return {'auc': auc, 'f1': f1}
