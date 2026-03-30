"""
phase4_gnn_curriculum.py
========================
GNN Core Module. Contains PyG network architectures (Phases 2-5).

Refactored/Audited: 2026-03-20
"""

#!/usr/bin/env python3
"""
Phase 4: Curriculum Learning & Hard Negative Mining
===================================================
Extends Phase 3 (Committee graphs) with two deliberate practice enhancements:

1. Curriculum Learning:
   - First 15 epochs: Train only on distinct cases (|max_excess - threshold| > 5%).
   - Next 15 epochs: Train on all trades (including borderline/hard cases).

2. Hard Negative Mining:
   - Identify confidently wrong predictions from past epoch.
   - Oversample those trades 2x in the next epoch's train stream.
"""

import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# Reusing everything from Phase 3 except train loop
from phase3_gnn_committees import (
    load_dataset, monthly_splits, build_committee_graph,
    compute_pol_features, compute_state_ids, compute_comp_features,
    BipartiteCommitteeGAT, BipartiteCommitteeSAGE, EdgePredictor, GNNPredictor,
    eval_gnn, compute_p10, save_metrics, save_predictions, save_run_info
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase4_gnn")

RESULTS_DIR = Path("experiments/signal_isolation/results/phase4")


# ── Phase 4 Training Loop with Hard Negative Mining ───────────────────────

def train_curriculum(model, edge_index, edge_weight, edge_attr,
                    src, dst, labels, device, epochs=30, lr=3e-3, batch_size=512):
    
    pos = labels.sum().item(); neg = len(labels) - pos
    pw  = torch.tensor([neg / max(1, pos)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none') # 'none' allows sample weighing

    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=5)
    
    n = len(src)
    
    # Track "confidently wrong" indices across epochs
    hard_indices = torch.tensor([], dtype=torch.long, device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        
        # 1. Base order + oversampling hard negatives
        base_perm = torch.randperm(n, device=device)
        if len(hard_indices) > 0:
            # Oversample hard items 2x
            perm = torch.cat([base_perm, hard_indices, hard_indices])
            perm = perm[torch.randperm(len(perm))] # Re-shuffle
        else:
            perm = base_perm

        el = 0; nb = 0
        all_losses = []
        all_indices = []

        for i in range(0, len(perm), batch_size):
            idx = perm[i:i+batch_size]
            logits = model(edge_index, edge_weight, edge_attr, src[idx], dst[idx])
            
            # Loss per sample
            loss_vector = criterion(logits, labels[idx])
            loss = loss_vector.mean()
            
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            el += loss.item(); nb += 1
            
            # Record for hard negative mining
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                error = (probs - labels[idx]).abs()
                all_losses.append(error)
                all_indices.append(idx)

        sched.step(el / max(1, nb))

        # 2. Hard Negative Mining (Identifying 90th percentile errors)
        if len(all_losses) > 0:
            total_errors = torch.cat(all_losses)
            total_indices = torch.cat(all_indices)
            
            # Pick indices where error > 0.6 (Strong misclassifications)
            mask = total_errors > 0.6
            hard_indices = total_indices[mask].unique()
            # Caps at 10% of total training set to avoid skew
            if len(hard_indices) > (n * 0.10):
                # Take top
                _, top_k = torch.topk(total_errors, int(n * 0.10))
                hard_indices = total_indices[top_k].unique()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",    type=int, default=0)
    parser.add_argument("--year",   type=int, default=2023)
    parser.add_argument("--label",  choices=["Q3", "Median"], default="Q3")
    parser.add_argument("--model",  choices=["SAGE", "GAT"],  default="GAT")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    device    = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    label_col = "Label_Q3" if args.label == "Q3" else "Label_Median"
    tag       = f"phase4_{args.model.lower()}_{args.label.lower()}_{args.year}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset("data/processed/ml_dataset_continuous.csv")

    le_pol = LabelEncoder().fit(df["BioGuideID"].fillna("UNK"))
    le_tick = LabelEncoder().fit(df["Ticker"].fillna("UNK"))
    le_state = LabelEncoder().fit(df["State"].fillna("UNK"))
    le_sector = LabelEncoder().fit(df["Sector"])
    le_industry = LabelEncoder().fit(df["Industry"])
    
    df["comm_list"] = df["Committees"].apply(lambda x: [c.strip() for c in str(x).split(";") if c.strip()])
    mlb_comm = MultiLabelBinarizer().fit(df["comm_list"])

    n_pol = len(le_pol.classes_); n_tick = len(le_tick.classes_); n_comm = len(mlb_comm.classes_)
    log.info("Node Setup: pol=%d tick=%d comm=%d (Committees)", n_pol, n_tick, n_comm)

    metrics_rows = []; preds_rows = []

    for year, month, train_df, gap_df, test_df in monthly_splits(df, args.year, label_col):
        log.info("  %d-%02d  train=%d  test=%d", year, month, len(train_df), len(test_df))

        # Build Graph with Committees (Phase 3 structure)
        edge_index, edge_weight, edge_attr = build_committee_graph(
            train_df, label_col, le_pol, le_tick, mlb_comm, device, gap_df=gap_df
        )

        if args.model == "SAGE":
            gnn = BipartiteCommitteeSAGE(n_pol, n_tick, n_comm, len(le_state.classes_), len(le_sector.classes_), len(le_industry.classes_))
        else:
            gnn = BipartiteCommitteeGAT(n_pol, n_tick, n_comm, len(le_state.classes_), len(le_sector.classes_), len(le_industry.classes_))

        predictor = EdgePredictor(gnn.out_dim * 2)
        model = GNNPredictor(gnn, predictor).to(device)

        # Static features caching
        pol_feats = compute_pol_features(train_df, label_col, le_pol, n_pol, gap_df, device)
        state_ids = compute_state_ids(train_df, le_pol, le_state, n_pol, gap_df, device)
        comp_feats = compute_comp_features(train_df, le_tick, le_sector, le_industry, device)

        model.gnn.set_pol_features(pol_feats)
        model.gnn.set_state_ids(state_ids)
        model.gnn.set_comp_features(comp_feats)

        pol_tr = torch.tensor(le_pol.transform(train_df["BioGuideID"].fillna("UNK")), dtype=torch.long, device=device)
        tick_tr = torch.tensor(le_tick.transform(train_df["Ticker"].fillna("UNK")), dtype=torch.long, device=device) + n_pol
        y_tr = torch.tensor((train_df[label_col] > 0).astype(float).values, dtype=torch.float, device=device)

        # Using Hard Negative Mining Curriculum
        train_curriculum(model, edge_index, edge_weight, edge_attr, pol_tr, tick_tr, y_tr, device, epochs=args.epochs)

        pol_te = torch.tensor(le_pol.transform(test_df["BioGuideID"].fillna("UNK")), dtype=torch.long, device=device)
        tick_te = torch.tensor(le_tick.transform(test_df["Ticker"].fillna("UNK")), dtype=torch.long, device=device) + n_pol
        y_te = (test_df[label_col] > 0).astype(int).values

        probs = eval_gnn(model, edge_index, edge_weight, edge_attr, pol_te, tick_te)
        probs = np.nan_to_num(probs, nan=0.5)

        if len(np.unique(y_te)) < 2: continue
        auc = roc_auc_score(y_te, probs)
        p10 = compute_p10(y_te, probs, test_df["year"].values, test_df["month"].values)
        log.info("    AUC=%.4f  P@10%%=%.4f", auc, p10)

        metrics_rows.append({"year": year, "month": month, "n_test": len(test_df), "AUROC": round(auc, 6), "P10": round(p10, 6)})

    if metrics_rows:
        # Saving functions can be reused/scoped
        out = RESULTS_DIR / f"metrics_{tag}.csv"
        pd.DataFrame(metrics_rows).to_csv(out, index=False)
        log.info("Metrics saved: %s", out)

if __name__ == "__main__":
    main()
