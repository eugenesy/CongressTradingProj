"""
sanity_debug.py
===============
Data Preparation Module. Handles raw transformation to ML dataset.

Refactored/Audited: 2026-03-20
"""

#!/usr/bin/env python3
"""
Sanity script to debug mode collapse in Phase 2 variables.
We load one month, execute forward, and check gradient norm and logit variance.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from phase2_gnn_extended import (
    load_dataset, build_enhanced_graph, compute_pol_features,
    compute_state_ids, compute_comp_features, BipartiteSAGEExtended,
    EdgePredictor, GNNPredictor, monthly_splits
)

def run_debug():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df = load_dataset("data/processed/ml_dataset_continuous.csv")

    le_pol      = LabelEncoder().fit(df["BioGuideID"].fillna("UNK"))
    le_tick     = LabelEncoder().fit(df["Ticker"].fillna("UNK"))
    le_state    = LabelEncoder().fit(df["State"].fillna("UNK"))
    le_sector   = LabelEncoder().fit(df["Sector"])
    le_industry = LabelEncoder().fit(df["Industry"])

    n_pol = len(le_pol.classes_)
    n_tick = len(le_tick.classes_)

    splits = list(monthly_splits(df, 2023, "Label_Q3"))
    if not splits: print("No splits"); return
    year, month, train_df, gap_df, test_df = splits[0]

    edge_index, edge_weight, edge_attr, _np, _nt = build_enhanced_graph(
        train_df, "Label_Q3", le_pol, le_tick, device, gap_df=gap_df,
        test_start=pd.Timestamp(2023, 1, 1)
    )

    gnn = BipartiteSAGEExtended(n_pol, n_tick, len(le_state.classes_),
                                len(le_sector.classes_), len(le_industry.classes_))
    predictor = EdgePredictor(gnn.out_dim * 2)
    model = GNNPredictor(gnn, predictor).to(device)

    pol_feats = compute_pol_features(train_df, "Label_Q3", le_pol, n_pol, gap_df, device)
    state_ids = compute_state_ids(train_df, le_pol, le_state, n_pol, gap_df, device)
    comp_feats = compute_comp_features(train_df, le_tick, le_sector, le_industry, device)

    model.gnn.set_pol_features(pol_feats)
    model.gnn.set_state_ids(state_ids)
    model.gnn.set_comp_features(comp_feats)

    pol_tr = torch.tensor(le_pol.transform(train_df["BioGuideID"].fillna("UNK")), dtype=torch.long, device=device)
    tick_tr = torch.tensor(le_tick.transform(train_df["Ticker"].fillna("UNK")), dtype=torch.long, device=device) + n_pol
    y_tr = torch.tensor((train_df["Label_Q3"] > 0).astype(float).values, dtype=torch.float, device=device)

    # test single batch
    model.train()
    logits = model(edge_index, edge_weight, edge_attr, pol_tr[:512], tick_tr[:512])

    print("\n=== Logits Summary ===")
    print(f"Logits min: {logits.min().item():.4f}, max: {logits.max().item():.4f}, mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
    if logits.std().item() < 1e-4:
        print("WARNING: Mode collapse immediately on forward pass!")
    else:
        print("Logits look healthy on forward pass.")

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, y_tr[:512])
    loss.backward()

    print("\n=== Gradient Norms ===")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"  {name:30s} norm: {param.grad.norm().item():.6f}")

if __name__ == "__main__":
    run_debug()
