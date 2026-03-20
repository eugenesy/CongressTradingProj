"""
check_nan_inputs.py
===================
Data Preparation Module. Handles raw transformation to ML dataset.

Refactored/Audited: 2026-03-20
"""

#!/usr/bin/env python3
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from phase2_gnn_extended import (
    load_dataset, build_enhanced_graph, compute_pol_features,
    compute_state_ids, compute_comp_features, monthly_splits
)

def check_nan():
    df = load_dataset("data/processed/ml_dataset_continuous.csv")
    label_col = "Label_Q3"

    le_pol      = LabelEncoder().fit(df["BioGuideID"].fillna("UNK"))
    le_tick     = LabelEncoder().fit(df["Ticker"].fillna("UNK"))
    le_state    = LabelEncoder().fit(df["State"].fillna("UNK"))
    le_sector   = LabelEncoder().fit(df["Sector"])
    le_industry = LabelEncoder().fit(df["Industry"])

    n_pol = len(le_pol.classes_)

    split_iter = monthly_splits(df, 2023, label_col)
    try:
        year, month, train_df, gap_df, test_df = next(split_iter)
    except StopIteration:
        print("No splits")
        return
    ts = pd.Timestamp(year, month, 1)

    edge_index, edge_weight, edge_attr, _np, _nt = build_enhanced_graph(
        train_df, label_col, le_pol, le_tick, "cpu",
        gap_df=gap_df, test_start=ts, max_age_years=4.0
    )

    print("--- NaN Checks ---")
    print(f"edge_weight NaN: {torch.isnan(edge_weight).any()}")
    print(f"edge_attr NaN: {torch.isnan(edge_attr).any()}")

    # Check columns individually
    print(f"edge_weight max: {edge_weight.max()}, min: {edge_weight.min()}")
    for i in range(edge_attr.shape[1]):
        print(f"edge_attr col {i} NaN: {torch.isnan(edge_attr[:, i]).any()}, max={edge_attr[:, i].max()}, min={edge_attr[:, i].min()}")

    pol_feats = compute_pol_features(train_df, label_col, le_pol, n_pol, gap_df, "cpu")
    print(f"pol_feats NaN: {torch.isnan(pol_feats).any()}")

    state_ids = compute_state_ids(train_df, le_pol, le_state, n_pol, gap_df, "cpu")
    print(f"state_ids NaN: {torch.isnan(state_ids).any()}")

    comp_feats = compute_comp_features(train_df, le_tick, le_sector, le_industry, "cpu")
    print(f"comp_feats NaN: {torch.isnan(comp_feats).any()}")

if __name__ == "__main__":
    check_nan()
