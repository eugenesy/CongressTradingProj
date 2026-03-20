"""
phase3_scaffold.py
==================
GNN Core Module. Contains PyG network architectures (Phases 2-5).

Refactored/Audited: 2026-03-20
"""

#!/usr/bin/env python3
"""
Phase 3 Scaffold: Committee Hyperedges [Planning Draft]
======================================================
1. Parse semicolon-separated 'Committees' from Politician entries.
2. Form Committee nodes: dim N_pol + N_tick + N_comm.
3. Draw edge weights=1.0 for Politician <-> Committee links.
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

def load_committees(csv_path="data/processed/ml_dataset_continuous.csv"):
    df = pd.read_csv(csv_path)
    
    # 1. Split committees
    df["Committees"] = df["Committees"].fillna("").astype(str)
    df["comm_list"] = df["Committees"].apply(lambda x: [c.strip() for c in x.split(";") if c.strip()])
    
    # 2. Fit MultiLabelBinarizer or LabelEncoder for unique committees
    mld = MultiLabelBinarizer()
    mld.fit(df["comm_list"])
    unique_comms = mld.classes_
    
    print(f"Unique Committees Count: {len(unique_comms)}")
    print(f"Sample Committees: {unique_comms[:5]}")
    
    # 3. Form map of BioGuideID -> Committee list
    pol_to_comm = df.groupby("BioGuideID")["comm_list"].first().to_dict()
    
    return unique_comms, pol_to_comm

if __name__ == "__main__":
    comms, p_map = load_committees()
