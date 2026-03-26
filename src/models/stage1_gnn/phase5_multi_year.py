"""
phase5_multi_year.py
====================
GNN Core Module. Contains PyG network architectures (Phases 2-5).

Refactored/Audited: 2026-03-20
"""

#!/usr/bin/env python3
"""
Phase 5: Full Multi-Year Validation (2020 - 2024)
==================================================
Extends Phase 3 (Winning Configuration: GAT + Committees) 
across multiple years with walk-forward aggregates.
"""

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from phase3_gnn_committees import (
    load_dataset, monthly_splits, build_committee_graph,
    compute_pol_features, compute_state_ids, compute_comp_features,
    BipartiteCommitteeGAT, EdgePredictor, GNNPredictor,
    eval_gnn, compute_p10, save_metrics, save_predictions
)

from phase4_gnn_curriculum import train_curriculum

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase5_multi")

RESULTS_DIR = Path("experiments/signal_isolation/results/phase5")

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",    type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset("data/processed/ml_dataset_continuous.csv")
    label_col = "Label_Q3"

    le_pol = LabelEncoder().fit(df["BioGuideID"].fillna("UNK"))
    le_tick = LabelEncoder().fit(df["Ticker"].fillna("UNK"))
    le_state = LabelEncoder().fit(df["State"].fillna("UNK"))
    le_sector = LabelEncoder().fit(df["Sector"])
    le_industry = LabelEncoder().fit(df["Industry"])
    mlb_comm = MultiLabelBinarizer().fit(df["comm_list"])

    n_pol = len(le_pol.classes_); n_tick = len(le_tick.classes_); n_comm = len(mlb_comm.classes_)
    log.info("Node Setup: pol=%d tick=%d comm=%d (Committees)", n_pol, n_tick, n_comm)

    # Years set to 2020 - 2024
    years_list = [2020, 2021, 2022, 2023, 2024]
    metrics_rows = []

    for year in years_list:
        log.info("=== Starting Year %d ===", year)
        for _, month, train_df, gap_df, test_df in monthly_splits(df, year, label_col):
            log.info("  %d-%02d  train=%d  test=%d", year, month, len(train_df), len(test_df))

            edge_index, edge_weight, edge_attr = build_committee_graph(
                train_df, label_col, le_pol, le_tick, mlb_comm, device, gap_df=gap_df
            )

            # Use winner: GAT
            gnn = BipartiteCommitteeGAT(n_pol, n_tick, n_comm, len(le_state.classes_), len(le_sector.classes_), len(le_industry.classes_))
            predictor = EdgePredictor(gnn.out_dim * 2)
            model = GNNPredictor(gnn, predictor).to(device)

            model.gnn.set_pol_features(compute_pol_features(train_df, label_col, le_pol, n_pol, gap_df, device))
            model.gnn.set_state_ids(compute_state_ids(train_df, le_pol, le_state, n_pol, gap_df, device))
            model.gnn.set_comp_features(compute_comp_features(train_df, le_tick, le_sector, le_industry, device))

            pol_tr = torch.tensor(le_pol.transform(train_df["BioGuideID"].fillna("UNK")), dtype=torch.long, device=device)
            tick_tr = torch.tensor(le_tick.transform(train_df["Ticker"].fillna("UNK")), dtype=torch.long, device=device) + n_pol
            y_tr = torch.tensor((train_df[label_col] > 0).astype(float).values, dtype=torch.float, device=device)

            # Using Phase 4 Curriculum method
            train_curriculum(model, edge_index, edge_weight, edge_attr, pol_tr, tick_tr, y_tr, device, epochs=args.epochs)

            pol_te = torch.tensor(le_pol.transform(test_df["BioGuideID"].fillna("UNK")), dtype=torch.long, device=device)
            tick_te = torch.tensor(le_tick.transform(test_df["Ticker"].fillna("UNK")), dtype=torch.long, device=device) + n_pol
            y_te = (test_df[label_col] > 0).astype(int).values

            probs = eval_gnn(model, edge_index, edge_weight, edge_attr, pol_te, tick_te)
            probs = np.nan_to_num(probs, nan=0.5)

            if len(np.unique(y_te)) < 2: continue
            auc = roc_auc_score(y_te, probs)
            p10 = compute_p10(y_te, probs, test_df["year"].values, test_df["month"].values)
            log.info("    [%d-%02d] AUC=%.4f  P@10%%=%.4f", year, month, auc, p10)

            metrics_rows.append({"year": year, "month": month, "n_test": len(test_df), "AUROC": round(auc, 6), "P10": round(p10, 6)})

    if metrics_rows:
        out = RESULTS_DIR / "metrics_multi_year_gat.csv"
        pd.DataFrame(metrics_rows).to_csv(out, index=False)
        log.info("Multi-year metrics saved to %s", out)

if __name__ == "__main__":
    main()
