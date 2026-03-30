import sys
import os
from pathlib import Path
import itertools

# Add the project root to the Python path so 'src' can be found
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from src.config import COMPANY_SIC_PATH, INDUSTRY_CROSSWALK_DIR
import src.data_prep.graph_builder as gb
from src.data_prep.lookups import CompanyCategoricalLookup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ablation")

def compute_p10(y_true, y_prob, years, months, k=0.10):
    df = pd.DataFrame({"l": y_true, "p": y_prob, "y": years, "m": months})
    precs = []
    for _, g in df.groupby(["y", "m"]):
        if len(g) < 5: continue
        n_top = max(1, int(len(g) * k))
        precs.append(g.nlargest(n_top, "p")["l"].mean())
    return float(np.mean(precs)) if precs else 0.0

def get_categorical_ids(df, entity_col, cat_col, encoder, num_entities, entity_encoder):
    ids = np.zeros(num_entities, dtype=np.int64)
    latest_cats = df.drop_duplicates(entity_col, keep='last')
    
    for _, row in latest_cats.iterrows():
        entity = row.get(entity_col)
        cat_val = row.get(cat_col)
        if pd.isna(entity) or pd.isna(cat_val): continue
        try:
            e_idx = entity_encoder.transform([entity])[0]
            c_idx = encoder.transform([cat_val])[0]
            ids[e_idx] = c_idx
        except ValueError:
            pass
    return torch.tensor(ids, dtype=torch.long)

def train_month(model, optimizer, criterion, x_pol, pol_state, x_comp, comp_sec, comp_ind, edge_index, labels, src_idx, dst_idx, epochs=30):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        h_pol, h_comp = model(x_pol, pol_state, x_comp, comp_sec, comp_ind, edge_index)
        logits = (h_pol[src_idx] * h_comp[dst_idx]).sum(dim=-1) 
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

def evaluate_month(model, x_pol, pol_state, x_comp, comp_sec, comp_ind, edge_index, src_idx, dst_idx):
    model.eval()
    with torch.no_grad():
        h_pol, h_comp = model(x_pol, pol_state, x_comp, comp_sec, comp_ind, edge_index)
        logits = (h_pol[src_idx] * h_comp[dst_idx]).sum(dim=-1)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Starting Ablation Study. Device: {device}")

    # 1. Load and Clean Data ONCE
    csv_path = "data/processed/v9_transactions.csv"
    log.info(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df["Filed"] = pd.to_datetime(df["Filed"])
    df = df.sort_values("Filed").reset_index(drop=True)
    
    for col in ["State", "Sector", "Industry"]:
        if col not in df.columns:
            df[col] = "Unknown"
        else:
            df[col] = df[col].fillna("Unknown")

    log.info("Resolving missing Sectors and Industries via temporal crosswalks...")
    cat_lookup = CompanyCategoricalLookup(COMPANY_SIC_PATH, INDUSTRY_CROSSWALK_DIR)
    mask = (df["Sector"] == "Unknown") | (df["Industry"] == "Unknown")
    if mask.any():
        resolved_cats = df.loc[mask].apply(lambda row: cat_lookup.get_sector_industry(row["Ticker"], row["Filed"]), axis=1)
        df.loc[mask, "Sector"] = resolved_cats.apply(lambda x: x[0])
        df.loc[mask, "Industry"] = resolved_cats.apply(lambda x: x[1])

    label_col = "Excess_Return_6M"

    le_pol = LabelEncoder().fit(df["BioGuideID"].fillna("UNK"))
    le_tick = LabelEncoder().fit(df["Ticker"].fillna("UNK"))
    le_state = LabelEncoder().fit(df["State"])
    le_sector = LabelEncoder().fit(df["Sector"])
    le_ind = LabelEncoder().fit(df["Industry"])

    n_pol = len(le_pol.classes_)
    n_tick = len(le_tick.classes_)

    # 2. Pre-slice the 12-month evaluation windows
    log.info(f"Pre-slicing walk-forward windows for {args.year}...")
    monthly_splits = []
    for month in range(1, 13):
        test_start = pd.Timestamp(args.year, month, 1)
        test_end = test_start + pd.DateOffset(months=1)
        
        train_df = df[df["Filed"] < test_start].copy()
        test_df = df[(df["Filed"] >= test_start) & (df["Filed"] < test_end)].copy()
        test_df = test_df.dropna(subset=[label_col])
        
        if len(train_df) > 0 and len(test_df) > 0:
            monthly_splits.append((month, test_start, train_df, test_df))

    # 3. Define the features to ablate
    ablation_flags = [
        "INCLUDE_IDEOLOGY",
        "INCLUDE_COMMITTEES",
        "INCLUDE_COMPANY_SIC"
    ]

    # Explicitly hold the fixed features
    gb.INCLUDE_PERFORMANCE = True
    gb.INCLUDE_POLITICIAN_BIO = True
    gb.INCLUDE_DISTRICT_ECON = False
    gb.INCLUDE_COMPANY_FINANCIALS = False

    # Generate all 8 combinations of True/False for the 3 active flags
    combinations = list(itertools.product([True, False], repeat=len(ablation_flags)))
    
    results = []
    log.info(f"Starting execution of {len(combinations)} feature combinations...")

    # 4. Run the combination search
    for combo_idx, combo in enumerate(combinations):
        config_state = dict(zip(ablation_flags, combo))
        for flag_name, flag_val in config_state.items():
            setattr(gb, flag_name, flag_val)

        combo_str = " | ".join([f"{k}={v}" for k, v in config_state.items() if v])
        if not combo_str: combo_str = "BASE GNN + Perf & Bio (No additional features)"
        
        log.info("-" * 60)
        log.info(f"Run {combo_idx + 1}/{len(combinations)} -> Active Optional: {combo_str}")

        builder = gb.DynamicGraphBuilder()
        combo_metrics = []

        # Walk-Forward Evaluation
        for month, test_start, train_df, test_df in monthly_splits:
            x_pol = builder.build_pol_features(train_df, le_pol, n_pol, test_start, device, label_col=label_col)
            x_comp = builder.build_comp_features(train_df, le_tick, n_tick, test_start, device)

            pol_state = get_categorical_ids(train_df, "BioGuideID", "State", le_state, n_pol, le_pol).to(device)
            comp_sec = get_categorical_ids(train_df, "Ticker", "Sector", le_sector, n_tick, le_tick).to(device)
            comp_ind = get_categorical_ids(train_df, "Ticker", "Industry", le_ind, n_tick, le_tick).to(device)

            src_train = le_pol.transform(train_df["BioGuideID"].fillna("UNK"))
            dst_train = le_tick.transform(train_df["Ticker"].fillna("UNK"))
            edge_index_train = torch.tensor(np.array([src_train, dst_train]), dtype=torch.long, device=device)
            
            y_train = torch.tensor((train_df[label_col] > 0).astype(float).values, dtype=torch.float, device=device)

            model = builder.init_model(len(le_state.classes_), len(le_sector.classes_), len(le_ind.classes_), out_dim=32, device=device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
            
            pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(1, y_train.sum())], device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            train_month(model, optimizer, criterion, x_pol, pol_state, x_comp, comp_sec, comp_ind, edge_index_train, y_train, src_train, dst_train, epochs=args.epochs)

            src_test = le_pol.transform(test_df["BioGuideID"].fillna("UNK"))
            dst_test = le_tick.transform(test_df["Ticker"].fillna("UNK"))
            y_test = (test_df[label_col] > 0).astype(int).values
            
            probs = evaluate_month(model, x_pol, pol_state, x_comp, comp_sec, comp_ind, edge_index_train, src_test, dst_test)
            
            if len(np.unique(y_test)) > 1:
                auc = roc_auc_score(y_test, probs)
                p10 = compute_p10(y_test, probs, test_df["Filed"].dt.year.values, test_df["Filed"].dt.month.values)
                combo_metrics.append({"month": month, "AUC": auc, "P10": p10})

        if combo_metrics:
            mdf = pd.DataFrame(combo_metrics)
            mean_auc = mdf['AUC'].mean()
            mean_p10 = mdf['P10'].mean()
            log.info(f"Score for Combo {combo_idx + 1} -> Mean AUC: {mean_auc:.4f} | Mean P@10%: {mean_p10:.4f}")
            
            res_dict = {"Mean_AUC": mean_auc, "Mean_P10": mean_p10}
            res_dict.update(config_state)
            # Include the static flags for completeness in the CSV
            res_dict.update({"INCLUDE_PERFORMANCE": True, "INCLUDE_POLITICIAN_BIO": True})
            results.append(res_dict)
            
    # 5. Save and display the optimal configurations
    log.info("==== ABLATION STUDY COMPLETE ====")
    results_df = pd.DataFrame(results)
    
    # Sort by best AUC score to identify the optimal attribute set
    results_df = results_df.sort_values(by="Mean_AUC", ascending=False).reset_index(drop=True)
    
    out_path = PROJECT_ROOT / "docs" / "ablation_results.csv"
    out_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(out_path, index=False)
    log.info(f"Detailed results matrix saved to {out_path}")
    
    print("\n🚀 Top 5 Feature Combinations (by AUC):")
    print(results_df.head(5).to_string())

if __name__ == "__main__":
    main()