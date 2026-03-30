import sys
import os
import itertools
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Import config globally for the single-threaded loop
import src.config as config
config.INCLUDE_POLITICIAN_BIO = True
config.INCLUDE_IDEOLOGY = True
config.INCLUDE_COMMITTEES = True
config.INCLUDE_COMPANY_SIC = False
config.INCLUDE_DISTRICT_ECON = False
config.INCLUDE_COMPANY_FINANCIALS = False

from src.config import COMPANY_SIC_PATH, INDUSTRY_CROSSWALK_DIR
from src.data_prep.graph_builder import DynamicGraphBuilder
from src.data_prep.lookups import CompanyCategoricalLookup

# Setup Logging to file and console
LOG_DIR = PROJECT_ROOT / "logs" / "stage3_ablation"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / "ablation_stage3.log"

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
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

def train_month(model, optimizer, criterion, x_pol, pol_state, x_comp, comp_sec, comp_ind, edge_index, aux_edges, labels, src_idx, dst_idx, epochs=30):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        h_pol, h_comp = model(x_pol, pol_state, x_comp, comp_sec, comp_ind, edge_index, **aux_edges)
        logits = (h_pol[src_idx] * h_comp[dst_idx]).sum(dim=-1) 
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

def evaluate_month(model, x_pol, pol_state, x_comp, comp_sec, comp_ind, edge_index, aux_edges, src_idx, dst_idx):
    model.eval()
    with torch.no_grad():
        h_pol, h_comp = model(x_pol, pol_state, x_comp, comp_sec, comp_ind, edge_index, **aux_edges)
        logits = (h_pol[src_idx] * h_comp[dst_idx]).sum(dim=-1)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs

def run_ablation_config(year, epochs, device, df, le_pol, le_tick, le_state, le_sector, le_ind):
    n_pol = len(le_pol.classes_)
    n_tick = len(le_tick.classes_)
    
    builder = DynamicGraphBuilder()
    metrics_rows = []
    edge_counts_rows = []

    for month in range(1, 13):
        test_start = pd.Timestamp(year, month, 1)
        test_end = test_start + pd.DateOffset(months=1)
        
        train_df = df[df["Filed"] < test_start].copy()
        test_df = df[(df["Filed"] >= test_start) & (df["Filed"] < test_end)].copy()
        test_df = test_df.dropna(subset=["Excess_Return_6M"])
        
        if len(train_df) == 0 or len(test_df) == 0:
            continue
            
        x_pol = builder.build_pol_features(train_df, le_pol, n_pol, test_start, device, label_col="Excess_Return_6M")
        x_comp = builder.build_comp_features(train_df, le_tick, n_tick, test_start, device)

        pol_state = get_categorical_ids(train_df, "BioGuideID", "State", le_state, n_pol, le_pol).to(device)
        comp_sec = get_categorical_ids(train_df, "Ticker", "Sector", le_sector, n_tick, le_tick).to(device)
        comp_ind = get_categorical_ids(train_df, "Ticker", "Industry", le_ind, n_tick, le_tick).to(device)

        src_train = le_pol.transform(train_df["BioGuideID"].fillna("UNK"))
        dst_train = le_tick.transform(train_df["Ticker"].fillna("UNK"))
        edge_index_train = torch.tensor(np.array([src_train, dst_train]), dtype=torch.long, device=device)
        
        # Build new Heterogeneous Edges dynamically
        aux_edges = builder.build_auxiliary_edges(train_df, le_pol, le_tick, test_start, device)
        
        # Track Edge Counts for diagnostics
        counts = {"month": month, "trade_edges": edge_index_train.shape[1], "lobby_strong": 0, "lobby_weak": 0, "campaign": 0}
        if 'lobby_strong' in aux_edges: counts["lobby_strong"] = aux_edges['lobby_strong']['edge_index'].shape[1]
        if 'lobby_weak' in aux_edges: counts["lobby_weak"] = aux_edges['lobby_weak']['edge_index'].shape[1]
        if 'campaign' in aux_edges: counts["campaign"] = aux_edges['campaign']['edge_index'].shape[1]
        edge_counts_rows.append(counts)

        y_train = torch.tensor((train_df["Excess_Return_6M"] > 0).astype(float).values, dtype=torch.float, device=device)

        model = builder.init_model(len(le_state.classes_), len(le_sector.classes_), len(le_ind.classes_), out_dim=32, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        
        pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(1, y_train.sum())], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_month(model, optimizer, criterion, x_pol, pol_state, x_comp, comp_sec, comp_ind, edge_index_train, aux_edges, y_train, src_train, dst_train, epochs=epochs)

        src_test = le_pol.transform(test_df["BioGuideID"].fillna("UNK"))
        dst_test = le_tick.transform(test_df["Ticker"].fillna("UNK"))
        y_test = (test_df["Excess_Return_6M"] > 0).astype(int).values
        
        probs = evaluate_month(model, x_pol, pol_state, x_comp, comp_sec, comp_ind, edge_index_train, aux_edges, src_test, dst_test)
        
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, probs)
            p10 = compute_p10(y_test, probs, test_df["Filed"].dt.year.values, test_df["Filed"].dt.month.values)
            log.info(f"Month {month:02d} | AUC: {auc:.4f} | P@10%: {p10:.4f}")
            metrics_rows.append({"month": month, "AUC": auc, "P10": p10})
            
    mdf = pd.DataFrame(metrics_rows) if metrics_rows else pd.DataFrame()
    edf = pd.DataFrame(edge_counts_rows) if edge_counts_rows else pd.DataFrame()
    return mdf, edf

def plot_diagnostics(results_df, all_monthly_metrics, all_edge_counts):
    sns.set_theme(style="whitegrid")
    
    # 1. Overall Performance Bar Chart
    plt.figure(figsize=(12, 6))
    results_melted = results_df.melt(id_vars=["Config_Name"], value_vars=["Mean_AUC", "Mean_P10"], 
                                     var_name="Metric", value_name="Score")
    sns.barplot(data=results_melted, y="Config_Name", x="Score", hue="Metric", palette="muted")
    plt.title("Stage 3 Ablation: Overall Performance by Edge Configuration", fontsize=14)
    plt.xlabel("Score")
    plt.ylabel("Configuration (Strong | Weak | Campaign)")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(LOG_DIR / "ablation_overall_performance.png", dpi=300)
    plt.close()

    # 2. Monthly AUC Timeline
    plt.figure(figsize=(12, 6))
    for cfg_name, mdf in all_monthly_metrics.items():
        if not mdf.empty:
            plt.plot(mdf["month"], mdf["AUC"], marker='o', label=cfg_name)
    plt.title("Monthly AUC Timeline across Configurations", fontsize=14)
    plt.xlabel("Month")
    plt.ylabel("AUC")
    plt.xticks(range(1, 13))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(LOG_DIR / "ablation_monthly_auc_timeline.png", dpi=300)
    plt.close()

    # 3. Auxiliary Edge Counts
    all_true_cfg = "S:True | W:True | C:True"
    if all_true_cfg in all_edge_counts and not all_edge_counts[all_true_cfg].empty:
        edf = all_edge_counts[all_true_cfg]
        plt.figure(figsize=(10, 5))
        plt.plot(edf["month"], edf["lobby_strong"], marker='s', label="Strong Lobbying")
        plt.plot(edf["month"], edf["lobby_weak"], marker='^', label="Weak Lobbying")
        plt.plot(edf["month"], edf["campaign"], marker='d', label="Campaign Finance")
        plt.title("Auxiliary Edge Volume Over Time (Cumulative Graph)", fontsize=14)
        plt.xlabel("Month")
        plt.ylabel("Number of Edges")
        plt.xticks(range(1, 13))
        plt.legend()
        plt.tight_layout()
        plt.savefig(LOG_DIR / "auxiliary_edge_counts.png", dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Starting Single-Threaded Edge Type Ablation Study. Device: {device}")

    # PRE-COMPRESSION CHECK
    log.info("Checking for uncompressed event logs and caching as parquets if needed...")
    from src.data_prep.lookups.lobbying_lookups import LobbyingLookup
    from src.data_prep.lookups.campaign_lookups import CampaignLookup
    _ = LobbyingLookup()
    _ = CampaignLookup()
    log.info("Caching complete.")

    csv_path = "data/processed/v9_transactions.csv"
    df = pd.read_csv(csv_path)
    df["Filed"] = pd.to_datetime(df["Filed"])
    df = df.sort_values("Filed").reset_index(drop=True)
    
    for col in ["State", "Sector", "Industry"]:
        df[col] = df[col] if col in df.columns else "Unknown"
        df[col] = df[col].fillna("Unknown")

    cat_lookup = CompanyCategoricalLookup(COMPANY_SIC_PATH, INDUSTRY_CROSSWALK_DIR)
    mask = (df["Sector"] == "Unknown") | (df["Industry"] == "Unknown")
    if mask.any():
        resolved_cats = df.loc[mask].apply(lambda row: cat_lookup.get_sector_industry(row["Ticker"], row["Filed"]), axis=1)
        df.loc[mask, "Sector"] = resolved_cats.apply(lambda x: x[0])
        df.loc[mask, "Industry"] = resolved_cats.apply(lambda x: x[1])

    le_pol = LabelEncoder().fit(df["BioGuideID"].fillna("UNK"))
    le_tick = LabelEncoder().fit(df["Ticker"].fillna("UNK"))
    le_state = LabelEncoder().fit(df["State"])
    le_sector = LabelEncoder().fit(df["Sector"])
    le_ind = LabelEncoder().fit(df["Industry"])

    edge_options = [False, True]
    combinations = list(itertools.product(edge_options, repeat=3))
    
    results = []
    all_monthly_metrics = {}
    all_edge_counts = {}
    
    for strong, weak, campaign in combinations:
        cfg_name = f"S:{strong} | W:{weak} | C:{campaign}"

        # --- INJECT PRECOMPUTED RESULTS TO SAVE TIME ---
        if strong is False and weak is False and campaign is False:
            log.info(f"Skipping previously completed Config -> {cfg_name}")
            mdf = pd.DataFrame({
                "month": range(1, 13),
                "AUC": [0.5831, 0.6980, 0.5964, 0.5748, 0.5590, 0.6377, 0.6176, 0.6018, 0.6479, 0.5864, 0.5351, 0.6051],
                "P10": [0.3000, 0.5424, 0.6552, 0.5068, 0.5663, 0.7108, 0.7455, 0.6316, 0.6400, 0.6491, 0.5872, 0.3488]
            })
            all_monthly_metrics[cfg_name] = mdf
            all_edge_counts[cfg_name] = pd.DataFrame() # Edge counts not plotted for baseline
            results.append({
                "Config_Name": cfg_name, "Strong_Lobby": strong, "Weak_Lobby": weak, "Campaign": campaign,
                "Mean_AUC": 0.6036, "Mean_P10": 0.5736
            })
            continue

        if strong is False and weak is False and campaign is True:
            log.info(f"Skipping previously completed Config -> {cfg_name}")
            mdf = pd.DataFrame({
                "month": range(1, 13),
                "AUC": [0.5732, 0.6844, 0.6188, 0.5903, 0.5731, 0.5981, 0.6619, 0.6135, 0.6290, 0.5382, 0.5337, 0.6178],
                "P10": [0.2800, 0.5254, 0.7586, 0.5616, 0.4940, 0.5904, 0.6364, 0.5053, 0.6000, 0.5263, 0.5229, 0.4651]
            })
            all_monthly_metrics[cfg_name] = mdf
            all_edge_counts[cfg_name] = pd.DataFrame() 
            results.append({
                "Config_Name": cfg_name, "Strong_Lobby": strong, "Weak_Lobby": weak, "Campaign": campaign,
                "Mean_AUC": 0.6027, "Mean_P10": 0.5388
            })
            continue
        # -----------------------------------------------

        config.INCLUDE_LOBBYING_SPONSORSHIP = strong
        config.INCLUDE_LOBBYING_VOTING = weak
        config.INCLUDE_CAMPAIGN_FINANCE = campaign
        
        log.info("\n" + "="*50)
        log.info(f"Testing Config -> {cfg_name}")
        log.info("="*50)
        
        mdf, edf = run_ablation_config(
            args.year, args.epochs, device, df, 
            le_pol, le_tick, le_state, le_sector, le_ind
        )
        
        all_monthly_metrics[cfg_name] = mdf
        all_edge_counts[cfg_name] = edf
        
        mean_auc = mdf['AUC'].mean() if not mdf.empty else 0.0
        mean_p10 = mdf['P10'].mean() if not mdf.empty else 0.0
        
        log.info(f"Config Average -> AUC: {mean_auc:.4f} | P@10%: {mean_p10:.4f}")
        results.append({
            "Config_Name": cfg_name,
            "Strong_Lobby": strong,
            "Weak_Lobby": weak,
            "Campaign": campaign,
            "Mean_AUC": mean_auc,
            "Mean_P10": mean_p10
        })
        
    log.info("\n" + "="*50)
    log.info("ABLATION STUDY FINAL RESULTS")
    log.info("="*50)
    res_df = pd.DataFrame(results).sort_values(by="Mean_AUC", ascending=False)
    for _, row in res_df.iterrows():
        log.info(f"{row['Config_Name']:<30} ==> AUC: {row['Mean_AUC']:.4f} | P@10%: {row['Mean_P10']:.4f}")
        
    # Generate Plots
    plot_diagnostics(res_df, all_monthly_metrics, all_edge_counts)
    log.info(f"Diagnostics saved to {LOG_DIR}")
    
if __name__ == "__main__":
    main()