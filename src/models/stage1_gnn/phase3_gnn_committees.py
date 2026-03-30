"""
phase3_gnn_committees.py
========================
GNN Core Module. Contains PyG network architectures (Phases 2-5).

Refactored/Audited: 2026-03-20
"""

#!/usr/bin/env python3
"""
Phase 3: Committee Hyperedges
=============================
Extends Phase 2 with Committee nodes for Politician <-> Committee diffusion.

Graph Structure:
  - Politician nodes: index 0 to N_pol-1
  - Company nodes: index N_pol to N_pol + N_tick - 1
  - Committee nodes: index N_pol + N_tick to N_pol + N_tick + N_comm - 1

Edges:
  - Politician <-> Company (Trade edges, 5-dim attrs, weight from list count)
  - Politician <-> Committee (Membership edges, 5-dim attrs padded, weight=1.0)
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch_geometric.nn import GraphConv, GATConv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase3_gnn")

RESULTS_DIR = Path("experiments/signal_isolation/results/phase3")

# ── Phase 2 Constants ──────────────────────────────────────────────────────
POL_FEAT_DIM  = 7
EDGE_ATTR_DIM = 5  # [win_rate, count_norm, ls, lg, ib]

TRADE_SIZE_MAP = {
    "$1,001 - $15,000":          1,
    "$15,001 - $50,000":         2,
    "$50,001 - $100,000":        3,
    "$100,001 - $250,000":       4,
    "$250,001 - $500,000":       5,
    "$500,001 - $1,000,000":     6,
    "$1,000,001 - $5,000,000":   7,
    "$5,000,001 - $25,000,000":  8,
}

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── Data loading ───────────────────────────────────────────────────────────

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Filed"]  = pd.to_datetime(df["Filed"])
    df["Traded"] = pd.to_datetime(df["Traded"])
    df = df.sort_values("Filed").reset_index(drop=True)

    # Preprocessing
    df["Filing_Gap"]     = pd.to_numeric(df["Filing_Gap"], errors="coerce").fillna(30).clip(0)
    df["log_filing_gap"] = np.log1p(df["Filing_Gap"])
    df["trade_size_ord"] = df["Trade_Size_USD"].map(TRADE_SIZE_MAP).fillna(2)
    df["log_trade_size"] = np.log1p(df["trade_size_ord"])

    lo = df["max_excess_6m"].quantile(0.01)
    hi = df["max_excess_6m"].quantile(0.99)
    df["max_excess_6m_w"] = df["max_excess_6m"].clip(lo, hi)

    df["Label_Q3"]     = (df["max_excess_6m_w"] > 0.178).astype(int)
    df["Label_Median"] = (df["max_excess_6m_w"] > 0.088).astype(int)
    df["is_buy"]       = df["Transaction"].str.lower().str.contains("purchase", na=False).astype(float)

    df["Sector"]   = df["Sector"].fillna("Unknown")
    df["Industry"] = df["Industry"].fillna("Unknown")
    df["In_SP100"]  = pd.to_numeric(df["In_SP100"], errors="coerce").fillna(0).astype(float)
    df["year"]     = df["Filed"].dt.year
    df["month"]    = df["Filed"].dt.month

    # Parse Commmittees
    df["Committees"] = df["Committees"].fillna("").astype(str)
    df["comm_list"] = df["Committees"].apply(lambda x: [c.strip() for c in x.split(";") if c.strip()])

    if "transaction_id" not in df.columns:
        df["transaction_id"] = df.index.astype(str)

    return df


# ── Splits ─────────────────────────────────────────────────────────────────

def monthly_splits(df, years, label_col):
    if isinstance(years, int): years = [years]
    h_days = 180  # 6M
    for year in years:
        for month in range(1, 13):
            ts = pd.Timestamp(year, month, 1)
            ne = ts + pd.DateOffset(months=1)
            test_df  = df[(df["Filed"] >= ts) & (df["Filed"] < ne) & df[label_col].notna()].copy()
            pre      = df[df["Filed"] < ts].copy()
            pre["Resolution"] = pre["Filed"] + pd.Timedelta(days=h_days)
            train_df = pre[(pre["Resolution"] < ts) & pre[label_col].notna()].copy()
            gap_df   = pre[pre["Resolution"] >= ts].copy()
            cutoff   = ts - pd.Timedelta(days=4 * 365.25)
            train_df = train_df[train_df["Filed"] >= cutoff].copy()
            if len(train_df) == 0 or len(test_df) == 0: continue
            yield year, month, train_df, gap_df, test_df


# ── Node Features (Unchanged logic) ─────────────────────────────────────────

def compute_pol_features(train_df, label_col, le_pol, n_pol, gap_df, device):
    feats = np.zeros((n_pol, POL_FEAT_DIM), dtype=np.float32)
    feats[:, 0] = 0.5; feats[:, 2] = 0.5
    df_c = train_df.copy(); df_c["_bio"] = df_c["BioGuideID"].fillna("UNK")
    df_c["_lbl"] = (df_c[label_col] > 0).astype(float)
    df_c["_buy"] = df_c["is_buy"]
    for bio_id, grp in df_c.groupby("_bio"):
        try: idx = le_pol.transform([bio_id])[0]
        except ValueError: continue
        feats[idx, 0] = grp["_lbl"].sum() / len(grp); feats[idx, 1] = np.log1p(len(grp)); feats[idx, 2] = grp["_buy"].mean()
        row = grp.iloc[0]
        feats[idx, 3] = 1.0 if str(row.get("Chamber","")).lower() == "senate" else 0.0
        py = str(row.get("Party","")).lower()
        if "democrat" in py: feats[idx, 4] = 1.0
        elif "republican" in py: feats[idx, 5] = 1.0
        else: feats[idx, 6] = 1.0
    return torch.tensor(feats, dtype=torch.float32, device=device)

def compute_state_ids(train_df, le_pol, le_state, n_pol, gap_df, device):
    s = np.zeros(n_pol, dtype=np.int64)
    for bio_id, state_val in zip(train_df["BioGuideID"].fillna("UNK"), train_df["State"].fillna("UNK")):
        try: s[le_pol.transform([bio_id])[0]] = le_state.transform([state_val])[0]
        except ValueError: pass
    return torch.tensor(s, dtype=torch.long, device=device)

def compute_comp_features(train_df, le_tick, le_sector, le_industry, device):
    n_tick = len(le_tick.classes_); comp = np.zeros((n_tick, 3), dtype=np.float32)
    t_info = {}
    for _, r in train_df.iterrows():
        tick = str(r.get("Ticker","UNK")).strip()
        if tick not in t_info:
            t_info[tick] = {"s": str(r.get("Sector","Unknown")), "i": str(r.get("Industry","Unknown")), "sp": r.get("In_SP100", 0)}
    for t, info in t_info.items():
        try:
            ti = le_tick.transform([t])[0]
            comp[ti, 0] = le_sector.transform([info["s"]])[0]; comp[ti, 1] = le_industry.transform([info["i"]])[0]; comp[ti, 2] = info["sp"]
        except ValueError: pass
    return torch.tensor(comp, dtype=torch.float32, device=device)


# ── Phase 3: Enhanced Graph with Committees ─────────────────────────────────

def build_committee_graph(train_df, label_col, le_pol, le_tick, mlb_comm, device, gap_df=None):
    """
    Combines Bipartite and Hyperedge links.
    Edges: 
      - p <-> c (weight = log1p count, attrs=aggregated)
      - p <-> comm (weight=1.0, attrs=padded default zeros)
    """
    n_pol  = len(le_pol.classes_)
    n_tick = len(le_tick.classes_)
    n_comm = len(mlb_comm.classes_)

    p_ids = le_pol.transform(train_df["BioGuideID"].fillna("UNK"))
    t_ids = le_tick.transform(train_df["Ticker"].fillna("UNK"))

    t_df = train_df.copy()
    t_df["p_id"] = p_ids; t_df["t_id"] = t_ids
    t_df["lbl"]  = (t_df[label_col] > 0).astype(float).values
    t_df["ls"]   = t_df["log_trade_size"].fillna(np.log1p(2)).values.astype(np.float32)
    t_df["lg"]   = t_df["log_filing_gap"].fillna(np.log1p(30)).values.astype(np.float32)
    t_df["ib"]   = t_df["is_buy"].fillna(0.5).values.astype(np.float32)

    agg = t_df.groupby(["p_id", "t_id"]).agg(
        cnt=("lbl", "count"), pos=("lbl", "sum"),
        sum_ls=("ls", "sum"), sum_lg=("lg", "sum"), sum_ib=("ib", "sum")
    ).reset_index()

    if gap_df is not None and len(gap_df) > 0:
        gap_df = gap_df.copy()
        gap_df["p_id"] = le_pol.transform(gap_df["BioGuideID"].fillna("UNK"))
        gap_df["t_id"] = le_tick.transform(gap_df["Ticker"].fillna("UNK"))
        gap_agg = gap_df.groupby(["p_id", "t_id"]).size().rename("gap_cnt").reset_index()
        merged = pd.merge(agg, gap_agg, on=["p_id", "t_id"], how="outer").fillna(0)
    else:
        merged = agg; merged["gap_cnt"] = 0

    merged["total"] = merged["cnt"] + merged["gap_cnt"]
    merged["wr"]    = np.where(merged["cnt"] > 0, merged["pos"] / merged["cnt"], 0.5)
    merged["avg_ls"] = np.where(merged["cnt"] > 0, merged["sum_ls"] / merged["cnt"], np.log1p(2))
    merged["avg_lg"] = np.where(merged["cnt"] > 0, merged["sum_lg"] / merged["cnt"], np.log1p(30))
    merged["avg_ib"] = np.where(merged["cnt"] > 0, merged["sum_ib"] / merged["cnt"], 0.5)
    merged["lc"]    = np.log1p(merged["total"])

    src_l, dst_l, lc_l, wr_l, ls_l, lg_l, ib_l = [], [], [], [], [], [], []
    
    # 1. Trade Edges (p <-> c)
    for _, r in merged.iterrows():
        p, t = int(r["p_id"]), int(r["t_id"])
        cnode = n_pol + t
        for u, v in [(p, cnode), (cnode, p)]:
            src_l.append(u); dst_l.append(v)
            lc_l.append(r["lc"]); wr_l.append(r["wr"])
            ls_l.append(r["avg_ls"]); lg_l.append(r["avg_lg"]); ib_l.append(r["avg_ib"])

    # 2. Committee Edges (p <-> comm)
    # Form a map of BioGuideID -> Comm list for faster iterations
    p_comm_df = train_df.groupby("BioGuideID")["comm_list"].first().reset_index()
    for _, r in p_comm_df.iterrows():
        bio = r["BioGuideID"]
        try: p = le_pol.transform([bio])[0]
        except ValueError: continue
        try:
            vec = mlb_comm.transform([r["comm_list"]])[0]  # binary membership
        except ValueError: continue
        
        for comm_idx in np.where(vec > 0)[0]:
            comm_node = n_pol + n_tick + comm_idx
            # Edges p <-> comm
            for u, v in [(p, comm_node), (comm_node, p)]:
                src_l.append(u); dst_l.append(v)
                lc_l.append(1.0) # Equal weight for hyperedges
                wr_l.append(0.5) # Pad default values
                ls_l.append(np.log1p(2)); lg_l.append(np.log1p(30)); ib_l.append(0.5)

    lc_arr = np.array(lc_l, dtype=np.float32)
    lc_norm = lc_arr / (lc_arr.max() if lc_arr.max() > 0 else 1.0)

    edge_index = torch.tensor([src_l, dst_l], dtype=torch.long, device=device)
    edge_weight = torch.tensor(lc_arr, dtype=torch.float, device=device)
    edge_attr = torch.stack([
        torch.tensor(wr_l, dtype=torch.float, device=device),
        torch.tensor(lc_norm, dtype=torch.float, device=device),
        torch.tensor(ls_l, dtype=torch.float, device=device),
        torch.tensor(lg_l, dtype=torch.float, device=device),
        torch.tensor(ib_l, dtype=torch.float, device=device),
    ], dim=1)

    return edge_index, edge_weight, edge_attr


# ── Hyperedge Models ───────────────────────────────────────────────────────

class BipartiteCommitteeSAGE(nn.Module):
    def __init__(self, n_pol, n_tick, n_comm, n_states, n_sectors, n_industries,
                 emb_dim=32, hidden_dim=64, out_dim=32, dropout=0.2):
        super().__init__()
        self.n_pol = n_pol; self.n_tick = n_tick; self.n_comm = n_comm
        self.pol_proj      = nn.Sequential(nn.Linear(POL_FEAT_DIM, emb_dim), nn.ReLU())
        self.state_emb     = nn.Embedding(n_states, emb_dim)
        self.sector_emb    = nn.Embedding(n_sectors, 8)
        self.industry_emb  = nn.Embedding(n_industries, 8)
        self.comp_proj     = nn.Sequential(nn.Linear(17, emb_dim), nn.ReLU())
        self.comm_emb      = nn.Embedding(n_comm, emb_dim)  # Committee learned embeddings
        self.norm          = nn.LayerNorm(emb_dim)

        self.conv1   = GraphConv(emb_dim, hidden_dim)
        self.conv2   = GraphConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim
        self._pol_features = self._state_ids = self._comp_features = None

    def set_pol_features(self, t):  self._pol_features  = t
    def set_state_ids(self, t):     self._state_ids     = t
    def set_comp_features(self, t): self._comp_features = t

    def get_node_features(self, device):
        pol  = self.pol_proj(self._pol_features) + self.state_emb(self._state_ids)
        s_e  = self.sector_emb(self._comp_features[:, 0].long())
        i_e  = self.industry_emb(self._comp_features[:, 1].long())
        comp = self.comp_proj(torch.cat([s_e, i_e, self._comp_features[:, 2].unsqueeze(1)], dim=1))
        # Committee Node features: Learned Embedding directly
        comm = self.comm_emb(torch.arange(self.n_comm, device=device))
        return self.norm(torch.cat([pol, comp, comm], dim=0))

    def forward(self, edge_index, edge_weight=None, edge_attr=None):
        x = self.get_node_features(edge_index.device)
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight)); x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x


class BipartiteCommitteeGAT(nn.Module):
    def __init__(self, n_pol, n_tick, n_comm, n_states, n_sectors, n_industries,
                 emb_dim=32, hidden_dim=64, out_dim=32, heads=4, dropout=0.2):
        super().__init__()
        self.n_pol = n_pol; self.n_tick = n_tick; self.n_comm = n_comm
        self.pol_proj      = nn.Sequential(nn.Linear(POL_FEAT_DIM, emb_dim), nn.ReLU())
        self.state_emb     = nn.Embedding(n_states, emb_dim)
        self.sector_emb    = nn.Embedding(n_sectors, 8)
        self.industry_emb  = nn.Embedding(n_industries, 8)
        self.comp_proj     = nn.Sequential(nn.Linear(17, emb_dim), nn.ReLU())
        self.comm_emb      = nn.Embedding(n_comm, emb_dim)
        self.norm          = nn.LayerNorm(emb_dim)

        self.conv1   = GATConv(emb_dim, hidden_dim // heads, heads=heads, dropout=dropout, edge_dim=EDGE_ATTR_DIM)
        self.conv2   = GATConv(hidden_dim, out_dim, heads=1, concat=False, dropout=dropout, edge_dim=EDGE_ATTR_DIM)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim
        self._pol_features = self._state_ids = self._comp_features = None

    def set_pol_features(self, t):  self._pol_features  = t
    def set_state_ids(self, t):     self._state_ids     = t
    def set_comp_features(self, t): self._comp_features = t

    def get_node_features(self, device):
        pol  = self.pol_proj(self._pol_features) + self.state_emb(self._state_ids)
        s_e  = self.sector_emb(self._comp_features[:, 0].long())
        i_e  = self.industry_emb(self._comp_features[:, 1].long())
        comp = self.comp_proj(torch.cat([s_e, i_e, self._comp_features[:, 2].unsqueeze(1)], dim=1))
        comm = self.comm_emb(torch.arange(self.n_comm, device=device))
        return self.norm(torch.cat([pol, comp, comm], dim=0))

    def forward(self, edge_index, edge_weight=None, edge_attr=None):
        x = self.get_node_features(edge_index.device)
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr)); x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x

from phase2_gnn_extended import EdgePredictor, GNNPredictor, train_month, eval_gnn, compute_p10, save_metrics, save_predictions, save_run_info

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",    type=int, default=0)
    parser.add_argument("--year",   type=int, default=2023)
    parser.add_argument("--label",  choices=["Q3", "Median"], default="Q3")
    parser.add_argument("--model",  choices=["SAGE", "GAT"],  default="GAT")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device    = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    label_col = "Label_Q3" if args.label == "Q3" else "Label_Median"
    tag       = f"comm_{args.model.lower()}_{args.label.lower()}_{args.year}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset("data/processed/ml_dataset_continuous.csv")

    le_pol = LabelEncoder().fit(df["BioGuideID"].fillna("UNK"))
    le_tick = LabelEncoder().fit(df["Ticker"].fillna("UNK"))
    le_state = LabelEncoder().fit(df["State"].fillna("UNK"))
    le_sector = LabelEncoder().fit(df["Sector"])
    le_industry = LabelEncoder().fit(df["Industry"])
    
    mlb_comm = MultiLabelBinarizer().fit(df["comm_list"])

    n_pol = len(le_pol.classes_); n_tick = len(le_tick.classes_); n_comm = len(mlb_comm.classes_)
    log.info("Node Setup: pol=%d tick=%d comm=%d (Committees)", n_pol, n_tick, n_comm)

    metrics_rows = []; preds_rows = []

    for year, month, train_df, gap_df, test_df in monthly_splits(df, args.year, label_col):
        log.info("  %d-%02d  train=%d  test=%d", year, month, len(train_df), len(test_df))

        # Build Graph with Committees
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

        train_month(model, edge_index, edge_weight, edge_attr, pol_tr, tick_tr, y_tr, device, epochs=args.epochs)

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
        save_metrics(metrics_rows, tag)
        save_predictions(preds_rows, tag)
    else: log.warning("No metrics computed")

if __name__ == "__main__":
    main()
