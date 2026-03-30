"""
phase2_gnn_extended.py
======================
GNN Core Module. Contains PyG network architectures (Phases 2-5).

Refactored/Audited: 2026-03-20
"""

#!/usr/bin/env python3
"""
Phase 2: Enhanced GNN with Company Node Features + Enriched Edge Attributes
============================================================================
Changes vs baseline:
  1. Company nodes: Sector embedding (12 classes) + Industry embedding (147 classes)
     + In_SP100 binary, projected to emb_dim via Linear layer.
  2. Edge attributes expanded 2→5 dims:
     [win_rate, log_count_norm, log_trade_size, log_filing_gap, is_buy]
  3. Phase 1 distribution corrections applied throughout.

Output (all under experiments/signal_isolation/results/phase2/):
  - metrics_{model}_{label}_{year}.csv   — per-month AUROC & P@10%
  - preds_{model}_{label}_{year}.csv     — per-trade probabilities + labels
  - run_info_{model}_{label}_{year}.json — full run metadata for reproducibility

Usage:
    python experiments/signal_isolation/phase2_gnn_extended.py \\
        --gpu 0 --year 2023 --label Q3 --model SAGE --epochs 30
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
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import GraphConv, GATConv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase2_gnn")

RESULTS_DIR = Path("experiments/signal_isolation/results/phase2")

# ── Constants ──────────────────────────────────────────────────────────────
POL_FEAT_DIM  = 7    # unchanged from baseline
EDGE_ATTR_DIM = 5    # win_rate, log_count_norm, log_trade_size, log_filing_gap, is_buy

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

HORIZON_DAYS = {
    "1M": 30, "2M": 60, "3M": 90, "6M": 180,
    "8M": 240, "12M": 365, "18M": 545, "24M": 730,
}


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── Data loading ───────────────────────────────────────────────────────────

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Filed"]  = pd.to_datetime(df["Filed"])
    df["Traded"] = pd.to_datetime(df["Traded"])
    df = df.sort_values("Filed").reset_index(drop=True)

    # Phase 1 distribution corrections
    df["Filing_Gap"]     = pd.to_numeric(df["Filing_Gap"], errors="coerce").fillna(30).clip(0)
    df["log_filing_gap"] = np.log1p(df["Filing_Gap"])
    df["trade_size_ord"] = df["Trade_Size_USD"].map(TRADE_SIZE_MAP).fillna(2)
    df["log_trade_size"] = np.log1p(df["trade_size_ord"])

    # Winsorize continuous label
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

    # Ensure transaction_id exists for prediction saving
    if "transaction_id" not in df.columns:
        df["transaction_id"] = df.index.astype(str)

    log.info("Dataset loaded: %d rows, %d cols", len(df), len(df.columns))
    return df


# ── Monthly walk-forward splits ────────────────────────────────────────────

def monthly_splits(df, years, label_col, max_age_years=4.0):
    """Yields (year, month, train_df, gap_df, test_df)."""
    if isinstance(years, int):
        years = [years]
    h_days = HORIZON_DAYS["6M"]
    for year in years:
        for month in range(1, 13):
            ts = pd.Timestamp(year, month, 1)
            ne = ts + pd.DateOffset(months=1)
            test_df  = df[(df["Filed"] >= ts) & (df["Filed"] < ne) & df[label_col].notna()].copy()
            pre      = df[df["Filed"] < ts].copy()
            pre["Resolution"] = pre["Filed"] + pd.Timedelta(days=h_days)
            train_df = pre[(pre["Resolution"] < ts) & pre[label_col].notna()].copy()
            gap_df   = pre[pre["Resolution"] >= ts].copy()
            if max_age_years > 0:
                cutoff   = ts - pd.Timedelta(days=max_age_years * 365.25)
                train_df = train_df[train_df["Filed"] >= cutoff].copy()
            if len(train_df) == 0 or len(test_df) == 0:
                continue
            yield year, month, train_df, gap_df, test_df


# ── Politician node features (identical to baseline) ──────────────────────

def compute_pol_features(train_df, label_col, le_pol, n_pol, gap_df, device):
    feats = np.zeros((n_pol, POL_FEAT_DIM), dtype=np.float32)
    feats[:, 0] = 0.5; feats[:, 2] = 0.5
    df_c = train_df.copy()
    df_c["_bio"] = df_c["BioGuideID"].fillna("UNK")
    df_c["_lbl"] = (df_c[label_col] > 0).astype(float)
    df_c["_buy"] = df_c["Transaction"].str.lower().str.contains("purchase", na=False).astype(float)
    filled = set()
    for bio_id, grp in df_c.groupby("_bio"):
        try: idx = le_pol.transform([bio_id])[0]
        except ValueError: continue
        filled.add(idx); n = len(grp)
        feats[idx, 0] = grp["_lbl"].sum() / n
        feats[idx, 1] = np.log1p(n)
        feats[idx, 2] = grp["_buy"].mean()
        row = grp.iloc[0]
        ch = str(row.get("Chamber", "")).lower()
        feats[idx, 3] = 1.0 if ch == "senate" else 0.0
        py = str(row.get("Party", "")).lower()
        if "democrat" in py: feats[idx, 4] = 1.0
        elif "republican" in py: feats[idx, 5] = 1.0
        else: feats[idx, 6] = 1.0
    if gap_df is not None and len(gap_df) > 0:
        g = gap_df.copy(); g["_bio"] = g["BioGuideID"].fillna("UNK")
        for bio_id, grp in g.groupby("_bio"):
            try: idx = le_pol.transform([bio_id])[0]
            except ValueError: continue
            if idx in filled: continue
            row = grp.iloc[0]
            ch = str(row.get("Chamber", "")).lower()
            feats[idx, 3] = 1.0 if ch == "senate" else 0.0
            py = str(row.get("Party", "")).lower()
            if "democrat" in py: feats[idx, 4] = 1.0
            elif "republican" in py: feats[idx, 5] = 1.0
            else: feats[idx, 6] = 1.0
            if "Transaction" in grp.columns:
                feats[idx, 2] = grp["Transaction"].str.lower().str.contains("purchase", na=False).astype(float).mean()
    return torch.tensor(feats, dtype=torch.float32, device=device)


def compute_state_ids(train_df, le_pol, le_state, n_pol, gap_df, device):
    state_ids = np.zeros(n_pol, dtype=np.int64)
    filled = set()
    for bio_id, state_val in zip(train_df["BioGuideID"].fillna("UNK"), train_df["State"].fillna("UNK")):
        try:
            pi = le_pol.transform([bio_id])[0]; si = le_state.transform([state_val])[0]
            state_ids[pi] = si; filled.add(pi)
        except ValueError: pass
    if gap_df is not None and len(gap_df) > 0:
        for bio_id, state_val in zip(gap_df["BioGuideID"].fillna("UNK"), gap_df["State"].fillna("UNK")):
            try:
                pi = le_pol.transform([bio_id])[0]
                if pi in filled: continue
                si = le_state.transform([state_val])[0]; state_ids[pi] = si
            except ValueError: pass
    return torch.tensor(state_ids, dtype=torch.long, device=device)


# ── Company node features (NEW in Phase 2) ────────────────────────────────

def compute_comp_features(train_df, le_tick, le_sector, le_industry, device):
    """Returns [n_tick, 3] tensor: [sector_idx, industry_idx, in_sp100]."""
    n_tick = len(le_tick.classes_)
    comp   = np.zeros((n_tick, 3), dtype=np.float32)

    tick_to_info = {}
    for _, row in train_df.iterrows():
        tick = str(row.get("Ticker", "UNK")).strip()
        if tick not in tick_to_info:
            tick_to_info[tick] = {
                "sector":   str(row.get("Sector",   "Unknown")).strip(),
                "industry": str(row.get("Industry", "Unknown")).strip(),
                "sp100":    float(row.get("In_SP100", 0) or 0),
            }

    for tick, info in tick_to_info.items():
        try: t_idx = le_tick.transform([tick])[0]
        except ValueError: continue
        try: s_idx = le_sector.transform([info["sector"]])[0]
        except ValueError: s_idx = 0
        try: i_idx = le_industry.transform([info["industry"]])[0]
        except ValueError: i_idx = 0
        comp[t_idx, 0] = float(s_idx)
        comp[t_idx, 1] = float(i_idx)
        comp[t_idx, 2] = info["sp100"]

    return torch.tensor(comp, dtype=torch.float32, device=device)


# ── Enhanced graph construction ────────────────────────────────────────────

def build_enhanced_graph(train_df, label_col, le_pol, le_tick, device,
                         gap_df=None, test_start=None, max_age_years=4.0):
    """5-dim edge_attr: [win_rate, log_count_norm, log_trade_size, log_filing_gap, is_buy]"""
    n_pol  = len(le_pol.classes_)
    n_tick = len(le_tick.classes_)

    if max_age_years > 0 and test_start is not None:
        cutoff   = test_start - pd.Timedelta(days=max_age_years * 365.25)
        train_df = train_df[train_df["Filed"] >= cutoff]

    pol_ids  = le_pol.transform(train_df["BioGuideID"].fillna("UNK"))
    tick_ids = le_tick.transform(train_df["Ticker"].fillna("UNK"))
    labels   = (train_df[label_col] > 0).astype(float).values
    log_sizes = train_df["log_trade_size"].fillna(np.log1p(2)).values.astype(np.float32)
    log_gaps  = train_df["log_filing_gap"].fillna(np.log1p(30)).values.astype(np.float32)
    is_buys   = train_df["is_buy"].fillna(0.5).values.astype(np.float32)

    # Vectorized Pandas aggregation using GroupBy (MUCH FASTER)
    t_df = train_df.copy()
    t_df["p_id"] = pol_ids
    t_df["t_id"] = tick_ids
    t_df["lbl"]  = labels
    t_df["ls"]   = log_sizes
    t_df["lg"]   = log_gaps
    t_df["ib"]   = is_buys

    agg = t_df.groupby(["p_id", "t_id"]).agg(
        cnt=("lbl", "count"),
        pos_sum=("lbl", "sum"),
        sum_ls=("ls", "sum"),
        sum_lg=("lg", "sum"),
        sum_ib=("ib", "sum")
    ).reset_index()

    # Gap edges for structural count
    if gap_df is not None and len(gap_df) > 0:
        gap_df = gap_df.copy()
        gap_df["p_id"] = le_pol.transform(gap_df["BioGuideID"].fillna("UNK"))
        gap_df["t_id"] = le_tick.transform(gap_df["Ticker"].fillna("UNK"))
        gap_agg = gap_df.groupby(["p_id", "t_id"]).size().rename("gap_cnt").reset_index()
        merged = pd.merge(agg, gap_agg, on=["p_id", "t_id"], how="outer").fillna(0)
    else:
        merged = agg; merged["gap_cnt"] = 0

    merged["total"]  = merged["cnt"] + merged["gap_cnt"]
    merged["wr"]     = np.where(merged["cnt"] > 0, merged["pos_sum"] / merged["cnt"], 0.5)
    merged["avg_ls"] = np.where(merged["cnt"] > 0, merged["sum_ls"] / merged["cnt"], np.log1p(2))
    merged["avg_lg"] = np.where(merged["cnt"] > 0, merged["sum_lg"] / merged["cnt"], np.log1p(30))
    merged["avg_ib"] = np.where(merged["cnt"] > 0, merged["sum_ib"] / merged["cnt"], 0.5)
    merged["lc"]     = np.log1p(merged["total"])

    src_l, dst_l, lc_l, wr_l, ls_l, lg_l, ib_l = [], [], [], [], [], [], []
    for _, row in merged.iterrows():
        p, t = int(row["p_id"]), int(row["t_id"])
        cnode = n_pol + t
        # Edge 1: p -> c
        src_l.append(p); dst_l.append(cnode)
        lc_l.append(row["lc"]); wr_l.append(row["wr"])
        ls_l.append(row["avg_ls"]); lg_l.append(row["avg_lg"]); ib_l.append(row["avg_ib"])
        # Edge 2: c -> p
        src_l.append(cnode); dst_l.append(p)
        lc_l.append(row["lc"]); wr_l.append(row["wr"])
        ls_l.append(row["avg_ls"]); lg_l.append(row["avg_lg"]); ib_l.append(row["avg_ib"])

    lc_arr = np.array(lc_l, dtype=np.float32)
    lc_norm = lc_arr / (lc_arr.max() if lc_arr.max() > 0 else 1.0)
    edge_index  = torch.tensor([src_l, dst_l], dtype=torch.long, device=device)
    edge_weight = torch.tensor(lc_arr, dtype=torch.float, device=device)
    edge_attr   = torch.stack([
        torch.tensor(wr_l,  dtype=torch.float, device=device),
        torch.tensor(lc_norm, dtype=torch.float, device=device),
        torch.tensor(ls_l,  dtype=torch.float, device=device),
        torch.tensor(lg_l,  dtype=torch.float, device=device),
        torch.tensor(ib_l,  dtype=torch.float, device=device),
    ], dim=1)

    return edge_index, edge_weight, edge_attr, n_pol, n_tick


# ── GNN Models ────────────────────────────────────────────────────────────

class BipartiteSAGEExtended(nn.Module):
    """SAGE with company sector/industry embeddings and 5-dim edge attrs."""
    def __init__(self, n_pol, n_tick, n_states, n_sectors, n_industries,
                 emb_dim=32, hidden_dim=64, out_dim=32, dropout=0.2):
        super().__init__()
        self.n_pol = n_pol
        self.pol_proj     = nn.Sequential(nn.Linear(POL_FEAT_DIM, emb_dim), nn.ReLU())
        self.state_emb    = nn.Embedding(n_states, emb_dim)
        self.sector_emb   = nn.Embedding(n_sectors,   8)
        self.industry_emb = nn.Embedding(n_industries, 8)
        self.comp_proj    = nn.Sequential(nn.Linear(17, emb_dim), nn.ReLU())  # 8+8+1=17
        self.norm         = nn.LayerNorm(emb_dim) # Add norm clamping
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
        sp   = self._comp_features[:, 2].unsqueeze(1)
        comp = self.comp_proj(torch.cat([s_e, i_e, sp], dim=1))
        return self.norm(torch.cat([pol, comp], dim=0)) # Normalized

    def forward(self, edge_index, edge_weight=None, edge_attr=None):
        x = self.get_node_features(edge_index.device)
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight)); x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x


class BipartiteGATExtended(nn.Module):
    """GAT with company sector/industry embeddings and 5-dim edge attrs."""
    def __init__(self, n_pol, n_tick, n_states, n_sectors, n_industries,
                 emb_dim=32, hidden_dim=64, out_dim=32, heads=4, dropout=0.2):
        super().__init__()
        self.n_pol = n_pol
        self.pol_proj     = nn.Sequential(nn.Linear(POL_FEAT_DIM, emb_dim), nn.ReLU())
        self.state_emb    = nn.Embedding(n_states, emb_dim)
        self.sector_emb   = nn.Embedding(n_sectors,   8)
        self.industry_emb = nn.Embedding(n_industries, 8)
        self.comp_proj    = nn.Sequential(nn.Linear(17, emb_dim), nn.ReLU())
        self.norm         = nn.LayerNorm(emb_dim) # norm clamping
        self.conv1   = GATConv(emb_dim, hidden_dim // heads, heads=heads,
                                dropout=dropout, edge_dim=EDGE_ATTR_DIM)
        self.conv2   = GATConv(hidden_dim, out_dim, heads=1, concat=False,
                                dropout=dropout, edge_dim=EDGE_ATTR_DIM)
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
        sp   = self._comp_features[:, 2].unsqueeze(1)
        comp = self.comp_proj(torch.cat([s_e, i_e, sp], dim=1))
        return self.norm(torch.cat([pol, comp], dim=0))

    def forward(self, edge_index, edge_weight=None, edge_attr=None):
        x = self.get_node_features(edge_index.device)
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr)); x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x


class EdgePredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, z_src, z_dst):
        return self.net(torch.cat([z_src, z_dst], dim=-1)).squeeze(-1)


class GNNPredictor(nn.Module):
    def __init__(self, gnn, predictor):
        super().__init__()
        self.gnn = gnn; self.predictor = predictor

    def forward(self, edge_index, edge_weight, edge_attr, src_ids, dst_ids):
        z = self.gnn(edge_index, edge_weight, edge_attr)
        return self.predictor(z[src_ids], z[dst_ids])


# ── Training loop ─────────────────────────────────────────────────────────

def train_month(model, edge_index, edge_weight, edge_attr,
                src, dst, labels, device, epochs=30, lr=3e-3, batch_size=512):
    pos = labels.sum().item(); neg = len(labels) - pos
    pw  = torch.tensor([neg / max(1, pos)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=5)
    n = len(src)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        el = 0; nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            logits = model(edge_index, edge_weight, edge_attr, src[idx], dst[idx])
            loss   = criterion(logits, labels[idx])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); el += loss.item(); nb += 1
        sched.step(el / max(1, nb))


def eval_gnn(model, edge_index, edge_weight, edge_attr, pol_test, tick_test):
    model.eval()
    with torch.no_grad():
        logits = model(edge_index, edge_weight, edge_attr, pol_test, tick_test)
        return torch.sigmoid(logits).cpu().numpy()


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_p10(y_true, y_prob, years, months, k=0.10):
    df = pd.DataFrame({"l": y_true, "p": y_prob, "y": years, "m": months})
    precs = []
    for _, g in df.groupby(["y", "m"]):
        if len(g) < 5: continue
        n_top = max(1, int(len(g) * k))
        precs.append(g.nlargest(n_top, "p")["l"].mean())
    return float(np.mean(precs)) if precs else 0.0


# ── Saving ────────────────────────────────────────────────────────────────

def save_predictions(preds_rows: list, tag: str):
    """Save per-trade predictions to CSV."""
    out = RESULTS_DIR / f"preds_{tag}.csv"
    pd.DataFrame(preds_rows).to_csv(out, index=False)
    log.info("Predictions saved → %s (%d trades)", out, len(preds_rows))


def save_metrics(metrics_rows: list, tag: str):
    """Save per-month metrics to CSV."""
    out = RESULTS_DIR / f"metrics_{tag}.csv"
    pd.DataFrame(metrics_rows).to_csv(out, index=False)
    log.info("Metrics saved → %s", out)


def save_run_info(info: dict, tag: str):
    """Save run metadata as JSON for reproducibility."""
    out = RESULTS_DIR / f"run_info_{tag}.json"
    with open(out, "w") as f:
        json.dump(info, f, indent=2, default=str)
    log.info("Run info saved → %s", out)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",    type=int, default=0)
    parser.add_argument("--year",   type=int, default=2023)
    parser.add_argument("--label",  choices=["Q3", "Median"], default="Q3")
    parser.add_argument("--model",  choices=["SAGE", "GAT"],  default="SAGE")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device    = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    label_col = "Label_Q3" if args.label == "Q3" else "Label_Median"
    tag       = f"{args.model.lower()}_{args.label.lower()}_{args.year}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_info = {
        "phase": 2,
        "model": args.model,
        "label": args.label,
        "year":  args.year,
        "gpu":   args.gpu,
        "device": str(device),
        "epochs": args.epochs,
        "seed":   args.seed,
        "label_col": label_col,
        "edge_attr_dim": EDGE_ATTR_DIM,
        "pol_feat_dim":  POL_FEAT_DIM,
        "changes": [
            "Company nodes: sector + industry embeddings + In_SP100",
            "Edge attrs: +log_trade_size, +log_filing_gap, +is_buy",
            "Phase 1: log1p(Filing_Gap), ordinal trade size, winsorized max_excess",
        ],
        "started_at": datetime.now().isoformat(),
    }

    csv = "data/processed/ml_dataset_continuous.csv"
    log.info("Loading dataset: %s", csv)
    df = load_dataset(csv)

    # Global encoders
    le_pol      = LabelEncoder().fit(df["BioGuideID"].fillna("UNK"))
    le_tick     = LabelEncoder().fit(df["Ticker"].fillna("UNK"))
    le_state    = LabelEncoder().fit(df["State"].fillna("UNK"))
    le_sector   = LabelEncoder().fit(df["Sector"])
    le_industry = LabelEncoder().fit(df["Industry"])

    n_pol        = len(le_pol.classes_)
    n_tick       = len(le_tick.classes_)
    n_states     = len(le_state.classes_)
    n_sectors    = len(le_sector.classes_)
    n_industries = len(le_industry.classes_)

    run_info.update({
        "n_pol": n_pol, "n_tick": n_tick,
        "n_states": n_states, "n_sectors": n_sectors, "n_industries": n_industries,
    })
    log.info("Nodes: pol=%d tick=%d | Sectors=%d Industries=%d States=%d",
             n_pol, n_tick, n_sectors, n_industries, n_states)

    metrics_rows = []
    preds_rows   = []

    for year, month, train_df, gap_df, test_df in monthly_splits(df, args.year, label_col):
        ts = pd.Timestamp(year, month, 1)
        log.info("  %d-%02d  train=%d  gap=%d  test=%d",
                 year, month, len(train_df), len(gap_df), len(test_df))

        edge_index, edge_weight, edge_attr, _np, _nt = build_enhanced_graph(
            train_df, label_col, le_pol, le_tick, device,
            gap_df=gap_df, test_start=ts, max_age_years=4.0,
        )

        # Build model fresh each month
        if args.model == "SAGE":
            gnn = BipartiteSAGEExtended(n_pol, n_tick, n_states, n_sectors, n_industries)
        else:
            gnn = BipartiteGATExtended(n_pol, n_tick, n_states, n_sectors, n_industries)

        predictor = EdgePredictor(gnn.out_dim * 2)
        model     = GNNPredictor(gnn, predictor).to(device)

        pol_feats  = compute_pol_features(train_df, label_col, le_pol, n_pol, gap_df, device)
        state_ids  = compute_state_ids(train_df, le_pol, le_state, n_pol, gap_df, device)
        comp_feats = compute_comp_features(train_df, le_tick, le_sector, le_industry, device)

        model.gnn.set_pol_features(pol_feats)
        model.gnn.set_state_ids(state_ids)
        model.gnn.set_comp_features(comp_feats)

        # Training tensors
        pol_tr  = torch.tensor(le_pol.transform(train_df["BioGuideID"].fillna("UNK")),
                               dtype=torch.long, device=device)
        tick_tr = torch.tensor(le_tick.transform(train_df["Ticker"].fillna("UNK")),
                               dtype=torch.long, device=device) + n_pol
        y_tr    = torch.tensor((train_df[label_col] > 0).astype(float).values,
                               dtype=torch.float, device=device)

        train_month(model, edge_index, edge_weight, edge_attr,
                    pol_tr, tick_tr, y_tr, device, epochs=args.epochs)

        # Eval
        pol_te  = torch.tensor(le_pol.transform(test_df["BioGuideID"].fillna("UNK")),
                               dtype=torch.long, device=device)
        tick_te = torch.tensor(le_tick.transform(test_df["Ticker"].fillna("UNK")),
                               dtype=torch.long, device=device) + n_pol
        y_te    = (test_df[label_col] > 0).astype(int).values

        probs = eval_gnn(model, edge_index, edge_weight, edge_attr, pol_te, tick_te)
        probs = np.nan_to_num(probs, nan=0.5)

        if len(np.unique(y_te)) < 2:
            log.warning("    Skipping %d-%02d: only one class in test set", year, month)
            continue

        auc = roc_auc_score(y_te, probs)
        p10 = compute_p10(y_te, probs,
                          test_df["year"].values, test_df["month"].values)

        log.info("    AUC=%.4f  P@10%%=%.4f  (n=%d)", auc, p10, len(test_df))

        metrics_rows.append({
            "year": year, "month": month, "n_test": len(test_df),
            "AUROC": round(auc, 6), "P10": round(p10, 6),
            "model": args.model, "label": args.label,
        })

        # Per-trade predictions
        for i, (_, row) in enumerate(test_df.iterrows()):
            preds_rows.append({
                "transaction_id": row.get("transaction_id", i),
                "year": year, "month": month,
                "BioGuideID": row.get("BioGuideID"),
                "Ticker":     row.get("Ticker"),
                "Filed":      row.get("Filed"),
                "true_label": int(y_te[i]),
                "prob":       float(probs[i]),
                "model": args.model, "label": args.label,
            })

    # ── Summary ───────────────────────────────────────────────────────────
    if metrics_rows:
        mdf = pd.DataFrame(metrics_rows)
        print(f"\n{'='*60}")
        print(f"Phase 2 | {args.model} | Label={args.label} | Year={args.year}")
        print(f"{'='*60}")
        print(f"  AUROC : {mdf['AUROC'].mean():.4f} ± {mdf['AUROC'].std():.4f}")
        print(f"  P@10% : {mdf['P10'].mean():.4f}  ± {mdf['P10'].std():.4f}")
        print(f"  Months evaluated: {len(mdf)}")
        print(f"  Total test trades: {sum(r['n_test'] for r in metrics_rows)}")

        run_info["finished_at"]  = datetime.now().isoformat()
        run_info["mean_auroc"]   = float(mdf["AUROC"].mean())
        run_info["std_auroc"]    = float(mdf["AUROC"].std())
        run_info["mean_p10"]     = float(mdf["P10"].mean())
        run_info["std_p10"]      = float(mdf["P10"].std())
        run_info["months_eval"]  = len(mdf)
        run_info["total_trades"] = int(sum(r["n_test"] for r in metrics_rows))

        save_metrics(metrics_rows, tag)
        save_predictions(preds_rows, tag)
        save_run_info(run_info, tag)
    else:
        log.warning("No results produced. Check data splits.")


if __name__ == "__main__":
    main()
