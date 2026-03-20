"""
run_study.py
============
Executable Pipeline Runners. Imports src.* modules to execute experiments.

Refactored/Audited: 2026-03-20
"""

#!/usr/bin/env python3
"""
Signal Isolation Study
======================
Question: Is there signal in politician identity x company identity for
predicting stock price direction, and does the graph/memory formulation help?

Models compared (all using the SAME gap-aware resolution-based split):
  1. Random          — pure noise baseline
  2. XGBoost(ID)     — LabelEncoded politician + ticker
  3. MLP(ID)         — learned embeddings for politician + ticker
  4. TGN(ID)         — graph structure + memory, NO msg/price features

Protocol:
  - Pooled 2023, 6M horizon, monthly walk-forward
  - Gap-aware resolution-based split (Train + Gap + Test)
  - Threshold fixed at 0.5
  - Primary metric: AUC

Usage:
  python experiments/signal_isolation/run_study.py --gpu 1 --seed 42
"""

import argparse
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.getcwd())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("signal_isolation")

# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Filed"] = pd.to_datetime(df["Filed"])
    df["Traded"] = pd.to_datetime(df["Traded"])
    df = df.sort_values("Filed").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Gap-aware resolution-based monthly split
# ---------------------------------------------------------------------------

HORIZON_DAYS = {
    "1M": 30, "2M": 60, "3M": 90, "6M": 180,
    "8M": 240, "12M": 365, "18M": 545, "24M": 730,
}
HORIZON_COL_IDX = {
    "1M": 0, "2M": 1, "3M": 2, "6M": 3,
    "8M": 4, "12M": 5, "18M": 6, "24M": 7,
}


def monthly_splits(df: pd.DataFrame, years, horizon: str):
    """
    Yield (year, month, train_df, gap_df, test_df) for each month of each year.

    `years` can be a single int or a list of ints.

    Train: Filed < test_start AND Filed + horizon_days < test_start AND label notna
    Gap:   Filed < test_start AND Filed + horizon_days >= test_start
    Test:  Filed within [test_start, next_month) AND label notna
    """
    if isinstance(years, int):
        years = [years]
    h_days = HORIZON_DAYS[horizon]
    target = f"Excess_Return_{horizon}"

    for year in years:
        for month in range(1, 13):
            test_start = pd.Timestamp(year, month, 1)
            next_month = test_start + pd.DateOffset(months=1)

            # Test set
            test_mask = (df["Filed"] >= test_start) & (df["Filed"] < next_month)
            test_df = df[test_mask].copy()
            test_df = test_df[test_df[target].notna()]

            # Pre-test events
            pre = df[df["Filed"] < test_start].copy()
            # Labels are computed from Filing date (see add_closing_prices.py),
            # so resolution must also anchor on Filed to avoid future leakage.
            pre["Resolution"] = pre["Filed"] + pd.Timedelta(days=h_days)

            # Train: resolved before test_start
            train_df = pre[(pre["Resolution"] < test_start) & (pre[target].notna())].copy()
            # Gap: not yet resolved
            gap_df = pre[pre["Resolution"] >= test_start].copy()

            if len(train_df) == 0 or len(test_df) == 0:
                log.warning("Skipping %d-%02d: train=%d test=%d", year, month, len(train_df), len(test_df))
                continue

            yield year, month, train_df, gap_df, test_df


def make_binary_label(df: pd.DataFrame, horizon: str) -> np.ndarray:
    """Excess_Return > 0 => 1, else 0."""
    return (df[f"Excess_Return_{horizon}"] > 0).astype(int).values


# ---------------------------------------------------------------------------
# Politician feature computation (shared, expanding window, no look-ahead)
# ---------------------------------------------------------------------------

# Politician feature columns (7 dims):
#   0: win_rate        — fraction of resolved trades beating SPY
#   1: log_trade_count — log(1 + n_trades), cold-start indicator
#   2: buy_ratio       — fraction of trades that are purchases
#   3: chamber         — House=0, Senate=1
#   4: party_dem       — one-hot Democrat
#   5: party_rep       — one-hot Republican
#   6: party_ind       — one-hot Independent/other
POL_FEAT_DIM = 7


def compute_pol_features_numpy(
    train_df: pd.DataFrame,
    horizon: str,
    le_pol: LabelEncoder,
) -> np.ndarray:
    """
    Compute per-politician features from the training window only (no look-ahead).

    Returns:
        np.ndarray [n_pol, POL_FEAT_DIM] — float32
    """
    target = f"Excess_Return_{horizon}"
    n_pol = len(le_pol.classes_)
    feats = np.zeros((n_pol, POL_FEAT_DIM), dtype=np.float32)

    # Sensible defaults for cold-start politicians
    feats[:, 0] = 0.5   # win rate prior (50/50)
    feats[:, 1] = 0.0   # no trade history
    feats[:, 2] = 0.5   # buy ratio prior

    df_copy = train_df.copy()
    df_copy["_bio"] = df_copy["BioGuideID"].fillna("UNK")
    df_copy["_label"] = (df_copy[target] > 0).astype(float)
    df_copy["_is_buy"] = (
        df_copy["Transaction"].str.lower().str.contains("purchase", na=False)
    ).astype(float)

    for bio_id, grp in df_copy.groupby("_bio"):
        try:
            idx = le_pol.transform([bio_id])[0]
        except ValueError:
            continue

        n_trades = len(grp)
        feats[idx, 0] = grp["_label"].sum() / n_trades        # win rate
        feats[idx, 1] = np.log1p(n_trades)                    # log trade count
        feats[idx, 2] = grp["_is_buy"].mean()                  # buy ratio

        row = grp.iloc[0]
        chamber = str(row.get("Chamber", "")).strip().lower()
        feats[idx, 3] = 1.0 if chamber == "senate" else 0.0

        party = str(row.get("Party", "")).strip().lower()
        if "democrat" in party:
            feats[idx, 4] = 1.0
        elif "republican" in party:
            feats[idx, 5] = 1.0
        else:
            feats[idx, 6] = 1.0  # Independent / other

    return feats


# ---------------------------------------------------------------------------
# Model 1: Random
# ---------------------------------------------------------------------------

def run_random(df, years, horizon, seed):
    set_seed(seed)
    all_preds, all_labels = [], []
    all_ids, all_filed, all_years_list, all_months_list = [], [], [], []
    month_results = []

    for year, month, train_df, gap_df, test_df in monthly_splits(df, years, horizon):
        y_test = make_binary_label(test_df, horizon)
        preds = np.random.rand(len(y_test))
        all_preds.extend(preds)
        all_labels.extend(y_test)
        all_ids.extend(test_df["transaction_id"].values)
        all_filed.extend(test_df["Filed"].values)
        all_years_list.extend([year] * len(y_test))
        all_months_list.extend([month] * len(y_test))

        auc = roc_auc_score(y_test, preds) if len(np.unique(y_test)) > 1 else 0.5
        month_results.append({"year": year, "month": month, "auc": auc, "n": len(y_test)})

    pooled_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    pooled_f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    pooled_acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    preds_arr = np.array(all_preds)
    preds_df = pd.DataFrame({
        "transaction_id": all_ids, "Filed": all_filed,
        "year": all_years_list, "month": all_months_list,
        "prob": preds_arr, "pred": (preds_arr > 0.5).astype(int), "label": all_labels,
    })
    return pooled_auc, pooled_f1, pooled_acc, month_results, preds_df


# ---------------------------------------------------------------------------
# Model 2: XGBoost (ID-only)
# ---------------------------------------------------------------------------

def run_xgboost_id(df, years, horizon, seed):
    import xgboost as xgb

    set_seed(seed)
    all_preds, all_labels = [], []
    all_ids, all_filed, all_years_list, all_months_list = [], [], [], []
    month_results = []

    for year, month, train_df, gap_df, test_df in monthly_splits(df, years, horizon):
        y_train = make_binary_label(train_df, horizon)
        y_test = make_binary_label(test_df, horizon)

        # Per-fold LabelEncoder (avoids leakage)
        le_pol = LabelEncoder().fit(
            pd.concat([train_df["BioGuideID"], test_df["BioGuideID"]]).fillna("UNK")
        )
        le_tick = LabelEncoder().fit(
            pd.concat([train_df["Ticker"], test_df["Ticker"]]).fillna("UNK")
        )

        pol_cats  = list(range(len(le_pol.classes_)))
        tick_cats = list(range(len(le_tick.classes_)))

        X_train = pd.DataFrame({
            "pol":  pd.Categorical(le_pol.transform(train_df["BioGuideID"].fillna("UNK")),  categories=pol_cats),
            "tick": pd.Categorical(le_tick.transform(train_df["Ticker"].fillna("UNK")), categories=tick_cats),
        })
        X_test = pd.DataFrame({
            "pol":  pd.Categorical(le_pol.transform(test_df["BioGuideID"].fillna("UNK")),  categories=pol_cats),
            "tick": pd.Categorical(le_tick.transform(test_df["Ticker"].fillna("UNK")), categories=tick_cats),
        })

        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric="logloss",
            enable_categorical=True,
            random_state=seed,
            verbosity=0,
        )

        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)[:, 1]

        all_preds.extend(preds)
        all_labels.extend(y_test)
        all_ids.extend(test_df["transaction_id"].values)
        all_filed.extend(test_df["Filed"].values)
        all_years_list.extend([year] * len(y_test))
        all_months_list.extend([month] * len(y_test))
        auc = roc_auc_score(y_test, preds) if len(np.unique(y_test)) > 1 else 0.5
        month_results.append({"year": year, "month": month, "auc": auc, "n": len(y_test)})

    pooled_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    pooled_f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    pooled_acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    preds_arr = np.array(all_preds)
    preds_df = pd.DataFrame({
        "transaction_id": all_ids, "Filed": all_filed,
        "year": all_years_list, "month": all_months_list,
        "prob": preds_arr, "pred": (preds_arr > 0.5).astype(int), "label": all_labels,
    })
    return pooled_auc, pooled_f1, pooled_acc, month_results, preds_df


# ---------------------------------------------------------------------------
# Model 2b: XGBoost (ID + politician features)
# ---------------------------------------------------------------------------

def run_xgboost_feat(df, years, horizon, seed):
    """
    XGBoost with politician background features added on top of ID columns.

    Features per trade:
      - pol_id (categorical)
      - tick_id (categorical)
      - pol_win_rate       — politician's historical win rate (training window)
      - pol_log_count      — log(1 + trade count) in training window
      - pol_buy_ratio      — fraction of purchases
      - pol_chamber        — House=0, Senate=1
      - pol_party_dem/rep/ind — one-hot party

    This is the fair comparison baseline for GNN-SAGE(feat):
    same features, no graph structure.
    """
    import xgboost as xgb

    set_seed(seed)
    all_preds, all_labels = [], []
    all_ids, all_filed, all_years_list, all_months_list = [], [], [], []
    month_results = []

    for year, month, train_df, gap_df, test_df in monthly_splits(df, years, horizon):
        y_train = make_binary_label(train_df, horizon)
        y_test = make_binary_label(test_df, horizon)

        le_pol = LabelEncoder().fit(
            pd.concat([train_df["BioGuideID"], test_df["BioGuideID"]]).fillna("UNK")
        )
        le_tick = LabelEncoder().fit(
            pd.concat([train_df["Ticker"], test_df["Ticker"]]).fillna("UNK")
        )
        le_state = LabelEncoder().fit(
            pd.concat([train_df["State"], test_df["State"]]).fillna("UNK")
        )

        # Compute politician features from training window only
        pol_feats = compute_pol_features_numpy(train_df, horizon, le_pol)  # [n_pol, 7]

        # Full category ranges from the jointly-fitted encoders — ensures train and
        # test DataFrames share identical Categorical dtypes so XGBoost never sees
        # an "unseen category" error at predict time.
        pol_cats   = list(range(len(le_pol.classes_)))
        tick_cats  = list(range(len(le_tick.classes_)))
        state_cats = list(range(len(le_state.classes_)))

        def make_features(fold_df):
            pol_ids = le_pol.transform(fold_df["BioGuideID"].fillna("UNK"))
            tick_ids = le_tick.transform(fold_df["Ticker"].fillna("UNK"))
            state_ids = le_state.transform(fold_df["State"].fillna("UNK"))
            # Look up politician features for each trade
            row_pol_feats = pol_feats[pol_ids]  # [n, 7]
            feat_df = pd.DataFrame({
                "pol_id":   pd.Categorical(pol_ids,   categories=pol_cats),
                "tick_id":  pd.Categorical(tick_ids,  categories=tick_cats),
                "state_id": pd.Categorical(state_ids, categories=state_cats),
                "pol_win_rate":  row_pol_feats[:, 0],
                "pol_log_count": row_pol_feats[:, 1],
                "pol_buy_ratio": row_pol_feats[:, 2],
                "pol_chamber":   row_pol_feats[:, 3],
                "pol_party_dem": row_pol_feats[:, 4],
                "pol_party_rep": row_pol_feats[:, 5],
                "pol_party_ind": row_pol_feats[:, 6],
            })
            return feat_df

        X_train = make_features(train_df)
        X_test = make_features(test_df)

        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric="logloss",
            enable_categorical=True,
            random_state=seed,
            verbosity=0,
        )
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)[:, 1]

        all_preds.extend(preds)
        all_labels.extend(y_test)
        all_ids.extend(test_df["transaction_id"].values)
        all_filed.extend(test_df["Filed"].values)
        all_years_list.extend([year] * len(y_test))
        all_months_list.extend([month] * len(y_test))
        auc = roc_auc_score(y_test, preds) if len(np.unique(y_test)) > 1 else 0.5
        month_results.append({"year": year, "month": month, "auc": auc, "n": len(y_test)})

    pooled_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    pooled_f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    pooled_acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    preds_arr = np.array(all_preds)
    preds_df = pd.DataFrame({
        "transaction_id": all_ids, "Filed": all_filed,
        "year": all_years_list, "month": all_months_list,
        "prob": preds_arr, "pred": (preds_arr > 0.5).astype(int), "label": all_labels,
    })
    return pooled_auc, pooled_f1, pooled_acc, month_results, preds_df

# ---------------------------------------------------------------------------
# Model 3: MLP (ID + politician features)
# ---------------------------------------------------------------------------

STATE_EMB_DIM = 8  # embedding size for US state (51 possible values)


class EmbeddingMLPFeat(nn.Module):
    """
    MLP baseline that receives politician features + state embedding in addition
    to ID embeddings.

    Input per trade:
      - pol_id   -> embedding [emb_dim]
      - tick_id  -> embedding [emb_dim]
      - state_id -> embedding [STATE_EMB_DIM]
      - pol_feat -> [POL_FEAT_DIM] (win_rate, log_count, buy_ratio, chamber, party×3)

    Concatenated and fed through a 2-layer MLP.
    """
    def __init__(self, n_pol, n_tick, n_states, emb_dim=16, hidden=64):
        super().__init__()
        self.emb_pol = nn.Embedding(n_pol, emb_dim)
        self.emb_tick = nn.Embedding(n_tick, emb_dim)
        self.emb_state = nn.Embedding(n_states, STATE_EMB_DIM)
        in_dim = 2 * emb_dim + STATE_EMB_DIM + POL_FEAT_DIM
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, pol_ids, tick_ids, state_ids, pol_feat):
        h = torch.cat([
            self.emb_pol(pol_ids),
            self.emb_tick(tick_ids),
            self.emb_state(state_ids),
            pol_feat,
        ], dim=-1)
        return self.net(h).squeeze(-1)


def run_mlp_feat(df, years, horizon, seed, device, epochs=50, lr=1e-3, batch_size=512):
    """
    MLP with politician background features — fair comparison baseline for GNN(feat).
    Same feature set as XGBoost(feat) but using learned ID embeddings.
    """
    set_seed(seed)
    all_preds, all_labels = [], []
    all_ids, all_filed, all_years_list, all_months_list = [], [], [], []
    month_results = []

    for year, month, train_df, gap_df, test_df in monthly_splits(df, years, horizon):
        y_train = make_binary_label(train_df, horizon)
        y_test = make_binary_label(test_df, horizon)

        le_pol = LabelEncoder().fit(
            pd.concat([train_df["BioGuideID"], test_df["BioGuideID"]]).fillna("UNK")
        )
        le_tick = LabelEncoder().fit(
            pd.concat([train_df["Ticker"], test_df["Ticker"]]).fillna("UNK")
        )
        le_state = LabelEncoder().fit(
            pd.concat([train_df["State"], test_df["State"]]).fillna("UNK")
        )

        # Politician features from training window only
        pol_feats_np = compute_pol_features_numpy(train_df, horizon, le_pol)  # [n_pol, 7]
        pol_feats_t = torch.tensor(pol_feats_np, dtype=torch.float, device=device)

        def make_tensors(fold_df):
            pol_ids = torch.tensor(
                le_pol.transform(fold_df["BioGuideID"].fillna("UNK")),
                dtype=torch.long, device=device,
            )
            tick_ids = torch.tensor(
                le_tick.transform(fold_df["Ticker"].fillna("UNK")),
                dtype=torch.long, device=device,
            )
            state_ids = torch.tensor(
                le_state.transform(fold_df["State"].fillna("UNK")),
                dtype=torch.long, device=device,
            )
            pf = pol_feats_t[pol_ids]  # [n, POL_FEAT_DIM]
            return pol_ids, tick_ids, state_ids, pf

        pol_tr, tick_tr, state_tr, pf_tr = make_tensors(train_df)
        pol_te, tick_te, state_te, pf_te = make_tensors(test_df)
        lab_train = torch.tensor(y_train, dtype=torch.float, device=device)

        model = EmbeddingMLPFeat(
            len(le_pol.classes_), len(le_tick.classes_), len(le_state.classes_)
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        pos = y_train.sum()
        neg = len(y_train) - pos
        pw = torch.tensor([neg / max(1, pos)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        model.train()
        n = len(pol_tr)
        for ep in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                logits = model(pol_tr[idx], tick_tr[idx], state_tr[idx], pf_tr[idx])
                loss = criterion(logits, lab_train[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(pol_te, tick_te, state_te, pf_te).sigmoid().cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(y_test)
        all_ids.extend(test_df["transaction_id"].values)
        all_filed.extend(test_df["Filed"].values)
        all_years_list.extend([year] * len(y_test))
        all_months_list.extend([month] * len(y_test))
        auc = roc_auc_score(y_test, preds) if len(np.unique(y_test)) > 1 else 0.5
        month_results.append({"year": year, "month": month, "auc": auc, "n": len(y_test)})

    pooled_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    pooled_f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    pooled_acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    preds_arr = np.array(all_preds)
    preds_df = pd.DataFrame({
        "transaction_id": all_ids, "Filed": all_filed,
        "year": all_years_list, "month": all_months_list,
        "prob": preds_arr, "pred": (preds_arr > 0.5).astype(int), "label": all_labels,
    })
    return pooled_auc, pooled_f1, pooled_acc, month_results, preds_df


class EmbeddingMLP(nn.Module):
    def __init__(self, n_pol, n_tick, emb_dim=16, hidden=64):
        super().__init__()
        self.emb_pol = nn.Embedding(n_pol, emb_dim)
        self.emb_tick = nn.Embedding(n_tick, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, pol_ids, tick_ids):
        h = torch.cat([self.emb_pol(pol_ids), self.emb_tick(tick_ids)], dim=-1)
        return self.net(h).squeeze(-1)


def run_mlp_id(df, years, horizon, seed, device, epochs=50, lr=1e-3, batch_size=512):
    set_seed(seed)
    all_preds, all_labels = [], []
    all_ids, all_filed, all_years_list, all_months_list = [], [], [], []
    month_results = []

    for year, month, train_df, gap_df, test_df in monthly_splits(df, years, horizon):
        y_train = make_binary_label(train_df, horizon)
        y_test = make_binary_label(test_df, horizon)

        # Per-fold LabelEncoder
        le_pol = LabelEncoder().fit(
            pd.concat([train_df["BioGuideID"], test_df["BioGuideID"]]).fillna("UNK")
        )
        le_tick = LabelEncoder().fit(
            pd.concat([train_df["Ticker"], test_df["Ticker"]]).fillna("UNK")
        )

        pol_train = torch.tensor(le_pol.transform(train_df["BioGuideID"].fillna("UNK")), dtype=torch.long, device=device)
        tick_train = torch.tensor(le_tick.transform(train_df["Ticker"].fillna("UNK")), dtype=torch.long, device=device)
        lab_train = torch.tensor(y_train, dtype=torch.float, device=device)

        pol_test = torch.tensor(le_pol.transform(test_df["BioGuideID"].fillna("UNK")), dtype=torch.long, device=device)
        tick_test = torch.tensor(le_tick.transform(test_df["Ticker"].fillna("UNK")), dtype=torch.long, device=device)

        model = EmbeddingMLP(len(le_pol.classes_), len(le_tick.classes_)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        pos = y_train.sum()
        neg = len(y_train) - pos
        pw = torch.tensor([neg / max(1, pos)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        # Train
        model.train()
        n = len(pol_train)
        for ep in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                logits = model(pol_train[idx], tick_train[idx])
                loss = criterion(logits, lab_train[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            preds = model(pol_test, tick_test).sigmoid().cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(y_test)
        all_ids.extend(test_df["transaction_id"].values)
        all_filed.extend(test_df["Filed"].values)
        all_years_list.extend([year] * len(y_test))
        all_months_list.extend([month] * len(y_test))
        auc = roc_auc_score(y_test, preds) if len(np.unique(y_test)) > 1 else 0.5
        month_results.append({"year": year, "month": month, "auc": auc, "n": len(y_test)})

    pooled_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    pooled_f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    pooled_acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    preds_arr = np.array(all_preds)
    preds_df = pd.DataFrame({
        "transaction_id": all_ids, "Filed": all_filed,
        "year": all_years_list, "month": all_months_list,
        "prob": preds_arr, "pred": (preds_arr > 0.5).astype(int), "label": all_labels,
    })
    return pooled_auc, pooled_f1, pooled_acc, month_results, preds_df


# ---------------------------------------------------------------------------
# Model 4: TGN (ID-only)
# ---------------------------------------------------------------------------
# Uses the existing ResearchTGN model configured with:
#   - raw_msg_dim = 1 (constant [1.0] as edge msg for memory)
#   - price_emb_dim = 0 (no price features)
#   - edge_extra_dim = 1 (just age_feat; NO winrate since that uses labels)
#   - edge_attr for GNN = [rel_t_enc, age_feat] (time encoding + edge age)
#   - Predictor uses only graph + static embeddings
# The TGN's advantage: graph structure and temporal memory.

def run_tgn_id(
    df,
    years,
    horizon,
    seed,
    device,
    epochs=5,
    lr=0.001,
    batch_size=200,
    memory_dim=100,
    time_dim=100,
    embedding_dim=100,
    grad_clip=1.0,
):
    from src.models.gap_tgn import ResearchTGN
    from torch_geometric.loader import TemporalDataLoader
    from torch_geometric.nn.models.tgn import LastNeighborLoader
    from torch_geometric.data import TemporalData

    set_seed(seed)

    # Load temporal data for graph structure
    data = torch.load("data/temporal_data.pt", weights_only=False).to(device)

    # Verify alignment
    if len(df) != len(data.src):
        log.warning("CSV/tensor length mismatch: csv=%d data=%d", len(df), len(data.src))
        min_len = min(len(df), len(data.src))
        df = df.iloc[:min_len].reset_index(drop=True)

    num_nodes = getattr(data, "num_nodes", int(torch.cat([data.src, data.dst]).max().item()) + 1)
    num_parties = getattr(data, "num_parties", int(data.x_static[:, 0].max().item()) + 1)
    num_states = getattr(data, "num_states", int(data.x_static[:, 1].max().item()) + 1)

    h_days = HORIZON_DAYS[horizon]
    h_idx = HORIZON_COL_IDX[horizon]

    # Prepare constant msg (1-dim) for memory updates
    # We use a constant [1.0] — the TGN only knows "an event happened"
    const_msg = torch.ones(len(data.src), 1, dtype=torch.float, device=device)

    # Helper to slice TemporalData by indices
    def slice_data(indices):
        idx = torch.as_tensor(indices, dtype=torch.long, device=device)
        if idx.numel() == 0:
            return TemporalData(
                src=torch.empty(0, dtype=torch.long, device=device),
                dst=torch.empty(0, dtype=torch.long, device=device),
                t=torch.empty(0, dtype=torch.long, device=device),
                msg=torch.empty(0, 1, dtype=torch.float, device=device),
                y=torch.empty(0, data.y.size(-1), dtype=torch.float, device=device),
            )
        return TemporalData(
            src=data.src[idx],
            dst=data.dst[idx],
            t=data.t[idx],
            msg=const_msg[idx],
            y=data.y[idx],
            trade_t=data.trade_t[idx],
        )

    def direction_targets(raw_return):
        has_label = ~torch.isnan(raw_return)
        up = raw_return > 0.0
        down = raw_return < 0.0
        mask = has_label & (up | down)
        return up.float(), mask

    # Walk-forward monthly evaluation
    all_preds, all_labels = [], []
    month_results = []

    # Build Resolution column once (anchor on Filed, not Traded)
    df_eval = df.copy()
    df_eval["Resolution"] = df_eval["Filed"] + pd.Timedelta(days=h_days)

    years_list = [years] if isinstance(years, int) else list(years)

    for yr in years_list:
        for month in range(1, 13):
            test_start = pd.Timestamp(yr, month, 1)
            next_month = test_start + pd.DateOffset(months=1)

            train_mask = (df_eval["Filed"] < test_start) & (df_eval["Resolution"] < test_start)
            gap_mask = (df_eval["Filed"] < test_start) & (df_eval["Resolution"] >= test_start)
            test_mask = (df_eval["Filed"] >= test_start) & (df_eval["Filed"] < next_month)

            train_data = slice_data(df_eval[train_mask].index.values)
            gap_data = slice_data(df_eval[gap_mask].index.values)
            test_data = slice_data(df_eval[test_mask].index.values)

            if len(test_data.src) == 0:
                log.warning("No test data for %d-%02d", yr, month)
                continue

            # Check for trainable labels
            raw_train_y = train_data.y[:, h_idx]
            _, train_label_mask = direction_targets(raw_train_y)
            if train_label_mask.sum() == 0:
                log.warning("No training labels for %d-%02d", yr, month)
                continue

            # Class weighting
            train_targets, _ = direction_targets(raw_train_y)
            pos = train_targets[train_label_mask].sum().item()
            neg = train_label_mask.sum().item() - pos
            pw = torch.tensor([neg / max(1, pos)], device=device)

            log.info(
                "  [TGN] %d-%02d: Train=%d Gap=%d Test=%d pos_weight=%.2f",
                yr, month, len(train_data.src), len(gap_data.src), len(test_data.src), pw.item(),
            )

            # ---- Build model (fresh per month) ----
            model = ResearchTGN(
                num_nodes=num_nodes,
                raw_msg_dim=1,          # constant msg
                memory_dim=memory_dim,
                time_dim=time_dim,
                embedding_dim=embedding_dim,
                num_parties=num_parties,
                num_states=num_states,
                price_emb_dim=0,        # NO price features
                edge_extra_dim=1,       # just age_feat
                predictor_extra_dim=0,  # no extra context for predictor
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
            neighbor_loader = LastNeighborLoader(num_nodes, size=30, device=device)

            train_loader = TemporalDataLoader(train_data, batch_size=batch_size, drop_last=False)
            gap_loader = TemporalDataLoader(gap_data, batch_size=batch_size)

            # ---- Train ----
            for epoch in range(1, epochs + 1):
                model.train()
                model.memory.reset_state()
                neighbor_loader.reset_state()
                model.memory.detach()
                epoch_losses = []

                for batch in train_loader:
                    batch = batch.to(device)
                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    raw_y = batch.y[:, h_idx]
                    targets, label_mask = direction_targets(raw_y)
                    batch_max_t = t.max()

                    optimizer.zero_grad()

                    aug_msg = msg  # constant [1.0]

                    n_id = torch.cat([src, dst]).unique()
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                    assoc = {node.item(): i for i, node in enumerate(n_id)}

                    age_feat = torch.log1p((batch_max_t - data.t[e_id]).float() / 86400.0).unsqueeze(-1)
                    rel_t = model.memory.last_update[n_id[edge_index[1]]] - data.t[e_id]
                    rel_t_enc = model.memory.time_enc(rel_t.to(torch.float))
                    hist_msg = torch.ones(e_id.size(0), 1, device=device)
                    edge_attr = torch.cat([rel_t_enc, hist_msg, age_feat], dim=-1)

                    src_local = torch.tensor([assoc[i.item()] for i in src], device=device, dtype=torch.long)
                    dst_local = torch.tensor([assoc[i.item()] for i in dst], device=device, dtype=torch.long)
                    curr_edge_index = torch.stack([src_local, dst_local], dim=0)
                    rel_t_curr = model.memory.last_update[dst] - t
                    rel_t_enc_curr = model.memory.time_enc(rel_t_curr.to(torch.float))
                    age_feat_curr = torch.log1p((batch_max_t - t).float() / 86400.0).unsqueeze(-1)
                    edge_attr_curr = torch.cat([rel_t_enc_curr, msg, age_feat_curr], dim=-1)

                    edge_index = torch.cat([edge_index, curr_edge_index], dim=1)
                    edge_attr = torch.cat([edge_attr, edge_attr_curr], dim=0)

                    z = model(n_id, edge_index, edge_attr)
                    z_src = z[[assoc[i.item()] for i in src]]
                    z_dst = z[[assoc[i.item()] for i in dst]]
                    s_src = model.encode_static(data.x_static[src])
                    s_dst = model.encode_static(data.x_static[dst])
                    p_context = torch.empty(len(src), 0, device=device)
                    batch_preds = model.predictor(z_src, z_dst, s_src, s_dst, p_context)

                    if label_mask.sum() > 0:
                        loss = criterion(batch_preds[label_mask].view(-1), targets[label_mask].view(-1))
                        loss.backward()
                        if grad_clip > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()
                        epoch_losses.append(loss.item())

                    model.memory.update_state(src, dst, t, aug_msg)
                    neighbor_loader.insert(src, dst)
                    model.memory.detach()

                # Gap: memory update only (after each epoch)
                if len(gap_data.src) > 0:
                    model.eval()
                    with torch.no_grad():
                        for batch in gap_loader:
                            batch = batch.to(device)
                            model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
                            neighbor_loader.insert(batch.src, batch.dst)

                avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
                if epoch == 1 or epoch == epochs:
                    log.info("    Epoch %d/%d loss=%.4f", epoch, epochs, avg_loss)

            # ---- Evaluate on test ----
            model.eval()
            test_loader = TemporalDataLoader(test_data, batch_size=batch_size)
            month_preds, month_targets = [], []

            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
                    raw_y = batch.y[:, h_idx]
                    targets, label_mask = direction_targets(raw_y)
                    batch_max_t = t.max()

                    aug_msg = msg

                    n_id = torch.cat([src, dst]).unique()
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                    assoc = {node.item(): i for i, node in enumerate(n_id)}

                    age_feat = torch.log1p((batch_max_t - data.t[e_id]).float() / 86400.0).unsqueeze(-1)
                    rel_t = model.memory.last_update[n_id[edge_index[1]]] - data.t[e_id]
                    rel_t_enc = model.memory.time_enc(rel_t.to(torch.float))
                    hist_msg = torch.ones(e_id.size(0), 1, device=device)
                    edge_attr = torch.cat([rel_t_enc, hist_msg, age_feat], dim=-1)

                    src_local = torch.tensor([assoc[i.item()] for i in src], device=device, dtype=torch.long)
                    dst_local = torch.tensor([assoc[i.item()] for i in dst], device=device, dtype=torch.long)
                    curr_edge_index = torch.stack([src_local, dst_local], dim=0)
                    rel_t_curr = model.memory.last_update[dst] - t
                    rel_t_enc_curr = model.memory.time_enc(rel_t_curr.to(torch.float))
                    age_feat_curr = torch.log1p((batch_max_t - t).float() / 86400.0).unsqueeze(-1)
                    edge_attr_curr = torch.cat([rel_t_enc_curr, msg, age_feat_curr], dim=-1)

                    edge_index = torch.cat([edge_index, curr_edge_index], dim=1)
                    edge_attr = torch.cat([edge_attr, edge_attr_curr], dim=0)

                    z = model(n_id, edge_index, edge_attr)
                    z_src = z[[assoc[i.item()] for i in src]]
                    z_dst = z[[assoc[i.item()] for i in dst]]
                    s_src = model.encode_static(data.x_static[src])
                    s_dst = model.encode_static(data.x_static[dst])
                    p_context = torch.empty(len(src), 0, device=device)
                    pred_probs = model.predictor(z_src, z_dst, s_src, s_dst, p_context).sigmoid()

                    if label_mask.sum() > 0:
                        month_preds.extend(pred_probs[label_mask].cpu().numpy().flatten())
                        month_targets.extend(targets[label_mask].cpu().numpy().flatten())

                    # Update memory with test events (for future months)
                    model.memory.update_state(src, dst, t, aug_msg)
                    neighbor_loader.insert(src, dst)

            if len(month_preds) == 0:
                log.warning("No predictions for %d-%02d", yr, month)
                continue

            all_preds.extend(month_preds)
            all_labels.extend(month_targets)
            auc = roc_auc_score(month_targets, month_preds) if len(np.unique(month_targets)) > 1 else 0.5
            month_results.append({"year": yr, "month": month, "auc": auc, "n": len(month_preds)})

    if not all_preds:
        return 0.5, 0.0, 0.5, month_results

    pooled_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    pooled_f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    pooled_acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    return pooled_auc, pooled_f1, pooled_acc, month_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Signal Isolation Study")
    parser.add_argument("--gpu", type=int, default=1, help="GPU device index")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=str, default="6M")
    # --years accepts one or more years; --year is kept as a deprecated alias
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="One or more test years (e.g. --years 2019 2020 2021)")
    parser.add_argument("--year", type=int, default=None,
                        help="Single test year (deprecated alias for --years)")
    parser.add_argument("--tgn-epochs", type=int, default=5)
    parser.add_argument("--mlp-epochs", type=int, default=50)
    parser.add_argument("--tgn-lr", type=float, default=0.001)
    parser.add_argument("--skip", nargs="*", default=["tgn"], help="Models to skip (random, xgb, xgb_feat, mlp, mlp_feat, tgn)")
    parser.add_argument("--data", type=str, default="data/processed/ml_dataset_clean.csv",
                        help="Path to the ML dataset CSV (default: ml_dataset_clean.csv)")
    parser.add_argument("--save-preds", action="store_true", default=False,
                        help="Save per-row predictions to a CSV alongside the summary")
    args = parser.parse_args()

    # Resolve years: --years takes priority, fall back to --year, then default 2023
    if args.years is not None:
        years = args.years
    elif args.year is not None:
        years = [args.year]
    else:
        years = [2023]

    years_str = "-".join(str(y) for y in years)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    log.info("Signal Isolation Study: years=%s horizon=%s seed=%d", years_str, args.horizon, args.seed)

    csv_path = args.data
    df = load_data(csv_path)
    log.info("Loaded %d rows from %s", len(df), csv_path)

    # Quick data summary
    target = f"Excess_Return_{args.horizon}"
    valid = df[target].notna()
    up_rate = (df.loc[valid, target] > 0).mean()
    log.info("Overall label rate: %.1f%% up (n=%d valid)", up_rate * 100, valid.sum())

    results_table = []
    preds_list = []  # collects (model_name, preds_df) when --save-preds

    # ---- 1. Random ----
    if "random" not in args.skip:
        log.info("=" * 60)
        log.info("Running: Random baseline")
        t0 = time.time()
        auc, f1, acc, months, preds_df = run_random(df, years, args.horizon, args.seed)
        dt = time.time() - t0
        log.info("Random: AUC=%.4f  F1=%.4f  Acc=%.4f  (%.1fs)", auc, f1, acc, dt)
        results_table.append(("Random", auc, f1, acc, months))
        preds_list.append(("Random", preds_df))

    # ---- 2. XGBoost (ID-only) ----
    if "xgb" not in args.skip:
        log.info("=" * 60)
        log.info("Running: XGBoost (ID-only)")
        t0 = time.time()
        auc, f1, acc, months, preds_df = run_xgboost_id(df, years, args.horizon, args.seed)
        dt = time.time() - t0
        log.info("XGBoost(ID): AUC=%.4f  F1=%.4f  Acc=%.4f  (%.1fs)", auc, f1, acc, dt)
        results_table.append(("XGBoost(ID)", auc, f1, acc, months))
        preds_list.append(("XGBoost(ID)", preds_df))

    # ---- 2b. XGBoost (ID + politician features) ----
    if "xgb_feat" not in args.skip:
        log.info("=" * 60)
        log.info("Running: XGBoost (ID + politician features)")
        t0 = time.time()
        auc, f1, acc, months, preds_df = run_xgboost_feat(df, years, args.horizon, args.seed)
        dt = time.time() - t0
        log.info("XGBoost(feat): AUC=%.4f  F1=%.4f  Acc=%.4f  (%.1fs)", auc, f1, acc, dt)
        results_table.append(("XGBoost(feat)", auc, f1, acc, months))
        preds_list.append(("XGBoost(feat)", preds_df))

    # ---- 3. MLP (ID-only) ----
    if "mlp" not in args.skip:
        log.info("=" * 60)
        log.info("Running: MLP (ID-only)")
        t0 = time.time()
        auc, f1, acc, months, preds_df = run_mlp_id(
            df, years, args.horizon, args.seed, device, epochs=args.mlp_epochs
        )
        dt = time.time() - t0
        log.info("MLP(ID): AUC=%.4f  F1=%.4f  Acc=%.4f  (%.1fs)", auc, f1, acc, dt)
        results_table.append(("MLP(ID)", auc, f1, acc, months))
        preds_list.append(("MLP(ID)", preds_df))

    # ---- 3b. MLP (ID + politician features) ----
    if "mlp_feat" not in args.skip:
        log.info("=" * 60)
        log.info("Running: MLP (ID + politician features)")
        t0 = time.time()
        auc, f1, acc, months, preds_df = run_mlp_feat(
            df, years, args.horizon, args.seed, device, epochs=args.mlp_epochs
        )
        dt = time.time() - t0
        log.info("MLP(feat): AUC=%.4f  F1=%.4f  Acc=%.4f  (%.1fs)", auc, f1, acc, dt)
        results_table.append(("MLP(feat)", auc, f1, acc, months))
        preds_list.append(("MLP(feat)", preds_df))

    # ---- 4. TGN (ID-only) ----
    if "tgn" not in args.skip:
        log.info("=" * 60)
        log.info("Running: TGN (ID-only)")
        t0 = time.time()
        auc, f1, acc, months = run_tgn_id(
            df, years, args.horizon, args.seed, device, epochs=args.tgn_epochs, lr=args.tgn_lr,
        )
        dt = time.time() - t0
        log.info("TGN(ID): AUC=%.4f  F1=%.4f  Acc=%.4f  (%.1fs)", auc, f1, acc, dt)
        results_table.append(("TGN(ID)", auc, f1, acc, months))

    # ---- Summary Table ----
    log.info("")
    log.info("=" * 70)
    log.info("  SIGNAL ISOLATION STUDY RESULTS — %s / %s / seed=%d", years_str, args.horizon, args.seed)
    log.info("=" * 70)
    log.info("%-20s  %8s  %8s  %8s", "Model", "AUC", "F1", "Acc")
    log.info("-" * 50)
    for name, auc, f1, acc, _ in results_table:
        log.info("%-20s  %8.4f  %8.4f  %8.4f", name, auc, f1, acc)
    log.info("-" * 50)

    # Year+Month breakdown
    log.info("")
    log.info("Year-Month AUC breakdown:")
    header = "Year-Mo  " + "  ".join(f"{name:>12s}" for name, *_ in results_table)
    log.info(header)
    log.info("-" * len(header))
    for yr in years:
        for m in range(1, 13):
            row = f"{yr}-{m:02d}  "
            has_data = False
            for name, _, _, _, month_list in results_table:
                mr = [x for x in month_list if x.get("year") == yr and x["month"] == m]
                if mr:
                    row += f"  {mr[0]['auc']:12.4f}"
                    has_data = True
                else:
                    row += f"  {'N/A':>12s}"
            if has_data:
                log.info(row)

    # Save results
    out_dir = Path("experiments/signal_isolation/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame([
        {"Model": name, "AUC": auc, "F1": f1, "Acc": acc}
        for name, auc, f1, acc, _ in results_table
    ])
    out_file = out_dir / f"study_{years_str}_{args.horizon}_seed{args.seed}.csv"
    results_df.to_csv(out_file, index=False)
    log.info("\nResults saved to %s", out_file)

    # Save per-row predictions if requested
    if args.save_preds and preds_list:
        all_preds_dfs = []
        for model_name, preds_df in preds_list:
            preds_df = preds_df.copy()
            preds_df.insert(0, "model", model_name)
            all_preds_dfs.append(preds_df)
        combined = pd.concat(all_preds_dfs, ignore_index=True)
        preds_file = out_dir / f"preds_study_{years_str}_{args.horizon}_seed{args.seed}.csv"
        combined.to_csv(preds_file, index=False)
        log.info("Per-row predictions saved to %s  (n=%d)", preds_file, len(combined))


if __name__ == "__main__":
    main()
