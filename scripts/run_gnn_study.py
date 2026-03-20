"""
run_gnn_study.py
================
Executable Pipeline Runners. Imports src.* modules to execute experiments.

Refactored/Audited: 2026-03-20
"""

#!/usr/bin/env python3
"""
GNN Signal Isolation Study
===========================
Hypothesis: A lightweight GNN on the bipartite politician-company graph can
learn neighbourhood structure that captures "who trades what" patterns.

If politician A and politician B trade similar stocks, the GNN propagates
information between them. A new trade by A for a stock that B already profited
from should receive a signal boost via message passing.

Architecture:
  1. Bipartite graph: politicians <-> companies with edges from training data.
     Edge weights = historical win rate (fraction of up-trades).
  2. Politician node features: hand-crafted from training window (no look-ahead):
       - rolling win rate (strongest signal, corr=0.141 with 6M label)
       - log(1 + trade count) — cold-start indicator
       - buy/sell ratio
       - chamber (House=0, Senate=1)
       - party one-hot (Democrat, Republican, Independent)
     These are projected to emb_dim via a linear layer.
     Company nodes keep learned embeddings (no structural features available).
  3. GNN: 2-layer GraphSAGE with edge weights for weighted message passing.
  4. Prediction head: concat GNN embeddings for (politician, company) pair,
     feed through a 2-layer MLP for binary classification.
  5. Training: end-to-end (GNN + projection + company embeddings + MLP)
     per walk-forward month.

Models compared:
  A. GNN-SAGE    — 2-layer GraphSAGE on bipartite graph + MLP head
  B. GNN-GAT     — 2-layer GAT on bipartite graph + MLP head
  C. GNN+XGB     — GNN embeddings as features for XGBoost (hybrid)

Protocol:
  - Pooled 2023 (configurable), 6M horizon, monthly walk-forward
  - Gap-aware resolution-based split (anchor on Filed, NOT Traded)
  - Threshold fixed at 0.5
  - Primary metric: AUC

Usage:
  python experiments/signal_isolation/run_gnn_study.py --gpu 0 --seed 42
  python experiments/signal_isolation/run_gnn_study.py --gpu 0 --seed 42 --horizon 6M --year 2023
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# PyG imports
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GraphConv, GATConv

sys.path.insert(0, os.getcwd())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gnn_study")


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
# Data loading (same as run_study.py)
# ---------------------------------------------------------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Filed"] = pd.to_datetime(df["Filed"])
    df["Traded"] = pd.to_datetime(df["Traded"])
    df = df.sort_values("Filed").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Gap-aware resolution-based monthly split (same as run_study.py)
# ---------------------------------------------------------------------------

HORIZON_DAYS = {
    "1M": 30, "2M": 60, "3M": 90, "6M": 180,
    "8M": 240, "12M": 365, "18M": 545, "24M": 730,
}


def monthly_splits(df: pd.DataFrame, years, horizon: str, label_col: str = None,
                   max_age_years: float = 0.0, dead_zone: float = 0.0):
    """
    Yield (year, month, train_df, gap_df, test_df) for each month of each year.

    `years` can be a single int or a list of ints.

    Resolution anchor: Filed + horizon_days (NOT Traded).
    Train: Filed < test_start AND Filed + horizon_days < test_start AND label notna
    Gap:   Filed < test_start AND Filed + horizon_days >= test_start
    Test:  Filed within [test_start, next_month) AND label notna

    Parameters
    ----------
    label_col : str, optional
        Column to use for label filtering (default: ``Excess_Return_{horizon}``).
    max_age_years : float
        If > 0, only training rows filed within this many years before
        test_start are included.  0 (default) uses all history.
    dead_zone : float
        If > 0, training rows where |label_col| < dead_zone are excluded.
        Test set is never filtered (full deployment simulation).
    """
    if isinstance(years, int):
        years = [years]
    h_days = HORIZON_DAYS[horizon]
    if label_col is None:
        label_col = f"Excess_Return_{horizon}"

    for year in years:
        for month in range(1, 13):
            test_start = pd.Timestamp(year, month, 1)
            next_month = test_start + pd.DateOffset(months=1)

            # Test set
            test_mask = (df["Filed"] >= test_start) & (df["Filed"] < next_month)
            test_df = df[test_mask].copy()
            test_df = test_df[test_df[label_col].notna()]

            # Pre-test events
            pre = df[df["Filed"] < test_start].copy()
            # Resolution anchored on Filed date (NOT Traded) to avoid leakage
            pre["Resolution"] = pre["Filed"] + pd.Timedelta(days=h_days)

            # Train: resolved before test_start
            train_df = pre[(pre["Resolution"] < test_start) & (pre[label_col].notna())].copy()
            # Gap: not yet resolved
            gap_df = pre[pre["Resolution"] >= test_start].copy()

            # Rolling window: drop training rows older than max_age_years
            if max_age_years > 0.0:
                cutoff = test_start - pd.Timedelta(days=max_age_years * 365.25)
                train_df = train_df[train_df["Filed"] >= cutoff].copy()

            # Dead zone: drop training rows with near-zero returns (coin-flip trades)
            if dead_zone > 0.0:
                train_df = train_df[train_df[label_col].abs() >= dead_zone].copy()

            if len(train_df) == 0 or len(test_df) == 0:
                log.warning("Skipping %d-%02d: train=%d test=%d",
                            year, month, len(train_df), len(test_df))
                continue

            yield year, month, train_df, gap_df, test_df


def make_binary_label(df: pd.DataFrame, horizon: str,
                      label_col: str = None) -> np.ndarray:
    """Return binary label array.  Default column: Excess_Return_{horizon}."""
    if label_col is None:
        label_col = f"Excess_Return_{horizon}"
    return (df[label_col] > 0).astype(int).values


# ---------------------------------------------------------------------------
# Politician node feature computation (expanding window, no look-ahead)
# ---------------------------------------------------------------------------

# Number of hand-crafted politician features:
#   win_rate (1) + log_trade_count (1) + buy_ratio (1) + chamber (1) + party (3) = 7
POL_FEAT_DIM = 7


def compute_politician_features(
    train_df: pd.DataFrame,
    horizon: str,
    le_pol: LabelEncoder,
    device: torch.device,
    gap_df: pd.DataFrame = None,
    label_col: str = None,
) -> torch.Tensor:
    """
    Compute hand-crafted features for each politician from the training window,
    optionally filling demographic features for gap-only politicians.

    Features (7 dims):
      0: rolling win rate — fraction of resolved trades with label > 0
      1: log(1 + trade count) — cold-start indicator / experience
      2: buy ratio — fraction of trades that are purchases
      3: chamber — House=0, Senate=1
      4: party Democrat — one-hot
      5: party Republican — one-hot
      6: party Independent — one-hot (or other)

    Label-derived features (0, 1, 2) come from training data only (no look-ahead).
    Demographic features (3-6) can be filled from gap_df for politicians not in
    train_df. Politicians with no resolved trades keep defaults (0.5 win rate,
    0 trades, 0.5 buy ratio).

    Parameters
    ----------
    label_col : str, optional
        Column to use as label (default: ``Excess_Return_{horizon}``).

    Returns:
        pol_features: [n_pol, 7] tensor on device
    """
    if label_col is None:
        label_col = f"Excess_Return_{horizon}"
    target = label_col
    n_pol = len(le_pol.classes_)
    feats = np.zeros((n_pol, POL_FEAT_DIM), dtype=np.float32)

    # Default values for cold-start politicians
    feats[:, 0] = 0.5   # win rate prior
    feats[:, 1] = 0.0   # log trade count = 0 (no trades)
    feats[:, 2] = 0.5   # buy ratio prior

    # Track which politicians have been filled (for gap fallback)
    filled = set()

    # Group training data by BioGuideID
    df_copy = train_df.copy()
    df_copy["_bio"] = df_copy["BioGuideID"].fillna("UNK")
    df_copy["_label"] = (df_copy[target] > 0).astype(float)
    df_copy["_is_buy"] = (df_copy["Transaction"].str.lower().str.contains(
        "purchase", na=False
    )).astype(float)

    # Aggregate per politician from training data
    for bio_id, grp in df_copy.groupby("_bio"):
        try:
            idx = le_pol.transform([bio_id])[0]
        except ValueError:
            continue  # politician not in encoder (shouldn't happen)

        filled.add(idx)
        n_trades = len(grp)
        n_positive = grp["_label"].sum()

        # Feature 0: win rate
        feats[idx, 0] = n_positive / n_trades

        # Feature 1: log(1 + trade count)
        feats[idx, 1] = np.log1p(n_trades)

        # Feature 2: buy ratio
        feats[idx, 2] = grp["_is_buy"].mean()

        # Static features from first record (constant per politician)
        row = grp.iloc[0]

        # Feature 3: chamber (House=0, Senate=1)
        chamber = str(row.get("Chamber", "")).strip().lower()
        feats[idx, 3] = 1.0 if chamber == "senate" else 0.0

        # Features 4-6: party one-hot
        party = str(row.get("Party", "")).strip().lower()
        if "democrat" in party:
            feats[idx, 4] = 1.0
        elif "republican" in party:
            feats[idx, 5] = 1.0
        else:
            feats[idx, 6] = 1.0  # Independent / other

    # Fill demographic features (3-6) from gap_df for gap-only politicians
    if gap_df is not None and len(gap_df) > 0:
        gap_copy = gap_df.copy()
        gap_copy["_bio"] = gap_copy["BioGuideID"].fillna("UNK")
        for bio_id, grp in gap_copy.groupby("_bio"):
            try:
                idx = le_pol.transform([bio_id])[0]
            except ValueError:
                continue
            if idx in filled:
                continue  # already have full features from training data

            row = grp.iloc[0]

            # Feature 3: chamber
            chamber = str(row.get("Chamber", "")).strip().lower()
            feats[idx, 3] = 1.0 if chamber == "senate" else 0.0

            # Features 4-6: party one-hot
            party = str(row.get("Party", "")).strip().lower()
            if "democrat" in party:
                feats[idx, 4] = 1.0
            elif "republican" in party:
                feats[idx, 5] = 1.0
            else:
                feats[idx, 6] = 1.0

            # buy_ratio from gap trades (no labels, but Transaction type is known)
            if "Transaction" in grp.columns:
                is_buy = grp["Transaction"].str.lower().str.contains(
                    "purchase", na=False
                ).astype(float)
                feats[idx, 2] = is_buy.mean()

    return torch.tensor(feats, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# State ID computation: per-politician, from training window
# ---------------------------------------------------------------------------

def compute_state_ids_per_pol(
    train_df: pd.DataFrame,
    le_pol: LabelEncoder,
    le_state: LabelEncoder,
    device: torch.device,
    gap_df: pd.DataFrame = None,
) -> torch.Tensor:
    """
    Return a [n_pol] long tensor mapping each politician to their state index.
    Politicians not seen in training get state 0 (UNK default), unless they
    appear in gap_df, in which case their state is filled from there.
    """
    n_pol = len(le_pol.classes_)
    state_ids = np.zeros(n_pol, dtype=np.int64)  # default: UNK (index 0)
    filled = set()

    bio_col = train_df["BioGuideID"].fillna("UNK")
    state_col = train_df["State"].fillna("UNK")

    for bio_id, state_val in zip(bio_col, state_col):
        try:
            pol_idx = le_pol.transform([bio_id])[0]
            state_idx = le_state.transform([state_val])[0]
            state_ids[pol_idx] = state_idx
            filled.add(pol_idx)
        except ValueError:
            pass  # unseen politician or state — keep default 0

    # Fill from gap_df for politicians not in training data
    if gap_df is not None and len(gap_df) > 0:
        gap_bio = gap_df["BioGuideID"].fillna("UNK")
        gap_state = gap_df["State"].fillna("UNK")
        for bio_id, state_val in zip(gap_bio, gap_state):
            try:
                pol_idx = le_pol.transform([bio_id])[0]
                if pol_idx in filled:
                    continue
                state_idx = le_state.transform([state_val])[0]
                state_ids[pol_idx] = state_idx
            except ValueError:
                pass

    return torch.tensor(state_ids, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Bipartite graph construction
# ---------------------------------------------------------------------------

def build_bipartite_graph(
    train_df: pd.DataFrame,
    horizon: str,
    le_pol: LabelEncoder,
    le_tick: LabelEncoder,
    device: torch.device,
    gap_df: pd.DataFrame = None,
    test_start: pd.Timestamp = None,
    temporal_decay: float = 0.0,
    max_age_years: float = 0.0,
    label_col: str = None,
):
    """
    Build a bipartite graph from training data, optionally augmented with gap
    (unresolved) transactions.

    Nodes: politicians [0, n_pol) and companies [n_pol, n_pol + n_tick).
    Edges: bidirectional (politician <-> company) for each unique (pol, tick) pair.

    Gap-awareness:
      If gap_df is provided, gap transactions contribute to edge counts but NOT
      to win_rate (since their outcomes are unresolved). This lets the GNN see
      "who is trading what RIGHT NOW" — structural information that flat models
      (XGBoost, MLP) fundamentally cannot access.

      Merge logic:
        - Pairs in both train and gap: total_count = train_count + gap_count,
          win_rate = train_win_rate (from resolved trades only).
        - Pairs only in gap: count = gap_count, win_rate = 0.5 (uninformative).
        - Pairs only in train: unchanged.

    Temporal decay:
      If temporal_decay > 0 and test_start is provided, each training trade is
      weighted by exp(-temporal_decay * age_years) where age_years is the time
      from the trade's Filed date to test_start. This down-weights stale edges.
      Gap edges always count as 1 (they are recent by definition).
      When temporal_decay == 0.0, behaviour is identical to the original.

    Temporal windowing:
      If max_age_years > 0 and test_start is provided, only training trades
      filed within max_age_years before test_start are included in the graph.
      This prevents unbounded graph growth from accumulating all history since
      2015. When max_age_years == 0.0, all history is used (original behaviour).

    Edge representation (two separate tensors):
      edge_weight [2*E]:   log1p(total_count) — structural connection strength,
                           used by GraphConv as a multiplicative message weight.
                           Now includes gap counts for connection strength.
      edge_attr   [2*E,2]: [win_rate, log1p_count_norm] — outcome + strength
                           as learnable features for GATConv attention.
                           win_rate is from resolved (training) trades only.

    Returns:
        edge_index:  [2, 2*E] long tensor (bidirectional)
        edge_weight: [2*E]    float tensor of log1p(total_count)
        edge_attr:   [2*E, 2] float tensor of [win_rate, log1p_count_norm]
        n_pol: number of politician nodes
        n_tick: number of company nodes
    """
    if label_col is None:
        label_col = f"Excess_Return_{horizon}"
    target = label_col
    n_pol = len(le_pol.classes_)
    n_tick = len(le_tick.classes_)

    # --- Temporal windowing: drop edges older than max_age_years ---
    if max_age_years > 0.0 and test_start is not None:
        cutoff = test_start - pd.Timedelta(days=max_age_years * 365.25)
        train_df = train_df[train_df["Filed"] >= cutoff]
        log.info("    Temporal window (%.1fy): kept %d / %d training trades (cutoff %s)",
                 max_age_years, len(train_df),
                 n_pol,  # reuse n_pol for log length; actual original len not available here
                 cutoff.strftime("%Y-%m-%d"))

    # --- Resolved (training) edge stats ---
    pol_ids = le_pol.transform(train_df["BioGuideID"].fillna("UNK"))
    tick_ids = le_tick.transform(train_df["Ticker"].fillna("UNK"))
    labels = (train_df[target] > 0).astype(float).values

    # Compute per-trade temporal weights (1.0 when decay is disabled)
    if temporal_decay > 0.0 and test_start is not None:
        filed_dates = pd.to_datetime(train_df["Filed"].values)
        age_years = np.array(
            [(test_start - pd.Timestamp(d)).days / 365.25 for d in filed_dates],
            dtype=np.float32,
        )
        age_years = np.clip(age_years, 0.0, None)
        trade_weights = np.exp(-temporal_decay * age_years)
    else:
        trade_weights = np.ones(len(train_df), dtype=np.float32)

    # edge_stats: {(pol, tick): [weighted_count, weighted_positive_sum]}
    edge_stats = {}
    for p, t, y, w in zip(pol_ids, tick_ids, labels, trade_weights):
        key = (int(p), int(t))
        if key not in edge_stats:
            edge_stats[key] = [0.0, 0.0]  # [weighted_count, weighted_positive_sum]
        edge_stats[key][0] += w
        edge_stats[key][1] += y * w

    # --- Gap (unresolved) edge stats ---
    gap_stats = {}
    if gap_df is not None and len(gap_df) > 0:
        gap_pol_ids = le_pol.transform(gap_df["BioGuideID"].fillna("UNK"))
        gap_tick_ids = le_tick.transform(gap_df["Ticker"].fillna("UNK"))
        for p, t in zip(gap_pol_ids, gap_tick_ids):
            key = (int(p), int(t))
            gap_stats[key] = gap_stats.get(key, 0) + 1

    # --- Merge: train + gap edge stats ---
    # merged: {(pol, tick): [total_count, win_rate]}
    all_pairs = set(edge_stats.keys()) | set(gap_stats.keys())
    merged_stats = {}
    for key in all_pairs:
        train_count, train_pos_sum = edge_stats.get(key, [0.0, 0.0])
        gap_count = gap_stats.get(key, 0)
        total_count = train_count + gap_count
        win_rate = (train_pos_sum / train_count) if train_count > 0 else 0.5
        merged_stats[key] = (total_count, win_rate)

    # Build edge lists (bidirectional: pol->company and company->pol)
    src_list, dst_list, log_count_list, win_rate_list = [], [], [], []
    for (p, t), (total_count, win_rate) in merged_stats.items():
        log_count = np.log1p(total_count)
        company_node = n_pol + t  # offset company IDs

        for src, dst in [(p, company_node), (company_node, p)]:
            src_list.append(src)
            dst_list.append(dst)
            log_count_list.append(log_count)
            win_rate_list.append(win_rate)

    log_counts = np.array(log_count_list, dtype=np.float32)
    win_rates = np.array(win_rate_list, dtype=np.float32)

    # Normalize log_count to [0, 1] for edge_attr so both dims are on same scale
    max_log_count = log_counts.max() if log_counts.max() > 0 else 1.0
    log_counts_norm = log_counts / max_log_count

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
    # Structural weight: raw log1p(total_count) — includes gap for connection strength
    edge_weight = torch.tensor(log_counts, dtype=torch.float, device=device)
    # Learnable features: [win_rate, log_count_norm] — for GATConv attention
    edge_attr = torch.stack([
        torch.tensor(win_rates, dtype=torch.float, device=device),
        torch.tensor(log_counts_norm, dtype=torch.float, device=device),
    ], dim=1)  # [2*E, 2]

    if gap_df is not None and len(gap_df) > 0:
        n_gap_only = len(set(gap_stats.keys()) - set(edge_stats.keys()))
        log.info("    Gap edges: %d gap pairs (%d gap-only new), "
                 "total unique pairs: %d (was %d)",
                 len(gap_stats), n_gap_only,
                 len(merged_stats), len(edge_stats))

    return edge_index, edge_weight, edge_attr, n_pol, n_tick


# ---------------------------------------------------------------------------
# GNN Models
# ---------------------------------------------------------------------------

class BipartiteSAGE(nn.Module):
    """
    2-layer GraphConv (SAGE-like) on the bipartite politician-company graph.

    Politician nodes: hand-crafted features (win_rate, trade_count, buy_ratio,
      chamber, party) projected to emb_dim via a linear layer.
    Company nodes: learned embeddings (no structural features available).

    Uses GraphConv instead of SAGEConv because it natively supports edge_weight
    in its message function (edge_weight * x_j aggregation).
    Edge weights (historical win rate per politician-company pair) are used
    in weighted message passing.
    """

    def __init__(self, n_pol: int, n_tick: int,
                 pol_feat_dim: int = POL_FEAT_DIM,
                 emb_dim: int = 32,
                 hidden_dim: int = 64, out_dim: int = 32, dropout: float = 0.2,
                 n_states: int = 60):
        super().__init__()
        self.n_pol = n_pol
        self.n_tick = n_tick

        # Politician: project hand-crafted features to emb_dim
        self.pol_proj = nn.Sequential(
            nn.Linear(pol_feat_dim, emb_dim),
            nn.ReLU(),
        )

        # Politician state embedding (added to pol_proj output)
        self.state_emb = nn.Embedding(n_states, emb_dim)

        # Company: learned embedding (no structural features available)
        self.emb_tick = nn.Embedding(n_tick, emb_dim)

        # 2-layer GraphConv (like SAGE but natively supports edge weights)
        self.conv1 = GraphConv(emb_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, out_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

        # Politician features and state IDs are set per-month
        self._pol_features = None
        self._state_ids = None

    def set_pol_features(self, pol_features: torch.Tensor):
        """Set the politician feature tensor [n_pol, pol_feat_dim] for this month."""
        self._pol_features = pol_features

    def set_state_ids(self, state_ids: torch.Tensor):
        """Set per-politician state IDs [n_pol] for this month."""
        self._state_ids = state_ids

    def get_node_features(self, device: torch.device) -> torch.Tensor:
        """Construct the full node feature matrix [n_pol + n_tick, emb_dim]."""
        pol_feats = self.pol_proj(self._pol_features)          # [n_pol, emb_dim]
        pol_feats = pol_feats + self.state_emb(self._state_ids)  # add state embedding
        tick_feats = self.emb_tick.weight                        # [n_tick, emb_dim]
        return torch.cat([pol_feats, tick_feats], dim=0)

    def forward(self, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None,
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass: compute node embeddings via weighted message passing.
        edge_weight: log1p(trade_count) — structural weight for GraphConv.
        edge_attr: ignored (accepted for interface consistency with BipartiteGAT).

        Returns: [n_pol + n_tick, out_dim] node embeddings.
        """
        x = self.get_node_features(edge_index.device)

        # Layer 1 — pass edge_weight for weighted aggregation
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index, edge_weight=edge_weight)

        return x


class BipartiteGAT(nn.Module):
    """
    2-layer GAT on the bipartite politician-company graph.

    Politician nodes: hand-crafted features projected to emb_dim.
    Company nodes: learned embeddings.
    Uses multi-head attention to weight neighbour contributions.
    Edge weights are passed as edge_attr to GATConv.
    """

    def __init__(self, n_pol: int, n_tick: int,
                 pol_feat_dim: int = POL_FEAT_DIM,
                 emb_dim: int = 32,
                 hidden_dim: int = 64, out_dim: int = 32, heads: int = 4,
                 dropout: float = 0.2, n_states: int = 60):
        super().__init__()
        self.n_pol = n_pol
        self.n_tick = n_tick

        # Politician: project hand-crafted features to emb_dim
        self.pol_proj = nn.Sequential(
            nn.Linear(pol_feat_dim, emb_dim),
            nn.ReLU(),
        )

        # Politician state embedding (added to pol_proj output)
        self.state_emb = nn.Embedding(n_states, emb_dim)

        self.emb_tick = nn.Embedding(n_tick, emb_dim)

        # 2-layer GAT: first layer uses multi-head, second concatenates
        # edge_dim=2: [win_rate, log1p_count_norm] as learnable edge features
        self.conv1 = GATConv(emb_dim, hidden_dim // heads, heads=heads,
                             dropout=dropout, edge_dim=2)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=1, concat=False,
                             dropout=dropout, edge_dim=2)

        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

        self._pol_features = None
        self._state_ids = None

    def set_pol_features(self, pol_features: torch.Tensor):
        """Set the politician feature tensor [n_pol, pol_feat_dim] for this month."""
        self._pol_features = pol_features

    def set_state_ids(self, state_ids: torch.Tensor):
        """Set per-politician state IDs [n_pol] for this month."""
        self._state_ids = state_ids

    def get_node_features(self, device: torch.device) -> torch.Tensor:
        pol_feats = self.pol_proj(self._pol_features)            # [n_pol, emb_dim]
        pol_feats = pol_feats + self.state_emb(self._state_ids)  # add state embedding
        tick_feats = self.emb_tick.weight
        return torch.cat([pol_feats, tick_feats], dim=0)

    def forward(self, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None,
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        """
        edge_weight: ignored (accepted for interface consistency with BipartiteSAGE).
        edge_attr: [E, 2] tensor of [win_rate, log1p_count_norm] for attention.
        """
        x = self.get_node_features(edge_index.device)

        # GATConv expects edge_attr as [E, edge_dim]
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr=edge_attr)

        return x


# ---------------------------------------------------------------------------
# Prediction head
# ---------------------------------------------------------------------------

class EdgePredictor(nn.Module):
    """
    Simple MLP that takes concatenated (src_embedding, dst_embedding)
    and predicts binary outcome (up/down).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_src: [batch, dim] — politician embeddings
            z_dst: [batch, dim] — company embeddings
        Returns:
            logits: [batch] — raw logits for binary classification
        """
        h = torch.cat([z_src, z_dst], dim=-1)
        return self.net(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Combined GNN + Predictor wrapper
# ---------------------------------------------------------------------------

class GNNPredictor(nn.Module):
    """
    End-to-end model: GNN produces node embeddings on the bipartite graph,
    then the EdgePredictor classifies (politician, company) pairs.
    """

    def __init__(self, gnn: nn.Module, predictor: EdgePredictor):
        super().__init__()
        self.gnn = gnn
        self.predictor = predictor

    def forward(self, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                edge_attr: torch.Tensor,
                src_ids: torch.Tensor, dst_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_index:  graph structure [2, E]
            edge_weight: log1p(trade_count) structural weights [E]
            edge_attr:   [win_rate, log1p_count_norm] learnable edge features [E, 2]
            src_ids:     politician node indices for prediction batch [B]
            dst_ids:     company node indices for prediction batch [B]
        Returns:
            logits: [B]
        """
        # Compute all node embeddings via GNN
        z = self.gnn(edge_index, edge_weight, edge_attr)

        # Look up embeddings for the prediction batch
        z_src = z[src_ids]
        z_dst = z[dst_ids]

        return self.predictor(z_src, z_dst)


# ---------------------------------------------------------------------------
# Training loop for one month
# ---------------------------------------------------------------------------

def train_gnn_month(
    model: GNNPredictor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_attr: torch.Tensor,
    train_src: torch.Tensor,
    train_dst: torch.Tensor,
    train_labels: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    grad_clip: float = 1.0,
    patience: int = 0,
):
    """
    Train the GNN + predictor end-to-end for one walk-forward month.

    Each epoch:
      1. Run GNN forward on the full graph to get node embeddings.
      2. For each mini-batch of training edges, compute predictions and loss.
      3. Backprop through predictor AND GNN (including embeddings).

    Args:
        patience: Early stopping patience. If > 0, training stops when loss
                  has not improved for `patience` consecutive epochs.
                  If 0 (default), always runs all `epochs`.
    """
    # Class weighting for imbalanced labels
    pos = train_labels.sum().item()
    neg = len(train_labels) - pos
    pw = torch.tensor([neg / max(1, pos)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    n = len(train_src)
    best_loss = float("inf")
    stagnant_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(edge_index, edge_weight, edge_attr,
                           train_src[idx], train_dst[idx])
            loss = criterion(logits, train_labels[idx])

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            log.info("      Epoch %d/%d  loss=%.4f  lr=%.6f",
                     epoch, epochs, avg_loss, current_lr)

        # Early stopping check
        if patience > 0:
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                stagnant_epochs = 0
            else:
                stagnant_epochs += 1
                if stagnant_epochs >= patience:
                    log.info("      Early stop at epoch %d/%d (patience=%d)",
                             epoch, epochs, patience)
                    break


@torch.no_grad()
def eval_gnn(
    model: GNNPredictor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    edge_attr: torch.Tensor,
    test_src: torch.Tensor,
    test_dst: torch.Tensor,
) -> np.ndarray:
    """Evaluate: return predicted probabilities."""
    model.eval()
    logits = model(edge_index, edge_weight, edge_attr, test_src, test_dst)
    return logits.sigmoid().cpu().numpy()


# ---------------------------------------------------------------------------
# Model A: GNN-SAGE
# ---------------------------------------------------------------------------

def run_gnn_sage(df, years, horizon, seed, device, epochs=50, lr=1e-3,
                 batch_size=512, emb_dim=32, hidden_dim=64, out_dim=32,
                 use_gap=False, warm_start=True, patience=10,
                 temporal_decay=0.0, max_age_years=0.0,
                 dead_zone: float = 0.0,
                 label_col: str = None):
    if label_col is None:
        label_col = f"Excess_Return_{horizon}"

    set_seed(seed)
    all_preds, all_labels = [], []
    all_ids, all_filed, all_years_list, all_months_list = [], [], [], []
    month_results = []

    # ------------------------------------------------------------------
    # Warm-start: fit encoders and build model ONCE on the full dataset
    # ------------------------------------------------------------------
    if warm_start:
        log.info("  [SAGE] Warm-start: fitting LabelEncoders on full dataset ...")
        le_pol = LabelEncoder().fit(df["BioGuideID"].fillna("UNK"))
        le_tick = LabelEncoder().fit(df["Ticker"].fillna("UNK"))
        le_state = LabelEncoder().fit(df["State"].fillna("UNK"))
        n_pol_global = len(le_pol.classes_)
        n_tick_global = len(le_tick.classes_)
        n_states_global = len(le_state.classes_)

        gnn = BipartiteSAGE(n_pol_global, n_tick_global,
                            pol_feat_dim=POL_FEAT_DIM,
                            emb_dim=emb_dim,
                            hidden_dim=hidden_dim, out_dim=out_dim,
                            n_states=n_states_global)
        predictor = EdgePredictor(in_dim=out_dim * 2, hidden_dim=64)
        model = GNNPredictor(gnn, predictor).to(device)
        log.info("  [SAGE] Warm-start model: n_pol=%d n_tick=%d n_states=%d",
                 n_pol_global, n_tick_global, n_states_global)
    else:
        le_pol = le_tick = le_state = model = None  # built per-fold below

    for year, month, train_df, gap_df, test_df in monthly_splits(
        df, years, horizon, label_col=label_col, max_age_years=max_age_years,
        dead_zone=dead_zone
    ):
        y_train = make_binary_label(train_df, horizon, label_col=label_col)
        y_test  = make_binary_label(test_df,  horizon, label_col=label_col)

        if warm_start:
            # Reuse global encoders; n_pol/n_tick are fixed across all folds
            n_pol = n_pol_global
            n_tick = n_tick_global
        else:
            # Per-fold LabelEncoder (fit on train+test+gap to handle all IDs)
            fit_frames_pol = [train_df["BioGuideID"], test_df["BioGuideID"]]
            fit_frames_tick = [train_df["Ticker"], test_df["Ticker"]]
            fit_frames_state = [train_df["State"], test_df["State"]]
            if use_gap and gap_df is not None and len(gap_df) > 0:
                fit_frames_pol.append(gap_df["BioGuideID"])
                fit_frames_tick.append(gap_df["Ticker"])
                fit_frames_state.append(gap_df["State"])
            le_pol = LabelEncoder().fit(pd.concat(fit_frames_pol).fillna("UNK"))
            le_tick = LabelEncoder().fit(pd.concat(fit_frames_tick).fillna("UNK"))
            le_state = LabelEncoder().fit(pd.concat(fit_frames_state).fillna("UNK"))
            n_states = len(le_state.classes_)
            n_pol = len(le_pol.classes_)
            n_tick = len(le_tick.classes_)

        # Determine test_start for temporal decay and/or max_age_years windowing
        test_start = pd.Timestamp(f"{year}-{month:02d}-01") \
            if (temporal_decay > 0 or max_age_years > 0) else None

        # Build bipartite graph (with gap edges if enabled)
        edge_index, edge_weight, edge_attr, _np, _nt = build_bipartite_graph(
            train_df, horizon, le_pol, le_tick, device,
            gap_df=gap_df if use_gap else None,
            test_start=test_start,
            temporal_decay=temporal_decay,
            max_age_years=max_age_years,
            label_col=label_col,
        )

        # Prepare training transaction edges (for the prediction task)
        pol_train = torch.tensor(
            le_pol.transform(train_df["BioGuideID"].fillna("UNK")),
            dtype=torch.long, device=device
        )
        # Company node IDs are offset by n_pol in the graph
        tick_train = torch.tensor(
            le_tick.transform(train_df["Ticker"].fillna("UNK")),
            dtype=torch.long, device=device
        ) + n_pol
        lab_train = torch.tensor(y_train, dtype=torch.float, device=device)

        pol_test = torch.tensor(
            le_pol.transform(test_df["BioGuideID"].fillna("UNK")),
            dtype=torch.long, device=device
        )
        tick_test = torch.tensor(
            le_tick.transform(test_df["Ticker"].fillna("UNK")),
            dtype=torch.long, device=device
        ) + n_pol

        # Compute politician node features from training window (+ gap demographics)
        _gap = gap_df if use_gap else None
        pol_features = compute_politician_features(
            train_df, horizon, le_pol, device, gap_df=_gap, label_col=label_col
        )
        state_ids = compute_state_ids_per_pol(
            train_df, le_pol, le_state, device, gap_df=_gap
        )

        if not warm_start:
            # Build a fresh model for this fold
            gnn = BipartiteSAGE(n_pol, n_tick, pol_feat_dim=POL_FEAT_DIM,
                                emb_dim=emb_dim,
                                hidden_dim=hidden_dim, out_dim=out_dim,
                                n_states=n_states)
            predictor = EdgePredictor(in_dim=out_dim * 2, hidden_dim=64)
            model = GNNPredictor(gnn, predictor).to(device)

        # Update dynamic node features (re-computed each fold from train window)
        model.gnn.set_pol_features(pol_features)
        model.gnn.set_state_ids(state_ids)

        n_edges = edge_index.size(1) // 2  # bidirectional, so divide by 2
        log.info("  [SAGE] %d-%02d: Train=%d Test=%d Edges=%d Nodes=%d",
                 year, month, len(train_df), len(test_df), n_edges,
                 n_pol + n_tick)

        # Train (fresh optimizer each month; model weights carried over if warm_start)
        train_gnn_month(model, edge_index, edge_weight, edge_attr,
                        pol_train, tick_train, lab_train,
                        epochs=epochs, lr=lr, batch_size=batch_size,
                        device=device, patience=patience)

        # Evaluate
        preds = eval_gnn(model, edge_index, edge_weight, edge_attr, pol_test, tick_test)

        all_preds.extend(preds)
        all_labels.extend(y_test)
        all_ids.extend(test_df["transaction_id"].values)
        all_filed.extend(test_df["Filed"].values)
        all_years_list.extend([year] * len(y_test))
        all_months_list.extend([month] * len(y_test))
        auc = roc_auc_score(y_test, preds) if len(np.unique(y_test)) > 1 else 0.5
        month_results.append({"year": year, "month": month, "auc": auc, "n": len(y_test)})
        log.info("    %d-%02d: AUC=%.4f (n=%d)", year, month, auc, len(y_test))

    if not all_preds:
        empty_df = pd.DataFrame(columns=["transaction_id", "Filed", "year", "month",
                                          "prob", "pred", "label"])
        return 0.5, 0.0, 0.5, month_results, empty_df

    preds_arr = np.array(all_preds)
    preds_df = pd.DataFrame({
        "transaction_id": all_ids,
        "Filed": all_filed,
        "year": all_years_list,
        "month": all_months_list,
        "prob": preds_arr,
        "pred": (preds_arr > 0.5).astype(int),
        "label": all_labels,
    })
    pooled_auc = roc_auc_score(all_labels, all_preds) \
        if len(np.unique(all_labels)) > 1 else 0.5
    pooled_f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    pooled_acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    return pooled_auc, pooled_f1, pooled_acc, month_results, preds_df



# ---------------------------------------------------------------------------
# Model B: GNN-GAT
# ---------------------------------------------------------------------------

def run_gnn_gat(df, years, horizon, seed, device, epochs=30, lr=1e-3,
                batch_size=512, emb_dim=32, hidden_dim=64, out_dim=32,
                use_gap=False):
    set_seed(seed)
    all_preds, all_labels = [], []
    month_results = []

    for year, month, train_df, gap_df, test_df in monthly_splits(df, years, horizon):
        y_train = make_binary_label(train_df, horizon)
        y_test = make_binary_label(test_df, horizon)

        # Per-fold LabelEncoder (fit on train+test+gap to handle all IDs)
        fit_frames_pol = [train_df["BioGuideID"], test_df["BioGuideID"]]
        fit_frames_tick = [train_df["Ticker"], test_df["Ticker"]]
        fit_frames_state = [train_df["State"], test_df["State"]]
        if use_gap and gap_df is not None and len(gap_df) > 0:
            fit_frames_pol.append(gap_df["BioGuideID"])
            fit_frames_tick.append(gap_df["Ticker"])
            fit_frames_state.append(gap_df["State"])
        le_pol = LabelEncoder().fit(pd.concat(fit_frames_pol).fillna("UNK"))
        le_tick = LabelEncoder().fit(pd.concat(fit_frames_tick).fillna("UNK"))
        le_state = LabelEncoder().fit(pd.concat(fit_frames_state).fillna("UNK"))
        n_states = len(le_state.classes_)

        # Build bipartite graph (with gap edges if enabled)
        edge_index, edge_weight, edge_attr, n_pol, n_tick = build_bipartite_graph(
            train_df, horizon, le_pol, le_tick, device,
            gap_df=gap_df if use_gap else None,
        )

        pol_train = torch.tensor(
            le_pol.transform(train_df["BioGuideID"].fillna("UNK")),
            dtype=torch.long, device=device
        )
        tick_train = torch.tensor(
            le_tick.transform(train_df["Ticker"].fillna("UNK")),
            dtype=torch.long, device=device
        ) + n_pol
        lab_train = torch.tensor(y_train, dtype=torch.float, device=device)

        pol_test = torch.tensor(
            le_pol.transform(test_df["BioGuideID"].fillna("UNK")),
            dtype=torch.long, device=device
        )
        tick_test = torch.tensor(
            le_tick.transform(test_df["Ticker"].fillna("UNK")),
            dtype=torch.long, device=device
        ) + n_pol

        # Compute politician node features from training window (+ gap demographics)
        _gap = gap_df if use_gap else None
        pol_features = compute_politician_features(
            train_df, horizon, le_pol, device, gap_df=_gap
        )
        state_ids = compute_state_ids_per_pol(
            train_df, le_pol, le_state, device, gap_df=_gap
        )

        gnn = BipartiteGAT(n_pol, n_tick, pol_feat_dim=POL_FEAT_DIM,
                            emb_dim=emb_dim,
                            hidden_dim=hidden_dim, out_dim=out_dim,
                            n_states=n_states)
        predictor = EdgePredictor(in_dim=out_dim * 2, hidden_dim=64)
        model = GNNPredictor(gnn, predictor).to(device)
        model.gnn.set_pol_features(pol_features)
        model.gnn.set_state_ids(state_ids)

        n_edges = edge_index.size(1) // 2
        log.info("  [GAT] %d-%02d: Train=%d Test=%d Edges=%d Nodes=%d",
                 year, month, len(train_df), len(test_df), n_edges,
                 n_pol + n_tick)

        train_gnn_month(model, edge_index, edge_weight, edge_attr,
                        pol_train, tick_train, lab_train,
                        epochs=epochs, lr=lr, batch_size=batch_size,
                        device=device)

        preds = eval_gnn(model, edge_index, edge_weight, edge_attr, pol_test, tick_test)

        all_preds.extend(preds)
        all_labels.extend(y_test)
        auc = roc_auc_score(y_test, preds) if len(np.unique(y_test)) > 1 else 0.5
        month_results.append({"year": year, "month": month, "auc": auc, "n": len(y_test)})
        log.info("    %d-%02d: AUC=%.4f (n=%d)", year, month, auc, len(y_test))

    if not all_preds:
        return 0.5, 0.0, 0.5, month_results

    pooled_auc = roc_auc_score(all_labels, all_preds) \
        if len(np.unique(all_labels)) > 1 else 0.5
    pooled_f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    pooled_acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    return pooled_auc, pooled_f1, pooled_acc, month_results


# ---------------------------------------------------------------------------
# Model C: GNN + XGBoost (hybrid)
# ---------------------------------------------------------------------------

def run_gnn_xgb(df, years, horizon, seed, device, epochs=30, lr=1e-3,
                batch_size=512, emb_dim=32, hidden_dim=64, out_dim=32,
                use_gap=False):
    """
    Hybrid approach:
      1. Train GNN end-to-end on training data (same as SAGE model).
      2. Extract learned node embeddings after GNN training.
      3. Use (pol_embedding, tick_embedding, pol_id, tick_id) as features
         for XGBoost.

    This tests whether GNN embeddings add value on top of identity features.
    """
    import xgboost as xgb

    set_seed(seed)
    all_preds, all_labels = [], []
    month_results = []

    for year, month, train_df, gap_df, test_df in monthly_splits(df, years, horizon):
        y_train = make_binary_label(train_df, horizon)
        y_test = make_binary_label(test_df, horizon)

        # Per-fold LabelEncoder (fit on train+test+gap to handle all IDs)
        fit_frames_pol = [train_df["BioGuideID"], test_df["BioGuideID"]]
        fit_frames_tick = [train_df["Ticker"], test_df["Ticker"]]
        fit_frames_state = [train_df["State"], test_df["State"]]
        if use_gap and gap_df is not None and len(gap_df) > 0:
            fit_frames_pol.append(gap_df["BioGuideID"])
            fit_frames_tick.append(gap_df["Ticker"])
            fit_frames_state.append(gap_df["State"])
        le_pol = LabelEncoder().fit(pd.concat(fit_frames_pol).fillna("UNK"))
        le_tick = LabelEncoder().fit(pd.concat(fit_frames_tick).fillna("UNK"))
        le_state = LabelEncoder().fit(pd.concat(fit_frames_state).fillna("UNK"))
        n_states = len(le_state.classes_)

        # Build bipartite graph (with gap edges if enabled)
        edge_index, edge_weight, edge_attr, n_pol, n_tick = build_bipartite_graph(
            train_df, horizon, le_pol, le_tick, device,
            gap_df=gap_df if use_gap else None,
        )

        pol_train_ids = le_pol.transform(train_df["BioGuideID"].fillna("UNK"))
        tick_train_ids = le_tick.transform(train_df["Ticker"].fillna("UNK"))
        pol_test_ids = le_pol.transform(test_df["BioGuideID"].fillna("UNK"))
        tick_test_ids = le_tick.transform(test_df["Ticker"].fillna("UNK"))

        pol_train_t = torch.tensor(pol_train_ids, dtype=torch.long, device=device)
        tick_train_t = torch.tensor(tick_train_ids, dtype=torch.long, device=device) + n_pol
        lab_train_t = torch.tensor(y_train, dtype=torch.float, device=device)

        # Compute politician node features from training window (+ gap demographics)
        _gap = gap_df if use_gap else None
        pol_features = compute_politician_features(
            train_df, horizon, le_pol, device, gap_df=_gap
        )
        state_ids = compute_state_ids_per_pol(
            train_df, le_pol, le_state, device, gap_df=_gap
        )

        # Step 1: Train GNN end-to-end
        gnn = BipartiteSAGE(n_pol, n_tick, pol_feat_dim=POL_FEAT_DIM,
                            emb_dim=emb_dim,
                            hidden_dim=hidden_dim, out_dim=out_dim,
                            n_states=n_states)
        predictor = EdgePredictor(in_dim=out_dim * 2, hidden_dim=64)
        model = GNNPredictor(gnn, predictor).to(device)
        model.gnn.set_pol_features(pol_features)
        model.gnn.set_state_ids(state_ids)

        n_edges = edge_index.size(1) // 2
        log.info("  [GNN+XGB] %d-%02d: Train=%d Test=%d Edges=%d",
                 year, month, len(train_df), len(test_df), n_edges)

        train_gnn_month(model, edge_index, edge_weight, edge_attr,
                        pol_train_t, tick_train_t, lab_train_t,
                        epochs=epochs, lr=lr, batch_size=batch_size,
                        device=device)

        # Step 2: Extract GNN embeddings (frozen)
        model.eval()
        with torch.no_grad():
            z = model.gnn(edge_index, edge_weight, edge_attr).cpu().numpy()

        # Build features: GNN embeddings + raw IDs
        def make_features(pol_ids, tick_ids):
            pol_emb = z[pol_ids]                     # [B, out_dim]
            tick_emb = z[tick_ids + n_pol]            # [B, out_dim]
            # Concatenate: GNN embeddings + categorical IDs
            feats = np.hstack([pol_emb, tick_emb])
            feat_df = pd.DataFrame(feats, columns=[
                f"pol_emb_{i}" for i in range(out_dim)
            ] + [
                f"tick_emb_{i}" for i in range(out_dim)
            ])
            feat_df["pol_id"] = pol_ids
            feat_df["tick_id"] = tick_ids
            # Mark ID columns as categorical for XGBoost
            feat_df["pol_id"] = feat_df["pol_id"].astype("category")
            feat_df["tick_id"] = feat_df["tick_id"].astype("category")
            return feat_df

        X_train = make_features(pol_train_ids, tick_train_ids)
        X_test = make_features(pol_test_ids, tick_test_ids)

        # Step 3: Train XGBoost on combined features
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
        auc = roc_auc_score(y_test, preds) if len(np.unique(y_test)) > 1 else 0.5
        month_results.append({"year": year, "month": month, "auc": auc, "n": len(y_test)})
        log.info("    %d-%02d: AUC=%.4f (n=%d)", year, month, auc, len(y_test))

    if not all_preds:
        return 0.5, 0.0, 0.5, month_results

    pooled_auc = roc_auc_score(all_labels, all_preds) \
        if len(np.unique(all_labels)) > 1 else 0.5
    pooled_f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    pooled_acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    return pooled_auc, pooled_f1, pooled_acc, month_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GNN Signal Isolation Study — Lightweight GNN on bipartite graph"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=str, default="6M",
                        choices=list(HORIZON_DAYS.keys()))
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="Test years (e.g. --years 2019 2020 2021 2022 2023 2024)")
    parser.add_argument("--year", type=int, default=None,
                        help="Deprecated alias for --years (single year)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="GNN training epochs per month (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--emb-dim", type=int, default=32,
                        help="Node embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="GNN hidden dimension")
    parser.add_argument("--out-dim", type=int, default=32,
                        help="GNN output embedding dimension")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Models to skip (sage, gat, gnn_xgb)")
    parser.add_argument("--gap", action="store_true", default=False,
                        help="Include gap (unresolved) transactions as graph edges")
    parser.add_argument("--no-warm-start", action="store_true", default=False,
                        help="Disable warm-start for GNN-SAGE (fit encoders+model "
                             "per fold instead of once globally)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience for GNN-SAGE (0=disabled, "
                             "default: 10)")
    parser.add_argument("--temporal-decay", type=float, default=0.5,
                        help="Temporal edge decay rate for GNN-SAGE "
                             "(0=disabled, default: 0.5)")
    parser.add_argument("--max-age-years", type=float, default=0.0,
                        help="Rolling window size in years (0 = unbounded expanding window).")
    parser.add_argument("--dead-zone", type=float, default=0.0,
                        help="Exclude training rows where |return| < dead_zone "
                             "(0 = no filtering, default). Test set is never filtered.")
    parser.add_argument("--data", type=str, default="data/processed/ml_dataset_clean.csv",
                        help="Path to the ML dataset CSV (default: ml_dataset_clean.csv)")
    parser.add_argument("--save-preds", action="store_true", default=False,
                        help="Save per-row predictions to a CSV alongside the summary")
    parser.add_argument("--label-type", type=str, default="excess",
                        choices=["raw", "excess"],
                        help="Label type: 'raw' = Stock_Return_{horizon} > 0, "
                             "'excess' = Excess_Return_{horizon} > 0 (default: excess)")
    args = parser.parse_args()

    # Resolve years list (--years takes precedence; --year is deprecated alias)
    if args.years is not None:
        years = args.years
    elif args.year is not None:
        years = [args.year]
    else:
        years = [2023]
    years_str = "_".join(str(y) for y in years)

    # Resolve label column
    if args.label_type == "raw":
        label_col = f"Stock_Return_{args.horizon}"
    else:
        label_col = f"Excess_Return_{args.horizon}"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    log.info("GNN Study: years=%s horizon=%s seed=%d epochs=%d lr=%.1e gap=%s "
             "warm_start=%s patience=%d temporal_decay=%.2f max_age_years=%.1f "
             "dead_zone=%.2f label_type=%s",
             years_str, args.horizon, args.seed, args.epochs, args.lr, args.gap,
             not args.no_warm_start, args.patience, args.temporal_decay,
             args.max_age_years, args.dead_zone, args.label_type)

    csv_path = args.data
    df = load_data(csv_path)
    log.info("Loaded %d rows from %s", len(df), csv_path)

    valid = df[label_col].notna()
    up_rate = (df.loc[valid, label_col] > 0).mean()
    log.info("Overall label rate (%s): %.1f%% up (n=%d valid)",
             label_col, up_rate * 100, valid.sum())

    results_table = []
    preds_list = []  # collects (model_name, preds_df) when --save-preds
    # Window suffix for filenames: "W1y", "W2y", etc., or "" for unbounded
    window_tag = f"_W{args.max_age_years:.0f}y" if args.max_age_years > 0 else ""
    dz_tag = f"_DZ{args.dead_zone * 100:.0f}pct" if args.dead_zone > 0 else ""
    file_tag = window_tag + dz_tag
    gnn_kwargs = dict(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        emb_dim=args.emb_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim,
        use_gap=args.gap,
    )
    # SAGE-specific kwargs (warm-start, early stopping, temporal decay, rolling window, label)
    sage_kwargs = dict(
        **gnn_kwargs,
        warm_start=not args.no_warm_start,
        patience=args.patience,
        temporal_decay=args.temporal_decay,
        max_age_years=args.max_age_years,
        dead_zone=args.dead_zone,
        label_col=label_col,
    )
    gap_suffix = " (gap)" if args.gap else ""

    # ---- A. GNN-SAGE ----
    if "sage" not in args.skip:
        log.info("=" * 60)
        log.info("Running: GNN-SAGE%s (2-layer GraphSAGE)", gap_suffix)
        t0 = time.time()
        auc, f1, acc, months, sage_preds_df = run_gnn_sage(
            df, years, args.horizon, args.seed, device, **sage_kwargs
        )
        dt = time.time() - t0
        log.info("GNN-SAGE%s: AUC=%.4f  F1=%.4f  Acc=%.4f  (%.1fs)",
                 gap_suffix, auc, f1, acc, dt)
        results_table.append((f"GNN-SAGE{gap_suffix}", auc, f1, acc, months))
        preds_list.append((f"GNN-SAGE{gap_suffix}", sage_preds_df))

    # ---- B. GNN-GAT ----
    if "gat" not in args.skip:
        log.info("=" * 60)
        log.info("Running: GNN-GAT%s (2-layer GAT)", gap_suffix)
        t0 = time.time()
        auc, f1, acc, months = run_gnn_gat(
            df, years, args.horizon, args.seed, device, **gnn_kwargs
        )
        dt = time.time() - t0
        log.info("GNN-GAT%s: AUC=%.4f  F1=%.4f  Acc=%.4f  (%.1fs)",
                 gap_suffix, auc, f1, acc, dt)
        results_table.append((f"GNN-GAT{gap_suffix}", auc, f1, acc, months))

    # ---- C. GNN+XGB ----
    if "gnn_xgb" not in args.skip:
        log.info("=" * 60)
        log.info("Running: GNN+XGB%s (SAGE embeddings + XGBoost)", gap_suffix)
        t0 = time.time()
        auc, f1, acc, months = run_gnn_xgb(
            df, years, args.horizon, args.seed, device, **gnn_kwargs
        )
        dt = time.time() - t0
        log.info("GNN+XGB%s: AUC=%.4f  F1=%.4f  Acc=%.4f  (%.1fs)",
                 gap_suffix, auc, f1, acc, dt)
        results_table.append((f"GNN+XGB{gap_suffix}", auc, f1, acc, months))

    # ---- Summary Table ----
    log.info("")
    log.info("=" * 70)
    log.info("  GNN STUDY RESULTS — %s / %s / seed=%d / label=%s",
             years_str, args.horizon, args.seed, args.label_type)
    log.info("=" * 70)
    log.info("%-20s  %8s  %8s  %8s", "Model", "AUC", "F1", "Acc")
    log.info("-" * 50)
    for name, auc, f1, acc, _ in results_table:
        log.info("%-20s  %8.4f  %8.4f  %8.4f", name, auc, f1, acc)
    log.info("-" * 50)

    # Reference baselines from signal isolation study (for context)
    log.info("")
    log.info("Reference baselines (from signal_isolation/run_study.py):")
    log.info("  Random:      AUC ~0.500")
    log.info("  XGBoost(ID): AUC ~0.577")
    log.info("  MLP(ID):     AUC ~0.530")
    log.info("  TGN(ID):     AUC ~0.512")

    # Monthly breakdown (per year)
    log.info("")
    log.info("Monthly AUC breakdown:")
    header = "Year  Month  " + "  ".join(f"{name:>12s}" for name, *_ in results_table)
    log.info(header)
    log.info("-" * len(header))
    for yr in years:
        for m in range(1, 13):
            row = f"{yr}  {m:02d}     "
            for name, _, _, _, months in results_table:
                mr = [x for x in months if x.get("year") == yr and x["month"] == m]
                if mr:
                    row += f"  {mr[0]['auc']:12.4f}"
                else:
                    row += f"  {'N/A':>12s}"
            log.info(row)

    # Save results
    out_dir = Path("experiments/signal_isolation/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame([
        {"Model": name, "AUC": auc, "F1": f1, "Acc": acc}
        for name, auc, f1, acc, _ in results_table
    ])
    label_suffix = f"_{args.label_type}" if args.label_type != "excess" else ""
    out_file = out_dir / (
        f"gnn_study_{years_str}_{args.horizon}_seed{args.seed}"
        f"{'_gap' if args.gap else ''}{label_suffix}{file_tag}.csv"
    )
    results_df.to_csv(out_file, index=False)
    log.info("\nResults saved to %s", out_file)

    # Save per-row predictions if requested
    # When --label-type is set, also save in raw_return dir for timeline plotting
    if args.save_preds and preds_list:
        # Standard save (original location)
        all_preds_dfs = []
        for model_name, preds_df in preds_list:
            preds_df = preds_df.copy()
            preds_df.insert(0, "model", model_name)
            all_preds_dfs.append(preds_df)
        combined = pd.concat(all_preds_dfs, ignore_index=True)
        preds_file = out_dir / (
            f"preds_gnn_{years_str}_{args.horizon}_seed{args.seed}"
            f"{'_gap' if args.gap else ''}{label_suffix}{file_tag}.csv"
        )
        combined.to_csv(preds_file, index=False)
        log.info("Per-row predictions saved to %s  (n=%d)", preds_file, len(combined))

        # Also save in raw_return directory with format matching run_raw_return_study.py
        # so that plot_raw_vs_excess_timeline.py can include GNN-SAGE
        raw_return_dir = Path("experiments/signal_isolation/results/raw_return")
        raw_return_dir.mkdir(parents=True, exist_ok=True)
        for model_name, preds_df in preds_list:
            if preds_df.empty:
                continue
            preds_df = preds_df.copy()
            preds_df.insert(0, "label_type", args.label_type)
            preds_df["model"] = model_name
            # Save one file per level=0 (threshold >0) matching naming convention
            # preds_{years_str}_{horizon}_{label_type}_L0_{model_slug}{window_tag}_seed{seed}.csv
            model_slug = model_name.lower().replace(" ", "_").replace("-", "_")
            rr_file = raw_return_dir / (
                f"preds_{years_str}_{args.horizon}_{args.label_type}_L0"
                f"_{model_slug}{file_tag}_seed{args.seed}.csv"
            )
            preds_df.to_csv(rr_file, index=False)
            log.info("Raw-return-format preds saved to %s", rr_file)


if __name__ == "__main__":
    main()
