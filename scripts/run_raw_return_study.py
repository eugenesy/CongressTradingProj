"""
run_raw_return_study.py
=======================
Executable Pipeline Runners. Imports src.* modules to execute experiments.

Refactored/Audited: 2026-03-20
"""

#!/usr/bin/env python3
"""
Raw vs Excess Return Label Comparison Study
============================================
Directly compares using raw stock returns vs SPY-excess returns as labels.
Motivation: the SPY benchmark adds noise because it requires the prediction
to be correct about relative outperformance, which is harder to learn than
absolute direction.

Models compared (non-neural, fast):
  1. Random          — noise floor
  2. XGBoost(ID)     — politician + ticker IDs
  3. XGBoost(feat)   — IDs + politician background features
  4. LogReg(feat)    — linear baseline with same feature set
  5. RF(feat)        — random forest with same feature set

Label types:
  raw    — Stock_Return_{horizon} > threshold
  excess — Excess_Return_{horizon} > threshold  (SPY-adjusted)

Thresholds (3 levels per horizon):
  Level 0: >0%   (any positive return)
  Level 1: horizon-scaled ~half-std threshold
  Level 2: horizon-scaled ~one-std threshold

Architecture (KEY DESIGN PRINCIPLE):
  For each (horizon, threshold, label_type) triple, build_folds() iterates
  monthly_splits() ONCE and precomputes labels, encoders, and all feature
  matrices for every fold.  Model runners receive only the precomputed fold
  list — they never touch the raw DataFrame or refit encoders.

Protocol: same gap-aware walk-forward as run_study.py.
Primary metric: Pooled AUROC, Precision@Top-10%.
Also saves: per-row probs/preds CSV and per-month AUC CSV per model/level/type.

Usage:
  python experiments/signal_isolation/run_raw_return_study.py \\
      --horizon 6M --start-year 2023 --end-year 2023 --seed 42
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.getcwd())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("raw_return_study")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Horizon-scaled thresholds (Level 0, Level 1, Level 2)
# Derived from Stock_Return distribution analysis on rebuilt ml_dataset_v2.csv
HORIZON_THRESHOLDS = {
    "1M":  (0.00, 0.03, 0.06),
    "2M":  (0.00, 0.04, 0.07),
    "3M":  (0.00, 0.03, 0.07),
    "6M":  (0.00, 0.05, 0.10),
    "12M": (0.00, 0.08, 0.16),
    "24M": (0.00, 0.10, 0.21),
}

HORIZON_DAYS = {
    "1M": 30, "2M": 60, "3M": 90, "6M": 180,
    "8M": 240, "12M": 365, "18M": 545, "24M": 730,
}

POL_FEAT_DIM = 7   # win_rate, log_count, buy_ratio, chamber, party×3


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Filed"]  = pd.to_datetime(df["Filed"])
    df["Traded"] = pd.to_datetime(df["Traded"])
    return df.sort_values("Filed").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Gap-aware monthly splits — called ONCE per (horizon, threshold) in
# build_folds(); never called directly by model runners.
# ---------------------------------------------------------------------------

def monthly_splits(df: pd.DataFrame, years, horizon: str, label_col: str,
                   max_age_years: float = 0.0, dead_zone: float = 0.0):
    """
    Yield (year, month, train_df, gap_df, test_df) for every month in years.

    Train : Filed < test_start AND resolution < test_start AND label notna
    Gap   : Filed < test_start AND resolution >= test_start
    Test  : Filed in [test_start, next_month) AND label notna

    Parameters
    ----------
    max_age_years : float
        If > 0, only training rows filed within this many years before
        test_start are included.  0 (default) uses all history.
    dead_zone : float
        If > 0, training rows where |label_col| < dead_zone are excluded.
        Applied to train only — test set is always kept complete so that
        evaluation reflects deployment conditions where the return is unknown.
    """
    if isinstance(years, int):
        years = [years]
    h_days = HORIZON_DAYS[horizon]
    target = label_col

    for year in years:
        for month in range(1, 13):
            test_start = pd.Timestamp(year, month, 1)
            next_month = test_start + pd.DateOffset(months=1)

            test_mask = (df["Filed"] >= test_start) & (df["Filed"] < next_month)
            test_df   = df[test_mask & df[target].notna()].copy()

            pre = df[df["Filed"] < test_start].copy()
            pre["_res"] = pre["Filed"] + pd.Timedelta(days=h_days)

            train_df = pre[(pre["_res"] < test_start) & pre[target].notna()].copy()
            gap_df   = pre[pre["_res"] >= test_start].copy()

            # Rolling window: drop training rows older than max_age_years
            if max_age_years > 0.0:
                cutoff = test_start - pd.Timedelta(days=max_age_years * 365.25)
                train_df = train_df[train_df["Filed"] >= cutoff].copy()

            # Dead zone: exclude near-zero training outcomes (train only)
            if dead_zone > 0.0:
                train_df = train_df[train_df[target].abs() >= dead_zone].copy()

            if len(train_df) == 0 or len(test_df) == 0:
                log.warning(
                    "Skipping %d-%02d: train=%d test=%d",
                    year, month, len(train_df), len(test_df),
                )
                continue

            yield year, month, train_df, gap_df, test_df


# ---------------------------------------------------------------------------
# Politician features
# ---------------------------------------------------------------------------

def _compute_pol_features(
    train_df: pd.DataFrame,
    label_col: str,
    threshold: float,
    le_pol: LabelEncoder,
) -> np.ndarray:
    """
    Build [n_pol, POL_FEAT_DIM] feature matrix from the training window only.
    win_rate is computed against the supplied threshold; all other features
    (log_count, buy_ratio, chamber, party) are threshold-independent.
    """
    target  = label_col
    n_pol   = len(le_pol.classes_)
    feats   = np.zeros((n_pol, POL_FEAT_DIM), dtype=np.float32)
    feats[:, 0] = 0.5   # win_rate prior
    feats[:, 2] = 0.5   # buy_ratio prior

    df = train_df.copy()
    df["_bio"]   = df["BioGuideID"].fillna("UNK")
    df["_win"]   = (df[target] > threshold).astype(float)
    df["_buy"]   = df["Transaction"].str.lower().str.contains("purchase", na=False).astype(float)

    for bio_id, grp in df.groupby("_bio"):
        try:
            idx = le_pol.transform([bio_id])[0]
        except ValueError:
            continue
        n = len(grp)
        feats[idx, 0] = grp["_win"].sum() / n
        feats[idx, 1] = np.log1p(n)
        feats[idx, 2] = grp["_buy"].mean()
        row    = grp.iloc[0]
        chamber = str(row.get("Chamber", "")).strip().lower()
        feats[idx, 3] = 1.0 if chamber == "senate" else 0.0
        party = str(row.get("Party", "")).strip().lower()
        if "democrat"   in party: feats[idx, 4] = 1.0
        elif "republican" in party: feats[idx, 5] = 1.0
        else:                        feats[idx, 6] = 1.0

    return feats


# ---------------------------------------------------------------------------
# Fold precomputation — THE SINGLE SOURCE OF TRUTH
# ---------------------------------------------------------------------------

def build_folds(
    df: pd.DataFrame,
    years,
    horizon: str,
    threshold: float,
    label_col: str,
    max_age_years: float = 0.0,
    dead_zone: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Iterate monthly_splits() exactly once and precompute everything every
    model needs.  Returns a list of fold dicts, each containing:

        year, month                          — fold identity
        y_train, y_test                      — binary labels [n_train/n_test]
        X_train_id, X_test_id                — pd.DataFrame for XGBoost(ID)
                                               (pol, tick as Categorical)
        X_train_feat_xgb, X_test_feat_xgb   — pd.DataFrame for XGBoost(feat)
                                               (pol/tick/state + 7 pol feats)
        X_train_np, X_test_np               — np.ndarray for LogReg / RF
                                               (pol_id int, tick_id int, +
                                                7 pol numeric feats) [n, 9]
    """
    folds = []

    for year, month, train_df, gap_df, test_df in monthly_splits(
        df, years, horizon, label_col, max_age_years=max_age_years, dead_zone=dead_zone
    ):
        # ---- Labels --------------------------------------------------------
        y_train = (train_df[label_col] > threshold).astype(int).values
        y_test  = (test_df[label_col]  > threshold).astype(int).values

        # ---- Encoders (fit on train ∪ test to prevent unseen-category
        #      errors at predict time without leaking test labels) -----------
        le_pol   = LabelEncoder().fit(
            pd.concat([train_df["BioGuideID"], test_df["BioGuideID"]]).fillna("UNK")
        )
        le_tick  = LabelEncoder().fit(
            pd.concat([train_df["Ticker"], test_df["Ticker"]]).fillna("UNK")
        )
        le_state = LabelEncoder().fit(
            pd.concat([train_df["State"], test_df["State"]]).fillna("UNK")
        )

        # ---- Integer IDs ---------------------------------------------------
        pol_ids_tr   = le_pol.transform(train_df["BioGuideID"].fillna("UNK"))
        tick_ids_tr  = le_tick.transform(train_df["Ticker"].fillna("UNK"))
        state_ids_tr = le_state.transform(train_df["State"].fillna("UNK"))
        pol_ids_te   = le_pol.transform(test_df["BioGuideID"].fillna("UNK"))
        tick_ids_te  = le_tick.transform(test_df["Ticker"].fillna("UNK"))
        state_ids_te = le_state.transform(test_df["State"].fillna("UNK"))

        pol_cats   = list(range(len(le_pol.classes_)))
        tick_cats  = list(range(len(le_tick.classes_)))
        state_cats = list(range(len(le_state.classes_)))

        # ---- Politician feature matrix (training window only) --------------
        pol_feats = _compute_pol_features(train_df, label_col, threshold, le_pol)

        rpf_tr = pol_feats[pol_ids_tr]   # [n_train, 7]
        rpf_te = pol_feats[pol_ids_te]   # [n_test,  7]

        # ---- XGBoost ID-only features (Categorical dtype) ------------------
        X_train_id = pd.DataFrame({
            "pol":  pd.Categorical(pol_ids_tr, categories=pol_cats),
            "tick": pd.Categorical(tick_ids_tr, categories=tick_cats),
        })
        X_test_id = pd.DataFrame({
            "pol":  pd.Categorical(pol_ids_te, categories=pol_cats),
            "tick": pd.Categorical(tick_ids_te, categories=tick_cats),
        })

        # ---- XGBoost full features (Categorical + numeric pol feats) -------
        X_train_feat_xgb = pd.DataFrame({
            "pol_id":        pd.Categorical(pol_ids_tr,   categories=pol_cats),
            "tick_id":       pd.Categorical(tick_ids_tr,  categories=tick_cats),
            "state_id":      pd.Categorical(state_ids_tr, categories=state_cats),
            "pol_win_rate":  rpf_tr[:, 0],
            "pol_log_count": rpf_tr[:, 1],
            "pol_buy_ratio": rpf_tr[:, 2],
            "pol_chamber":   rpf_tr[:, 3],
            "pol_party_dem": rpf_tr[:, 4],
            "pol_party_rep": rpf_tr[:, 5],
            "pol_party_ind": rpf_tr[:, 6],
        })
        X_test_feat_xgb = pd.DataFrame({
            "pol_id":        pd.Categorical(pol_ids_te,   categories=pol_cats),
            "tick_id":       pd.Categorical(tick_ids_te,  categories=tick_cats),
            "state_id":      pd.Categorical(state_ids_te, categories=state_cats),
            "pol_win_rate":  rpf_te[:, 0],
            "pol_log_count": rpf_te[:, 1],
            "pol_buy_ratio": rpf_te[:, 2],
            "pol_chamber":   rpf_te[:, 3],
            "pol_party_dem": rpf_te[:, 4],
            "pol_party_rep": rpf_te[:, 5],
            "pol_party_ind": rpf_te[:, 6],
        })

        # ---- Numpy arrays for LogReg / RF ----------------------------------
        # Columns: [pol_id_int, tick_id_int, win_rate, log_count, buy_ratio,
        #           chamber, party_dem, party_rep, party_ind]
        X_train_np = np.hstack([
            pol_ids_tr.reshape(-1, 1).astype(np.float32),
            tick_ids_tr.reshape(-1, 1).astype(np.float32),
            rpf_tr,
        ])
        X_test_np = np.hstack([
            pol_ids_te.reshape(-1, 1).astype(np.float32),
            tick_ids_te.reshape(-1, 1).astype(np.float32),
            rpf_te,
        ])

        folds.append({
            "year": year, "month": month,
            "y_train": y_train, "y_test": y_test,
            "X_train_id": X_train_id, "X_test_id": X_test_id,
            "X_train_feat_xgb": X_train_feat_xgb, "X_test_feat_xgb": X_test_feat_xgb,
            "X_train_np": X_train_np, "X_test_np": X_test_np,
        })

    return folds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def precision_at_top_k(labels: np.ndarray, scores: np.ndarray, k_frac: float = 0.10) -> float:
    k = max(1, int(len(labels) * k_frac))
    top_idx = np.argsort(scores)[::-1][:k]
    return float(labels[top_idx].mean())


def _pool(all_labels, all_preds, month_results, all_years, all_months):
    labels    = np.array(all_labels)
    preds_arr = np.array(all_preds)
    pooled_auc = roc_auc_score(labels, preds_arr) if len(np.unique(labels)) > 1 else 0.5
    p10        = precision_at_top_k(labels, preds_arr)
    preds_df   = pd.DataFrame({
        "year":  all_years,
        "month": all_months,
        "prob":  preds_arr,
        "pred":  (preds_arr > 0.5).astype(int),
        "label": labels,
    })
    return pooled_auc, p10, month_results, preds_df


# ---------------------------------------------------------------------------
# Model runners — receive precomputed folds, do model training only
# ---------------------------------------------------------------------------

def run_random(folds: List[Dict], seed: int):
    set_seed(seed)
    all_preds, all_labels, month_results, all_years, all_months = [], [], [], [], []
    for fold in folds:
        y_test = fold["y_test"]
        preds  = np.random.rand(len(y_test))
        all_preds.extend(preds)
        all_labels.extend(y_test)
        all_years.extend([fold["year"]] * len(y_test))
        all_months.extend([fold["month"]] * len(y_test))
        auc = roc_auc_score(y_test, preds) if len(np.unique(y_test)) > 1 else 0.5
        month_results.append({"year": fold["year"], "month": fold["month"], "auc": auc, "n": len(y_test)})
    return _pool(all_labels, all_preds, month_results, all_years, all_months)


def run_xgboost_id(folds: List[Dict], seed: int):
    import xgboost as xgb
    set_seed(seed)
    all_preds, all_labels, month_results, all_years, all_months = [], [], [], [], []
    for fold in folds:
        clf = xgb.XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1, subsample=0.8,
            use_label_encoder=False, eval_metric="logloss",
            enable_categorical=True, random_state=seed, verbosity=0, nthread=4,
        )
        clf.fit(fold["X_train_id"], fold["y_train"])
        preds = clf.predict_proba(fold["X_test_id"])[:, 1]
        all_preds.extend(preds)
        all_labels.extend(fold["y_test"])
        all_years.extend([fold["year"]] * len(preds))
        all_months.extend([fold["month"]] * len(preds))
        auc = roc_auc_score(fold["y_test"], preds) if len(np.unique(fold["y_test"])) > 1 else 0.5
        month_results.append({"year": fold["year"], "month": fold["month"], "auc": auc, "n": len(preds)})
    return _pool(all_labels, all_preds, month_results, all_years, all_months)


def run_xgboost_feat(folds: List[Dict], seed: int):
    import xgboost as xgb
    set_seed(seed)
    all_preds, all_labels, month_results, all_years, all_months = [], [], [], [], []
    for fold in folds:
        clf = xgb.XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1, subsample=0.8,
            use_label_encoder=False, eval_metric="logloss",
            enable_categorical=True, random_state=seed, verbosity=0, nthread=4,
        )
        clf.fit(fold["X_train_feat_xgb"], fold["y_train"])
        preds = clf.predict_proba(fold["X_test_feat_xgb"])[:, 1]
        all_preds.extend(preds)
        all_labels.extend(fold["y_test"])
        all_years.extend([fold["year"]] * len(preds))
        all_months.extend([fold["month"]] * len(preds))
        auc = roc_auc_score(fold["y_test"], preds) if len(np.unique(fold["y_test"])) > 1 else 0.5
        month_results.append({"year": fold["year"], "month": fold["month"], "auc": auc, "n": len(preds)})
    return _pool(all_labels, all_preds, month_results, all_years, all_months)


def run_logreg_feat(folds: List[Dict], seed: int):
    set_seed(seed)
    all_preds, all_labels, month_results, all_years, all_months = [], [], [], [], []
    for fold in folds:
        y_train, y_test = fold["y_train"], fold["y_test"]
        if len(np.unique(y_train)) < 2:
            preds = np.full(len(y_test), 0.5)
        else:
            clf = LogisticRegression(C=1.0, max_iter=300, random_state=seed, solver="lbfgs")
            clf.fit(fold["X_train_np"], y_train)
            preds = clf.predict_proba(fold["X_test_np"])[:, 1]
        all_preds.extend(preds)
        all_labels.extend(y_test)
        all_years.extend([fold["year"]] * len(preds))
        all_months.extend([fold["month"]] * len(preds))
        auc = roc_auc_score(y_test, preds) if len(np.unique(y_test)) > 1 else 0.5
        month_results.append({"year": fold["year"], "month": fold["month"], "auc": auc, "n": len(preds)})
    return _pool(all_labels, all_preds, month_results, all_years, all_months)


def run_rf_feat(folds: List[Dict], seed: int):
    set_seed(seed)
    all_preds, all_labels, month_results, all_years, all_months = [], [], [], [], []
    for fold in folds:
        y_train, y_test = fold["y_train"], fold["y_test"]
        if len(np.unique(y_train)) < 2:
            preds = np.full(len(y_test), 0.5)
        else:
            clf = RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=seed, n_jobs=4
            )
            clf.fit(fold["X_train_np"], y_train)
            preds = clf.predict_proba(fold["X_test_np"])[:, 1]
        all_preds.extend(preds)
        all_labels.extend(y_test)
        all_years.extend([fold["year"]] * len(preds))
        all_months.extend([fold["month"]] * len(preds))
        auc = roc_auc_score(y_test, preds) if len(np.unique(y_test)) > 1 else 0.5
        month_results.append({"year": fold["year"], "month": fold["month"], "auc": auc, "n": len(preds)})
    return _pool(all_labels, all_preds, month_results, all_years, all_months)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Raw vs Excess Return Label Comparison Study")
    parser.add_argument("--horizon", type=str, default="6M",
                        choices=list(HORIZON_THRESHOLDS.keys()))
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--end-year",   type=int, default=2023)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--data",       type=str,
                        default="data/processed/ml_dataset_v2.csv")
    parser.add_argument("--skip",       nargs="*", default=[],
                        help="Models to skip: random xgb xgb_feat logreg rf")
    parser.add_argument("--threshold-level", type=int, default=None, choices=[0, 1, 2],
                        help="Run only one threshold level (0/1/2). Default: all 3.")
    parser.add_argument("--max-age-years", type=float, default=0.0,
                        help="Rolling window size in years (0 = unbounded expanding window).")
    parser.add_argument("--dead-zone", type=float, default=0.0,
                        help="Exclude training rows where |return| < dead_zone "
                             "(e.g. 0.10 drops trades with < 10%% absolute return). "
                             "Test set is always kept complete. Default: 0 (disabled).")
    args = parser.parse_args()

    years     = list(range(args.start_year, args.end_year + 1))
    years_str = f"{args.start_year}-{args.end_year}"
    skip      = set(args.skip or [])

    # Window suffix for filenames: "W1y", "W2y", etc., or "" for unbounded
    window_tag = f"_W{args.max_age_years:.0f}y" if args.max_age_years > 0 else ""
    # Dead zone suffix for filenames: "DZ10pct", etc., or "" for disabled
    dz_tag = f"_DZ{args.dead_zone * 100:.0f}pct" if args.dead_zone > 0 else ""
    file_tag = window_tag + dz_tag

    log.info("Raw vs Excess Return Label Study  |  horizon=%s  years=%s  seed=%d  max_age_years=%.1f  dead_zone=%.2f",
             args.horizon, years_str, args.seed, args.max_age_years, args.dead_zone)
    log.info("data=%s", args.data)

    df = load_data(args.data)
    log.info("Loaded %d rows", len(df))

    thresholds_all = HORIZON_THRESHOLDS[args.horizon]
    levels_to_run  = [args.threshold_level] if args.threshold_level is not None else [0, 1, 2]

    # Both label types to compare
    label_types = [
        ("raw",    f"Stock_Return_{args.horizon}"),
        ("excess", f"Excess_Return_{args.horizon}"),
    ]

    out_dir = Path("experiments/signal_isolation/results/raw_return")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary_rows = []

    for label_type, label_col in label_types:
        log.info("")
        log.info("=" * 70)
        log.info("LABEL TYPE: %s  (%s)", label_type.upper(), label_col)
        log.info("=" * 70)

        for level in levels_to_run:
            threshold  = thresholds_all[level]
            thresh_pct = threshold * 100

            log.info("")
            log.info("  LEVEL %d: %s > %.0f%%", level, label_col, thresh_pct)
            log.info("  " + "-" * 60)

            valid    = df[label_col].notna()
            pos_rate = (df.loc[valid, label_col] > threshold).mean()
            log.info("  Positive rate: %.1f%%  (n=%d valid)", pos_rate * 100, valid.sum())

            # ------------------------------------------------------------------
            # Precompute folds ONCE — all models share this
            # ------------------------------------------------------------------
            log.info("  Precomputing folds...")
            t_prep = time.time()
            folds = build_folds(df, years, args.horizon, threshold, label_col,
                                max_age_years=args.max_age_years,
                                dead_zone=args.dead_zone)
            log.info("  %d folds ready in %.1fs", len(folds), time.time() - t_prep)

            # ------------------------------------------------------------------
            # Run models
            # ------------------------------------------------------------------
            RUNNERS = [
                ("random",   "Random",        run_random),
                ("xgb",      "XGBoost(ID)",   run_xgboost_id),
                ("xgb_feat", "XGBoost(feat)", run_xgboost_feat),
                ("logreg",   "LogReg(feat)",  run_logreg_feat),
                ("rf",       "RF(feat)",      run_rf_feat),
            ]

            for key, name, runner in RUNNERS:
                if key in skip:
                    continue
                t0 = time.time()
                auc, p10, months, preds_df = runner(folds, args.seed)
                elapsed = time.time() - t0
                log.info("  %-15s  AUC=%.4f  P@10%%=%.4f  (%.1fs)", name, auc, p10, elapsed)

                # Per-month AUC breakdown
                for m in months:
                    log.info("    %d-%02d  AUC=%.4f  n=%d",
                             m["year"], m["month"], m["auc"], m["n"])

                # Save per-row predictions
                preds_df.insert(0, "label_type", label_type)
                preds_df.insert(1, "horizon", args.horizon)
                preds_df.insert(2, "threshold_level", level)
                preds_df.insert(3, "threshold", threshold)
                preds_df.insert(4, "model", name)
                preds_file = (out_dir /
                    f"preds_{years_str}_{args.horizon}_{label_type}_L{level}_{key}{file_tag}_seed{args.seed}.csv")
                preds_df.to_csv(preds_file, index=False)

                # Save per-month AUC
                months_df = pd.DataFrame(months)
                months_df.insert(0, "label_type", label_type)
                months_df.insert(1, "horizon", args.horizon)
                months_df.insert(2, "threshold_level", level)
                months_df.insert(3, "model", name)
                months_file = (out_dir /
                    f"monthly_{years_str}_{args.horizon}_{label_type}_L{level}_{key}{file_tag}_seed{args.seed}.csv")
                months_df.to_csv(months_file, index=False)

                all_summary_rows.append({
                    "label_type":         label_type,
                    "horizon":            args.horizon,
                    "threshold_level":    level,
                    "threshold":          threshold,
                    "model":              name,
                    "pooled_auc":         auc,
                    "precision_at_top10": p10,
                })

    # ------------------------------------------------------------------
    # Combined summary
    # ------------------------------------------------------------------
    summary_df   = pd.DataFrame(all_summary_rows)
    summary_file = out_dir / f"comparison_{years_str}_{args.horizon}_all{file_tag}_seed{args.seed}.csv"
    summary_df.to_csv(summary_file, index=False)

    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY  →  %s", summary_file)
    log.info("=" * 70)
    log.info("%-10s  %-15s  %5s  %8s  %8s  %8s",
             "LabelType", "Model", "Level", "Thresh%", "AUC", "P@10%")
    log.info("-" * 65)
    for row in all_summary_rows:
        log.info("%-10s  %-15s  L%d     %6.0f%%  %8.4f  %8.4f",
                 row["label_type"], row["model"], row["threshold_level"],
                 row["threshold"] * 100,
                 row["pooled_auc"],
                 row["precision_at_top10"])


if __name__ == "__main__":
    main()
