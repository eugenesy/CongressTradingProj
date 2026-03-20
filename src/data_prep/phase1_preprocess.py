"""
phase1_preprocess.py
====================
Data Preparation Module. Handles raw transformation to ML dataset.

Refactored/Audited: 2026-03-20
"""

#!/usr/bin/env python3
"""
Phase 1: Distribution-Aware Preprocessing
==========================================
Applies log1p(Filing_Gap), ordinal Trade_Size_USD, and winsorized
max_excess_6m to the continuous dataset, then re-runs XGBoost comparison
on 2023 dev set to measure improvement over the raw baseline.

Run:
    python experiments/signal_isolation/phase1_preprocess.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# ── Trade size ordinal map ─────────────────────────────────────────────────
TRADE_SIZE_MAP = {
    "$1,001 - $15,000":       1,
    "$15,001 - $50,000":      2,
    "$50,001 - $100,000":     3,
    "$100,001 - $250,000":    4,
    "$250,001 - $500,000":    5,
    "$500,001 - $1,000,000":  6,
    "$1,000,001 - $5,000,000":7,
    "$5,000,001 - $25,000,000":8,
}

def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Filed"] = pd.to_datetime(df["Filed"])
    df["year"]  = df["Filed"].dt.year
    df["month"] = df["Filed"].dt.month

    # ── 1. Log-scale Filing_Gap ────────────────────────────────────────────
    df["Filing_Gap"] = pd.to_numeric(df["Filing_Gap"], errors="coerce").fillna(30)
    df["log_filing_gap"] = np.log1p(df["Filing_Gap"].clip(0))

    # ── 2. Ordinal Trade_Size_USD ──────────────────────────────────────────
    df["trade_size_ord"] = df["Trade_Size_USD"].map(TRADE_SIZE_MAP).fillna(2).astype(int)
    # Power-law correction: log1p of ordinal buckets
    df["log_trade_size"] = np.log1p(df["trade_size_ord"])

    # ── 3. Winsorize max_excess_6m at 1st / 99th percentile ───────────────
    lo = df["max_excess_6m"].quantile(0.01)
    hi = df["max_excess_6m"].quantile(0.99)
    df["max_excess_6m_w"] = df["max_excess_6m"].clip(lo, hi)

    # ── Labels (continuous targets, winsorized) ────────────────────────────
    median_thr = df["max_excess_6m_w"].median()
    q3_thr     = df["max_excess_6m_w"].quantile(0.75)
    df["Label_Median"] = (df["max_excess_6m_w"] > 0.088).astype(int)
    df["Label_Q3"]     = (df["max_excess_6m_w"] > 0.178).astype(int)
    print(f"After winsorize — Median thr: {median_thr:.3f}, Q3 thr: {q3_thr:.3f}")
    print(f"Label_Median pos rate: {df['Label_Median'].mean():.2%}")
    print(f"Label_Q3     pos rate: {df['Label_Q3'].mean():.2%}")
    return df


def compute_p10(y_true, y_prob, years, months, k=0.10):
    preds = pd.DataFrame({"label": y_true, "prob": y_prob, "year": years, "month": months})
    precs = []
    for _, grp in preds.groupby(["year", "month"]):
        if len(grp) < 5:
            continue
        n_top = max(1, int(len(grp) * k))
        prec  = grp.nlargest(n_top, "prob")["label"].mean()
        precs.append(prec)
    return float(np.mean(precs)) if precs else 0.0


def run_xgb(df, feature_sets, target_col):
    train = df[df["Filed"] < "2023-01-01"]
    test  = df[(df["Filed"] >= "2023-01-01") & (df["Filed"] < "2024-01-01")]

    results = {}
    for name, feats in feature_sets.items():
        # Encode categoricals inline
        X_tr = train[feats].copy()
        X_te = test[feats].copy()
        for c in X_tr.select_dtypes("object").columns:
            le = LabelEncoder()
            X_tr[c] = le.fit_transform(X_tr[c].astype(str))
            classes = set(le.classes_)
            X_te[c] = X_te[c].astype(str).apply(lambda x: x if x in classes else le.classes_[0])
            X_te[c] = le.transform(X_te[c])

        y_tr = train[target_col]
        y_te = test[target_col]

        neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=neg/max(1, pos),
            eval_metric="auc", random_state=42,
            n_jobs=8, tree_method="hist",
        )
        clf.fit(X_tr, y_tr, verbose=False)
        prob = clf.predict_proba(X_te)[:, 1]

        auc = roc_auc_score(y_te, prob)
        p10 = compute_p10(y_te, prob, test["year"], test["month"])
        results[name] = {"AUC": round(auc, 4), "P@10%": round(p10, 4)}
    return results


def main():
    csv = "data/processed/ml_dataset_continuous.csv"
    df  = load_and_preprocess(csv)

    # ── Encode IDs for node-index features ────────────────────────────────
    df["BioGuideID_enc"] = LabelEncoder().fit_transform(df["BioGuideID"].astype(str))
    df["Ticker_enc"]     = LabelEncoder().fit_transform(df["Ticker"].astype(str))
    for c in ["Chamber", "Party", "State", "Transaction", "Industry", "Sector"]:
        df[f"{c}_enc"] = LabelEncoder().fit_transform(df[c].astype(str))

    # ── Feature sets to compare ────────────────────────────────────────────
    baseline_feats = [
        "BioGuideID_enc", "Ticker_enc",
        "Chamber_enc", "Party_enc", "State_enc",
        "Transaction_enc", "Industry_enc", "Sector_enc",
        "Filing_Gap", "Late_Filing", "In_SP100",
    ]
    phase1_feats = [
        "BioGuideID_enc", "Ticker_enc",
        "Chamber_enc", "Party_enc", "State_enc",
        "Transaction_enc", "Industry_enc", "Sector_enc",
        "log_filing_gap",          # NEW: log-scaled gap
        "log_trade_size",          # NEW: ordinal log trade size
        "Late_Filing", "In_SP100",
    ]

    print("\n" + "="*60)
    print("Target: Label_Q3")
    print("="*60)
    res_q3 = run_xgb(df, {"Baseline": baseline_feats, "Phase1 (log feats)": phase1_feats}, "Label_Q3")
    for name, m in res_q3.items():
        print(f"  {name:25s} AUC={m['AUC']:.4f}  P@10%={m['P@10%']:.4f}")

    print("\n" + "="*60)
    print("Target: Label_Median")
    print("="*60)
    res_med = run_xgb(df, {"Baseline": baseline_feats, "Phase1 (log feats)": phase1_feats}, "Label_Median")
    for name, m in res_med.items():
        print(f"  {name:25s} AUC={m['AUC']:.4f}  P@10%={m['P@10%']:.4f}")

    # Random baseline
    print("\n  Random baseline    AUC=0.5000  P@10%(Q3)~0.25  P@10%(Med)~0.50")


if __name__ == "__main__":
    main()
