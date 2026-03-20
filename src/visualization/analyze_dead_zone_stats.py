"""
Dead Zone Statistical Analysis
--------------------------------
Compute per-year AUROCs for DZ=0 and DZ=10%, run paired tests across years,
and bootstrap pooled AUROC differences to produce CIs.

Outputs (saved to experiments/signal_isolation/results/visualizations/):
 - dead_zone_per_year_auroc_table.csv
 - dead_zone_stats_summary.csv

Run:
  conda run -n chocolate python experiments/signal_isolation/analyze_dead_zone_stats.py
"""

import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon, ttest_rel


# --------------------
# Config
# --------------------
RESULTS_DIR = Path("experiments/signal_isolation/results/raw_return")
OUT_DIR = Path("experiments/signal_isolation/results/visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["random", "xgb", "xgb_feat", "logreg", "rf", "gnn_sage"]
DEAD_ZONES = [0.0, 0.10]
YEARS = list(range(2019, 2025))
SEED = 42
N_BOOT = 1000


def dz_tag(dz: float) -> str:
    return "" if dz <= 0.0 else f"_DZ{int(round(dz*100))}pct"


def flat_path(model: str, dz: float) -> Path:
    years_str = "2019-2024"
    return RESULTS_DIR / f"preds_{years_str}_6M_excess_L0_{model}{dz_tag(dz)}_seed{SEED}.csv"


def gnn_path(dz: float) -> Path:
    years_str = "2019_2020_2021_2022_2023_2024"
    return RESULTS_DIR / f"preds_{years_str}_6M_excess_L0_gnn_sage{dz_tag(dz)}_seed{SEED}.csv"


def load_preds(model: str, dz: float) -> pd.DataFrame:
    # GNN predictions: prefer combined multi-year file, fall back to per-year files
    if model == "gnn_sage":
        p = gnn_path(dz)
        if p.exists():
            df = pd.read_csv(p)
            df.columns = [c.lower() for c in df.columns]
            # try to sort by filed date to create a stable row order for paired bootstrap
            if "filed" in df.columns:
                try:
                    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
                    df = df.sort_values("filed").reset_index(drop=True)
                except Exception:
                    pass
            return df

        # fallback: stitch per-year GNN files (older baseline format)
        parts: List[pd.DataFrame] = []
        for y in YEARS:
            py = RESULTS_DIR / f"preds_{y}_6M_excess_L0_gnn_sage{dz_tag(dz)}_seed{SEED}.csv"
            if py.exists():
                tmp = pd.read_csv(py)
                tmp.columns = [c.lower() for c in tmp.columns]
                parts.append(tmp)
        if parts:
            df = pd.concat(parts, ignore_index=True)
            if "filed" in df.columns:
                try:
                    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
                    df = df.sort_values("filed").reset_index(drop=True)
                except Exception:
                    pass
            return df

        # not found
        raise FileNotFoundError(p)

    # Flat model predictions (single combined CSV)
    p = flat_path(model, dz)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    # normalize columns lower-case
    df.columns = [c.lower() for c in df.columns]
    return df


def pooled_auroc_from_df(df: pd.DataFrame) -> float:
    # find label and score columns
    label_col = next((c for c in df.columns if c in ("label", "y_true", "true_label")), "label")
    score_col = next((c for c in df.columns if c in ("prob", "score", "prob_1", "pred_proba", "prediction")), None)
    if score_col is None:
        raise RuntimeError("No score column")
    y = df[label_col].values
    s = df[score_col].values
    return float(roc_auc_score(y, s))


def per_year_auroc(df: pd.DataFrame) -> Dict[int, float]:
    out = {}
    # prefer integer year column
    if "year" in df.columns and pd.api.types.is_integer_dtype(df["year"]):
        for y in YEARS:
            sub = df[df["year"] == y]
            if sub.empty:
                continue
            try:
                out[y] = pooled_auroc_from_df(sub)
            except Exception:
                out[y] = float("nan")
        return out
    # fallback: look for datetime filed column
    date_col = next((c for c in df.columns if c in ("filed", "date", "trade_date")), None)
    if date_col:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        for y in YEARS:
            sub = df[df[date_col].dt.year == y]
            if sub.empty:
                continue
            try:
                out[y] = pooled_auroc_from_df(sub)
            except Exception:
                out[y] = float("nan")
    return out


def paired_year_tests(baseline: Dict[int, float], dz: Dict[int, float]):
    # get years present in both and not NaN
    years = sorted([y for y in baseline.keys() if y in dz and not math.isnan(baseline[y]) and not math.isnan(dz[y])])
    a = np.array([baseline[y] for y in years])
    b = np.array([dz[y] for y in years])
    res = {}
    if len(years) >= 2:
        # Wilcoxon (paired)
        try:
            w_stat, w_p = wilcoxon(a, b)
        except Exception:
            w_stat, w_p = float('nan'), float('nan')
        # paired t-test
        try:
            t_stat, t_p = ttest_rel(a, b)
        except Exception:
            t_stat, t_p = float('nan'), float('nan')
        res.update({"years_used": years, "n": len(years), "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p), "ttest_stat": float(t_stat), "ttest_p": float(t_p)})
    else:
        res.update({"years_used": years, "n": len(years), "wilcoxon_stat": float('nan'), "wilcoxon_p": float('nan'), "ttest_stat": float('nan'), "ttest_p": float('nan')})
    return res


def bootstrap_pooled_delta(df0: pd.DataFrame, df1: pd.DataFrame, n_boot: int = 2000, seed: int = 42):
    # assume rows align by position (test set same for both DZ files)
    if len(df0) != len(df1):
        raise RuntimeError("Baseline and DZ files have different row counts; cannot do paired bootstrap")
    label_col = next((c for c in df0.columns if c in ("label", "y_true", "true_label")), "label")
    score_col = next((c for c in df0.columns if c in ("prob", "score", "prob_1", "pred_proba", "prediction")), None)
    if score_col is None:
        raise RuntimeError("No score column")
    y = df0[label_col].values
    s0 = df0[score_col].values
    s1 = df1[score_col].values
    rng = random.Random(seed)
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        try:
            auc0 = roc_auc_score(y[idx], s0[idx])
            auc1 = roc_auc_score(y[idx], s1[idx])
            diffs.append(auc1 - auc0)
        except Exception:
            # skip failed resamples
            continue
    if not diffs:
        return {"n_boot": 0, "median": float('nan'), "ci_lower": float('nan'), "ci_upper": float('nan')}
    arr = np.array(diffs)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {"n_boot": len(arr), "median": float(np.median(arr)), "ci_lower": float(lo), "ci_upper": float(hi)}


def main():
    summary_rows = []
    per_year_rows = []

    for model in MODELS:
        print(f"Processing model: {model}")
        df0 = load_preds(model, 0.0)
        df1 = load_preds(model, 0.10)

        # quick sanity: same number of rows? if GNN produced combined file, both should match
        same_len = len(df0) == len(df1)
        print(f"  rows: baseline={len(df0)} dz10={len(df1)} same_len={same_len}")

        pooled0 = pooled_auroc_from_df(df0)
        pooled1 = pooled_auroc_from_df(df1)

        # per-year
        py0 = per_year_auroc(df0)
        py1 = per_year_auroc(df1)
        for y in sorted(set(list(py0.keys()) + list(py1.keys()))):
            per_year_rows.append({"model": model, "year": y, "dz": 0.0, "auroc": py0.get(y, float('nan'))})
            per_year_rows.append({"model": model, "year": y, "dz": 0.10, "auroc": py1.get(y, float('nan'))})

        paired = paired_year_tests(py0, py1)

        # bootstrap pooled delta (paired bootstrap requires same row order)
        try:
            boot = bootstrap_pooled_delta(df0, df1, n_boot=N_BOOT, seed=SEED)
        except Exception as exc:
            boot = {"n_boot": 0, "median": float('nan'), "ci_lower": float('nan'), "ci_upper": float('nan'), "error": str(exc)}

        row = {
            "model": model,
            "pooled_baseline": pooled0,
            "pooled_dz10": pooled1,
            "delta": pooled1 - pooled0,
            "boot_median": boot.get("median"),
            "boot_ci_lower": boot.get("ci_lower"),
            "boot_ci_upper": boot.get("ci_upper"),
            "boot_n": boot.get("n_boot"),
            "wilcoxon_stat": paired.get("wilcoxon_stat"),
            "wilcoxon_p": paired.get("wilcoxon_p"),
            "ttest_stat": paired.get("ttest_stat"),
            "ttest_p": paired.get("ttest_p"),
            "years_used": paired.get("years_used"),
            "n_years": paired.get("n"),
        }
        summary_rows.append(row)

    # Save per-year table
    py_df = pd.DataFrame(per_year_rows)
    py_out = OUT_DIR / "dead_zone_per_year_auroc_table.csv"
    py_df.to_csv(py_out, index=False)
    print("Saved per-year AUROC table to", py_out)

    # Save summary
    sum_df = pd.DataFrame(summary_rows)
    sum_out = OUT_DIR / "dead_zone_stats_summary.csv"
    sum_df.to_csv(sum_out, index=False)
    print("Saved stats summary to", sum_out)

    # Print concise results
    print("\nSummary:\n")
    print(sum_df.to_string(index=False))


if __name__ == "__main__":
    main()
