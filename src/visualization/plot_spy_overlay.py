"""
plot_spy_overlay.py
===================
Visualization Utilities. Generates presentations plots and distributions.

Refactored/Audited: 2026-03-20
"""

#!/usr/bin/env python3
"""
SPY Overlay: Raw vs Excess AUROC Timeline with SPY Background
=============================================================
Produces a single-panel figure showing per-year AUROC for raw and excess
labels (for all available models) with the SPY 6M forward return plotted
as a shaded background.

The visual argument: excess AUROC should track SPY regime (high SPY return
→ low excess positive rate → harder problem or noisy AUROC); raw AUROC should
be decoupled from SPY regime.

Inputs:
  - experiments/signal_isolation/results/raw_return/  (prediction CSVs)
  - data/processed/ml_dataset_v2.csv                 (SPY_Return_6M column)

Outputs:
  - experiments/signal_isolation/results/visualizations/spy_overlay_auroc.png

Usage:
  python experiments/signal_isolation/plot_spy_overlay.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.getcwd())

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("experiments/signal_isolation/results/raw_return")
VIS_DIR     = Path("experiments/signal_isolation/results/visualizations")
ML_DATASET  = Path("data/processed/ml_dataset_v2.csv")

YEARS   = [2019, 2020, 2021, 2022, 2023, 2024]
SEED    = 42
HORIZON = "6M"
LEVEL   = 0

# Models to include (slug → display name)
FLAT_MODELS = {
    "random":   "Random",
    "xgb_feat": "XGBoost(feat)",
}
GNN_SLUG = "gnn_sage"
GNN_NAME = "GNN-SAGE"

MODEL_ORDER = ["Random", "XGBoost(feat)", "GNN-SAGE"]

# Visual style
COLORS: Dict[str, str] = {
    "raw":    "#2166ac",   # blue
    "excess": "#d6604d",   # red-orange
}
LINESTYLES: Dict[str, str] = {
    "Random":        ":",
    "XGBoost(feat)": "--",
    "GNN-SAGE":      "-",
}
MARKERS: Dict[str, str] = {
    "Random":        "o",
    "XGBoost(feat)": "s",
    "GNN-SAGE":      "^",
}
LINEWIDTHS: Dict[str, float] = {
    "Random":        1.5,
    "XGBoost(feat)": 1.8,
    "GNN-SAGE":      2.5,
}

SPY_POS_COLOR = "#b8d9b0"   # light green for positive SPY 6M return years
SPY_NEG_COLOR = "#f5b8b0"   # light red for negative SPY 6M return years


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_preds(path: Path) -> Optional[pd.DataFrame]:
    """Load predictions CSV; return None if missing or malformed."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if not {"label", "prob"}.issubset(df.columns):
            return None
        return df
    except Exception:
        return None


def _auroc(df: pd.DataFrame) -> float:
    labels = df["label"].values
    probs  = df["prob"].values
    if len(np.unique(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, probs)


def load_results() -> pd.DataFrame:
    """Load all available prediction CSVs and compute per-year AUROC."""
    rows: List[dict] = []

    for year in YEARS:
        yr_str = f"{year}-{year}"
        for label_type in ("raw", "excess"):
            # Flat models
            for slug, name in FLAT_MODELS.items():
                candidates = [
                    RESULTS_DIR / f"preds_{yr_str}_{HORIZON}_{label_type}_L{LEVEL}_{slug}_seed{SEED}.csv",
                    RESULTS_DIR / f"preds_{year}_{HORIZON}_{label_type}_L{LEVEL}_{slug}_seed{SEED}.csv",
                ]
                for path in candidates:
                    df = _load_preds(path)
                    if df is not None:
                        rows.append({
                            "year": year, "model": name,
                            "label_type": label_type, "auroc": _auroc(df),
                        })
                        break

            # GNN-SAGE
            candidates = [
                RESULTS_DIR / f"preds_{yr_str}_{HORIZON}_{label_type}_L{LEVEL}_{GNN_SLUG}_seed{SEED}.csv",
                RESULTS_DIR / f"preds_{year}_{HORIZON}_{label_type}_L{LEVEL}_{GNN_SLUG}_seed{SEED}.csv",
            ]
            for path in candidates:
                df = _load_preds(path)
                if df is not None:
                    rows.append({
                        "year": year, "model": GNN_NAME,
                        "label_type": label_type, "auroc": _auroc(df),
                    })
                    break

    return pd.DataFrame(rows)


def load_spy_return_by_year() -> Dict[int, float]:
    """
    Compute the median SPY_Return_6M per test year from ml_dataset_v2.csv.
    Each row's Filed date determines the year bucket; SPY_Return_6M is the
    forward 6M SPY return from that filing date.
    """
    df = pd.read_csv(ML_DATASET, usecols=["Filed", "SPY_Return_6M"])
    df["Filed"] = pd.to_datetime(df["Filed"])
    df["year"]  = df["Filed"].dt.year
    # Use median to avoid outlier distortion; mean also fine
    spy_by_year = (
        df[df["year"].isin(YEARS)]
        .groupby("year")["SPY_Return_6M"]
        .median()
        .to_dict()
    )
    return spy_by_year


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_spy_overlay(results: pd.DataFrame, spy_by_year: Dict[int, float],
                     out_path: Path) -> None:
    """
    Single-panel figure:
      - Background: green/red shading per year based on sign of SPY 6M return
      - Secondary y-axis (right): SPY 6M forward return as a grey line
      - Primary y-axis (left): AUROC lines per model × label_type
    """
    years_present = sorted(results["year"].unique())
    models_present = [m for m in MODEL_ORDER if m in results["model"].unique()]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- Background shading: one bar per year ---
    bar_width = 0.8
    for yr in years_present:
        ret = spy_by_year.get(yr, 0.0)
        color = SPY_POS_COLOR if ret >= 0 else SPY_NEG_COLOR
        ax1.axvspan(yr - bar_width / 2, yr + bar_width / 2,
                    facecolor=color, alpha=0.35, zorder=0)

    # --- SPY return on secondary y-axis ---
    ax2 = ax1.twinx()
    spy_vals = [spy_by_year.get(yr, float("nan")) for yr in years_present]
    ax2.plot(years_present, [v * 100 for v in spy_vals],
             color="gray", linewidth=1.5, linestyle="-.", marker="D",
             markersize=5, alpha=0.6, zorder=1, label="SPY 6M fwd return (median)")
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.4)
    ax2.set_ylabel("SPY 6M Forward Return (%)", color="gray", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="gray")
    # Keep SPY axis range symmetric around 0 for visual clarity
    spy_abs_max = max(abs(v * 100) for v in spy_vals if not np.isnan(v))
    ax2.set_ylim(-spy_abs_max * 1.6, spy_abs_max * 1.6)

    # --- AUROC lines ---
    legend_handles = []
    for model in models_present:
        for label_type in ("raw", "excess"):
            sub = (results[(results["model"] == model) &
                           (results["label_type"] == label_type)]
                   .set_index("year")
                   .reindex(years_present))
            aucs = sub["auroc"].values
            if np.all(np.isnan(aucs)):
                continue
            line, = ax1.plot(
                years_present, aucs,
                color=COLORS[label_type],
                linestyle=LINESTYLES[model],
                linewidth=LINEWIDTHS[model],
                marker=MARKERS[model],
                markersize=7,
                alpha=0.92,
                zorder=3,
                label=f"{model} ({label_type})",
            )
            # Annotate AUROC values
            for yr, auc in zip(years_present, aucs):
                if not np.isnan(auc):
                    ax1.annotate(
                        f"{auc:.3f}",
                        (yr, auc),
                        textcoords="offset points",
                        xytext=(0, 7),
                        fontsize=6.5,
                        ha="center",
                        color=COLORS[label_type],
                        zorder=4,
                    )
            legend_handles.append(line)

    ax1.axhline(0.5, color="black", linewidth=1.0, linestyle=":", alpha=0.5,
                zorder=2, label="Random baseline (0.5)")
    ax1.set_ylabel("Pooled AUROC", fontsize=11)
    ax1.set_xlabel("Test Year", fontsize=11)
    ax1.set_ylim(0.40, 0.80)
    ax1.set_xticks(years_present)
    ax1.grid(axis="y", alpha=0.25, zorder=0)

    # --- Legend ---
    # AUROC lines
    ax1.legend(handles=legend_handles + [
        plt.Line2D([0], [0], color="black", linewidth=1, linestyle=":",
                   label="Random baseline (0.5)")
    ], loc="upper left", fontsize=8, framealpha=0.85)

    # SPY legend (secondary axis)
    spy_handle = ax2.get_lines()[0]
    pos_patch  = mpatches.Patch(facecolor=SPY_POS_COLOR, alpha=0.5,
                                 label="SPY 6M return > 0 (bull)")
    neg_patch  = mpatches.Patch(facecolor=SPY_NEG_COLOR, alpha=0.5,
                                 label="SPY 6M return ≤ 0 (bear)")
    ax2.legend(handles=[spy_handle, pos_patch, neg_patch],
               loc="upper right", fontsize=8, framealpha=0.85)

    plt.title(
        "Raw vs Excess Return AUROC by Year — SPY Regime Overlay\n"
        "6M horizon · L0 threshold · seed 42\n"
        "Blue = raw label  ·  Red = excess label  ·  Shading = SPY 6M forward return",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading experiment results ...")
    results = load_results()
    if results.empty:
        print("ERROR: No prediction CSVs found. Have the experiments finished?")
        sys.exit(1)

    n_models   = results["model"].nunique()
    n_years    = results["year"].nunique()
    n_rows     = len(results)
    print(f"  Found {n_rows} rows ({n_models} models × {n_years} years × 2 label types)")

    # Report coverage
    pivot = results.pivot_table(
        index="year", columns=["model", "label_type"], values="auroc",
        aggfunc="first",
    )
    print("\nAUROC coverage (NaN = missing):")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nLoading SPY 6M forward return by year ...")
    spy_by_year = load_spy_return_by_year()
    for yr, ret in sorted(spy_by_year.items()):
        print(f"  {yr}: SPY 6M fwd return = {ret * 100:+.2f}%")

    out_path = VIS_DIR / "spy_overlay_auroc.png"
    print(f"\nGenerating SPY overlay figure -> {out_path} ...")
    plot_spy_overlay(results, spy_by_year, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
