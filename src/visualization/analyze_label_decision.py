"""
analyze_label_decision.py
=========================
Visualization Utilities. Generates presentations plots and distributions.

Refactored/Audited: 2026-03-20
"""

#!/usr/bin/env python3
"""
Label Decision Analysis
=======================
Aggregates all raw vs excess experiment results across years (2019-2024) and
produces a definitive answer: which label type should we use?

Inputs (from results/raw_return/):
  - preds_{year}-{year}_6M_{raw|excess}_L0_{model}_seed42.csv  (flat models)
  - preds_{year}_6M_{raw|excess}_L0_gnn_sage_seed42.csv        (GNN-SAGE, from run_gnn_study.py)

Outputs (in results/visualizations/):
  - label_decision_auroc_by_year.png   — AUROC per year per model, raw vs excess
  - label_decision_summary.csv         — full table

Usage:
  python experiments/signal_isolation/analyze_label_decision.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.getcwd())

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR  = Path("experiments/signal_isolation/results/raw_return")
VIS_DIR      = Path("experiments/signal_isolation/results/visualizations")
YEARS        = [2019, 2020, 2021, 2022, 2023, 2024]
SEED         = 42
HORIZON      = "6M"
LEVEL        = 0

# Models to include in the analysis (slug -> display name)
FLAT_MODELS = {
    "random":   "Random",
    "xgb_feat": "XGBoost(feat)",
}
GNN_SLUG    = "gnn_sage"
GNN_NAME    = "GNN-SAGE"

# All models for display order
MODEL_ORDER = ["Random", "XGBoost(feat)", "GNN-SAGE"]

COLORS = {
    "raw":    "#2166ac",   # blue
    "excess": "#d6604d",   # red-orange
}
MARKERS = {
    "Random":        "o",
    "XGBoost(feat)": "s",
    "GNN-SAGE":      "^",
}


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def _load_preds(path: Path) -> pd.DataFrame:
    """Load a predictions CSV; ensure required columns exist."""
    df = pd.read_csv(path)
    required = {"label", "prob"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns {required - set(df.columns)} in {path}")
    return df


def load_flat_model_results() -> List[Dict]:
    """
    Load flat-model prediction CSVs for all years and return a list of dicts:
      {year, model, label_type, auroc, n, positive_rate}
    """
    rows = []
    for year in YEARS:
        years_str = f"{year}-{year}"
        for label_type in ("raw", "excess"):
            for slug, name in FLAT_MODELS.items():
                fname = (RESULTS_DIR /
                    f"preds_{years_str}_{HORIZON}_{label_type}_L{LEVEL}_{slug}_seed{SEED}.csv")
                if not fname.exists():
                    # Try alternate naming (year only, no range)
                    fname2 = (RESULTS_DIR /
                        f"preds_{year}_{HORIZON}_{label_type}_L{LEVEL}_{slug}_seed{SEED}.csv")
                    if not fname2.exists():
                        continue
                    fname = fname2
                try:
                    df = _load_preds(fname)
                    labels = df["label"].values
                    probs  = df["prob"].values
                    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
                    rows.append({
                        "year": year, "model": name, "label_type": label_type,
                        "auroc": auc, "n": len(df),
                        "positive_rate": labels.mean(),
                    })
                except Exception as e:
                    print(f"  [WARN] Could not load {fname}: {e}")
    return rows


def load_gnn_results() -> List[Dict]:
    """
    Load GNN-SAGE prediction CSVs saved by run_gnn_study.py.
    Supports both naming conventions:
      preds_{year}-{year}_6M_{label_type}_L0_gnn_sage_seed42.csv  (raw_return dir)
      preds_{year}_6M_{label_type}_L0_gnn_sage_seed42.csv         (alternate)
    """
    rows = []
    for year in YEARS:
        for label_type in ("raw", "excess"):
            candidates = [
                RESULTS_DIR / f"preds_{year}-{year}_{HORIZON}_{label_type}_L{LEVEL}_{GNN_SLUG}_seed{SEED}.csv",
                RESULTS_DIR / f"preds_{year}_{HORIZON}_{label_type}_L{LEVEL}_{GNN_SLUG}_seed{SEED}.csv",
            ]
            found = None
            for c in candidates:
                if c.exists():
                    found = c
                    break
            if found is None:
                continue
            try:
                df = _load_preds(found)
                labels = df["label"].values
                probs  = df["prob"].values
                auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
                rows.append({
                    "year": year, "model": GNN_NAME, "label_type": label_type,
                    "auroc": auc, "n": len(df),
                    "positive_rate": labels.mean(),
                })
            except Exception as e:
                print(f"  [WARN] Could not load {found}: {e}")
    return rows


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_stability(summary: pd.DataFrame) -> pd.DataFrame:
    """
    For each (model, label_type): mean AUROC, std, min, max across years.
    Also compute raw-vs-excess gap per model.
    """
    stats = (summary
             .groupby(["model", "label_type"])["auroc"]
             .agg(["mean", "std", "min", "max"])
             .reset_index())
    stats.columns = ["model", "label_type", "mean_auroc", "std_auroc", "min_auroc", "max_auroc"]

    # Add random-adjusted gain (mean - 0.5)
    stats["gain_over_random"] = stats["mean_auroc"] - 0.5

    # Raw vs excess gap per model
    pivot = stats.pivot(index="model", columns="label_type", values="mean_auroc")
    if "raw" in pivot.columns and "excess" in pivot.columns:
        pivot["raw_minus_excess"] = pivot["raw"] - pivot["excess"]
        stats = stats.merge(
            pivot[["raw_minus_excess"]].reset_index(),
            on="model", how="left"
        )

    return stats


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_auroc_by_year(summary: pd.DataFrame, out_path: Path) -> None:
    """
    3-panel figure (one per model): AUROC per year, colored by label type.
    Also includes a bottom panel showing positive rate per year per label.
    """
    models = [m for m in MODEL_ORDER if m in summary["model"].unique()]
    n_models = len(models)

    fig, axes = plt.subplots(
        n_models + 1, 1,
        figsize=(10, 3.5 * (n_models + 1)),
        sharex=True,
    )
    if n_models + 1 == 1:
        axes = [axes]

    years_present = sorted(summary["year"].unique())

    for i, model in enumerate(models):
        ax = axes[i]
        for label_type in ("raw", "excess"):
            sub = (summary[(summary["model"] == model) &
                           (summary["label_type"] == label_type)]
                   .set_index("year")
                   .reindex(years_present))
            aucs = sub["auroc"].values
            ax.plot(years_present, aucs,
                    color=COLORS[label_type],
                    marker=MARKERS[model],
                    linewidth=2, markersize=7,
                    label=f"{label_type}  (mean={np.nanmean(aucs):.3f})")
            # Annotate each point
            for yr, auc in zip(years_present, aucs):
                if not np.isnan(auc):
                    ax.annotate(f"{auc:.3f}", (yr, auc),
                                textcoords="offset points", xytext=(0, 6),
                                fontsize=7, ha="center",
                                color=COLORS[label_type])
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Random (0.5)")
        ax.set_ylabel("Pooled AUROC")
        ax.set_title(f"{model}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0.42, 0.78)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(years_present)

    # Bottom panel: positive rate per label type
    ax_pos = axes[n_models]
    pos_df = summary.groupby(["year", "label_type"])["positive_rate"].mean().reset_index()
    for label_type in ("raw", "excess"):
        sub = (pos_df[pos_df["label_type"] == label_type]
               .set_index("year")
               .reindex(years_present))
        ax_pos.plot(years_present, sub["positive_rate"].values,
                    color=COLORS[label_type], marker="D", linewidth=2,
                    markersize=6, label=label_type, linestyle="--")
    ax_pos.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax_pos.set_ylabel("Positive Rate")
    ax_pos.set_title("Positive Rate per Label Type (base rate indicator)", fontsize=10)
    ax_pos.legend(fontsize=9)
    ax_pos.set_ylim(0.25, 0.80)
    ax_pos.set_xlabel("Test Year")
    ax_pos.grid(axis="y", alpha=0.3)
    ax_pos.set_xticks(years_present)

    plt.suptitle(
        "Raw vs Excess Return Labels — AUROC by Year (6M horizon, L0, seed 42)\n"
        "Blue = raw (Stock_Return_6M > 0)   Red = excess (Excess_Return_6M > 0)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure: {out_path}")


def print_summary_table(summary: pd.DataFrame, stability: pd.DataFrame) -> None:
    """Print a clean console summary table."""
    print("\n" + "=" * 80)
    print("AUROC BY YEAR  (6M, L0, seed 42)")
    print("=" * 80)

    # Wide table: rows = year, columns = model x label_type
    pivot = summary.pivot_table(
        index="year", columns=["model", "label_type"], values="auroc"
    )
    # Reorder columns
    col_order = [(m, lt) for m in MODEL_ORDER for lt in ("raw", "excess")
                 if (m, lt) in pivot.columns]
    pivot = pivot[col_order]
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 80)
    print("STABILITY SUMMARY (mean ± std across years)")
    print("=" * 80)
    print(f"{'Model':<18} {'LabelType':<8} {'Mean':>7} {'Std':>6} {'Min':>7} {'Max':>7} {'GainVsRandom':>13} {'RawMinusExcess':>15}")
    print("-" * 80)
    for _, row in stability.sort_values(["model", "label_type"]).iterrows():
        gap = f"{row.get('raw_minus_excess', float('nan')):+.4f}" if "raw_minus_excess" in row.index else "  N/A"
        print(
            f"{row['model']:<18} {row['label_type']:<8} "
            f"{row['mean_auroc']:7.4f} {row['std_auroc']:6.4f} "
            f"{row['min_auroc']:7.4f} {row['max_auroc']:7.4f} "
            f"{row['gain_over_random']:13.4f} {gap:>15}"
        )

    print("\n" + "=" * 80)
    print("POSITIVE RATE BY YEAR (base rate = % of trades that go up)")
    print("=" * 80)
    pos_tbl = summary.groupby(["year", "label_type"])["positive_rate"].mean().unstack()
    print(pos_tbl.to_string(float_format=lambda x: f"{x:.3f}"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading flat-model results ...")
    flat_rows = load_flat_model_results()
    print(f"  Found {len(flat_rows)} flat-model result rows")

    print("Loading GNN-SAGE results ...")
    gnn_rows  = load_gnn_results()
    print(f"  Found {len(gnn_rows)} GNN-SAGE result rows")

    all_rows = flat_rows + gnn_rows
    if not all_rows:
        print("ERROR: No results found. Have the experiments finished?")
        sys.exit(1)

    summary = pd.DataFrame(all_rows)
    summary = summary.sort_values(["model", "year", "label_type"]).reset_index(drop=True)

    # Save summary CSV
    summary_path = VIS_DIR / "label_decision_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Saved summary CSV: {summary_path}")

    # Stability stats
    stability = compute_stability(summary)
    stability_path = VIS_DIR / "label_decision_stability.csv"
    stability.to_csv(stability_path, index=False)
    print(f"  Saved stability CSV: {stability_path}")

    # Console print
    print_summary_table(summary, stability)

    # Figure
    fig_path = VIS_DIR / "label_decision_auroc_by_year.png"
    print("\nGenerating figure ...")
    plot_auroc_by_year(summary, fig_path)

    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    gnn_raw    = stability[(stability["model"] == "GNN-SAGE") & (stability["label_type"] == "raw")]
    gnn_excess = stability[(stability["model"] == "GNN-SAGE") & (stability["label_type"] == "excess")]
    if not gnn_raw.empty and not gnn_excess.empty:
        raw_mean    = gnn_raw["mean_auroc"].iloc[0]
        excess_mean = gnn_excess["mean_auroc"].iloc[0]
        raw_std     = gnn_raw["std_auroc"].iloc[0]
        excess_std  = gnn_excess["std_auroc"].iloc[0]
        print(f"  GNN-SAGE  raw:    mean AUROC = {raw_mean:.4f}  (std={raw_std:.4f})")
        print(f"  GNN-SAGE  excess: mean AUROC = {excess_mean:.4f}  (std={excess_std:.4f})")
        if abs(raw_mean - excess_mean) < 0.010:
            print("\n  --> BOTH labels perform comparably. Prefer EXCESS (alpha, regime-invariant).")
        elif raw_mean > excess_mean:
            print(f"\n  --> RAW is {raw_mean - excess_mean:.4f} AUROC higher on average. Check if driven by base rate.")
        else:
            print(f"\n  --> EXCESS is {excess_mean - raw_mean:.4f} AUROC higher on average. Prefer EXCESS.")
    else:
        print("  GNN-SAGE results not yet fully available — re-run once jobs complete.")


if __name__ == "__main__":
    main()
