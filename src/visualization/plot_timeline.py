"""
plot_timeline.py
----------------
Render per-horizon timeline comparison plots from saved prediction CSVs.

For each horizon, produces one PNG with 4 subplots:
  [0,0] Year  × AUROC         — bar chart, one group of bars per year
  [0,1] Year  × Prec@Top-10%  — bar chart, one group of bars per year
  [1,0] Month × AUROC         — line chart, x=year-month, one line per model
  [1,1] Month × Prec@Top-10%  — line chart, x=year-month, one line per model

Prec@Top-10% = precision among the top-10% highest-confidence predictions
per period.  This is the standard actionable metric for copy-trading / stock
selection tasks: each period you copy only the top-k most confident calls and
measure what fraction were correct.  The random baseline equals the overall
positive label rate.

Academic precedent: Feng et al. "Temporal Relational Ranking for Stock
Prediction" (ACM TOIS 2019, arXiv:1809.09441) and Xu et al. "HIST" (2021,
arXiv:2110.13716) both use precision@top-k as a primary evaluation metric for
GNN-based stock ranking/selection tasks.

Usage
-----
  # Render all available horizons:
  python experiments/signal_isolation/plot_timeline.py

  # Render specific horizons only:
  python experiments/signal_isolation/plot_timeline.py --horizons 6M 12M

  # Override results dir or output dir:
  python experiments/signal_isolation/plot_timeline.py \
      --results-dir experiments/signal_isolation/results \
      --out-dir experiments/signal_isolation/results/visualizations

Output
------
  results/visualizations/timeline_<HORIZON>.png
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_auc_score

# Top-k percentage used for Precision@Top-k (10 % = copy the top decile)
TOP_K_PCT = 0.10

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HORIZONS = ["1M", "2M", "3M", "6M", "12M", "24M"]

# Display order and colours — deterministic regardless of file load order
MODEL_ORDER = [
    "GNN-SAGE",
    "GNN-SAGE (gap)",
    "XGBoost(ID)",
    "XGBoost(feat)",
    "MLP(ID)",
    "MLP(feat)",
    "Random",
]

MODEL_COLORS = {
    "GNN-SAGE":        "#e6194b",   # red
    "GNN-SAGE (gap)":  "#f58231",   # orange
    "XGBoost(ID)":     "#3cb44b",   # green
    "XGBoost(feat)":   "#4363d8",   # blue
    "MLP(ID)":         "#911eb4",   # purple
    "MLP(feat)":       "#42d4f4",   # cyan
    "Random":          "#a9a9a9",   # grey
}

MODEL_MARKERS = {
    "GNN-SAGE":        "o",
    "GNN-SAGE (gap)":  "D",
    "XGBoost(ID)":     "s",
    "XGBoost(feat)":   "^",
    "MLP(ID)":         "v",
    "MLP(feat)":       "P",
    "Random":          "x",
}

FIGSIZE = (20, 12)
BAR_ALPHA = 0.85
LINE_ALPHA = 0.85
LINE_WIDTH = 1.8
MARKER_SIZE = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prec_at_k(y_true, y_prob, k_pct: float = TOP_K_PCT) -> float:
    """Precision among the top k_pct highest-probability predictions.

    Simulates a copy-trading strategy that only copies the top-decile most
    confident "buy" calls each period.  The random baseline equals the overall
    positive label rate.  Returns NaN if the group is too small.
    """
    n = len(y_true)
    k = max(1, int(round(n * k_pct)))
    if k >= n:
        return float("nan")
    top_idx = np.argsort(y_prob.values)[::-1][:k]
    return float(np.array(y_true.values)[top_idx].mean())


def safe_auc(y_true, y_prob):
    """AUROC; returns NaN if only one class present."""
    if len(y_true.unique()) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_prob)


def load_preds_for_horizon(results_dir: Path, horizon: str) -> pd.DataFrame:
    """
    Load and concatenate all preds CSVs for a given horizon.
    Handles both naming conventions:
      preds_gnn_..._<HORIZON>_seed<N>.csv
      preds_study_..._<HORIZON>_seed<N>.csv
      preds_gnn_..._<HORIZON>_seed<N>_gap.csv  (gap variant)
    """
    pattern = str(results_dir / f"preds_*_{horizon}_seed*.csv")
    files = sorted(glob.glob(pattern))
    # also catch gap variant: preds_gnn_..._<HORIZON>_seed<N>_gap.csv
    gap_pattern = str(results_dir / f"preds_gnn_*_{horizon}_seed*_gap.csv")
    files += sorted(glob.glob(gap_pattern))
    # deduplicate while preserving order
    seen = set()
    files = [f for f in files if not (f in seen or seen.add(f))]

    if not files:
        return pd.DataFrame()

    frames = []
    for fpath in files:
        df = pd.read_csv(fpath)
        # Rename gap variant model label
        if "_gap.csv" in fpath:
            df["model"] = df["model"].str.replace(r"^GNN-SAGE$", "GNN-SAGE (gap)", regex=True)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["year"] = combined["year"].astype(int)
    combined["month"] = combined["month"].astype(int)
    return combined


def compute_year_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame[model, year, auc, prec_k] pooled per year.

    For the yearly bar, prec_k is the mean of per-month Prec@Top-10% values
    (rather than a single pooled top-10% cut across all months), so that every
    month contributes equally regardless of trade volume.
    """
    rows = []
    for (model, year), grp in df.groupby(["model", "year"], sort=True):
        auc = safe_auc(grp["label"], grp["prob"])
        # Average Prec@Top-10% across constituent months
        monthly_prec = []
        for _, mgrp in grp.groupby("month"):
            p = prec_at_k(mgrp["label"], mgrp["prob"])
            if not np.isnan(p):
                monthly_prec.append(p)
        pk = float(np.mean(monthly_prec)) if monthly_prec else float("nan")
        rows.append({"model": model, "year": year, "auc": auc, "prec_k": pk})
    return pd.DataFrame(rows)


def compute_month_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame[model, year, month, period, auc, prec_k] per calendar month."""
    rows = []
    for (model, year, month), grp in df.groupby(["model", "year", "month"], sort=True):
        auc = safe_auc(grp["label"], grp["prob"])
        pk = prec_at_k(grp["label"], grp["prob"])
        period = year + (month - 1) / 12.0  # float for x-axis ordering
        rows.append({
            "model": model, "year": year, "month": month,
            "period": period, "label_str": f"{year}-{month:02d}",
            "auc": auc, "prec_k": pk,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bar_panel(ax, year_df: pd.DataFrame, metric: str, title: str, ylabel: str,
                   ref_line: float = 0.5, ref_label: str = "random"):
    """Grouped bar chart: x=year, groups=models.

    Parameters
    ----------
    ref_line : float
        Y-value for the horizontal reference line (0.5 for AUROC; overall
        positive-label rate for Prec@Top-k).
    ref_label : str
        Short label appended to the reference line annotation.
    """
    models = [m for m in MODEL_ORDER if m in year_df["model"].unique()]
    years = sorted(year_df["year"].unique())
    n_models = len(models)
    n_years = len(years)

    width = 0.8 / n_models
    x = np.arange(n_years)

    for i, model in enumerate(models):
        sub = year_df[year_df["model"] == model].set_index("year")
        vals = [sub.loc[y, metric] if y in sub.index else float("nan") for y in years]
        offset = (i - n_models / 2 + 0.5) * width
        color = MODEL_COLORS.get(model, "#888888")
        ax.bar(x + offset, vals, width=width * 0.92,
               label=model, color=color, alpha=BAR_ALPHA, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axhline(ref_line, color="black", linewidth=0.8, linestyle="--", alpha=0.5,
               label="_nolegend_")
    ax.annotate(f"{ref_label} ({ref_line:.2f})",
                xy=(x[-1] + 0.5, ref_line), xycoords="data",
                fontsize=7, color="black", alpha=0.6, va="bottom", ha="right")
    # Zoom y-axis to the data range so small differences are visible
    all_vals = year_df[metric].dropna().tolist() + [ref_line]
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        pad = max((hi - lo) * 0.4, 0.002)
        ax.set_ylim(lo - pad, hi + pad)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.grid(axis="y", which="minor", alpha=0.15, linewidth=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_line_panel(ax, month_df: pd.DataFrame, metric: str, title: str, ylabel: str,
                    rolling: int = 3, ref_line: float = 0.5, ref_label: str = "random"):
    """Line chart: x=year-month, one line per model.

    Raw monthly values are shown as faint thin lines; a rolling average of
    `rolling` months is overlaid as a bold line with markers.

    Parameters
    ----------
    ref_line : float
        Y-value for the horizontal reference line.
    ref_label : str
        Short label appended to the reference line annotation.
    """
    models = [m for m in MODEL_ORDER if m in month_df["model"].unique()]

    # Build a shared x-axis of all periods in order
    all_periods = sorted(month_df["period"].unique())
    period_to_label = (
        month_df[["period", "label_str"]]
        .drop_duplicates()
        .set_index("period")["label_str"]
        .to_dict()
    )

    for model in models:
        sub = month_df[month_df["model"] == model].sort_values("period").reset_index(drop=True)
        color = MODEL_COLORS.get(model, "#888888")
        marker = MODEL_MARKERS.get(model, "o")

        # Raw values — faint background
        ax.plot(sub["period"], sub[metric],
                color=color, linewidth=0.7, alpha=0.25, zorder=1)

        # Rolling average — bold foreground
        rolled = sub[metric].rolling(window=rolling, min_periods=1, center=True).mean()
        ax.plot(sub["period"], rolled,
                label=model, color=color, marker=marker,
                markersize=MARKER_SIZE, linewidth=LINE_WIDTH,
                alpha=LINE_ALPHA, zorder=2)

    # x-ticks: show only Jan of each year to avoid crowding
    tick_periods = [p for p in all_periods if period_to_label.get(p, "").endswith("-01")]
    tick_labels = [period_to_label[p][:4] for p in tick_periods]  # just the year
    ax.set_xticks(tick_periods)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{title}  (3-month rolling avg)", fontsize=11, fontweight="bold")
    ax.axhline(ref_line, color="black", linewidth=0.8, linestyle="--", alpha=0.5,
               label="_nolegend_")
    if all_periods:
        ax.annotate(f"{ref_label} ({ref_line:.2f})",
                    xy=(all_periods[-1], ref_line), xycoords="data",
                    fontsize=7, color="black", alpha=0.6, va="bottom", ha="right")
    # Zoom y-axis to the data range so small differences are visible
    all_vals = month_df[metric].dropna().tolist() + [ref_line]
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        pad = max((hi - lo) * 0.15, 0.002)
        ax.set_ylim(lo - pad, hi + pad)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.grid(axis="both", which="minor", alpha=0.15, linewidth=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_horizon(horizon: str, results_dir: Path, out_dir: Path) -> bool:
    """Load preds, compute metrics, render 2×2 figure. Returns True if rendered."""
    df = load_preds_for_horizon(results_dir, horizon)
    if df.empty:
        print(f"  [{horizon}] No prediction files found — skipping.")
        return False

    models_found = sorted(df["model"].unique())
    print(f"  [{horizon}] Models: {models_found}  |  rows: {len(df):,}")

    year_df = compute_year_metrics(df)
    month_df = compute_month_metrics(df)

    # Random baseline for Prec@Top-10%: overall positive-label rate
    pos_rate = float(df["label"].mean())
    pos_rate_rounded = round(pos_rate, 2)

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle(
        f"Horizon: {horizon}  —  Model Comparison Timeline (2019–2024)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    plot_bar_panel(axes[0, 0], year_df, "auc",
                   title="AUROC by Year", ylabel="Pooled AUROC",
                   ref_line=0.5, ref_label="random")
    plot_bar_panel(axes[0, 1], year_df, "prec_k",
                   title=f"Prec@Top-10% by Year", ylabel="Precision (top 10%)",
                   ref_line=pos_rate_rounded, ref_label="random")
    plot_line_panel(axes[1, 0], month_df, "auc",
                    title="AUROC by Month", ylabel="AUROC",
                    ref_line=0.5, ref_label="random")
    plot_line_panel(axes[1, 1], month_df, "prec_k",
                    title=f"Prec@Top-10% by Month", ylabel="Precision (top 10%)",
                    ref_line=pos_rate_rounded, ref_label="random")

    # Shared legend below all subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Supplement with any models only in line plots
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    seen_labels = set(labels)
    for h, l in zip(h2, l2):
        if l not in seen_labels:
            handles.append(h)
            labels.append(l)
            seen_labels.add(l)

    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(labels),
        fontsize=9, frameon=True, framealpha=0.9,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])

    out_path = out_dir / f"timeline_{horizon}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{horizon}] Saved → {out_path}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Render per-horizon timeline comparison plots.")
    parser.add_argument(
        "--horizons", nargs="+", default=HORIZONS,
        help="Which horizons to render (default: all 6).",
    )
    parser.add_argument(
        "--results-dir",
        default="experiments/signal_isolation/results",
        help="Directory containing preds_*.csv files.",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/signal_isolation/results/visualizations",
        help="Directory to write PNG files into.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results dir : {results_dir}")
    print(f"Output dir  : {out_dir}")
    print(f"Horizons    : {args.horizons}")
    print()

    rendered = 0
    for h in args.horizons:
        if render_horizon(h, results_dir, out_dir):
            rendered += 1

    print(f"\nDone. {rendered}/{len(args.horizons)} horizon(s) rendered.")


if __name__ == "__main__":
    main()
