"""
Rolling Window Sweep Analysis
==============================
Aggregate per-row prediction CSVs from the rolling window sweep and produce:

1. Per-year AUROC table  (rows=year, columns=window size, one panel per model)
2. Mean AUROC vs window size line plot (with ± std error bars across years)
3. Delta heatmap: AUROC(W) - AUROC(unbounded) per model × year

Usage:
    python experiments/signal_isolation/analyze_window_sweep.py [--results-dir PATH]

Outputs (saved to experiments/signal_isolation/results/visualizations/):
    window_sweep_auroc_table.csv
    window_sweep_auroc_table.png
    window_sweep_mean_auroc.png
    window_sweep_delta_heatmap.png
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("research")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HORIZON = "6M"
LABEL_TYPE = "excess"
LEVEL = "L0"
SEED = 42

# Models tracked (slug as it appears in filenames)
FLAT_MODELS = ["random", "xgb", "xgb_feat", "logreg", "rf"]
GNN_MODELS = ["gnn_sage"]
ALL_MODELS = FLAT_MODELS + GNN_MODELS

# Human-readable display names
MODEL_DISPLAY = {
    "random":   "Random",
    "xgb":      "XGBoost(ID)",
    "xgb_feat": "XGBoost(feat)",
    "logreg":   "LogReg(feat)",
    "rf":       "RF(feat)",
    "gnn_sage": "GNN-SAGE",
}

# Window sizes (0 = unbounded)
WINDOWS = [0, 1, 2, 3, 5]
WINDOW_LABELS = {0: "Unbounded", 1: "W=1y", 2: "W=2y", 3: "W=3y", 5: "W=5y"}

YEARS = list(range(2019, 2025))


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def _window_tag(w: int) -> str:
    return f"_W{w}y" if w > 0 else ""


def flat_preds_path(results_dir: Path, model: str, w: int) -> Path:
    """Expected path for a flat model's prediction CSV."""
    tag = _window_tag(w)
    return results_dir / (
        f"preds_2019-2024_{HORIZON}_{LABEL_TYPE}_{LEVEL}_{model}{tag}_seed{SEED}.csv"
    )


def gnn_preds_path(results_dir: Path, w: int) -> Path:
    """Expected path for a GNN-SAGE prediction CSV (all years combined)."""
    tag = _window_tag(w)
    years_str = "_".join(str(y) for y in YEARS)
    return results_dir / (
        f"preds_{years_str}_{HORIZON}_{LABEL_TYPE}_{LEVEL}_gnn_sage{tag}_seed{SEED}.csv"
    )


def gnn_preds_path_per_year(results_dir: Path, year: int, w: int) -> Path:
    """Fallback: per-year GNN prediction CSV (older format)."""
    tag = _window_tag(w)
    return results_dir / (
        f"preds_{year}_{HORIZON}_{LABEL_TYPE}_{LEVEL}_gnn_sage{tag}_seed{SEED}.csv"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_flat_preds(results_dir: Path, model: str, w: int) -> Optional[pd.DataFrame]:
    p = flat_preds_path(results_dir, model, w)
    if not p.exists():
        logger.debug("Missing flat preds: %s", p.name)
        return None
    df = pd.read_csv(p)
    # Normalise column names: some older files use different capitalisation
    df.columns = [c.lower() for c in df.columns]
    return df


def load_gnn_preds(results_dir: Path, w: int) -> Optional[pd.DataFrame]:
    """Load GNN preds: try combined multi-year file first, then per-year files."""
    # Combined file (new format from --years 2019 2020 ... 2024)
    p = gnn_preds_path(results_dir, w)
    if p.exists():
        df = pd.read_csv(p)
        df.columns = [c.lower() for c in df.columns]
        return df
    # Fallback: stitch per-year files (old format, unbounded baseline)
    parts = []
    for year in YEARS:
        py = gnn_preds_path_per_year(results_dir, year, w)
        if py.exists():
            sub = pd.read_csv(py)
            sub.columns = [c.lower() for c in sub.columns]
            parts.append(sub)
    if parts:
        return pd.concat(parts, ignore_index=True)
    logger.debug("Missing GNN preds for W=%s", w)
    return None


# ---------------------------------------------------------------------------
# AUROC computation
# ---------------------------------------------------------------------------

def auroc_from_df(df: pd.DataFrame) -> Optional[float]:
    """Compute AUROC from a preds DataFrame with columns label / score (or prob_1)."""
    if df is None or df.empty:
        return None
    label_col = next((c for c in df.columns if c in ("label", "y_true", "true_label")), None)
    score_col = next((c for c in df.columns if c in ("prob", "score", "prob_1", "pred_proba", "prediction")), None)
    if label_col is None or score_col is None:
        logger.warning("Cannot find label/score columns in DataFrame (cols=%s)", list(df.columns))
        return None
    y = df[label_col].values
    s = df[score_col].values
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, s))
    except Exception as exc:
        logger.warning("AUROC failed: %s", exc)
        return None


def compute_year_auroc(df: pd.DataFrame, year: int) -> Optional[float]:
    """Filter preds DataFrame to a single year and compute AUROC."""
    if df is None:
        return None
    # Prefer an explicit integer 'year' column (flat model CSVs).
    if "year" in df.columns and pd.api.types.is_integer_dtype(df["year"]):
        sub = df[df["year"] == year]
        return auroc_from_df(sub) if not sub.empty else None
    # Fall back: look for a proper datetime column (GNN CSVs use 'Filed').
    date_col = next(
        (c for c in df.columns if c.lower() in ("filed", "date", "trade_date")), None
    )
    if date_col:
        try:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            sub = df[df[date_col].dt.year == year]
            if sub.empty:
                return None
            return auroc_from_df(sub)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Build results matrix
# ---------------------------------------------------------------------------

def build_results(results_dir: Path) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        model, year, window, auroc
    """
    records = []

    for w in WINDOWS:
        logger.info("Loading window W=%s ...", WINDOW_LABELS[w])

        # --- Flat models ---
        for model in FLAT_MODELS:
            df = load_flat_preds(results_dir, model, w)
            if df is None:
                continue
            for year in YEARS:
                auc = compute_year_auroc(df, year)
                if auc is not None:
                    records.append({
                        "model": model,
                        "year": year,
                        "window": w,
                        "auroc": auc,
                    })

        # --- GNN-SAGE (combined multi-year file, split by year column) ---
        gnn_df = load_gnn_preds(results_dir, w)
        if gnn_df is not None:
            for year in YEARS:
                auc = compute_year_auroc(gnn_df, year)
                if auc is not None:
                    records.append({
                        "model": "gnn_sage",
                        "year": year,
                        "window": w,
                        "auroc": auc,
                    })

    result = pd.DataFrame(records)
    logger.info("Loaded %d (model, year, window) AUROC observations", len(result))
    return result


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------

def make_pivot_tables(df: pd.DataFrame, out_dir: Path) -> None:
    """Save per-model AUROC pivot tables (year × window) and a combined CSV."""
    all_rows = []

    for model in ALL_MODELS:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="year", columns="window", values="auroc")
        pivot.columns = [WINDOW_LABELS.get(c, str(c)) for c in pivot.columns]
        pivot.index.name = "Year"
        display_name = MODEL_DISPLAY.get(model, model)
        logger.info("\n%s\n%s", display_name, pivot.to_string())
        # append to all_rows
        for _, row in pivot.reset_index().iterrows():
            row_dict = {"model": display_name, "year": row["Year"]}
            row_dict.update({k: v for k, v in row.items() if k != "Year"})
            all_rows.append(row_dict)

    if all_rows:
        combined = pd.DataFrame(all_rows)
        out_path = out_dir / "window_sweep_auroc_table.csv"
        combined.to_csv(out_path, index=False)
        logger.info("AUROC table saved to %s", out_path)


# ---------------------------------------------------------------------------
# Figure 1: Per-year AUROC table heatmap
# ---------------------------------------------------------------------------

def plot_auroc_table(df: pd.DataFrame, out_dir: Path) -> None:
    models_present = [m for m in ALL_MODELS if m in df["model"].unique()]
    n_models = len(models_present)
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 5), squeeze=False)
    fig.suptitle(
        f"Per-Year AUROC by Window Size\n"
        f"(6M horizon, excess labels, L0, seed 42)",
        fontsize=12, y=1.02,
    )

    windows_present = sorted(df["window"].unique())
    col_labels = [WINDOW_LABELS.get(w, str(w)) for w in windows_present]

    for ax, model in zip(axes[0], models_present):
        sub = df[df["model"] == model]
        pivot = sub.pivot_table(index="year", columns="window", values="auroc")
        pivot = pivot.reindex(columns=windows_present, index=YEARS)

        vals = pivot.values
        im = ax.imshow(
            vals, aspect="auto", cmap="RdYlGn",
            vmin=max(0.45, np.nanmin(vals) - 0.02),
            vmax=min(0.70, np.nanmax(vals) + 0.02),
        )

        ax.set_xticks(range(len(windows_present)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(YEARS)))
        ax.set_yticklabels(YEARS, fontsize=8)
        ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=10)

        for i in range(len(YEARS)):
            for j in range(len(windows_present)):
                v = vals[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=6, color="black")

    plt.colorbar(im, ax=axes[0, -1], label="AUROC", fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path = out_dir / "window_sweep_auroc_table.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("AUROC table heatmap saved to %s", out_path)


# ---------------------------------------------------------------------------
# Figure 2: Mean AUROC vs window size (line plot)
# ---------------------------------------------------------------------------

def plot_mean_auroc(df: pd.DataFrame, out_dir: Path) -> None:
    models_present = [m for m in ALL_MODELS if m in df["model"].unique()]
    if not models_present:
        return

    windows_present = sorted(df["window"].unique())
    col_labels = [WINDOW_LABELS.get(w, str(w)) for w in windows_present]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(models_present)))  # type: ignore[attr-defined]
    for model, color in zip(models_present, colors):
        sub = df[df["model"] == model]
        means, stds = [], []
        for w in windows_present:
            vals = sub[sub["window"] == w]["auroc"].dropna().values
            means.append(np.mean(vals) if len(vals) > 0 else np.nan)
            stds.append(np.std(vals) if len(vals) > 0 else np.nan)
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        label = MODEL_DISPLAY.get(model, model)
        ax.plot(range(len(windows_present)), means_arr, marker="o", label=label, color=color)
        ax.fill_between(
            range(len(windows_present)),
            means_arr - stds_arr,
            means_arr + stds_arr,
            alpha=0.15, color=color,
        )

    ax.set_xticks(range(len(windows_present)))
    ax.set_xticklabels(col_labels)
    ax.set_xlabel("Training Window Size")
    ax.set_ylabel("Mean AUROC (± std across years)")
    ax.set_title(
        "Mean AUROC vs Training Window Size\n"
        "(6M horizon, excess labels, L0, seed 42, years 2019-2024)"
    )
    ax.legend(loc="best", fontsize=9)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, label="Unbounded (baseline)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "window_sweep_mean_auroc.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Mean AUROC line plot saved to %s", out_path)


# ---------------------------------------------------------------------------
# Figure 3: Delta heatmap vs unbounded baseline
# ---------------------------------------------------------------------------

def plot_delta_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    models_present = [m for m in ALL_MODELS if m in df["model"].unique()]
    windowed = [w for w in sorted(df["window"].unique()) if w != 0]
    if not models_present or not windowed:
        return

    # Build delta matrix: shape (n_models, n_windows)
    delta_matrix = np.full((len(models_present), len(windowed)), np.nan)

    for mi, model in enumerate(models_present):
        sub = df[df["model"] == model]
        unbounded = sub[sub["window"] == 0]["auroc"].mean()
        if np.isnan(unbounded):
            continue
        for wi, w in enumerate(windowed):
            wval = sub[sub["window"] == w]["auroc"].mean()
            delta_matrix[mi, wi] = wval - unbounded

    fig, ax = plt.subplots(figsize=(len(windowed) * 1.5 + 2, len(models_present) * 0.8 + 1.5))

    vabs = np.nanmax(np.abs(delta_matrix)) + 0.005
    im = ax.imshow(delta_matrix, aspect="auto", cmap="RdYlGn", vmin=-vabs, vmax=vabs)

    ax.set_xticks(range(len(windowed)))
    ax.set_xticklabels([WINDOW_LABELS.get(w, str(w)) for w in windowed])
    ax.set_yticks(range(len(models_present)))
    ax.set_yticklabels([MODEL_DISPLAY.get(m, m) for m in models_present])
    ax.set_xlabel("Window Size")
    ax.set_title(
        "AUROC Delta vs Unbounded Baseline (mean over 2019-2024)\n"
        "Green = rolling window beats unbounded"
    )

    for i in range(len(models_present)):
        for j in range(len(windowed)):
            v = delta_matrix[i, j]
            if not np.isnan(v):
                sign = "+" if v >= 0 else ""
                ax.text(j, i, f"{sign}{v:.4f}", ha="center", va="center",
                        fontsize=8, color="black")

    plt.colorbar(im, ax=ax, label="ΔAUROC (windowed − unbounded)", fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path = out_dir / "window_sweep_delta_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Delta heatmap saved to %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze rolling window sweep results")
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("experiments/signal_isolation/results/raw_return"),
        help="Directory containing prediction CSVs",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("experiments/signal_isolation/results/visualizations"),
        help="Directory to save figures and tables",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = build_results(args.results_dir)
    if df.empty:
        logger.error(
            "No results found in %s — have the sweep jobs completed?", args.results_dir
        )
        return

    # Summary
    logger.info(
        "Models found    : %s",
        sorted(df["model"].unique()),
    )
    logger.info(
        "Windows found   : %s",
        sorted(df["window"].unique()),
    )
    logger.info(
        "Years found     : %s",
        sorted(df["year"].unique()),
    )

    make_pivot_tables(df, args.out_dir)
    plot_auroc_table(df, args.out_dir)
    plot_mean_auroc(df, args.out_dir)
    plot_delta_heatmap(df, args.out_dir)

    logger.info("Done. Outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()
