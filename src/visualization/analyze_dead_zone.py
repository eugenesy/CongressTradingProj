"""
Dead Zone Filtering Analysis
==============================
Compare per-row prediction CSVs from the dead zone sweep against the
DZ=0% (unfiltered) baseline. Produces:

1. Pooled AUROC table: model × dead zone threshold (CSV + console)
2. Per-year AUROC bar chart: baseline vs DZ=10% side by side per model
3. Delta bar chart: AUROC(DZ=10%) - AUROC(DZ=0%) per model
4. Differential GNN vs flat: does GNN benefit more than flat models?

Usage:
    python experiments/signal_isolation/analyze_dead_zone.py [--results-dir PATH]

Outputs (saved to experiments/signal_isolation/results/visualizations/):
    dead_zone_auroc_table.csv
    dead_zone_pooled_auroc.png
    dead_zone_per_year_auroc.png
    dead_zone_delta.png
"""

import argparse
import logging
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

FLAT_MODELS = ["random", "xgb", "xgb_feat", "logreg", "rf"]
GNN_MODELS = ["gnn_sage"]
ALL_MODELS = FLAT_MODELS + GNN_MODELS

MODEL_DISPLAY = {
    "random":   "Random",
    "xgb":      "XGBoost(ID)",
    "xgb_feat": "XGBoost(feat)",
    "logreg":   "LogReg(feat)",
    "rf":       "RF(feat)",
    "gnn_sage": "GNN-SAGE",
}

# Dead zone thresholds to analyse (0 = baseline, 0.10 = 10%)
DEAD_ZONES = [0.0, 0.10]
DZ_LABELS: Dict[float, str] = {0.0: "DZ=0% (baseline)", 0.10: "DZ=10%"}

YEARS = list(range(2019, 2025))
YEARS_STR_FLAT = "2019-2024"
YEARS_STR_GNN = "_".join(str(y) for y in YEARS)


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def _dz_tag(dz: float) -> str:
    """Filename suffix for dead zone, e.g. '' or '_DZ10pct'."""
    if dz <= 0.0:
        return ""
    return f"_DZ{int(round(dz * 100))}pct"


def flat_preds_path(results_dir: Path, model: str, dz: float) -> Path:
    tag = _dz_tag(dz)
    return results_dir / (
        f"preds_{YEARS_STR_FLAT}_{HORIZON}_{LABEL_TYPE}_{LEVEL}_{model}{tag}_seed{SEED}.csv"
    )


def gnn_preds_path(results_dir: Path, dz: float) -> Path:
    """Combined multi-year GNN prediction CSV."""
    tag = _dz_tag(dz)
    return results_dir / (
        f"preds_{YEARS_STR_GNN}_{HORIZON}_{LABEL_TYPE}_{LEVEL}_gnn_sage{tag}_seed{SEED}.csv"
    )


def gnn_preds_path_per_year(results_dir: Path, year: int, dz: float) -> Path:
    """Fallback per-year GNN CSV (older baseline format)."""
    tag = _dz_tag(dz)
    return results_dir / (
        f"preds_{year}_{HORIZON}_{LABEL_TYPE}_{LEVEL}_gnn_sage{tag}_seed{SEED}.csv"
    )


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lower() for c in df.columns]
    return df


def load_flat_preds(results_dir: Path, model: str, dz: float) -> Optional[pd.DataFrame]:
    p = flat_preds_path(results_dir, model, dz)
    if not p.exists():
        logger.debug("Missing flat preds: %s", p.name)
        return None
    return _normalise(pd.read_csv(p))


def load_gnn_preds(results_dir: Path, dz: float) -> Optional[pd.DataFrame]:
    """Combined file first, then per-year fallback."""
    p = gnn_preds_path(results_dir, dz)
    if p.exists():
        return _normalise(pd.read_csv(p))
    parts = []
    for year in YEARS:
        py = gnn_preds_path_per_year(results_dir, year, dz)
        if py.exists():
            parts.append(_normalise(pd.read_csv(py)))
    if parts:
        return pd.concat(parts, ignore_index=True)
    logger.debug("Missing GNN preds for DZ=%.2f", dz)
    return None


# ---------------------------------------------------------------------------
# AUROC helpers
# ---------------------------------------------------------------------------

def auroc_from_df(df: Optional[pd.DataFrame]) -> Optional[float]:
    if df is None or df.empty:
        return None
    label_col = next((c for c in df.columns if c in ("label", "y_true", "true_label")), None)
    score_col = next(
        (c for c in df.columns if c in ("prob", "score", "prob_1", "pred_proba", "prediction")),
        None,
    )
    if label_col is None or score_col is None:
        logger.warning("Cannot find label/score columns (cols=%s)", list(df.columns))
        return None
    y = df[label_col].values
    s = df[score_col].values
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, s))
    except Exception as exc:
        logger.warning("AUROC error: %s", exc)
        return None


def auroc_for_year(df: Optional[pd.DataFrame], year: int) -> Optional[float]:
    if df is None:
        return None
    # Integer 'year' column (flat model CSVs)
    if "year" in df.columns and pd.api.types.is_integer_dtype(df["year"]):
        sub = df[df["year"] == year]
        return auroc_from_df(sub) if not sub.empty else None
    # Datetime 'filed' column (GNN CSVs)
    date_col = next((c for c in df.columns if c in ("filed", "date", "trade_date")), None)
    if date_col:
        try:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            sub = df[df[date_col].dt.year == year]
            return auroc_from_df(sub) if not sub.empty else None
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Build results matrix
# ---------------------------------------------------------------------------

def build_results(results_dir: Path) -> pd.DataFrame:
    """Returns DataFrame with columns: model, year, dead_zone, auroc."""
    records = []

    for dz in DEAD_ZONES:
        dz_label = DZ_LABELS[dz]
        logger.info("Loading %s ...", dz_label)

        for model in FLAT_MODELS:
            df = load_flat_preds(results_dir, model, dz)
            if df is None:
                continue
            # Pooled AUROC (all years)
            pooled = auroc_from_df(df)
            if pooled is not None:
                records.append({"model": model, "year": "pooled", "dead_zone": dz, "auroc": pooled})
            # Per-year
            for year in YEARS:
                auc = auroc_for_year(df, year)
                if auc is not None:
                    records.append({"model": model, "year": year, "dead_zone": dz, "auroc": auc})

        gnn_df = load_gnn_preds(results_dir, dz)
        if gnn_df is not None:
            pooled = auroc_from_df(gnn_df)
            if pooled is not None:
                records.append({"model": "gnn_sage", "year": "pooled", "dead_zone": dz, "auroc": pooled})
            for year in YEARS:
                auc = auroc_for_year(gnn_df, year)
                if auc is not None:
                    records.append({"model": "gnn_sage", "year": year, "dead_zone": dz, "auroc": auc})

    result = pd.DataFrame(records)
    logger.info("Loaded %d observations", len(result))
    return result


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------

def make_auroc_table(df: pd.DataFrame, out_dir: Path) -> None:
    """Print and save pooled AUROC table: model × dead_zone."""
    pooled = df[df["year"] == "pooled"].copy()
    if pooled.empty:
        logger.warning("No pooled AUROC entries found")
        return

    pivot = pooled.pivot_table(index="model", columns="dead_zone", values="auroc")
    pivot.index = [MODEL_DISPLAY.get(m, m) for m in pivot.index]
    pivot.columns = [DZ_LABELS.get(float(c), str(c)) for c in pivot.columns]

    # Add delta column if both DZ=0 and DZ=10% are present
    baseline_col = DZ_LABELS[0.0]
    dz10_col = DZ_LABELS[0.10]
    if baseline_col in pivot.columns and dz10_col in pivot.columns:
        pivot["Delta (DZ10% − baseline)"] = pivot[dz10_col] - pivot[baseline_col]

    logger.info("\nPooled AUROC Summary\n%s", pivot.to_string())

    out_path = out_dir / "dead_zone_auroc_table.csv"
    pivot.to_csv(out_path)
    logger.info("Saved AUROC table to %s", out_path)


# ---------------------------------------------------------------------------
# Figure 1: Pooled AUROC grouped bar chart
# ---------------------------------------------------------------------------

def plot_pooled_auroc(df: pd.DataFrame, out_dir: Path) -> None:
    pooled = df[df["year"] == "pooled"].copy()
    if pooled.empty:
        return

    models_present = [m for m in ALL_MODELS if m in pooled["model"].unique()]
    dz_present = sorted(pooled["dead_zone"].unique())

    x = np.arange(len(models_present))
    width = 0.35
    n_dz = len(dz_present)
    offsets = np.linspace(-(n_dz - 1) * width / 2, (n_dz - 1) * width / 2, n_dz)

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["steelblue", "tomato", "seagreen", "goldenrod"]

    for i, dz in enumerate(dz_present):
        vals = [
            pooled.loc[(pooled["model"] == m) & (pooled["dead_zone"] == dz), "auroc"].values
            for m in models_present
        ]
        y = [v[0] if len(v) > 0 else np.nan for v in vals]
        bars = ax.bar(x + offsets[i], y, width, label=DZ_LABELS.get(dz, str(dz)),
                      color=colors[i % len(colors)], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, y):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models_present], rotation=20, ha="right")
    ax.set_ylabel("Pooled AUROC (2019-2024)")
    ax.set_title(
        "Pooled AUROC: Baseline vs Dead Zone Filtering\n"
        "(6M horizon, excess labels, L0, seed 42)"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0.48, ax.get_ylim()[1] + 0.015)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "dead_zone_pooled_auroc.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Pooled AUROC bar chart saved to %s", out_path)


# ---------------------------------------------------------------------------
# Figure 2: Per-year AUROC (baseline vs DZ=10%), one panel per model
# ---------------------------------------------------------------------------

def plot_per_year_auroc(df: pd.DataFrame, out_dir: Path) -> None:
    year_df = df[df["year"] != "pooled"].copy()
    year_df = year_df[year_df["year"].apply(lambda y: isinstance(y, int))]
    if year_df.empty:
        return

    models_present = [m for m in ALL_MODELS if m in year_df["model"].unique()]
    dz_present = sorted(year_df["dead_zone"].unique())
    n_models = len(models_present)
    if n_models == 0:
        return

    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(5 * ((n_models + 1) // 2), 8),
                              squeeze=False)
    flat_axes = [ax for row in axes for ax in row]
    colors = ["steelblue", "tomato"]

    for ax, model in zip(flat_axes, models_present):
        for i, dz in enumerate(dz_present):
            sub = year_df[(year_df["model"] == model) & (year_df["dead_zone"] == dz)]
            if sub.empty:
                continue
            sub = sub.sort_values("year")
            ax.plot(sub["year"], sub["auroc"], marker="o",
                    label=DZ_LABELS.get(dz, str(dz)), color=colors[i % len(colors)])
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=10)
        ax.set_xlabel("Year")
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.44, 0.70)
        ax.set_xticks(YEARS)
        ax.set_xticklabels([str(y)[2:] for y in YEARS])
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Hide any extra axes
    for ax in flat_axes[n_models:]:
        ax.set_visible(False)

    fig.suptitle(
        "Per-Year AUROC: Baseline vs Dead Zone Filtering\n"
        "(6M horizon, excess labels, L0, seed 42)",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = out_dir / "dead_zone_per_year_auroc.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Per-year AUROC chart saved to %s", out_path)


# ---------------------------------------------------------------------------
# Figure 3: Delta AUROC (DZ=10% − baseline) per model
# ---------------------------------------------------------------------------

def plot_delta(df: pd.DataFrame, out_dir: Path) -> None:
    pooled = df[df["year"] == "pooled"].copy()
    if pooled.empty:
        return
    if 0.0 not in pooled["dead_zone"].values or 0.10 not in pooled["dead_zone"].values:
        logger.warning("Need both DZ=0 and DZ=10% to plot deltas")
        return

    models_present = [m for m in ALL_MODELS if m in pooled["model"].unique()]
    deltas, labels = [], []
    for model in models_present:
        base = pooled.loc[(pooled["model"] == model) & (pooled["dead_zone"] == 0.0), "auroc"]
        dz10 = pooled.loc[(pooled["model"] == model) & (pooled["dead_zone"] == 0.10), "auroc"]
        if base.empty or dz10.empty:
            continue
        deltas.append(float(dz10.values[0]) - float(base.values[0]))
        labels.append(MODEL_DISPLAY.get(model, model))

    if not deltas:
        return

    colors = ["tomato" if d < 0 else "seagreen" for d in deltas]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(range(len(labels)), deltas, color=colors, edgecolor="white", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("ΔAUROC (DZ=10% − baseline)")
    ax.set_title(
        "Pooled AUROC Gain from Dead Zone Filtering (DZ=10%)\n"
        "(positive = filtering improved discrimination)"
    )
    for bar, val in zip(bars, deltas):
        sign = "+" if val >= 0 else ""
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.0003 if val >= 0 else -0.0008),
                f"{sign}{val:.4f}",
                ha="center", va="bottom" if val >= 0 else "top",
                fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "dead_zone_delta.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Delta bar chart saved to %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dead zone filtering results")
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
            "No results found in %s — have the dead zone sweep jobs completed?",
            args.results_dir,
        )
        return

    logger.info("Models found    : %s", sorted(df["model"].unique()))
    logger.info("Dead zones found: %s", sorted(df["dead_zone"].unique()))
    # Year values can be integers (per-year rows) or the string 'pooled'.
    years_present = df["year"].unique()
    years_int = sorted([int(y) for y in years_present if isinstance(y, (int, np.integer))])
    logger.info("Years found     : %s", years_int)

    make_auroc_table(df, args.out_dir)
    plot_pooled_auroc(df, args.out_dir)
    plot_per_year_auroc(df, args.out_dir)
    plot_delta(df, args.out_dir)

    logger.info("Done. Outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()
