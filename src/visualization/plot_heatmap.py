"""
plot_heatmap.py
---------------
Render one heatmap PNG per horizon showing Year × Model performance.

For each horizon, produces one PNG with 2 side-by-side heatmaps:
  [left]  AUROC         — rows=years, cols=models
  [right] Prec@Top-10%  — rows=years, cols=models

Colour scale is diverging, centred on the random baseline:
  AUROC        → centred at 0.50
  Prec@Top-10% → centred at the overall positive-label rate (~0.42)

Usage
-----
  python experiments/signal_isolation/plot_heatmap.py

  # Specific horizons only:
  python experiments/signal_isolation/plot_heatmap.py --horizons 6M 12M

  # Override dirs:
  python experiments/signal_isolation/plot_heatmap.py \
      --results-dir experiments/signal_isolation/results \
      --out-dir experiments/signal_isolation/results/visualizations

Output
------
  results/visualizations/heatmap_<HORIZON>.png
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HORIZONS = ["1M", "2M", "3M", "6M", "12M", "24M"]

MODEL_ORDER = [
    "GNN-SAGE",
    "GNN-SAGE (gap)",
    "XGBoost(ID)",
    "XGBoost(feat)",
    "MLP(ID)",
    "MLP(feat)",
    "Random",
]

TOP_K_PCT = 0.10

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prec_at_k(y_true, y_prob, k_pct: float = TOP_K_PCT) -> float:
    n = len(y_true)
    k = max(1, int(round(n * k_pct)))
    if k >= n:
        return float("nan")
    top_idx = np.argsort(np.asarray(y_prob))[::-1][:k]
    return float(np.asarray(y_true)[top_idx].mean())


def safe_auc(y_true, y_prob) -> float:
    if len(pd.Series(y_true).unique()) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_prob)


def load_preds_for_horizon(results_dir: Path, horizon: str) -> pd.DataFrame:
    pattern = str(results_dir / f"preds_*_{horizon}_seed*.csv")
    files = sorted(glob.glob(pattern))
    gap_pattern = str(results_dir / f"preds_gnn_*_{horizon}_seed*_gap.csv")
    files += sorted(glob.glob(gap_pattern))
    seen: set = set()
    files = [f for f in files if not (f in seen or seen.add(f))]

    if not files:
        return pd.DataFrame()

    frames = []
    for fpath in files:
        df = pd.read_csv(fpath)
        if "_gap.csv" in fpath:
            df["model"] = df["model"].str.replace(
                r"^GNN-SAGE$", "GNN-SAGE (gap)", regex=True
            )
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["year"] = combined["year"].astype(int)
    combined["month"] = combined["month"].astype(int)
    return combined


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame[model, year, auc, prec_k].

    AUROC is pooled across all months in (model, year).
    Prec@Top-10% is the mean of per-month values.
    """
    rows = []
    for (model, year), grp in df.groupby(["model", "year"], sort=True):
        auc = safe_auc(grp["label"], grp["prob"])
        monthly_prec = []
        for _, mgrp in grp.groupby("month"):
            p = prec_at_k(mgrp["label"].values, mgrp["prob"].values)
            if not np.isnan(p):
                monthly_prec.append(p)
        pk = float(np.mean(monthly_prec)) if monthly_prec else float("nan")
        rows.append({"model": model, "year": year, "auc": auc, "prec_k": pk})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def draw_heatmap(ax, pivot: pd.DataFrame, vmin: float, vcenter: float,
                 vmax: float, cmap: str, title: str, ref_label: str):
    """
    Draw an annotated heatmap.
    pivot : rows=years, cols=models (in MODEL_ORDER subset).
    """
    data = pivot.values.astype(float)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)

    # Annotate cells
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val = data[r, c]
            if np.isnan(val):
                txt, color = "—", "grey"
            else:
                txt = f"{val:.3f}"
                normed = norm(val)
                color = "white" if (normed < 0.25 or normed > 0.75) else "black"
            ax.text(c, r, txt, ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Model", fontsize=9)
    ax.set_ylabel("Year", fontsize=9)
    ax.set_title(f"{title}\n(dashed = {ref_label})", fontsize=10, fontweight="bold")

    return im


def render_horizon(horizon: str, results_dir: Path, out_dir: Path) -> bool:
    df = load_preds_for_horizon(results_dir, horizon)
    if df.empty:
        print(f"  [{horizon}] No prediction files found — skipping.")
        return False

    models_found = sorted(df["model"].unique())
    print(f"  [{horizon}] Models: {models_found}  |  rows: {len(df):,}")

    grid = compute_metrics(df)
    models = [m for m in MODEL_ORDER if m in grid["model"].unique()]
    years  = sorted(grid["year"].unique())
    pos_rate = round(float(df["label"].mean()), 3)

    def make_pivot(metric):
        return (
            grid.pivot_table(index="year", columns="model",
                             values=metric, aggfunc="mean")
            .reindex(index=years)
            .reindex(columns=models)
        )

    pivot_auc = make_pivot("auc")
    pivot_pk  = make_pivot("prec_k")

    # Shared colour bounds per metric
    def bounds(vals, center):
        lo = max(center - 0.12, float(vals.min()) - 0.005)
        hi = min(center + 0.12, float(vals.max()) + 0.005)
        # ensure lo < center < hi
        lo = min(lo, center - 0.005)
        hi = max(hi, center + 0.005)
        return lo, hi

    auc_lo, auc_hi = bounds(grid["auc"].dropna(), 0.50)
    pk_lo,  pk_hi  = bounds(grid["prec_k"].dropna(), pos_rate)

    n_models = len(models)
    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(12, n_models * 1.6 + 3), max(5, len(years) * 0.85 + 2.5)),
        gridspec_kw={"wspace": 0.45},
    )
    fig.suptitle(
        f"Horizon: {horizon}  —  Year × Model Heatmap",
        fontsize=13, fontweight="bold", y=1.02,
    )

    im0 = draw_heatmap(
        axes[0], pivot_auc,
        vmin=auc_lo, vcenter=0.50, vmax=auc_hi,
        cmap="RdYlGn", title="AUROC", ref_label="random = 0.50",
    )
    im1 = draw_heatmap(
        axes[1], pivot_pk,
        vmin=pk_lo, vcenter=pos_rate, vmax=pk_hi,
        cmap="RdYlGn",
        title="Prec@Top-10%",
        ref_label=f"random ≈ {pos_rate:.2f}",
    )

    # Colourbars
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)

    out_path = out_dir / f"heatmap_{horizon}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{horizon}] Saved → {out_path}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Render Year×Model heatmaps, one PNG per horizon."
    )
    parser.add_argument("--horizons", nargs="+", default=HORIZONS)
    parser.add_argument(
        "--results-dir", default="experiments/signal_isolation/results",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/signal_isolation/results/visualizations",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results dir : {results_dir}")
    print(f"Output dir  : {out_dir}")
    print(f"Horizons    : {args.horizons}\n")

    rendered = 0
    for h in args.horizons:
        if render_horizon(h, results_dir, out_dir):
            rendered += 1

    print(f"\nDone. {rendered}/{len(args.horizons)} horizon(s) rendered.")


if __name__ == "__main__":
    main()
