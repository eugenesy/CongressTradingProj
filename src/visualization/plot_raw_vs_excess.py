"""
plot_raw_vs_excess.py
---------------------
Grid visualization comparing raw stock return labels vs SPY-excess return
labels for the 6M horizon, 2023.

Layout: 3 columns (L0, L1, L2) × 2 rows (AUROC, P@Top-10%)
Each cell: grouped bar chart — models on x-axis, raw vs excess side by side.

Output: results/visualizations/raw_vs_excess_6M_2023.png
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("experiments/signal_isolation/results/raw_return")
OUT_DIR     = Path("experiments/signal_isolation/results/visualizations")
OUT_FILE    = OUT_DIR / "raw_vs_excess_6M_2023.png"

HORIZON     = "6M"
YEAR        = "2023-2023"
SEED        = 42

MODEL_ORDER = ["Random", "XGBoost(ID)", "XGBoost(feat)", "LogReg(feat)", "RF(feat)"]
MODEL_SHORT = ["Random", "XGB(ID)", "XGB(feat)", "LogReg", "RF"]

LEVEL_LABELS = {
    0: "L0: >0%\n(any gain)",
    1: "L1: >5%\n(moderate gain)",
    2: "L2: >10%\n(strong gain)",
}

COLOR_RAW    = "#2196F3"   # blue
COLOR_EXCESS = "#FF5722"   # orange-red
RANDOM_LINE  = "#888888"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_summary() -> pd.DataFrame:
    path = RESULTS_DIR / f"comparison_{YEAR}_{HORIZON}_all_seed{SEED}.csv"
    df = pd.read_csv(path)
    return df

def load_monthly() -> pd.DataFrame:
    """Load all monthly CSVs for 6M and concatenate."""
    pattern = str(RESULTS_DIR / f"monthly_{YEAR}_{HORIZON}_*_seed{SEED}.csv")
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(summary: pd.DataFrame, monthly: pd.DataFrame):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    levels = [0, 1, 2]
    metrics = [
        ("pooled_auc",         "Pooled AUROC",       0.45, 0.65),
        ("precision_at_top10", "Precision @ Top-10%", 0.0,  1.0),
    ]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 8),
        gridspec_kw={"hspace": 0.45, "wspace": 0.30},
    )
    fig.suptitle(
        f"Raw Return vs Excess Return Labels — 6M Horizon, 2023\n"
        f"Walk-forward (monthly folds, train on all prior data)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    x = np.arange(len(MODEL_ORDER))
    bar_w = 0.35

    for row, (metric_col, metric_label, ymin, ymax) in enumerate(metrics):
        for col, level in enumerate(levels):
            ax = axes[row, col]

            raw_vals    = []
            excess_vals = []

            for model in MODEL_ORDER:
                r = summary.loc[
                    (summary["label_type"] == "raw") &
                    (summary["threshold_level"] == level) &
                    (summary["model"] == model),
                    metric_col
                ]
                e = summary.loc[
                    (summary["label_type"] == "excess") &
                    (summary["threshold_level"] == level) &
                    (summary["model"] == model),
                    metric_col
                ]
                raw_vals.append(r.values[0] if len(r) else np.nan)
                excess_vals.append(e.values[0] if len(e) else np.nan)

            bars_raw    = ax.bar(x - bar_w/2, raw_vals,    bar_w,
                                 label="Raw return",    color=COLOR_RAW,    alpha=0.85)
            bars_excess = ax.bar(x + bar_w/2, excess_vals, bar_w,
                                 label="Excess return", color=COLOR_EXCESS, alpha=0.85)

            # Random baseline horizontal line
            rand_raw = summary.loc[
                (summary["label_type"] == "raw") &
                (summary["threshold_level"] == level) &
                (summary["model"] == "Random"),
                metric_col
            ]
            if len(rand_raw):
                ax.axhline(rand_raw.values[0], color=RANDOM_LINE,
                           linestyle="--", linewidth=1.0, alpha=0.7)

            # Value labels on bars
            for bar in list(bars_raw) + list(bars_excess):
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                            f"{h:.3f}", ha="center", va="bottom",
                            fontsize=6.5, rotation=0)

            ax.set_xticks(x)
            ax.set_xticklabels(MODEL_SHORT, fontsize=8, rotation=15, ha="right")
            ax.set_ylim(ymin, ymax)
            ax.set_ylabel(metric_label if col == 0 else "", fontsize=9)
            ax.set_title(LEVEL_LABELS[level], fontsize=9, pad=6)
            ax.yaxis.grid(True, alpha=0.3, linestyle=":")
            ax.set_axisbelow(True)

            if row == 0 and col == 2:
                ax.legend(fontsize=8, loc="upper right")

    # -----------------------------------------------------------------------
    # Bottom panel: per-month AUROC line chart (L0, XGB(ID) raw vs excess)
    # -----------------------------------------------------------------------
    if not monthly.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 3.5))
        fig2.suptitle(
            "Per-Month AUROC — XGBoost(ID), 6M Horizon L0 (>0%), 2023",
            fontsize=11, fontweight="bold",
        )

        for label_type, color, label in [
            ("raw",    COLOR_RAW,    "Raw return (Stock_Return_6M > 0%)"),
            ("excess", COLOR_EXCESS, "Excess return (Excess_Return_6M > 0%)"),
        ]:
            sub = monthly.loc[
                (monthly["label_type"] == label_type) &
                (monthly["threshold_level"] == 0) &
                (monthly["model"] == "XGBoost(ID)")
            ].sort_values(["year", "month"])
            if sub.empty:
                continue
            month_labels = [f"{int(r.month):02d}" for _, r in sub.iterrows()]
            ax2.plot(month_labels, sub["auc"].values, marker="o",
                     color=color, label=label, linewidth=1.8, markersize=5)

        ax2.axhline(0.5, color=RANDOM_LINE, linestyle="--", linewidth=1.0,
                    alpha=0.7, label="Random (0.50)")
        ax2.set_xlabel("Month (2023)", fontsize=9)
        ax2.set_ylabel("AUROC", fontsize=9)
        ax2.set_ylim(0.40, 0.75)
        ax2.legend(fontsize=8)
        ax2.yaxis.grid(True, alpha=0.3, linestyle=":")
        ax2.set_axisbelow(True)

        monthly_file = OUT_DIR / "raw_vs_excess_6M_2023_monthly.png"
        fig2.tight_layout()
        fig2.savefig(monthly_file, dpi=150, bbox_inches="tight")
        print(f"Saved: {monthly_file}")
        plt.close(fig2)

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_FILE}")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Print conclusion to stdout
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("CONCLUSION — Raw vs Excess Return Labels (6M, 2023)")
    print("=" * 65)
    for level in levels:
        print(f"\n  Level {level} (threshold={LEVEL_LABELS[level].split(chr(10))[0]}):")
        for model in MODEL_ORDER:
            r = summary.loc[
                (summary["label_type"] == "raw") &
                (summary["threshold_level"] == level) &
                (summary["model"] == model), "pooled_auc"
            ]
            e = summary.loc[
                (summary["label_type"] == "excess") &
                (summary["threshold_level"] == level) &
                (summary["model"] == model), "pooled_auc"
            ]
            if len(r) and len(e):
                diff = r.values[0] - e.values[0]
                winner = "RAW  +" if diff > 0 else "EXCS +"
                print(f"    {model:<16s}  raw={r.values[0]:.4f}  excess={e.values[0]:.4f}"
                      f"  Δ={diff:+.4f}  {'← raw wins' if diff > 0 else '← excess wins'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    summary = load_summary()
    monthly = load_monthly()
    plot(summary, monthly)
