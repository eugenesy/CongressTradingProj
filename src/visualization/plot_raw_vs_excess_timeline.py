"""
plot_raw_vs_excess_timeline.py
------------------------------
Timeline-style visualization comparing raw stock return labels vs SPY-excess
return labels for the 6M horizon, 2023.

Layout (mirrors timeline_6M.png):
  [0,0] Model × AUROC       — grouped bar chart per model, raw vs excess side by side
  [0,1] Model × Prec@Top-10% — grouped bar chart per model, raw vs excess side by side
  [1,0] Month × AUROC        — per-month line chart, one line per model×label_type
  [1,1] Month × Prec@Top-10% — per-month line chart, one line per model×label_type

Per-row prediction CSVs (preds_2023-2023_6M_{raw|excess}_L0_{model}_seed42.csv)
are used as the data source for both panels.

Output:
  results/visualizations/raw_vs_excess_timeline_6M_2023.png
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("experiments/signal_isolation/results/raw_return")
OUT_DIR     = Path("experiments/signal_isolation/results/visualizations")
OUT_FILE    = OUT_DIR / "raw_vs_excess_timeline_6M_2023.png"

HORIZON     = "6M"
YEAR_RANGE  = "2023-2023"
SEED        = 42
LEVEL       = 0         # L0: >0% threshold (most data, clearest signal)

TOP_K_PCT   = 0.10

# Models to show (display order)
MODEL_ORDER = ["Random", "XGBoost(ID)", "XGBoost(feat)", "LogReg(feat)", "RF(feat)", "GNN-SAGE"]
MODEL_SHORT = ["Random", "XGB(ID)", "XGB(feat)", "LogReg", "RF", "GNN-SAGE"]

# Two series per model: raw (solid) and excess (dashed)
COLOR_RAW    = "#2196F3"   # blue  — raw stock return
COLOR_EXCESS = "#FF5722"   # orange-red — excess over SPY

# Per-model colours for line charts (same palette as timeline_6M.png)
MODEL_COLORS = {
    "Random":          "#a9a9a9",  # grey
    "XGBoost(ID)":     "#3cb44b",  # green
    "XGBoost(feat)":   "#4363d8",  # blue
    "LogReg(feat)":    "#f58231",  # orange
    "RF(feat)":        "#911eb4",  # purple
    # GNN-SAGE added when available
    "GNN-SAGE":        "#e6194b",  # red
}
MODEL_MARKERS = {
    "Random":          "x",
    "XGBoost(ID)":     "s",
    "XGBoost(feat)":   "^",
    "LogReg(feat)":    "D",
    "RF(feat)":        "v",
    "GNN-SAGE":        "o",
}

FIGSIZE    = (20, 12)
BAR_ALPHA  = 0.85
LINE_ALPHA = 0.85
LINE_WIDTH = 2.2
MARKER_SZ  = 6

# Only these 3 models appear in the line panels — keeps the chart readable.
# The bar panels still show all MODEL_ORDER models.
LINE_MODELS = ["Random", "XGBoost(feat)", "GNN-SAGE"]

# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def prec_at_k(y_true, y_prob, k_pct: float = TOP_K_PCT) -> float:
    """Precision among the top k_pct highest-probability predictions."""
    n = len(y_true)
    k = max(1, int(round(n * k_pct)))
    if k >= n:
        return float("nan")
    top_idx = np.argsort(np.array(y_prob))[::-1][:k]
    return float(np.array(y_true)[top_idx].mean())


def safe_auc(y_true, y_prob):
    """AUROC; returns NaN if only one class present."""
    u = np.unique(y_true)
    if len(u) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_prob)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_preds_for_level(level: int) -> pd.DataFrame:
    """
    Load all per-row prediction CSVs for the given threshold level.
    Returns combined DataFrame with columns:
      label_type, model, year, month, prob, label
    """
    frames = []
    for label_type in ["raw", "excess"]:
        pattern = str(
            RESULTS_DIR /
            f"preds_{YEAR_RANGE}_{HORIZON}_{label_type}_L{level}_*_seed{SEED}.csv"
        )
        files = glob.glob(pattern)
        for fpath in sorted(files):
            df = pd.read_csv(fpath)
            # Ensure label_type column is present (older files may lack it)
            if "label_type" not in df.columns:
                df["label_type"] = label_type
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No preds CSVs found in {RESULTS_DIR} for level L{level}."
        )
    combined = pd.concat(frames, ignore_index=True)
    combined["year"]  = combined["year"].astype(int)
    combined["month"] = combined["month"].astype(int)
    return combined


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_pooled_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pool all predictions across months per (model, label_type) and compute
    AUROC and mean(per-month Prec@Top-10%).
    """
    rows = []
    for (model, label_type), grp in df.groupby(["model", "label_type"], sort=False):
        auc = safe_auc(grp["label"], grp["prob"])
        # Average per-month Prec@Top-10%
        monthly_prec = []
        for _, mgrp in grp.groupby(["year", "month"]):
            p = prec_at_k(mgrp["label"].values, mgrp["prob"].values)
            if not np.isnan(p):
                monthly_prec.append(p)
        pk = float(np.mean(monthly_prec)) if monthly_prec else float("nan")
        rows.append({
            "model": model, "label_type": label_type,
            "auc": auc, "prec_k": pk,
        })
    return pd.DataFrame(rows)


def compute_monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-month AUROC and Prec@Top-10% for each (model, label_type, year, month).
    """
    rows = []
    for (model, label_type, year, month), grp in df.groupby(
        ["model", "label_type", "year", "month"], sort=True
    ):
        auc = safe_auc(grp["label"].values, grp["prob"].values)
        pk  = prec_at_k(grp["label"].values, grp["prob"].values)
        period = year + (month - 1) / 12.0
        rows.append({
            "model": model, "label_type": label_type,
            "year": year, "month": month,
            "period": period,
            "label_str": f"{year}-{month:02d}",
            "auc": auc, "prec_k": pk,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

def plot_bar_panel(ax, pooled_df: pd.DataFrame, metric: str,
                   title: str, ylabel: str,
                   ref_line_raw: float, ref_line_excess: float):
    """
    Grouped bar chart: x=model, each model has two bars (raw, excess).

    Parameters
    ----------
    ref_line_raw    : horizontal reference line for raw (e.g. random baseline)
    ref_line_excess : horizontal reference line for excess
    """
    models_in_order = [m for m in MODEL_ORDER if m in pooled_df["model"].unique()]
    # Also append any extra models (e.g. GNN-SAGE) not in MODEL_ORDER
    extra = [m for m in pooled_df["model"].unique() if m not in MODEL_ORDER]
    models_in_order = models_in_order + extra

    x = np.arange(len(models_in_order))
    bar_w = 0.35

    raw_vals    = []
    excess_vals = []
    for m in models_in_order:
        sub = pooled_df[pooled_df["model"] == m]
        rv = sub.loc[sub["label_type"] == "raw",    metric]
        ev = sub.loc[sub["label_type"] == "excess", metric]
        raw_vals.append(rv.values[0] if len(rv) else float("nan"))
        excess_vals.append(ev.values[0] if len(ev) else float("nan"))

    bars_raw    = ax.bar(x - bar_w / 2, raw_vals,    bar_w,
                         label="Raw return",    color=COLOR_RAW,    alpha=BAR_ALPHA,
                         edgecolor="white", linewidth=0.4)
    bars_excess = ax.bar(x + bar_w / 2, excess_vals, bar_w,
                         label="Excess return", color=COLOR_EXCESS, alpha=BAR_ALPHA,
                         edgecolor="white", linewidth=0.4)

    # Reference lines
    ax.axhline(ref_line_raw,    color=COLOR_RAW,    linewidth=0.9, linestyle="--",
               alpha=0.6, label="_nolegend_")
    ax.axhline(ref_line_excess, color=COLOR_EXCESS, linewidth=0.9, linestyle=":",
               alpha=0.6, label="_nolegend_")

    # Value labels on bars
    for bar in list(bars_raw) + list(bars_excess):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=6, rotation=0)

    short_labels = []
    for m in models_in_order:
        idx = MODEL_ORDER.index(m) if m in MODEL_ORDER else -1
        short_labels.append(MODEL_SHORT[idx] if idx >= 0 else m)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    all_vals = [v for v in raw_vals + excess_vals if not np.isnan(v)]
    all_vals += [ref_line_raw, ref_line_excess]
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        pad = max((hi - lo) * 0.35, 0.005)
        ax.set_ylim(lo - pad, hi + pad)

    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.grid(axis="y", which="minor", alpha=0.15, linewidth=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="upper right")


def plot_line_panel(ax, month_df: pd.DataFrame, metric: str,
                    title: str, ylabel: str,
                    ref_line: float, rolling: int = 3):
    """
    Per-month line chart — only LINE_MODELS for readability.

    Raw lines: solid  (raw return)
    Excess lines: dashed (excess return)
    Colour: per-model palette.
    """
    # Restrict to key models only
    models_in_order = [m for m in LINE_MODELS if m in month_df["model"].unique()]

    all_periods = sorted(month_df["period"].unique())
    period_to_label = (
        month_df[["period", "label_str"]]
        .drop_duplicates()
        .set_index("period")["label_str"]
        .to_dict()
    )

    for model in models_in_order:
        color  = MODEL_COLORS.get(model, "#888888")
        marker = MODEL_MARKERS.get(model, "o")

        for label_type, ls, lw, alpha, zord in [
            ("raw",    "-",  LINE_WIDTH,       LINE_ALPHA,       3),
            ("excess", "--", LINE_WIDTH * 0.8, LINE_ALPHA * 0.75, 2),
        ]:
            sub = month_df[
                (month_df["model"] == model) &
                (month_df["label_type"] == label_type)
            ].sort_values("period").reset_index(drop=True)

            if sub.empty:
                continue

            # Faint raw signal
            ax.plot(sub["period"], sub[metric],
                    color=color, linewidth=0.5, alpha=0.15, linestyle=ls, zorder=1)

            # Rolling average — bold foreground
            rolled = sub[metric].rolling(window=rolling, min_periods=1, center=True).mean()
            label_str = f"{model} ({'raw' if label_type == 'raw' else 'exc. SPY'})"
            ax.plot(sub["period"], rolled,
                    label=label_str, color=color, linestyle=ls,
                    marker=marker, markersize=MARKER_SZ,
                    linewidth=lw, alpha=alpha, zorder=zord)

    # x-axis: one tick per month, label every other
    ax.set_xticks(all_periods)
    xticklabels = [period_to_label.get(p, "") for p in all_periods]
    ax.set_xticklabels(xticklabels, fontsize=8, rotation=45, ha="right")

    ax.axhline(ref_line, color="black", linewidth=0.9, linestyle=":", alpha=0.5,
               label="_nolegend_")
    if all_periods:
        ax.annotate(f"random ({ref_line:.2f})",
                    xy=(all_periods[-1], ref_line), xycoords="data",
                    fontsize=7, color="black", alpha=0.6, va="bottom", ha="right")

    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{title}  (3-month rolling avg, key models only)", fontsize=11, fontweight="bold")

    all_vals = month_df.loc[month_df["model"].isin(LINE_MODELS), metric].dropna().tolist() + [ref_line]
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        pad = max((hi - lo) * 0.15, 0.005)
        ax.set_ylim(lo - pad, hi + pad)

    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.grid(axis="both", which="minor", alpha=0.15, linewidth=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render(level: int = LEVEL):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading prediction CSVs for L{level} …")
    df = load_preds_for_level(level)
    models_found = sorted(df["model"].unique())
    label_types  = sorted(df["label_type"].unique())
    print(f"  Models   : {models_found}")
    print(f"  Label types: {label_types}")
    print(f"  Rows     : {len(df):,}")

    pooled_df  = compute_pooled_metrics(df)
    monthly_df = compute_monthly_metrics(df)

    # Random baselines (pooled)
    rand_raw    = pooled_df.loc[
        (pooled_df["model"] == "Random") & (pooled_df["label_type"] == "raw"), "prec_k"
    ]
    rand_excess = pooled_df.loc[
        (pooled_df["model"] == "Random") & (pooled_df["label_type"] == "excess"), "prec_k"
    ]
    rand_pk_raw    = rand_raw.values[0]    if len(rand_raw)    else 0.5
    rand_pk_excess = rand_excess.values[0] if len(rand_excess) else 0.5

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.suptitle(
        f"Raw Return vs Excess Return Labels — 6M Horizon, 2023  "
        f"[L{level}: >{'0%' if level==0 else '5%' if level==1 else '10%'} threshold]\n"
        f"Solid lines = raw return  |  Dashed lines = excess over SPY",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # Top row: bar charts (pooled across all 2023 months)
    plot_bar_panel(
        axes[0, 0], pooled_df, "auc",
        title="Pooled AUROC (2023, all months)",
        ylabel="Pooled AUROC",
        ref_line_raw=0.5, ref_line_excess=0.5,
    )
    plot_bar_panel(
        axes[0, 1], pooled_df, "prec_k",
        title="Mean Prec@Top-10% (2023, all months)",
        ylabel="Precision (top 10%)",
        ref_line_raw=rand_pk_raw,
        ref_line_excess=rand_pk_excess,
    )

    # Bottom row: per-month line charts
    plot_line_panel(
        axes[1, 0], monthly_df, "auc",
        title="AUROC by Month",
        ylabel="AUROC",
        ref_line=0.5,
    )
    plot_line_panel(
        axes[1, 1], monthly_df, "prec_k",
        title="Prec@Top-10% by Month",
        ylabel="Precision (top 10%)",
        ref_line=(rand_pk_raw + rand_pk_excess) / 2,
    )

    # Shared legend below — deduplicate across subplots
    all_handles, all_labels_leg = [], []
    seen = set()
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in seen and li != "_nolegend_":
                all_handles.append(hi)
                all_labels_leg.append(li)
                seen.add(li)

    # Remove per-panel legends
    for ax in axes.flat:
        leg = ax.get_legend()
        if leg:
            leg.remove()

    ncol = min(len(all_labels_leg), 6)
    fig.legend(
        all_handles, all_labels_leg,
        loc="lower center", ncol=ncol,
        fontsize=8, frameon=True, framealpha=0.9,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.97])

    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_FILE}")

    # ------------------------------------------------------------------
    # Print key numbers to stdout
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"KEY NUMBERS  — Level L{level}, 6M, 2023")
    print("=" * 70)
    print(f"{'Model':<18}  {'AUROC(raw)':>10}  {'AUROC(exc)':>10}  "
          f"{'ΔAUC':>7}  {'P@10%(raw)':>10}  {'P@10%(exc)':>10}  {'ΔP@10%':>8}")
    print("-" * 80)
    models_all = [m for m in MODEL_ORDER if m in pooled_df["model"].unique()]
    for m in models_all:
        sub = pooled_df[pooled_df["model"] == m]
        ar = sub.loc[sub["label_type"] == "raw",    "auc"].values
        ae = sub.loc[sub["label_type"] == "excess", "auc"].values
        pr = sub.loc[sub["label_type"] == "raw",    "prec_k"].values
        pe = sub.loc[sub["label_type"] == "excess", "prec_k"].values
        ar = ar[0] if len(ar) else float("nan")
        ae = ae[0] if len(ae) else float("nan")
        pr = pr[0] if len(pr) else float("nan")
        pe = pe[0] if len(pe) else float("nan")
        print(f"{m:<18}  {ar:10.4f}  {ae:10.4f}  {ar-ae:+7.4f}  "
              f"{pr:10.4f}  {pe:10.4f}  {pr-pe:+8.4f}")


if __name__ == "__main__":
    render(level=LEVEL)
