"""
plot_sweep.py
=============
Visualization Utilities. Generates presentations plots and distributions.

Refactored/Audited: 2026-03-20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_sweep():
    viz_dir = Path("experiments/signal_isolation/results/visualizations")
    summary_path = viz_dir / "sweep_stats_summary.csv"
    per_year_path = viz_dir / "sweep_per_year_auroc.csv"

    if not summary_path.exists():
        print(f"Summary file not found: {summary_path}")
        return

    df = pd.read_csv(summary_path)
    df_py = pd.read_csv(per_year_path)

    # Set plot style
    sns.set_theme(style="whitegrid")

    # 1. Pooled AUROC by DZ threshold — grouped bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='model', y='pooled_auc', hue='dz')
    plt.title("Pooled AUROC by Dead-Zone Threshold", fontsize=15)
    plt.ylabel("AUROC")
    plt.ylim(0.48, 0.55)
    plt.legend(title="Dead Zone")
    plt.tight_layout()
    plt.savefig(viz_dir / "sweep_pooled_auroc_bar.png")

    # 2. ΔAUROC vs DZ threshold — line chart per model
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df[df['dz'] > 0], x='dz', y='delta', hue='model', marker='o')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("ΔAUROC vs Dead-Zone Threshold (v. Baseline DZ=0)", fontsize=15)
    plt.ylabel("ΔAUROC")
    plt.tight_layout()
    plt.savefig(viz_dir / "sweep_delta_auroc_line.png")

    # 3. Bootstrap CI Heatmap
    # Pivot for heatmap
    pivot_delta = df.pivot(index='model', columns='dz', values='delta')
    # Filter out dz=0
    pivot_delta = pivot_delta.drop(columns=[0.0], errors='ignore')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_delta, annot=True, fmt=".4f", cmap="RdYlGn", center=0)
    plt.title("Heatmap of ΔAUROC by Model and Dead-Zone %", fontsize=15)
    plt.tight_layout()
    plt.savefig(viz_dir / "sweep_delta_heatmap.png")

    # 4. Per-year AUROC (GNN-SAGE only)
    gnn_py = df_py[df_py['model'] == 'gnn_sage']
    if not gnn_py.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=gnn_py, x='year', y='auroc', hue='dz', marker='s')
        plt.title("GNN-SAGE Per-Year AUROC by Dead-Zone", fontsize=15)
        plt.ylabel("AUROC")
        plt.tight_layout()
        plt.savefig(viz_dir / "sweep_gnn_per_year_auroc.png")

    print(f"Visualizations saved to {viz_dir}")

if __name__ == "__main__":
    plot_sweep()
