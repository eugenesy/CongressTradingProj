"""
aggregate_phase2.py
===================
Evaluation Aggregators. Groups metrics across N=60 rolling horizons.

Refactored/Audited: 2026-03-20
"""

#!/usr/bin/env python3
"""
Summaries aggregator to read Phase 2 outputs
"""

import pandas as pd
from pathlib import Path

def aggregate():
    d = Path("experiments/signal_isolation/results/phase2")
    if not d.exists():
        print("No results directory.")
        return

    metrics = list(d.glob("metrics_*.csv"))
    if not metrics:
        print("No metric csvs yet.")
        return

    print("=== Phase 2 Validation summaries ===")
    for m in metrics:
        df = pd.read_csv(m)
        print(f"\nModel/Label: {m.stem}")
        print(f"  AUROC : {df['AUROC'].mean():.4f} \u00B1 {df['AUROC'].std():.4f}")
        print(f"  P@10% : {df['P10'].mean():.4f} \u00B1 {df['P10'].std():.4f}")
        print(f"  Tested: {df['n_test'].sum()} trades over {len(df)} months")

if __name__ == "__main__":
    aggregate()
