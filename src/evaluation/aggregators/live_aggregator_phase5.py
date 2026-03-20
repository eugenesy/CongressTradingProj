"""
live_aggregator_phase5.py
=========================
Evaluation Aggregators. Groups metrics across N=60 rolling horizons.

Refactored/Audited: 2026-03-20
"""

#!/usr/bin/env python3
"""
Live log aggregator for Phase 5 (Multi-Year)
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

def live_agg():
    log = "/tmp/phase5_multi_year_gat.log"
    p = Path(log)
    if not p.exists():
        print("Log not found")
        return
        
    with open(p, "r") as f:
        content = f.read()
        
    # Match: 16:19:29 [INFO]     [2022-10] AUC=0.5223  P@10%=0.4444
    matches = re.findall(r"\[(\d{4})-\d+\]\s+AUC=([0-9.]+)\s+P@10%=([0-9.]+)", content)
    if not matches:
        print("No months evaluated yet.")
        return
        
    df = pd.DataFrame(matches, columns=["year", "auc", "p10"])
    df["auc"] = df["auc"].astype(float)
    df["p10"] = df["p10"].astype(float)
    
    print("=== Phase 5 Multi-Year LIVE Aggregates ===")
    for yr, grp in df.groupby("year"):
        print(f"\nYear {yr} ({len(grp)} months):")
        print(f"  AUROC : {grp['auc'].mean():.4f} \u00B1 {grp['auc'].std():.4f}")
        print(f"  P@10% : {grp['p10'].mean():.4f} \u00B1 {grp['p10'].std():.4f}")
        
    print(f"\nOverall ({len(df)} months):")
    print(f"  AUROC : {df['auc'].mean():.4f} \u00B1 {df['auc'].std():.4f}")
    print(f"  P@10% : {df['p10'].mean():.4f} \u00B1 {df['p10'].std():.4f}")

if __name__ == "__main__":
    live_agg()
