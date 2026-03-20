"""
live_aggregator_phase3.py
=========================
Evaluation Aggregators. Groups metrics across N=60 rolling horizons.

Refactored/Audited: 2026-03-20
"""

#!/usr/bin/env python3
"""
Live log aggregator for Phase 3
"""

import re
import numpy as np
from pathlib import Path

def live_agg():
    logs = ["/tmp/phase3_sage_q3.log", "/tmp/phase3_gat_q3.log", "/tmp/phase3_sage_med.log"]
    
    print("=== Phase 3 LIVE Log Summaries ===")
    for log in logs:
        p = Path(log)
        if not p.exists(): continue
        
        with open(p, "r") as f:
            content = f.read()
            
        matches = re.findall(r"AUC=([0-9.]+)\s+P@10%=([0-9.]+)", content)
        if not matches:
            print(f"\n{p.name}: No rows evaluated yet.")
            continue
            
        aucs = [float(m[0]) for m in matches]
        p10s = [float(m[1]) for m in matches]
        
        print(f"\n{p.name}:")
        print(f"  Months evaluated: {len(aucs)}")
        print(f"  AUROC : {np.mean(aucs):.4f} \u00B1 {np.std(aucs):.4f}")
        print(f"  P@10% : {np.mean(p10s):.4f} \u00B1 {np.std(p10s):.4f}")

if __name__ == "__main__":
    live_agg()
