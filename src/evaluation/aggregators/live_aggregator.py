"""
live_aggregator.py
==================
Evaluation Aggregators. Groups metrics across N=60 rolling horizons.

Refactored/Audited: 2026-03-20
"""

#!/usr/bin/env python3
"""
Live log aggregator to read continuous AUC prints without waiting for complete exit.
"""

import re
import numpy as np
from pathlib import Path

def live_agg():
    logs = ["/tmp/phase2_sage_q3.log", "/tmp/phase2_gat_q3.log", "/tmp/phase2_sage_med.log"]
    
    print("=== Phase 2 LIVE Log Summaries ===")
    for log in logs:
        p = Path(log)
        if not p.exists(): continue
        
        with open(p, "r") as f:
            content = f.read()
            
        # match: 15:09:41 [INFO]     AUC=0.6080  P@10%=0.3091  (n=556)
        matches = re.findall(r"AUC=([0-9.]+)\s+P@10%=([0-9.]+)\s+\(n=(\d+)\)", content)
        if not matches:
            print(f"\n{p.name}: No rows evaluated yet.")
            continue
            
        aucs = [float(m[0]) for m in matches]
        p10s = [float(m[1]) for m in matches]
        
        print(f"\n{p.name}:")
        print(f"  Months evaluated: {len(aucs)}")
        print(f"  AUROC : {np.mean(aucs):.4f} \u00B1 {np.std(aucs):.4f}")
        print(f"  P@10% : {np.mean(p10s):.4f} \u00B1 {np.std(p10s):.4f}")
        print(f"  Sample row-01: AUC={aucs[0]:.4f}")

if __name__ == "__main__":
    live_agg()
