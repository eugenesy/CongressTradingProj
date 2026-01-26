
import json
import glob
import os
import numpy as np

def check_files(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found for pattern: {pattern}")
        return

    print(f"Checking {len(files)} files for pattern: {pattern}")
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            keys = data.keys()
            y_prob = np.array(data.get('y_prob', []))
            y_true = np.array(data.get('y_true', []))
            
            missing = []
            if 'y_prob' not in keys: missing.append('y_prob')
            if 'y_true' not in keys: missing.append('y_true')
            
            if missing:
                print(f"  [FAIL] {os.path.basename(fpath)} missing keys: {missing}")
            elif len(y_prob) == 0:
                print(f"  [WARN] {os.path.basename(fpath)} is empty")
            else:
                # Check for frozen probabilities (all same)
                if np.all(y_prob == y_prob[0]):
                     print(f"  [WARN] {os.path.basename(fpath)} has FROZEN probabilities (std={np.std(y_prob):.4f})")
                else:
                    # tailored check
                    pass
                    # print(f"  [OK] {os.path.basename(fpath)} (n={len(y_prob)})")

        except Exception as e:
            print(f"  [ERR] {os.path.basename(fpath)}: {e}")

print("--- Checking TGN (Full) ---")
check_files("results/experiments/H_1M_A_0.0/probs/probs_full_*.json")

print("\n--- Checking Baselines (XGBoost) ---")
check_files("results/baselines/H_1M_A_0.0/probs/probs_xgboost_*.json")

print("\n--- Checking Baselines (MLP) ---")
check_files("results/baselines/H_1M_A_0.0/probs/probs_mlp_*.json")
