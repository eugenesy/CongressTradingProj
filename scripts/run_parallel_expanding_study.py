"""
run_parallel_expanding_study.py
===============================
Executable Pipeline Runners. Imports src.* modules to execute experiments.

Refactored/Audited: 2026-03-20
"""

import subprocess
import time
import pandas as pd
from pathlib import Path

def main():
    years = [2020, 2021, 2022, 2023, 2024]
    processes = []
    out_dir = Path("experiments/signal_isolation/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Launching Multi-Year Parallel Evaluations (2020-2023) on GPUs 0-3...")
    
    # Phase 1: Parallel 2020-2023
    for i, year in enumerate([2020, 2021, 2022, 2023]):
        gpu_id = str(i)
        env_vars = {"CUDA_VISIBLE_DEVICES": gpu_id}
        
        # 1. XGBoost
        xgb_cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} /home/syeugene/miniconda3/envs/chocolate/bin/python experiments/signal_isolation/run_path_dependent_ml.py --test-year {year} > {out_dir}/xgb_{year}.log 2>&1"
        print(f"-> Scheduling XGBoost Year {year} on GPU {gpu_id}")
        p_xgb = subprocess.Popen(xgb_cmd, shell=True)
        processes.append((f"XGB_{year}", p_xgb))
        
        # 2. GNN
        gnn_cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} /home/syeugene/miniconda3/envs/chocolate/bin/python experiments/signal_isolation/run_gnn_continuous_reach.py --test-year {year} > {out_dir}/gnn_{year}.log 2>&1"
        print(f"-> Scheduling GNN Year {year} on GPU {gpu_id}")
        p_gnn = subprocess.Popen(gnn_cmd, shell=True)
        processes.append((f"GNN_{year}", p_gnn))

    print("⏳ Waiting for parallel fits (2020-2023) to complete...")
    for name, p in processes:
        p.wait()
    print("✅ Parallel Phase 1 Complete.")

    # Phase 2: Year 2024 (sequential overlap)
    print("\n🚀 Launching trailing Year 2024 evaluation on GPU 0...")
    xgb_cmd_24 = f"CUDA_VISIBLE_DEVICES=0 /home/syeugene/miniconda3/envs/chocolate/bin/python experiments/signal_isolation/run_path_dependent_ml.py --test-year 2024 > {out_dir}/xgb_2024.log 2>&1"
    gnn_cmd_24 = f"CUDA_VISIBLE_DEVICES=0 /home/syeugene/miniconda3/envs/chocolate/bin/python experiments/signal_isolation/run_gnn_continuous_reach.py --test-year 2024 > {out_dir}/gnn_2024.log 2>&1"
    
    p_xgb_24 = subprocess.Popen(xgb_cmd_24, shell=True)
    p_gnn_24 = subprocess.Popen(gnn_cmd_24, shell=True)
    
    p_xgb_24.wait()
    p_gnn_24.wait()
    print("✅ Phase 2 (2024) Complete.")

    # Phase 3: Aggregation
    print("\n📊 Aggregating Results...")
    xgb_frames = []
    gnn_frames = []
    
    for year in years:
        # Load XGB
        x_path = out_dir / f"xgb_res_{year}.csv"
        if x_path.exists():
            dx = pd.read_csv(x_path)
            dx['test_year'] = year
            xgb_frames.append(dx)
            
        # Load GNN
        g_path = out_dir / f"gnn_res_{year}.csv"
        if g_path.exists():
            dg = pd.read_csv(g_path)
            dg['test_year'] = year
            gnn_frames.append(dg)

    if not xgb_frames or not gnn_frames:
        print("Error: No results found to aggregate.")
        return

    df_xgb = pd.concat(xgb_frames)
    df_gnn = pd.concat(gnn_frames)

    # Calculate Mean & Std
    summary = []
    targets = ['Label_Median', 'Label_Q3']
    configs = ['Combined', 'Structure-Only']

    for target in targets:
        target_label = "Median" if "Median" in target else "Q3"
        for config in configs:
            # XGB Filter
            x_match = df_xgb[df_xgb['target'] == target]
            if config == 'Structure-Only':
                xgb_auc = x_match['auc_id'] # ID equivalent
                xgb_p10 = x_match['p10_id']
            else:
                xgb_auc = x_match['auc_all'] 
                xgb_p10 = x_match['p10_all']

            # GNN Filter
            g_match = df_gnn[(df_gnn['Target'] == target_label) & (df_gnn['Config'] == config)]
            gnn_auc = g_match['AUROC'] if not g_match.empty else pd.Series([0])
            gnn_p10 = g_match['P@10%'] if not g_match.empty else pd.Series([0])

            summary.append({
                'Target': target_label,
                'Config': config,
                'XGB_AUC_Avg': xgb_auc.mean(),
                'XGB_AUC_Std': xgb_auc.std(),
                'GNN_AUC_Avg': gnn_auc.mean(),
                'GNN_AUC_Std': gnn_auc.std(),
                'XGB_P10_Avg': xgb_p10.mean(),
                'GNN_P10_Avg': gnn_p10.mean()
            })

    summary_df = pd.DataFrame(summary)
    print("\n" + "#" * 50)
    print("Aggregate Multi-Year (2020-2024) Ablation Summary")
    print("#" * 50)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(out_dir / "aggregate_study_summary.csv", index=False)
    
    print("\n📈 Triggering Timeline Plotter...")
    subprocess.run(["/home/syeugene/miniconda3/envs/chocolate/bin/python", "experiments/signal_isolation/plot_multi_year_timeline.py"])

if __name__ == "__main__":
    main()
