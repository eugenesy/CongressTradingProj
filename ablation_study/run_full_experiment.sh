
#!/bin/bash
# Run the full 6-year ablation study
# Usage: ./run_full_experiment.sh

echo "Starting Full 6-Year Ablation Study..."
echo "Modes: pol_only, mkt_only, full"
echo "Years: 2019-2024"
echo "Logs: ablation_study/logs/ablation_log.txt"

# Ensure we are in the project root
cd /data1/user_syeugene/fintech/chocolate

# Run with full-run flag
# Uses nohup to persist if terminal closes, but we run in fg for now to show output? 
# Better to just run straight python if user is watching.
python3 ablation_study/run_ablation.py --full-run

echo "Experiment Complete. Check results/ablation_monthly_breakdown.csv"
