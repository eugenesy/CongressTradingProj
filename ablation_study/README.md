
# Ablation Study: Politician vs Market Signal

This directory contains the code for the 3-Way Ablation Study (2019-2024).

## configurations
1. **Politician Signal Only (`pol_only`)**: Price features (sequence + history) are zeroed out.
2. **Market Signal Only (`mkt_only`)**: Politician embeddings (Static Party/State) are zeroed out.
3. **Full Model (`full`)**: All features active.

## Execution
To run the full study (approx 10 epochs per month per year):
```bash
bash ablation_study/run_full_experiment.sh
```

## Results
*   **Metrics CSV**: `results/ablation_monthly_breakdown.csv`
*   **Classification Reports**: `results/reports/report_{mode}_{year}_{month}.json`
*   **Logs**: `ablation_study/logs/ablation_log.txt`

## Logic Details
The partial ablation is implemented in `run_ablation.py` by dynamically zeroing tensors before they enter the model or the price encoder.
