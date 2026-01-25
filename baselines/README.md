# Baseline Model Comparison

Compare TGN against traditional ML baselines using identical features and evaluation methodology.

## Features Used (Matching TGN Exactly)

| Feature Type | Features | Encoding |
|--------------|----------|----------|
| **Politician** | Party, State, BioGuideID, District | One-Hot |
| **Transaction** | Trade_Size, Is_Buy, Filing_Gap | Numerical (log-scaled) |
| **Market** | 14-dim price features | Numerical (from `price_sequences.pt`) |

## Available Models

1. **XGBoost** - Gradient Boosting
2. **LightGBM** - Fast Gradient Boosting
3. **Random Forest** - Ensemble of Decision Trees
4. **MLP** - Multi-Layer Perceptron (Neural Network)
5. **Logistic Regression** - Linear Baseline
6. **KNN** - K-Nearest Neighbors

## Usage

```bash
# Single model
python baselines/run_baselines.py --horizon 1M --alpha 0.0 --model xgboost

# All models (sweep)
python baselines/run_baselines.py --horizon 1M --alpha 0.0 --all
```

### Parameters
- `--horizon`: Prediction horizon (1M, 2M, 3M, 6M, 8M, 12M, 18M, 24M)
- `--alpha`: Win/Loss threshold (0.0 = any positive excess return is a win)
- `--model`: Specific model to run
- `--all`: Run all 6 models

## Output Format

Results are saved in the same format as TGN ablation:

```
results/baselines/H_{horizon}_A_{alpha}/
├── reports/
│   ├── report_xgboost_2019_01.json          # Win/Loss metrics
│   ├── report_xgboost_2019_01_directional.json  # Price direction metrics
│   ├── report_lightgbm_2019_01.json
│   └── ...
├── summary_xgboost.csv
├── summary_lightgbm.csv
└── summary_all_models.csv  # Aggregated comparison
```

## Evaluation Methodology

- **Training**: Growing window (all data before test month)
- **Testing**: Monthly (same as TGN ablation)
- **Target**: Win/Loss (excess return > alpha)
- **Reports**: Both Win/Loss and Directional (price up/down)

## Metrics Reported

- **Accuracy**: Overall classification accuracy
- **F1 (Class 1)**: F1 score for positive class
- **AUC**: ROC Area Under Curve
- **PR-AUC**: Precision-Recall AUC
- **Dir_***: Same metrics for directional (price up/down)
