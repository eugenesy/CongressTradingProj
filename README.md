# Project Chocolate: Congressional Trading Prediction with TGN

This repository contains the implementation of a Temporal Graph Network (TGN) for predicting excess returns of US Congressperson stock trades.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/chocolate.git  # TODO: Update with actual repo URL
cd chocolate
```

### 2. Create environment (Conda recommended)
```bash
conda env create -f environment.yml
conda activate chocolate
```

**Alternative (pip):**
```bash
pip install -r requirements.txt
```

### 3. Install package in editable mode
```bash
pip install -e .
```

### 4. Verify installation
```bash
python -c "import src; print('Installation successful!')"
```

## Directory Structure

*   `scripts/`: Executable entry points (`run_rolling.py`, `run_ablation.py`, `build_dataset.py`).
*   `src/`: Core library code (`temporal_data.py`, `models_tgn.py`, `config.py`, `financial_pipeline/`).
*   `data/`: Data storage (gitignored - proprietary).
*   `results/`: Training metrics and evaluation JSON reports (gitignored).
*   `tests/`: Unit tests (run with `pytest`).
*   `docs/`: Methodological documentation ([TGN Explainer](docs/TGN_EXPLAINER.md), [Model Overview](docs/MODEL_OVERVIEW.md), [Experiment Log](docs/EXPERIMENT_LOG.md)).
*   `FUTURE_WORK.md`: Roadmap and Next Steps.

## Data Generation (From Raw Source)

If you need to regenerate the dataset from scratch:

### Prerequisites
1. Place the raw transaction file (`v5_transactions.csv`) in `data/raw/`.
2. Ensure you have API access for downloading historical price data (used by the pipeline).

### Run the Pipeline
```bash
python scripts/build_dataset.py
```

This will:
1. **Download historical price data** for all tickers and SPY → `data/parquet/*.parquet` (~3000 files, ~473MB)
2. Add benchmark columns, closing prices, and excess returns.
3. Clean and standardize the data.
4. Generate `data/processed/ml_dataset_reduced_attributes.csv`.
5. Build engineered price features → `data/price_sequences.pt`.

**Note**: 
- This process may take several hours depending on the number of tickers.
- The parquet files and generated datasets are **proprietary** and excluded from git via `.gitignore`.
- If you already have the parquet files and processed CSV, you can skip this step.

## Quick Start
### 1. Build Graph
```bash
python src/temporal_data.py
# Generates data/temporal_data.pt (using Filed Date)
```

### 2. Rolling Window Evaluation
Run a standard rolling evaluation (default 1M horizon):
```bash
python scripts/run_rolling.py --horizon 1M --alpha 0.0
```

### 3. Ablation Study
Run the full ablation study (Politician vs Market Signal) or targeted experiments:

**New: Targeted Experiment (Full Model Only)**
```bash
python scripts/run_ablation.py --full-only --horizon 6M --alpha 0.05
# Results saved to: results/experiments/H_6M_A_0.05/
```

**Full 3-Way Ablation (2019-2024)**
```bash
python scripts/run_ablation.py --full-run
```

**All Horizons Sequential (1M, 2M, 3M, 6M, 8M, 12M, 18M, 24M)**
```bash
bash scripts/run_all_horizons.sh
```
Runs full model for all 8 horizons sequentially (2019-2024, α=0.0). Results saved to `results/experiments/H_{horizon}_A_0.0/`.

## Baseline Model Comparison

Compare TGN against traditional ML models (XGBoost, LightGBM, Random Forest, MLP, Logistic Regression, KNN) using identical features:

```bash
# Single model
python baselines/run_baselines.py --horizon 1M --alpha 0.0 --model xgboost

# All models
python baselines/run_baselines.py --horizon 1M --alpha 0.0 --all
```

See [`baselines/README.md`](baselines/README.md) for details.

## Testing

Run unit tests:
```bash
pytest tests/ -v
```

### End-to-End Pipeline Test

**Option 1: Quick test (assumes data already exists)**
```bash
bash scripts/test_pipeline.sh
```
Tests: Temporal graph building → Ablation study (full-only, 1M, α=0.0)

**Option 2: Full test (includes dataset building - takes hours)**
```bash
bash scripts/test_full_pipeline.sh
```
Tests: Dataset building → Temporal graph → Ablation study

**For tmux:**
```bash
# Start a tmux session
tmux new -s chocolate_test

# Run the test
bash scripts/test_pipeline.sh

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t chocolate_test
```

## Results & Metrics
*   **Standard Report**: `report_{mode}_{year}_{month}.json`
*   **Directional Report**: `report_{mode}_{year}_{month}_directional.json` (Interprets "Sell" success as "Stock Up").

## Documentation

- **[TGN Explainer](docs/TGN_EXPLAINER.md)**: Introduction to Temporal Graph Networks
- **[Model Overview](docs/MODEL_OVERVIEW.md)**: Architecture details
- **[Experiment Log](docs/EXPERIMENT_LOG.md)**: Complete experiment history
- **[Future Work](FUTURE_WORK.md)**: Roadmap and planned improvements

## Requirements
*   Python 3.8+
*   PyTorch 2.0+, PyTorch Geometric 2.3+
*   Pandas, NumPy, Matplotlib, Tqdm
*   See `requirements.txt` or `environment.yml` for full dependencies

## Contributing

<!-- TODO: Add contribution guidelines -->
See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Citation

<!-- TODO: Add BibTeX citation if this is published research -->

## License

<!-- TODO: Add license information -->
See [LICENSE](LICENSE) for details.

## Troubleshooting

<!-- TODO: Add common issues and solutions -->

**Common Issues:**
- **Import errors**: Ensure you've installed the package with `pip install -e .`
- **CUDA errors**: Check PyTorch and CUDA compatibility in `environment.yml`
- **Missing data**: Run `scripts/build_dataset.py` to generate required files
