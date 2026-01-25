# Project Chocolate: Congressional Trading Prediction with TGN

This repository contains the implementation of a Temporal Graph Network (TGN) for predicting excess returns of US Congressperson stock trades.

## Directory Structure

*   `scripts/`: Executable entry points (`run_rolling.py`, `run_ablation.py`).
*   `src/`: Core library code (`temporal_data.py`, `models_tgn.py`, `config.py`).
*   `data/`: Data storage (`temporal_data.pt`, `price_sequences.pt`).
*   `results/`: Training metrics and evaluation JSON reports.
*   `docs/`: Methodological documentation.
*   `FUTURE_WORK.md`: Roadmap and Next Steps.

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

## Results & Metrics
*   **Standard Report**: `report_{mode}_{year}_{month}.json`
*   **Directional Report**: `report_{mode}_{year}_{month}_directional.json` (Interprets "Sell" success as "Stock Up").

## Requirements
*   Python 3.8+
*   PyTorch, PyTorch Geometric
*   Pandas, NumPy, Matplotlib, Tqdm
