# Project Chocolate: Congressional Trading Prediction with TGN

This repository contains the implementation of a Temporal Graph Network (TGN) for predicting excess returns of US Congressperson stock trades.

## Directory Structure

* `src/`: Source code for models, data processing, and training.
* `data/`: Data storage (temporal_data.pt, price_sequences.pt).
* `results/`: Training metrics, learning curves, and evaluation JSONs.
* `logs/`: Execution logs.
* `docs/`: Methodological documentation and experiment logs.
    * [`EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md): Detailed history of Experiments 001-013.
    * [`TGN_EXPLAINER.md`](docs/TGN_EXPLAINER.md): Technical explanation of the model architecture.
* `presentations/`: LaTeX source for project presentations.
* [`FUTURE_WORK.md`](FUTURE_WORK.md): **Start Here for Next Steps** (Baseline Plan, To-Dos).

## Quick Start & Workflows

### 0. Data Preparation (Required for all)
**Download Data**:
Please request access and download the required datasets (`ml_dataset_reduced_attributes.csv`, `all_tickers_historical_data.pkl`) from this [Google Drive Link](https://drive.google.com/drive/folders/1ku1EeknWuz5h3PVxHtRhYs6Rd-GuHxfa?usp=share_link).

**Setup**:
Place the downloaded files into `data/raw/` inside the project directory.

**Generate Base Labels**:
Before running any specific workflow, you must generate the multiclass labels. This is the prerequisite for all use cases below:
```bash
python src/generate_multiclass_labels.py
```

### Use Case 1: Single Binary Labeling
Use this workflow to predict a single binary outcome (e.g., Buy vs. Sell, or Profitable vs. Not).

1.  **Build Temporal Data**:
    This script references `config_single_label.py`.
    ```bash
    python src/temporal_data_single_label.py
    ```

2.  **Run Training**:
    This script references `models_tgn_binary_labels.py`.
    ```bash
    python src/train_rolling_binary_labels.py
    ```

### Use Case 2: Single Multiclass Labeling
Use this workflow to predict a single target with multiple classes (e.g., Buy/Sell/Hold or High/Medium/Low returns).

1.  **Build Temporal Data**:
    This script references `config_single_label.py`.
    ```bash
    python src/temporal_data_single_label.py
    ```

2.  **Run Training**:
    This script references `models_tgn_multiclass_label.py`.
    ```bash
    python src/train_rolling_multiclass_label.py
    ```

### Use Case 3: Multiple Binary Labels
Use this workflow to predict multiple independent binary targets simultaneously (multi-label classification).

1.  **Build Temporal Data**:
    This script references `config_multiple_binary_labels.py`.
    ```bash
    python src/temporal_data_multiple_binary_labels.py
    ```

2.  **Run Training**:
    This script references `models_tgn_binary_labels.py` (adapted for multi-label output).
    ```bash
    python src/train_rolling_binary_labels.py
    ```

## Reproducing Presentation Results (Ablation Study)
To reproduce the full 6-year study (2019-2024) presented in the LaTeX slides:
```bash
./ablation_study/run_full_experiment.sh
```
This script:
* Runs 3 model configurations (Politician Only, Market Only, Full Model).
* Evaluates over 72 months (2019-2024).
* Saves consolidated results to `results/ablation_monthly_breakdown.csv`.

## Handover & Next Steps

Please refer to **[`FUTURE_WORK.md`](FUTURE_WORK.md)** for a detailed list of next steps, including:
1.  **Baseline Model Implementation** (Priority for teammate).
2.  Technical Paper drafting.
3.  Planned Graph Enhancements (Social Graph, etc.).

## Requirements
* Python 3.8+
* PyTorch, PyTorch Geometric
* Pandas, NumPy, Matplotlib, Tqdm