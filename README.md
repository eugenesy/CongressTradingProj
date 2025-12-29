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