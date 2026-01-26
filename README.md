# GAP-TGN: Graph Alpha Prediction for Congressional Trading

**GAP-TGN** is a specialized Temporal Graph Network (TGN) designed to detect alpha signals in congressional trading disclosures. Unlike traditional tabular methods, GAP-TGN explicitly models the evolving network of legislators and corporate entities, using an asynchronous propagation strategy to handle reporting delays.

## Key Features
*   **Asynchronous Propagation**: Updates node states during "gap" periods using resolved transactions, preventing data staleness.
*   **Gated Multi-Modal Fusion**: Dynamically weights graph embeddings against immediate market signals.
*   **Production-Ready Pipeline**: Clean separation of data processing, training, and evaluation.

## Installation

```bash
# Clone the repository
git clone https://github.com/syeugene/chocolate.git
cd chocolate

# Create environment
conda create -n chocolate python=3.10
conda activate chocolate

# Install dependencies and package
pip install -r requirements.txt
pip install -e .
```

## Usage

### 1. Build Dataset
Process raw transactions into the temporal graph format.
```bash
chocolate-build
```

### 2. Train GAP-TGN (Proposed Model)
Train the main graph model on the 2019-2024 dataset.
```bash
# Using the installed entry point
chocolate-train --horizon 6M --epochs 5 --seed 42

# Or directly via python
python scripts/train_gap_tgn.py --horizon 6M
```

### 2. Train Baselines (Benchmarking)
Run comparative baselines (XGBoost, Logistic Regression, MLP) on the same data split.
```bash
chocolate-baselines --horizon 6M --start-year 2019 --end-year 2024
```

### 3. Evaluate Results
Results are saved to `results/experiments/` and `results/baselines/`.

## Directory Structure
*   `src/gap_tgn.py`: Core TGN model definition.
*   `scripts/`: Operational scripts for training and evaluation.
*   `legacy/`: Archived code (ablation studies, old baselines).
*   `data/`: Data storage (Parquet files are git-ignored).

## Citation
If you use this code found in `src/gap_tgn.py`, please cite the accompanying workshop paper.
