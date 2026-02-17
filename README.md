# GAP-TGN: Graph Alpha Prediction for Congressional Trading

**GAP-TGN** is a specialized Temporal Graph Network (TGN) designed to detect alpha signals in congressional trading disclosures. Unlike traditional tabular methods, GAP-TGN explicitly models the evolving network of legislators and corporate entities, using an asynchronous propagation strategy to handle reporting delays.

## Key Features

* **Asynchronous Propagation**: Updates node states during "gap" periods using resolved transactions, preventing data staleness.
* **Gated Multi-Modal Fusion**: Dynamically weights graph embeddings against immediate market signals.
* **Dynamic Node Features**: Integrates rich, evolving metadata for nodes (e.g., Politician Ideology, Committee Assignments, Corporate Financials) directly into the graph learning process.
* **Production-Ready Pipeline**: Clean separation of data processing, training, and evaluation.

## Installation

```bash
# Clone the repository
git clone [https://github.com/syeugene/chocolate.git](https://github.com/syeugene/chocolate.git)
cd chocolate

# Create environment
conda create -n chocolate python=3.10
conda activate chocolate

# Install dependencies and package
pip install -r requirements.txt
pip install -e .
```

## Data Preparation

Before running the pipeline, you must ensure the raw data files are present in `data/raw/` to support the enhanced feature lookups.

### Required Raw Data

Ensure the following files exist in your `data/raw/` directory. These are used to generate the dynamic node features:

* **congress_terms_all_github.csv**: Legislator terms and biographical data.
* **ideology_scores_quarterly.csv**: DW-NOMINATE ideology scores.
* **committee_assignments.csv**: Historical committee memberships.
* **company_sic_data.csv**: SIC industry codes for tickers.
* **sec_quarterly_financials_unzipped.csv**: Corporate financial fundamentals.
* **district_industries/**: Directory containing Census Bureau economic data for districts.

## Usage

### 1. Configure Feature Flags

You can toggle which features are included in the graph by editing `src/config.py`.

```python
# src/config.py

# Set these to True/False to control feature generation
INCLUDE_POLITICIAN_BIO = True 
INCLUDE_IDEOLOGY = True
INCLUDE_COMMITTEES = True
INCLUDE_COMPANY_SIC = True
INCLUDE_DISTRICT_ECON = False
INCLUDE_COMPANY_FINANCIALS = False
```

### 2. Build Temporal Data

Once your config is set and raw data is in place, run the temporal data script. This will generate the `data/temporal_data.pt` file containing the graph structure and your enabled feature vectors (`x_pol`, `x_comp`).

```bash
# Generates data/temporal_data.pt
python src/temporal_data.py
```

> **Note**: This script will print the mapped dimensions of your features (e.g., `Pol_Dim=95`) during execution.

### 3. Train GAP-TGN

Train the model. The script automatically detects the feature dimensions in `temporal_data.pt` and adjusts the model architecture accordingly.

```bash
# Using the installed entry point
chocolate-train --horizon 6M --epochs 5 --seed 42

# Or directly via python
python scripts/train_gap_tgn.py --horizon 6M
```

### 4. Train Baselines (Benchmarking)

Run comparative baselines (XGBoost, Logistic Regression, MLP) on the same data split.

```bash
chocolate-baselines --horizon 6M --start-year 2019 --end-year 2024
```

### 5. Evaluate Results

Results are saved to `results/experiments/` and `results/baselines/`.

## Directory Structure

* **src/gap_tgn.py**: Core TGN model definition (includes `ResearchTGN` and `encode_node_features`).
* **src/temporal_data.py**: Data builder that integrates `src/data_processing` lookups.
* **src/data_processing/**: Contains `feature_lookups.py` for handling bio, ideology, and financial data.
* **src/config.py**: Global configuration and feature flags.
* **scripts/**: Operational scripts for training and evaluation.
* **data/**: Data storage (Parquet files are git-ignored).

## Citation

If you use this code found in `src/gap_tgn.py`, please cite the accompanying workshop paper: