# Capitol Gains: GAP-TGN (Graph Alpha Prediction for Congressional Trading)

**GAP-TGN** is the specialized Temporal Graph Network (TGN) driving the Capitol Gains project, designed to detect alpha signals in congressional trading disclosures. Unlike traditional tabular methods, GAP-TGN explicitly models the evolving network of legislators and corporate entities, using an asynchronous propagation strategy to handle reporting delays.

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

### 1. Download Processed Data (Recommended)
To get started quickly without handling 90GB of raw files, you can download the pre-compiled, processed dataset. 

1. Download the processed data zip file from this Google Drive folder: [Data Repository](https://drive.google.com/drive/u/0/folders/1oYfCPYj5FR3GoIQp2zWkE4IQ1I5yIP7V)
2. Unzip the contents directly into your project's `data/` directory. It should contain the `processed/` subfolder, `temporal_data.pt`, and `price_sequences.pt`.

**Pre-configured Feature Flags:**
Because the graph features are baked directly into the tensor data during generation, this processed dataset comes with the following configuration hard-coded:

```python
INCLUDE_POLITICIAN_BIO = True 
INCLUDE_IDEOLOGY = True
INCLUDE_COMMITTEES = True
INCLUDE_COMPANY_SIC = True
INCLUDE_DISTRICT_ECON = False
INCLUDE_COMPANY_FINANCIALS = False
INCLUDE_LOBBYING_SPONSORSHIP = True
INCLUDE_LOBBYING_VOTING = True
INCLUDE_CAMPAIGN_FINANCE = True
```

### 2. Customizing Data Features (Raw Data)
If you would like to re-generate the processed data with a different combination of the feature flags listed above, you will need the full 90GB raw dataset. **Please reach out to the project creators to request access to the raw data files.**

Once obtained, you will place the raw files in the `data/raw/` directory, adjust the flags in `src/config.py`, and run `python src/temporal_data.py` to build a custom `temporal_data.pt` file.

## Usage

### 1. Train GAP-TGN

Assuming you have downloaded the processed data, you can immediately train the model. The script automatically detects the feature dimensions in `temporal_data.pt` and adjusts the model architecture accordingly.

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

* **src/gap_tgn.py**: Core TGN model definition (includes `ResearchTGN` and `encode_node_features`).
* **src/temporal_data.py**: Data builder that integrates `src/data_processing` lookups.
* **src/data_processing/**: Contains `feature_lookups.py` for handling bio, ideology, and financial data.
* **src/config.py**: Global configuration and feature flags.
* **scripts/**: Operational scripts for training and evaluation.
* **data/**: Data storage (Parquet files are git-ignored).

## Citation

If you use this code found in `src/gap_tgn.py`, please cite the accompanying workshop paper.