# GAP-TGN: Graph Alpha Prediction for Congressional Trading

**GAP-TGN** is a specialized Temporal Graph Network (TGN) designed to detect alpha signals in congressional trading disclosures. Unlike traditional tabular methods, GAP-TGN explicitly models the evolving network of legislators and corporate entities, using an asynchronous propagation strategy to handle reporting delays.

## Key Features
* **Asynchronous Propagation**: Updates node states during "gap" periods using resolved transactions, preventing data staleness.
* **Gated Multi-Modal Fusion**: Dynamically weights graph embeddings against immediate market signals.
* **Production-Ready Pipeline**: Clean separation of data processing, training, and evaluation.
* **Rich Feature Engineering**: Configurable inclusion of multi-modal data sources, including ideology, district economics, and corporate financials.

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

## Data Setup
**Crucial Step:** The raw datasets required for this project are hosted externally. You must download and place them correctly before running any scripts.

1.  Download the data archive from [Google Drive](https://drive.google.com/drive/folders/1NoA9RH53IeafKRcF0i7N78fUO47Np-F8).
2.  Unzip the downloaded content.
3.  Create the directory `data/raw/` if it does not exist.
4.  Move the unzipped files and folders into `data/raw/` so that your structure looks like this:

```text
data/
  raw/
    ├── ideology_scores_quarterly.csv
    ├── committee_assignments.csv
    ├── company_sic_data.csv
    ├── sec_quarterly_financials_unzipped.csv
    ├── congress_terms_all_github.csv
    └── district_industries/
          ├── 2011_CB_estimates.csv
          ├── ...
          └── survey_release_dates.csv
```

## Configuration
You can toggle the inclusion of specific data sources in the temporal graph by modifying flags in `src/config.py`. Adjust these before running the build step:

* `INCLUDE_IDEOLOGY`: Adds W-Nominate ideology scores (dim 1 & 2) for legislators.
* `INCLUDE_DISTRICT_ECON`: Adds employment data by industry sector for the legislator's district (updated annually).
* `INCLUDE_COMMITTEES`: Adds binary vectors representing committee assignments (updated per Congress).
* `INCLUDE_COMPANY_SIC`: Adds one-hot encoding for company industry sectors (SIC).
* `INCLUDE_COMPANY_FINANCIALS`: Adds quarterly financial metrics from SEC 10-Q filings.

## Usage

### 1. Build Dataset
Process raw transactions and configured features into the temporal graph format.
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

### 3. Train Baselines (Benchmarking)
Run comparative baselines (XGBoost, Logistic Regression, MLP) on the same data split.
```bash
chocolate-baselines --horizon 6M --start-year 2019 --end-year 2024
```

### 4. Evaluate Results
Results are saved to `results/experiments/` and `results/baselines/`.

## Directory Structure
* `src/gap_tgn.py`: Core TGN model definition.
* `src/config.py`: Configuration flags for feature engineering.
* `src/data_processing/`: Scripts for handling feature lookups and dataset construction.
* `scripts/`: Operational scripts for training and evaluation.
* `legacy/`: Archived code (ablation studies, old baselines).
* `data/`: Data storage (Parquet files are git-ignored).

## Citation
If you use this code found in `src/gap_tgn.py`, please cite the accompanying workshop paper.