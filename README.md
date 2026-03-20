# Fintech Signal Isolation Repository

An enterprise-grade, path-dependent evaluation framework for dynamic Graph Neural Networks (GNNs) on continuous congressional trade topologies.

## Project Structure
```text
fintech_repo/
├── data/                       <- Data handling (Datasets intentionally excluded from vcs)
│   ├── raw/                    <- User drops `new_v5_transactions_with_committee_indsutry.csv` here
│   ├── processed/              <- Outputs like `ml_dataset_v2.csv` land here
│   └── parquet/                <- Ticker parquet dependencies 
├── src/                        <- Core modular source code
│   ├── data_prep/              <- Data conversion logic
│   ├── models/                 <- Core algorithmic implementations (baselines & gnn)
│   ├── evaluation/             <- Aggregation and metric engines
│   └── visualization/          <- Plotting and boundary analysis scripts
├── scripts/                    <- Executable pipelines (The Runner Layer)
├── results/                    <- Grouped historical outputs for easy reference
├── docs/                       <- Presentation and paper assets
└── legacy/                     <- Preserved historical frameworks (e.g. TGN)
```

## Dataset Dependencies (Critical)
To preserve repository scalability, massive dataset CSVs and `.pt` generated tensor maps are ignored by `.gitignore`. 

**To fully execute this pipeline from scratch, you must provide:**
1. `new_v5_transactions_with_committee_indsutry.csv` placed into `data/raw/` (or run it via the path provided to `scripts/run_*` via flags).
2. Ticker Parquet directories (optional but recommended for precise SPY baselines) placed in `data/parquet/`.

## Pipeline Execution Order
1. **Data Preprocessing**: `src/data_prep/build_clean_dataset.py` builds the V2 cleanly encoded `ml_dataset_v2.csv` and computes forward paths.
2. **Tabular Baselines**: `scripts/run_path_dependent_ml.py` builds standardized XGBoost trees.
3. **Continuous GNN Core**: `scripts/run_gnn_continuous_reach.py` initiates the Continuous Reach bounding logic across GNN permutations.
4. **Aggregation**: Process raw output logs sequentially via `src/evaluation/aggregators/live_aggregator.py`.

*Refactored & Audited: March 2026*
