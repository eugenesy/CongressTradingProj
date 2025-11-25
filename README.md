# Congressional Trading Analysis

A Python-based financial data analysis system for processing and analyzing congressional trading transaction data using machine learning.

---

## 📋 Table of Contents
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Features](#features)
- [Data Pipeline](#data-pipeline)
- [Machine Learning](#machine-learning)
- [Development](#development)
- [Contributing](#contributing)

---

## 📁 Project Structure

```
apple/
├── bin/                        # Executable entry points
│   ├── run_pipeline.py         # Run data processing pipeline
│   └── run_experiments.py      # Run ML experiments
├── data/                       # All data files (gitignored)
│   ├── raw/                    # Raw transaction data (CSV files)
│   ├── processed/              # Cleaned data at various stages
│   │   ├── v5-v9_transactions.csv   # Progressive cleaning stages
│   │   ├── ml_dataset_reduced_attributes.csv
│   │   └── ml_dataset_preprocessed.csv
│   ├── parquet/                # Historical stock prices (3000+ files)
│   │   ├── SPY.parquet         # S&P 500 benchmark
│   │   └── *.parquet           # Individual stock data
│   ├── models/                 # Trained ML models (.joblib)
│   └── results/                # Experiment results
│       └── <model_name>/       # Per-model subdirectories
│           ├── <year>/         # Per-year results
│           │   ├── metrics.csv
│           │   └── predictions.json
│           ├── all_predictions.json
│           └── summary_metrics.csv
├── src/                        # Source code
│   ├── data_pipeline/          # ETL pipeline modules
│   │   ├── download_tickers.py      # Download stock prices
│   │   ├── download_spy.py          # Download SPY benchmark
│   │   ├── add_spy_columns.py       # Add benchmark data
│   │   ├── add_closing_prices.py    # Enrich with prices
│   │   ├── add_excess_returns.py    # Calculate returns vs SPY
│   │   ├── add_trading_labels.py    # Generate profit labels
│   │   ├── clean_data.py            # Data cleaning
│   │   ├── add_transaction_ids.py   # ID assignment
│   │   └── embedding_enhanced.py    # Generate embeddings
│   ├── ml/                     # Machine learning
│   │   ├── training/           # Model training scripts
│   │   │   ├── train_knn.py
│   │   │   ├── train_lightgbm.py
│   │   │   ├── train_logistic_regression.py
│   │   │   ├── train_random_forest.py
│   │   │   ├── train_mlp.py
│   │   │   ├── train_catboost.py
│   │   │   └── train_xgboost.py
│   │   ├── create_ml_dataset.py     # Feature selection
│   │   └── preprocess.py            # Feature engineering
│   ├── analysis/               # Analysis tools
│   │   └── compare_data.py     # Data comparison utilities
│   └── utils.py                # Shared utilities
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

### Directory Descriptions

**`bin/`** - Command-line executable scripts
- Entry points for running workflows
- Not imported as modules, executed directly

**`data/`** - All project data (excluded from git)
- Organized by processing stage and purpose
- Contains raw, processed, and ML-ready datasets
- Historical stock data in efficient parquet format
- Trained models and experiment results

**`src/`** - Source code organized by functionality
- `data_pipeline/`: ETL operations (download, enrich, clean)
- `ml/`: Machine learning (preprocessing, training)
- `analysis/`: Data analysis and comparison tools
- `utils.py`: Shared utilities (paths, I/O, evaluation, training)

---

## 🚀 Quick Start

### Prerequisites

**Required Data Files (Not Included in Repository)**

Due to file size limitations, the following data files are hosted on Google Drive:

📂 **Google Drive Location:**  
`FinTech Insider Trading Project/For Baseline/`

**Required Files:**
1. **`v5_transactions_with_approp_ticker.csv`** - Initial transaction data
2. **`parquet/`** folder - Historical stock price data (3000+ files)

### Installation

#### Step 1: Clone the Repository
```bash
git clone https://github.com/eugenesy/CongressTradingProj.git
cd CongressTradingProj
```

#### Step 2: Install Dependencies
```bash
pip install -r requirements.txt

# Optional: Install additional models
pip install catboost xgboost
```

#### Step 3: Download Required Data Files

1. **Access Google Drive:**
   - Navigate to: `FinTech Insider Trading Project/For Baseline/`
   
2. **Download the following:**
   - `v5_transactions_with_approp_ticker.csv`
   - `parquet/` folder (entire folder with all .parquet files)

3. **Place files in the correct locations:**
   ```bash
   # Create data directories if they don't exist
   mkdir -p data/processed
   mkdir -p data/parquet
   
   # Move the downloaded files:
   # - Move v5_transactions_with_approp_ticker.csv to data/processed/
   # - Move all .parquet files to data/parquet/
   ```

   Your directory structure should look like:
   ```
   apple/
   ├── data/
   │   ├── processed/
   │   │   └── v5_transactions_with_approp_ticker.csv
   │   └── parquet/
   │       ├── SPY.parquet
   │       ├── AAPL.parquet
   │       └── ... (3000+ files)
   ```

### Usage

#### First-Time Setup: Run Data Processing Pipeline
```bash
python bin/run_pipeline.py
```

**Prerequisites:** Make sure you've downloaded the required data files from Google Drive (see Step 3 above).

This will:
1. Load transaction data from `data/processed/v5_transactions_with_approp_ticker.csv`
2. Load historical stock data from `data/parquet/`
3. Download SPY benchmark data
4. Enrich transactions with prices and returns
5. Add trading labels (profitable/unprofitable)
6. Clean and standardize data
7. Generate final dataset: `data/processed/v9_transactions.csv`

#### Run ML Experiments
```bash
python bin/run_experiments.py
```

This will:
1. Create ML dataset from processed transactions
2. Preprocess features (encoding, scaling, engineering)
3. Train 7 models with sliding window cross-validation
4. Save predictions and metrics

**Trained Models:**
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Random Forest
- LightGBM
- Multi-Layer Perceptron (MLP)
- CatBoost (optional)
- XGBoost (optional)

---

## ✨ Features

### Data Pipeline
- **Historical Data Download**: Loads stock prices from local parquet files
- **Benchmark Integration**: Adds S&P 500 (SPY) data for comparison
- **Return Calculation**: Computes excess returns (stock return - SPY return)
- **Label Generation**: Creates binary labels for profitable trades
- **Data Cleaning**: Removes duplicates, standardizes values
- **Checkpointing**: Resumes from failure points

### ML Experiments
- **Sliding Window Evaluation**: Uses concrete calendar years (e.g., train on 2020-2024, test on 2025)
- **Robust Cross-Validation**: Prevents data leakage
- **Transaction-Level Predictions**: Each prediction linked to transaction ID
- **Aggregated Results**: Combined predictions across all test years
- **Comprehensive Metrics**: Precision, recall, F1-score by class

### Prediction Output Format
```json
{
  "transaction_id": {
    "prob": [0.3, 0.7],
    "pred": 1.0,
    "true_value": 1.0
  }
}
```

---

## 📊 Data Pipeline

### Processing Stages

```
v5_transactions_with_approp_ticker.csv    (Initial cleaned data)
           ↓  + SPY prices
v5_transactions_with_benchmark.csv         (With SPY benchmark)
           ↓  + closing prices
v6_transactions.csv                        (With stock prices)
           ↓  + excess returns
v7_transactions.csv                        (With returns calculated)
           ↓  + trading labels
v8_transactions.csv                        (With profit labels)
           ↓  + cleaning
v8_transactions_final_cleaned.csv          (Cleaned data)
           ↓  + transaction IDs
v9_transactions.csv                        (Final dataset)
           ↓  + feature selection
ml_dataset_reduced_attributes.csv          (ML subset)
           ↓  + preprocessing
ml_dataset_preprocessed.csv                (Ready for models)
```

### Label Definition
A transaction is labeled **profitable (1)** if:
- **Purchase** AND excess return > 6%, OR
- **Sale** AND excess return < 6%

Otherwise labeled **unprofitable (0)**

---

## 🤖 Machine Learning

### Model Training

All models use a common training function that:
1. Loads preprocessed data
2. Sorts by filing date
3. Creates sliding windows (5 years train, 1 year test)
4. Aligns features between train/test sets
5. Trains model and evaluates
6. Saves predictions with transaction IDs

### Adding a New Model

Create `src/ml/training/train_<model_name>.py`:

```python
from src.utils import run_sliding_window_training, get_data_path
from your_library import YourModel

def train_your_model():
    """Trains and evaluates your model with a rolling window."""
    input_path = get_data_path("processed", "ml_dataset_preprocessed.csv")
    
    model_factory = lambda: YourModel(param1=value1, param2=value2)
    
    run_sliding_window_training(
        model_factory=model_factory,
        model_name="your_model",
        input_path=str(input_path)
    )

if __name__ == "__main__":
    train_your_model()
```

Then add to `bin/run_experiments.py`:
```python
from src.ml.training.train_your_model import train_your_model

print("\n=== Training Your Model ===")
train_your_model()
```

### Results Structure

For each model and test year:
- `metrics.csv`: Classification metrics (precision, recall, F1)
- `predictions.json`: Per-transaction predictions with probabilities
- `all_predictions.json`: Aggregated predictions across all years
- `summary_metrics.csv`: Average metrics across all test years

---

## 🛠️ Development

### Project Conventions

- **Modular Design**: Each component has a single responsibility
- **Path Portability**: All paths use `get_data_path()` for consistency
- **No Hardcoded Paths**: Works on any team member's machine
- **Reusable Training Logic**: Common `run_sliding_window_training()` eliminates duplication
- **Type Consistency**: Path objects converted to strings for function calls

### Code Quality Improvements

| Aspect | Before | After | Improvement |
|--------|--------|-------|-------------|
| Training script lines | ~90 each | ~20 each | **-78%** |
| Total duplicated code | ~630 lines | ~270 lines | **-57%** |
| Hardcoded paths | Many | 0 | **100% removed** |
| `sys.path` hacks | Many | 0 | **100% removed** |

### Path Management

All file paths use the centralized helper:

```python
from src.utils import get_data_path

# Get paths
parquet_dir = get_data_path('parquet')
input_csv = get_data_path('processed', 'v9_transactions.csv')
model_path = get_data_path('models', 'knn_model.joblib')
```

### Running Individual Components

```bash
# Run specific data processing steps
python -m src.data_pipeline.download_tickers
python -m src.data_pipeline.clean_data

# Train individual models
python -m src.ml.training.train_knn
python -m src.ml.training.train_lightgbm
```

---

## 🤝 Contributing

### For Team Members

1. **Clone and setup:**
   ```bash
   git clone <repo-url>
   cd apple
   pip install -r requirements.txt
   ```

2. **Create your branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make changes** following the project conventions

4. **Test your changes:**
   ```bash
   python bin/run_experiments.py  # or specific scripts
   ```

5. **Commit and push:**
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin feature/your-feature-name
   ```

### Code Style

- Follow PEP 8
- Use type hints where applicable
- Document functions with docstrings
- Keep functions focused and single-purpose

---

## 📝 License

[Your License Here]

---

## 🙏 Acknowledgments

- Data source: [Your data source]
- Built with Python, scikit-learn, LightGBM, and other open-source tools

---

**Ready for collaboration! 🚀**
