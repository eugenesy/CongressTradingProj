"""
Entry point script to run all ML experiments.
"""

import sys
from pathlib import Path
# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.training.train_knn import train_knn
from src.ml.training.train_lightgbm import train_lightgbm
from src.ml.training.train_logistic_regression import train_logistic_regression
from src.ml.training.train_random_forest import train_random_forest
from src.ml.training.train_mlp import train_mlp
from src.ml.create_ml_dataset import create_ml_dataset
from src.ml.preprocess import preprocess_data
from src.utils import get_data_path
import os


def main():
    """Main function to orchestrate the ML workflow."""
    print("Starting ML experiments workflow...")
    
    # Step 1: Create the ML dataset
    input_csv = get_data_path("processed", "v9_transactions.csv")
    ml_dataset_csv = get_data_path("processed", "ml_dataset_reduced_attributes.csv")
    
    if not ml_dataset_csv.exists():
        print("Creating ML dataset...")
        create_ml_dataset(str(input_csv), str(ml_dataset_csv))
    else:
        print(f"Skipping ML dataset creation: {ml_dataset_csv} already exists.")

    # Step 2: Preprocess the data
    preprocessed_csv = get_data_path("processed", "ml_dataset_preprocessed.csv")
    
    if not preprocessed_csv.exists():
        print("Preprocessing data...")
        preprocess_data(str(ml_dataset_csv), str(preprocessed_csv))
    else:
        print(f"Skipping preprocessing: {preprocessed_csv} already exists.")

    # Step 3: Train models
    print("\nTraining models...")
    
    print("\n=== Training LightGBM ===")
    train_lightgbm()
    
    print("\n=== Training Logistic Regression ===")
    train_logistic_regression()
    
    print("\n=== Training Random Forest ===")
    train_random_forest()
    
    print("\n=== Training KNN ===")
    train_knn()
    
    print("\n=== Training MLP ===")
    train_mlp()

    # Optional models (require additional libraries)
    try:
        import catboost
        print("\n=== Training CatBoost ===")
        from src.ml.training.train_catboost import train_catboost
        train_catboost()
    except ImportError:
        print("\nSkipping CatBoost: library not installed. Install with 'pip install catboost'")

    try:
        import xgboost
        print("\n=== Training XGBoost ===")
        from src.ml.training.train_xgboost import train_xgboost
        train_xgboost()
    except ImportError:
        print("\nSkipping XGBoost: library not installed. Install with 'pip install xgboost'")

    print("\n✅ ML experiments workflow completed successfully!")


if __name__ == "__main__":
    main()
