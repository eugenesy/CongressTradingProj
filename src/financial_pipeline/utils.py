"""
Consolidated utility functions for the financial data analysis project.
"""

import pandas as pd
import joblib
from sklearn.metrics import classification_report
import os
import json
import pickle
from pathlib import Path


def get_project_root():
    """Returns the absolute path to the chocolate project root directory."""
    # src/financial_pipeline/utils.py -> src/ -> chocolate/
    return Path(__file__).parent.parent.parent.absolute()


def get_data_path(*args):
    """Returns a path relative to the data directory."""
    return get_project_root() / "data" / Path(*args)


# ===== Data I/O =====

def load_data(file_path):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)


def load_csv_with_path(file_path, **kwargs):
    """Loads CSV with path resolution."""
    full_path = get_project_root() / file_path
    return pd.read_csv(full_path, **kwargs)


def save_csv_with_path(df, file_path, **kwargs):
    """Saves CSV with path resolution."""
    full_path = get_project_root() / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(full_path, **kwargs)


# ===== Checkpointing =====

def save_checkpoint(data, checkpoint_file):
    """Saves a checkpoint to disk."""
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved to {checkpoint_file}")


def load_checkpoint(checkpoint_file):
    """Loads a checkpoint from disk."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded checkpoint from {checkpoint_file}")
        return data
    return None


# ===== Model Management =====

def save_model(model, file_path):
    """Saves a trained model to a file."""
    joblib.dump(model, file_path)


def load_model(file_path):
    """Loads a trained model from a file."""
    return joblib.load(file_path)


# ===== Evaluation & Results =====

def evaluate_model(y_true, y_pred):
    """Evaluates a model and returns a classification report."""
    return classification_report(y_true, y_pred, output_dict=True)


def save_results(model_name, test_year, metrics, y_true, y_pred, y_prob, transaction_ids=None, results_base_dir=None):
    """Saves the model's results to a year-specific subdirectory."""
    if results_base_dir is None:
        results_base_dir = get_data_path("results")
    
    results_dir = results_base_dir / model_name / str(test_year)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_df = pd.DataFrame(metrics).transpose()
    metrics_df.to_csv(results_dir / "metrics.csv")

    # Save predictions
    predictions = {}
    
    y_true_list = y_true.tolist()
    y_pred_list = y_pred.tolist()
    y_prob_list = y_prob.tolist()
    
    if transaction_ids is not None:
        t_ids_list = transaction_ids.tolist()
        for i, t_id in enumerate(t_ids_list):
            predictions[str(t_id)] = {
                "prob": y_prob_list[i],
                "pred": y_pred_list[i],
                "true_value": y_true_list[i]
            }
    else:
        # Fallback if no transaction IDs
        predictions = {
            "true_label": y_true_list,
            "predicted_label": y_pred_list,
            "predicted_probability": y_prob_list
        }

    with open(results_dir / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)
        
    return predictions


def save_aggregated_predictions(model_name, all_predictions, results_base_dir=None):
    """Saves the aggregated predictions to the model directory."""
    if results_base_dir is None:
        results_base_dir = get_data_path("results")
    
    results_dir = results_base_dir / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "all_predictions.json", "w") as f:
        json.dump(all_predictions, f, indent=4)


def save_summary_results(model_name, all_metrics, results_base_dir=None):
    """Calculates and saves the summary of metrics across all years."""
    if results_base_dir is None:
        results_base_dir = get_data_path("results")
    
    flat_metrics = []
    for report in all_metrics:
        weighted_avg = report.get('weighted avg', {})
        flat_metrics.append({
            'precision': weighted_avg.get('precision'),
            'recall': weighted_avg.get('recall'),
            'f1-score': weighted_avg.get('f1-score'),
        })

    if not flat_metrics:
        return

    summary_df = pd.DataFrame(flat_metrics).mean().to_frame().T
    
    results_dir = results_base_dir / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(results_dir / "summary_metrics.csv", index=False)


# ===== Training Utilities =====

def run_sliding_window_training(
    model_factory,
    model_name,
    input_path,
    train_window=5,
    results_base_dir=None
):
    """
    Run sliding window training and evaluation for any sklearn-compatible model.
    
    Args:
        model_factory: A callable that returns a new model instance
        model_name: Name of the model (for saving results)
        input_path: Path to the preprocessed dataset
        train_window: Number of years to use for training (default: 5)
        results_base_dir: Base directory for results (default: data/results)
    
    Returns:
        The last trained model instance
    """
    df = load_data(input_path)
    df = df.sort_values(by=["Filed_year", "Filed_month", "Filed_day"])

    all_metrics = []
    all_predictions = {}

    min_year = df["Filed_year"].min()
    max_year = df["Filed_year"].max()

    for year in range(min_year, max_year - train_window + 1):
        train_years = list(range(year, year + train_window))
        test_year = year + train_window

        print(f"Training on years: {train_years}, Testing on year: {test_year}")

        train_df = df[df["Filed_year"].isin(train_years)]
        test_df = df[df["Filed_year"] == test_year]

        if test_df.empty or train_df.empty:
            print(f"Skipping year {test_year} due to no data.")
            continue

        # Extract features and labels
        if "transaction_id" in train_df.columns:
            X_train = train_df.drop(["Label_1M", "Filed_year", "Filed_month", "Filed_day", "transaction_id"], axis=1)
        else:
            X_train = train_df.drop(["Label_1M", "Filed_year", "Filed_month", "Filed_day"], axis=1)
            
        y_train = train_df["Label_1M"]

        if "transaction_id" in test_df.columns:
            transaction_ids = test_df["transaction_id"]
            X_test = test_df.drop(["Label_1M", "Filed_year", "Filed_month", "Filed_day", "transaction_id"], axis=1)
        else:
            transaction_ids = None
            X_test = test_df.drop(["Label_1M", "Filed_year", "Filed_month", "Filed_day"], axis=1)

        y_test = test_df["Label_1M"]

        # Align features between train and test
        train_cols = X_train.columns
        test_cols = X_test.columns

        missing_in_test = set(train_cols) - set(test_cols)
        for c in missing_in_test:
            X_test[c] = 0

        missing_in_train = set(test_cols) - set(train_cols)
        for c in missing_in_train:
            X_train[c] = 0

        X_test = X_test[train_cols]

        # Train and evaluate
        model = model_factory()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        metrics = evaluate_model(y_test, y_pred)
        all_metrics.append(metrics)
        print(f"Metrics for test year {test_year}: {metrics}")

        preds = save_results(model_name, test_year, metrics, y_test, y_pred, y_prob, transaction_ids, results_base_dir)
        all_predictions.update(preds)

    if all_metrics:
        print("\nSaved metrics and predictions for each window.")
        save_summary_results(model_name, all_metrics, results_base_dir)
        save_aggregated_predictions(model_name, all_predictions, results_base_dir)
        print("Saved summary of metrics across all windows.")

        # Save the last model
        model_path = get_data_path("models") / f"{model_name}_model.joblib"
        save_model(model, str(model_path))
        print(f"Saved the last trained model to {model_path}")
        return model
    else:
        print("No models were trained.")
        return None
