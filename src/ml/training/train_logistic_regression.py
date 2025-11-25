"""Logistic Regression model training script."""

from sklearn.linear_model import LogisticRegression
from src.utils import run_sliding_window_training, get_data_path


def train_logistic_regression():
    """Trains and evaluates a Logistic Regression model with a rolling window."""
    input_path = get_data_path("processed", "ml_dataset_preprocessed.csv")
    
    model_factory = lambda: LogisticRegression(max_iter=1000, random_state=42)
    
    run_sliding_window_training(
        model_factory=model_factory,
        model_name="logistic_regression",
        input_path=str(input_path)
    )


if __name__ == "__main__":
    train_logistic_regression()
