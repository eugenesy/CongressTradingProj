"""CatBoost model training script."""

from catboost import CatBoostClassifier
from src.utils import run_sliding_window_training, get_data_path


def train_catboost():
    """Trains and evaluates a CatBoost model with a rolling window."""
    input_path = get_data_path("processed", "ml_dataset_preprocessed.csv")
    
    model_factory = lambda: CatBoostClassifier(random_state=42, verbose=0)
    
    run_sliding_window_training(
        model_factory=model_factory,
        model_name="catboost",
        input_path=str(input_path)
    )


if __name__ == "__main__":
    train_catboost()
