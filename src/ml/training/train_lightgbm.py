"""LightGBM model training script."""

import lightgbm as lgb
from src.utils import run_sliding_window_training, get_data_path


def train_lightgbm():
    """Trains and evaluates a LightGBM model with a rolling window."""
    input_path = get_data_path("processed", "ml_dataset_preprocessed.csv")
    
    model_factory = lambda: lgb.LGBMClassifier(random_state=42)
    
    run_sliding_window_training(
        model_factory=model_factory,
        model_name="lightgbm",
        input_path=str(input_path)
    )


if __name__ == "__main__":
    train_lightgbm()
