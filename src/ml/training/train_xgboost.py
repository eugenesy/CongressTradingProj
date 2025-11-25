"""XGBoost model training script."""

import xgboost as xgb
from src.utils import run_sliding_window_training, get_data_path


def train_xgboost():
    """Trains and evaluates an XGBoost model with a rolling window."""
    input_path = get_data_path("processed", "ml_dataset_preprocessed.csv")
    
    model_factory = lambda: xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    run_sliding_window_training(
        model_factory=model_factory,
        model_name="xgboost",
        input_path=str(input_path)
    )


if __name__ == "__main__":
    train_xgboost()
