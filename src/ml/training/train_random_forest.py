"""Random Forest model training script."""

from sklearn.ensemble import RandomForestClassifier
from src.utils import run_sliding_window_training, get_data_path


def train_random_forest():
    """Trains and evaluates a Random Forest model with a rolling window."""
    input_path = get_data_path("processed", "ml_dataset_preprocessed.csv")
    
    model_factory = lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    run_sliding_window_training(
        model_factory=model_factory,
        model_name="random_forest",
        input_path=str(input_path)
    )


if __name__ == "__main__":
    train_random_forest()
