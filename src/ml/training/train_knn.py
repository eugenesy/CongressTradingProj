"""K-NN model training script."""

from sklearn.neighbors import KNeighborsClassifier
from src.utils import run_sliding_window_training, get_data_path


def train_knn():
    """Trains and evaluates a k-NN model with a rolling window."""
    input_path = get_data_path("processed", "ml_dataset_preprocessed.csv")
    
    model_factory = lambda: KNeighborsClassifier()
    
    run_sliding_window_training(
        model_factory=model_factory,
        model_name="knn",
        input_path=str(input_path)
    )


if __name__ == "__main__":
    train_knn()
