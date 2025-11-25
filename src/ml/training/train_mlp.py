"""MLP (Multi-Layer Perceptron) model training script."""

from sklearn.neural_network import MLPClassifier
from src.utils import run_sliding_window_training, get_data_path


def train_mlp():
    """Trains and evaluates an MLP model with a rolling window."""
    input_path = get_data_path("processed", "ml_dataset_preprocessed.csv")
    
    model_factory = lambda: MLPClassifier(random_state=42, max_iter=1000)
    
    run_sliding_window_training(
        model_factory=model_factory,
        model_name="mlp",
        input_path=str(input_path)
    )


if __name__ == "__main__":
    train_mlp()
