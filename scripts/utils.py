# utils.py

import pickle
import os
import pandas as pd

# Placeholder for common utility functions

def save_checkpoint(data, checkpoint_file):
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved to {checkpoint_file}")

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded checkpoint from {checkpoint_file}")
        return data
    return None

def load_csv_with_path(file_path, **kwargs):
    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
    return pd.read_csv(full_path, **kwargs)

def save_csv_with_path(df, file_path, **kwargs):
    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
    df.to_csv(full_path, **kwargs)

