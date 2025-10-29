import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the project root to sys.path for direct execution
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.utils import load_csv_with_path

# Configuration
INPUT_CSV = '../data/v9_transactions.csv'
OUTPUT_DIR = '../data/exploration_results/'

def explore_v9_data(input_csv=INPUT_CSV, output_dir=OUTPUT_DIR):
    print(f"Starting data exploration for {input_csv}...")

    # Ensure output directory exists
    os.makedirs(os.path.join(os.path.dirname(__file__), output_dir), exist_ok=True)

    # Load data
    df = load_csv_with_path(input_csv)

    # --- 1. Basic Data Overview ---
    print("\n--- Basic Data Overview ---")
    print("Shape:", df.shape)
    print("\nColumns and Data Types:")
    print(df.info())
    print("\nDescriptive Statistics:")
    print(df.describe(include='all'))

    # --- 2. Missing Values ---
    print("\n--- Missing Values Analysis ---")
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    if not missing_values.empty:
        print(missing_values)
        missing_values.plot(kind='barh', figsize=(10, 6))
        plt.title('Missing Values per Column')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), output_dir, 'missing_values.png'))
        plt.close()
    else:
        print("No missing values found.")

    # --- 3. Label Analysis ---
    print("\n--- Label Analysis ---")
    labels = ['Label_1M', 'Label_3M', 'Label_6M']
    for label_col in labels:
        if label_col in df.columns:
            print(f"\nDistribution of {label_col}:")
            print(df[label_col].value_counts(dropna=False))
            print(f"Percentage Distribution of {label_col}:")
            print(df[label_col].value_counts(dropna=False, normalize=True) * 100)

            plt.figure(figsize=(6, 4))
            sns.countplot(x=label_col, data=df, palette='viridis')
            plt.title(f'Distribution of {label_col}')
            plt.savefig(os.path.join(os.path.dirname(__file__), output_dir, f'{label_col}_distribution.png'))
            plt.close()

    # --- 4. Categorical Feature Analysis vs. Labels ---
    print("\n--- Categorical Feature Analysis vs. Labels ---")
    categorical_cols = ['Party', 'Transaction', 'Standardized_Trade_Size', 'Chamber']
    for col in categorical_cols:
        if col in df.columns:
            for label_col in labels:
                if label_col in df.columns:
                    print(f"\n{col} vs {label_col}:")
                    cross_tab = pd.crosstab(df[col], df[label_col], normalize='index') * 100
                    print(cross_tab)

                    cross_tab.plot(kind='bar', stacked=True, figsize=(10, 6))
                    plt.title(f'{col} vs {label_col}')
                    plt.ylabel('Percentage')
                    plt.tight_layout()
                    plt.savefig(os.path.join(os.path.dirname(__file__), output_dir, f'{col}_vs_{label_col}.png'))
                    plt.close()

    # --- 5. Numerical Feature Analysis ---
    print("\n--- Numerical Feature Analysis ---")
    numerical_cols = ['Excess_Return_1M', 'Excess_Return_3M', 'Excess_Return_6M']
    for col in numerical_cols:
        if col in df.columns:
            print(f"\nDescriptive Statistics for {col}:")
            print(df[col].describe())

            plt.figure(figsize=(8, 5))
            sns.histplot(df[col].dropna(), kde=True, bins=50)
            plt.title(f'Distribution of {col}')
            plt.savefig(os.path.join(os.path.dirname(__file__), output_dir, f'{col}_hist.png'))
            plt.close()

            plt.figure(figsize=(8, 5))
            sns.boxplot(y=df[col].dropna())
            plt.title(f'Box Plot of {col}')
            plt.savefig(os.path.join(os.path.dirname(__file__), output_dir, f'{col}_boxplot.png'))
            plt.close()

            # Numerical vs Label
            for label_col in labels:
                if label_col in df.columns:
                    plt.figure(figsize=(8, 5))
                    sns.boxplot(x=label_col, y=col, data=df, palette='viridis')
                    plt.title(f'{col} by {label_col}')
                    plt.savefig(os.path.join(os.path.dirname(__file__), output_dir, f'{col}_by_{label_col}_boxplot.png'))
                    plt.close()

    print(f"\nData exploration complete. Results saved to {os.path.join(os.path.dirname(__file__), output_dir)}")

if __name__ == "__main__":
    explore_v9_data()
