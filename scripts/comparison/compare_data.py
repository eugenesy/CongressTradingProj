import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add the project root to sys.path for direct execution
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import utility functions
from scripts.utils import load_csv_with_path

# Configuration
INITIAL_CSV = '../data/v5_transactions_with_approp_ticker.csv'
FINAL_CSV = '../data/v9_transactions.csv'
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'comparison_results'))
REPORT_FILE = os.path.join(OUTPUT_DIR, 'comparison_report.md')

def generate_report_header():
    return """
# Data Processing Comparison Report

This report details the changes and transformations applied to the financial transaction data throughout the processing pipeline, comparing the initial `v5_transactions_with_approp_ticker.csv` with the final `v9_transactions.csv`.

"""

def compare_data(
    initial_csv=INITIAL_CSV,
    final_csv=FINAL_CSV,
    output_dir=OUTPUT_DIR,
    report_file=REPORT_FILE
):
    print("Starting data comparison...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load datasets
    df_initial = load_csv_with_path(initial_csv)
    df_final = load_csv_with_path(final_csv)

    report_content = generate_report_header()

    # 1. Basic Information
    report_content += "## 1. Basic Data Information\n\n"
    report_content += f"### Initial Data (`{os.path.basename(initial_csv)}`)\n"
    report_content += f"- Rows: {len(df_initial)}\n"
    report_content += f"- Columns: {len(df_initial.columns)}\n"
    report_content += f"- Column Names: {', '.join(df_initial.columns.tolist())}\n\n"

    report_content += f"### Final Data (`{os.path.basename(final_csv)}`)\n"
    report_content += f"- Rows: {len(df_final)}\n"
    report_content += f"- Columns: {len(df_final.columns)}\n"
    report_content += f"- Column Names: {', '.join(df_final.columns.tolist())}\n\n"

    # 2. Column Differences
    report_content += "## 2. Column Differences\n\n"
    initial_cols = set(df_initial.columns)
    final_cols = set(df_final.columns)

    added_cols = list(final_cols - initial_cols)
    removed_cols = list(initial_cols - final_cols)
    common_cols = list(initial_cols.intersection(final_cols))

    report_content += f"- Columns Added: {', '.join(added_cols) if added_cols else 'None'}\n"
    report_content += f"- Columns Removed: {', '.join(removed_cols) if removed_cols else 'None'}\n"
    report_content += f"- Common Columns: {', '.join(common_cols) if common_cols else 'None'}\n\n"

    # 3. Row Count Change
    report_content += "## 3. Row Count Change\n\n"
    row_diff = len(df_initial) - len(df_final)
    report_content += f"- Number of rows removed during processing: {row_diff}\n\n"

    # 4. Specific Column Analysis
    report_content += "## 4. Detailed Column Analysis\n\n"

    # Transaction Column
    report_content += "### Transaction Column Changes\n\n"
    initial_transactions = df_initial['Transaction'].value_counts()
    final_transactions = df_final['Transaction'].value_counts()

    report_content += "#### Initial Transaction Distribution\n"
    report_content += f"```\n{initial_transactions}\n```\n\n"
    report_content += "#### Final Transaction Distribution\n"
    report_content += f"```\n{final_transactions}\n```\n\n"

    # Trade_Size_USD vs Standardized_Trade_Size
    if 'Trade_Size_USD' in df_initial.columns and 'Standardized_Trade_Size' in df_final.columns:
        report_content += "### Trade Size Standardization\n\n"
        report_content += "The `Trade_Size_USD` column from the initial data was standardized into `Standardized_Trade_Size` in the final data.\n\n"
        report_content += "#### Initial Trade Size Distribution (`Trade_Size_USD`)\n"
        report_content += f"```\n{df_initial['Trade_Size_USD'].value_counts().head()}\n```\n\n"
        report_content += "#### Final Standardized Trade Size Distribution (`Standardized_Trade_Size`)\n"
        report_content += f"```\n{df_final['Standardized_Trade_Size'].value_counts()}\n```\n\n"

        # Visualization for Trade Size
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        df_initial['Trade_Size_USD'].value_counts().head(10).plot(kind='bar')
        plt.title('Initial Trade Size Distribution (Top 10)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.subplot(1, 2, 2)
        df_final['Standardized_Trade_Size'].value_counts().plot(kind='bar')
        plt.title('Final Standardized Trade Size Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plot_path = os.path.join(output_dir, 'trade_size_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        report_content += f"![Trade Size Distribution]({os.path.basename(plot_path)})\n\n"

    # Party Column
    if 'Party' in df_initial.columns and 'Party' in df_final.columns:
        report_content += "### Party Affiliation Standardization\n\n"
        report_content += "The `Party` column abbreviations were mapped to full names.\n\n"
        report_content += "#### Initial Party Distribution\n"
        report_content += f"```\n{df_initial['Party'].value_counts()}\n```\n\n"
        report_content += "#### Final Party Distribution\n"
        report_content += f"```\n{df_final['Party'].value_counts()}\n```\n\n"

        # Visualization for Party
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        df_initial['Party'].value_counts().plot(kind='bar')
        plt.title('Initial Party Distribution')
        plt.xticks(rotation=45, ha='right')

        plt.subplot(1, 2, 2)
        df_final['Party'].value_counts().plot(kind='bar')
        plt.title('Final Party Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plot_path = os.path.join(output_dir, 'party_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        report_content += f"![Party Distribution]({os.path.basename(plot_path)})\n\n"

    # ID Column
    if 'ID' in df_final.columns:
        report_content += "### Transaction ID Addition\n\n"
        report_content += "A unique `ID` column was added to the final dataset.\n\n"
        report_content += f"- Example IDs: {df_final['ID'].head().tolist()}\n\n"

    # Label Columns
    label_cols = [col for col in df_final.columns if col.startswith('Label_')]
    if label_cols:
        report_content += "### Trading Labels\n\n"
        report_content += "New binary trading labels (`Label_1M`, `Label_3M`, `Label_6M`) were added.\n\n"
        for col in label_cols:
            report_content += f"#### {col} Distribution\n"
            report_content += f"```\n{df_final[col].value_counts(dropna=False)}\n```\n\n"
            # Visualization for Labels
            plt.figure(figsize=(6, 4))
            df_final[col].value_counts(dropna=False).plot(kind='bar')
            plt.title(f'{col} Distribution')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'{col.lower()}_distribution.png')
            plt.savefig(plot_path)
            plt.close()
            report_content += f"![{col} Distribution]({os.path.basename(plot_path)})\n\n"

    # Excess Return Columns
    excess_return_cols = [col for col in df_final.columns if col.startswith('Excess_Return_')]
    if excess_return_cols:
        report_content += "### Excess Returns\n\n"
        report_content += "Excess return columns (`Excess_Return_1M`, `Excess_Return_3M`, `Excess_Return_6M`) were calculated.\n\n"
        for col in excess_return_cols:
            report_content += f"#### {col} Summary Statistics\n"
            report_content += f"```\n{df_final[col].describe()}\n```\n\n"
            # Visualization for Excess Returns
            plt.figure(figsize=(8, 5))
            sns.histplot(df_final[col].dropna(), kde=True)
            plt.title(f'{col} Distribution')
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'{col.lower()}_distribution.png')
            plt.savefig(plot_path)
            plt.close()
            report_content += f"![{col} Distribution]({os.path.basename(plot_path)})\n\n"

    # 5. Missing Values Analysis
    report_content += "## 5. Missing Values Analysis\n\n"
    missing_initial = df_initial.isnull().sum()
    missing_final = df_final.isnull().sum()

    missing_initial = missing_initial[missing_initial > 0].sort_values(ascending=False)
    missing_final = missing_final[missing_final > 0].sort_values(ascending=False)

    report_content += "### Initial Data Missing Values\n"
    if not missing_initial.empty:
        report_content += f"```\n{missing_initial}\n```\n\n"
    else:
        report_content += "No missing values in initial data.\n\n"

    report_content += "### Final Data Missing Values\n"
    if not missing_final.empty:
        report_content += f"```\n{missing_final}\n```\n\n"
    else:
        report_content += "No missing values in final data.\n\n"

    # Save the report
    with open(report_file, 'w') as f:
        f.write(report_content)

    print(f"Comparison report generated at {report_file}")
    print("Data comparison completed.")


if __name__ == "__main__":
    compare_data()
