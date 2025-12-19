
import pandas as pd

# Load Results
try:
    df = pd.read_csv('results/ablation_monthly_breakdown.csv')
except FileNotFoundError:
    print("Error: Results file not found.")
    exit()

# Group by Mode and Year
yearly_stats = df.groupby(['AblationMode', 'Year'])[['F1', 'ROC_AUC']].mean().reset_index()

# Pivot for better readability
f1_table = yearly_stats.pivot(index='Year', columns='AblationMode', values='F1')
auc_table = yearly_stats.pivot(index='Year', columns='AblationMode', values='ROC_AUC')

print("\n=== Yearly Average F1 Score ===")
print(f1_table[['pol_only', 'mkt_only', 'full']].round(4))

print("\n=== Yearly Average ROC AUC ===")
print(auc_table[['pol_only', 'mkt_only', 'full']].round(4))

# Calculate Overall Mean across all years
print("\n=== Overall Mean (2019-2024) ===")
overall = df.groupby('AblationMode')[['F1', 'ROC_AUC']].mean()
print(overall.round(4))
