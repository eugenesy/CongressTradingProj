
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set aesthetic style for "Research Quality"
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
# plt.rcParams['font.family'] = 'serif'

# 1. Load Data
try:
    df = pd.read_csv('results/ablation_monthly_breakdown.csv')
    df['Date'] = pd.to_datetime(df.assign(Day=1)[['Year', 'Month', 'Day']])
except FileNotFoundError:
    print("Error: Results CSV not found.")
    exit()

os.makedirs('results/plots', exist_ok=True)

# Define Colors (Professional Palette)
colors = {"pol_only": "#d62728",    # Red
          "mkt_only": "#1f77b4",    # Blue
          "full": "#2ca02c"}        # Green

# === Plot 1: Yearly Average F1 Score (Bar Chart) ===
yearly_f1 = df.groupby(['Year', 'AblationMode'])['F1'].mean().reset_index()

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=yearly_f1, x='Year', y='F1', hue='AblationMode', palette=colors, dodge=True)
plt.title("Yearly Average F1 Score by Signal Source", fontsize=16, fontweight='bold', pad=20)
plt.ylabel("F1 Score")
plt.xlabel("Year")
plt.legend(title='Configuration', loc='upper left', bbox_to_anchor=(1, 1))
plt.ylim(0, 1.0)
sns.despine()
plt.tight_layout()
plt.savefig('results/plots/ablation_yearly_f1.png', dpi=300)
plt.close()

# === Plot 2: Yearly Average ROC-AUC (Bar Chart) ===
yearly_auc = df.groupby(['Year', 'AblationMode'])['ROC_AUC'].mean().reset_index()

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=yearly_auc, x='Year', y='ROC_AUC', hue='AblationMode', palette=colors, dodge=True)
plt.title("Yearly Average ROC-AUC by Signal Source", fontsize=16, fontweight='bold', pad=20)
plt.ylabel("ROC AUC")
plt.xlabel("Year")
plt.legend(title='Configuration', loc='upper left', bbox_to_anchor=(1, 1))
plt.ylim(0.4, 0.8) # Zoom in a bit as AUC usually > 0.5
sns.despine()
plt.tight_layout()
plt.savefig('results/plots/ablation_yearly_auc.png', dpi=300)
plt.close()

# === Plot 3: Monthly F1 Trend (Smoothed Line Plot) ===
# We use seaborn lineplot which automatically aggregates or we can just plot raw with smoothing
plt.figure(figsize=(14, 7))
# Create rolling mean for smoothing visual
df_sorted = df.sort_values('Date')
df_sorted['Rolling_F1'] = df_sorted.groupby('AblationMode')['F1'].transform(lambda x: x.rolling(3, min_periods=1).mean())

sns.lineplot(data=df_sorted, x='Date', y='Rolling_F1', hue='AblationMode', palette=colors, linewidth=2.5)
plt.title("Monthly F1 Score Trend (3-Month Rolling Average)", fontsize=16, fontweight='bold', pad=20)
plt.ylabel("F1 Score (Rolling Avg)")
plt.xlabel("Date")
plt.legend(title='Configuration')
plt.ylim(0, 1.0)
sns.despine()
plt.tight_layout()
plt.savefig('results/plots/ablation_monthly_trend.png', dpi=300)
plt.close()

print("Visualization Complete. Plots saved to results/plots/")
