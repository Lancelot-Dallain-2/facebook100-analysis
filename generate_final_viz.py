#!/usr/bin/env python3
"""
Génération rapide des visualisations finales pour Q4, Q5, Q6
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

base_dir = Path(__file__).parent
results_dir = base_dir / "results"

print("="*80)
print("GÉNÉRATION DES VISUALISATIONS FINALES")
print("="*80)

# Q4 - Link Prediction
print("\n[Q4] Link Prediction...")
df_q4 = pd.read_csv(results_dir / "q4" / "Q4_link_prediction_results.csv")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics = ['common_neighbors', 'jaccard', 'adamic_adar']

for idx, metric in enumerate(metrics):
    df_metric = df_q4[df_q4['Metric'] == metric]
    
    for k_val in [50, 100, 200]:
        df_k = df_metric[df_metric['k'] == k_val]
        axes[idx].plot(df_k['Fraction'], df_k['Precision'], 
                      marker='o', label=f'k={k_val}', linewidth=2)
    
    axes[idx].set_xlabel('Fraction Removed', fontsize=12)
    axes[idx].set_ylabel('Precision', fontsize=12)
    axes[idx].set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / "q4" / "Q4_precision_analysis.png", dpi=300, bbox_inches='tight')
print(f"  Saved: Q4_precision_analysis.png")
plt.close()

# Q4 - Summary stats
print(f"  Networks: {df_q4['Network'].nunique()}")
print(f"  Total evaluations: {len(df_q4)}")

# Q5 - Label Propagation
print("\n[Q5] Label Propagation...")
df_q5 = pd.read_csv(results_dir / "q5" / "Q5_label_propagation_results.csv")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
attributes = df_q5['Attribute'].unique()

for attr in attributes:
    df_attr = df_q5[df_q5['Attribute'] == attr]
    ax.plot(df_attr['Fraction Removed'], df_attr['Accuracy'], 
           marker='o', label=attr, linewidth=2, markersize=8)

ax.set_xlabel('Fraction Removed', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Label Propagation Accuracy by Attribute', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / "q5" / "Q5_accuracy_analysis.png", dpi=300, bbox_inches='tight')
print(f"  Saved: Q5_accuracy_analysis.png")
print(f"  Attributes: {', '.join(attributes)}")
print(f"  Mean accuracy: {df_q5['Accuracy'].mean():.4f}")
plt.close()

# Q6 - Community Detection
print("\n[Q6] Community Detection...")
df_q6 = pd.read_csv(results_dir / "q6" / "Q6_community_detection_results.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Modularity distribution
axes[0, 0].hist(df_q6['Modularity'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Modularity', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Modularity Distribution', fontsize=13, fontweight='bold')
axes[0, 0].axvline(df_q6['Modularity'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df_q6["Modularity"].mean():.3f}')
axes[0, 0].legend()

# Number of communities
axes[0, 1].hist(df_q6['Num Communities'], bins=15, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].set_xlabel('Number of Communities', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('Community Count Distribution', fontsize=13, fontweight='bold')

# Average community size
top_10 = df_q6.nlargest(10, 'Avg Community Size')
axes[1, 0].barh(range(len(top_10)), top_10['Avg Community Size'])
axes[1, 0].set_yticks(range(len(top_10)))
axes[1, 0].set_yticklabels(top_10['Network'], fontsize=9)
axes[1, 0].set_xlabel('Average Community Size', fontsize=12)
axes[1, 0].set_title('Top 10 Networks by Avg Community Size', fontsize=13, fontweight='bold')

# Modularity vs Num Communities
axes[1, 1].scatter(df_q6['Num Communities'], df_q6['Modularity'], alpha=0.6, s=50)
axes[1, 1].set_xlabel('Number of Communities', fontsize=12)
axes[1, 1].set_ylabel('Modularity', fontsize=12)
axes[1, 1].set_title('Modularity vs Number of Communities', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / "q6" / "Q6_community_analysis.png", dpi=300, bbox_inches='tight')
print(f"  Saved: Q6_community_analysis.png")
print(f"  Networks: {len(df_q6)}")
print(f"  Mean modularity: {df_q6['Modularity'].mean():.4f}")
print(f"  Mean communities: {df_q6['Num Communities'].mean():.1f}")
plt.close()

print("\n" + "="*80)
print("VISUALISATIONS GÉNÉRÉES AVEC SUCCÈS!")
print("="*80)
