#!/usr/bin/env python3
"""
Complete plot of volatility vs market cap with binning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import seaborn as sns

PARQUET_DIR = Path('/home/yahweh/code/ml/processed/parquet')

def process_file(file_path):
    try:
        parts = file_path.stem.split('_')
        if len(parts) >= 4:
            mc = int(parts[2])
            if mc > 0:
                df = pd.read_parquet(file_path)
                if len(df) > 10:
                    prices = df['last'].values
                    price_cv = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
                    returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else [0]
                    returns_std = np.std(returns) * 100
                    return mc / 1_000_000, price_cv, returns_std
    except:
        pass
    return None

print("Processing files for complete volatility analysis...")
files = list(PARQUET_DIR.glob('*.parquet'))[:30000]  # Large sample

results = []
with ProcessPoolExecutor(max_workers=20) as executor:
    for i, result in enumerate(executor.map(process_file, files)):
        if result:
            results.append(result)
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(files)} files...")

print(f"\nTotal valid samples: {len(results)}")

df = pd.DataFrame(results, columns=['marketcap', 'cv', 'returns_std'])
df = df[df['marketcap'] < 5000]  # Focus on <$5B for clarity

# Create bins - mix of linear and log scale
bins = []
# Fine-grained from 0-200M
bins.extend(range(0, 200, 20))  # 0, 20, 40, 60, 80, 100, 120, 140, 160, 180
# Coarser from 200M-1B
bins.extend(range(200, 1000, 100))  # 200, 300, 400, 500, 600, 700, 800, 900
# Even coarser above 1B
bins.extend([1000, 1500, 2000, 3000, 5000])

# Calculate statistics for each bin
bin_stats = []
for i in range(len(bins)-1):
    mask = (df['marketcap'] >= bins[i]) & (df['marketcap'] < bins[i+1])
    subset = df[mask]
    if len(subset) > 5:
        bin_stats.append({
            'bin_center': (bins[i] + bins[i+1]) / 2,
            'bin_label': f'${bins[i]}-{bins[i+1]}M',
            'mean_cv': subset['cv'].mean(),
            'std_cv': subset['cv'].std(),
            'median_cv': subset['cv'].median(),
            'mean_returns_std': subset['returns_std'].mean(),
            'count': len(subset),
            'q25': subset['cv'].quantile(0.25),
            'q75': subset['cv'].quantile(0.75)
        })

bin_df = pd.DataFrame(bin_stats)

# Create comprehensive plot
fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# 1. Main plot: Mean CV with error bars
ax1 = axes[0, 0]
ax1.errorbar(bin_df['bin_center'], bin_df['mean_cv'], yerr=bin_df['std_cv'], 
             marker='o', capsize=5, capthick=2, markersize=8, linewidth=2)
ax1.set_xlabel('Market Cap (Millions $)', fontsize=11)
ax1.set_ylabel('Mean Price CV', fontsize=11)
ax1.set_title('Volatility (CV) vs Market Cap - With Std Dev', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Mark potential breakpoints
for mc in [40, 60, 80, 100, 200, 300, 500, 1000]:
    ax1.axvline(x=mc, color='red', linestyle='--', alpha=0.3)

# 2. Median CV (less affected by outliers)
ax2 = axes[0, 1]
ax2.plot(bin_df['bin_center'], bin_df['median_cv'], marker='s', markersize=8, linewidth=2, color='green')
ax2.fill_between(bin_df['bin_center'], bin_df['q25'], bin_df['q75'], alpha=0.3, color='green')
ax2.set_xlabel('Market Cap (Millions $)', fontsize=11)
ax2.set_ylabel('Median Price CV', fontsize=11)
ax2.set_title('Median Volatility with IQR', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

# 3. Sample count per bin
ax3 = axes[1, 0]
ax3.bar(range(len(bin_df)), bin_df['count'], color='blue', alpha=0.7)
ax3.set_xlabel('Bin Index', fontsize=11)
ax3.set_ylabel('Sample Count', fontsize=11)
ax3.set_title('Sample Distribution Across Bins', fontsize=12, fontweight='bold')
ax3.set_xticks(range(0, len(bin_df), 2))
ax3.set_xticklabels([bin_df.iloc[i]['bin_label'] for i in range(0, len(bin_df), 2)], rotation=45)

# 4. Returns volatility
ax4 = axes[1, 1]
ax4.plot(bin_df['bin_center'], bin_df['mean_returns_std'], marker='^', markersize=8, linewidth=2, color='purple')
ax4.set_xlabel('Market Cap (Millions $)', fontsize=11)
ax4.set_ylabel('Mean Returns Std (%)', fontsize=11)
ax4.set_title('Returns Volatility vs Market Cap', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xscale('log')

# 5. Scatter plot with all points (subsample for visibility)
ax5 = axes[2, 0]
sample = df.sample(min(5000, len(df)))
scatter = ax5.scatter(sample['marketcap'], sample['cv'], alpha=0.3, s=1, c=sample['cv'], cmap='hot')
ax5.plot(bin_df['bin_center'], bin_df['mean_cv'], 'b-', linewidth=3, label='Mean')
ax5.plot(bin_df['bin_center'], bin_df['median_cv'], 'g-', linewidth=2, label='Median')
ax5.set_xlabel('Market Cap (Millions $)', fontsize=11)
ax5.set_ylabel('Price CV', fontsize=11)
ax5.set_title('All Data Points with Mean/Median Lines', fontsize=12, fontweight='bold')
ax5.set_xscale('log')
ax5.legend()
plt.colorbar(scatter, ax=ax5)

# 6. Derivative plot - rate of change
ax6 = axes[2, 1]
cv_changes = np.diff(bin_df['mean_cv'].values)
mc_changes = bin_df['bin_center'].values[1:]
ax6.plot(mc_changes, cv_changes, marker='o', linewidth=2, color='red')
ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax6.set_xlabel('Market Cap (Millions $)', fontsize=11)
ax6.set_ylabel('Change in Mean CV', fontsize=11)
ax6.set_title('Rate of Volatility Change', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.set_xscale('log')

# Mark significant changes
threshold = np.std(cv_changes) * 0.5
for i, (mc, change) in enumerate(zip(mc_changes, cv_changes)):
    if abs(change) > threshold:
        ax6.scatter(mc, change, s=200, color='yellow', edgecolors='red', linewidth=2, zorder=5)
        ax6.text(mc, change, f'${mc:.0f}M', fontsize=8, ha='center', va='bottom')

plt.suptitle('Complete Volatility vs Market Cap Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('volatility_vs_marketcap_complete.png', dpi=150, bbox_inches='tight')
plt.show()

# Print table of results
print("\nDetailed Bin Statistics:")
print("-" * 90)
print(f"{'Range':20} | {'Count':>6} | {'Mean CV':>8} | {'Median CV':>8} | {'Std CV':>8}")
print("-" * 90)
for _, row in bin_df.iterrows():
    print(f"{row['bin_label']:20} | {row['count']:6d} | {row['mean_cv']:.6f} | {row['median_cv']:.6f} | {row['std_cv']:.6f}")

# Identify natural breakpoints
print("\nSuggested Breakpoints Based on Volatility Changes:")
cv_threshold = bin_df['mean_cv'].std() * 0.3
last_cv = bin_df.iloc[0]['mean_cv']
breakpoints = []

for i in range(1, len(bin_df)):
    current_cv = bin_df.iloc[i]['mean_cv']
    if abs(current_cv - last_cv) > cv_threshold:
        breakpoints.append(bin_df.iloc[i]['bin_center'])
        print(f"  ${bin_df.iloc[i]['bin_center']:.0f}M: CV changes from {last_cv:.6f} to {current_cv:.6f}")
        last_cv = current_cv

print(f"\nFinal suggested buckets based on data:")
if len(breakpoints) > 0:
    all_points = [0] + sorted(breakpoints) + [10000]
    for i in range(len(all_points)-1):
        if i < 5:  # Limit to 5 buckets
            print(f"  Bucket {i+1}: ${all_points[i]:.0f}M - ${all_points[i+1]:.0f}M")
