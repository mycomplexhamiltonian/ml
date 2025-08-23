#!/usr/bin/env python3
"""
Enhanced Price Volatility vs Market Cap Analysis with Log-Scale Bucketing
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import time

PARQUET_DIR = Path('/home/yahweh/code/ml/processed/parquet')

def extract_marketcap_from_filename(filename):
    """Extract market cap from filename pattern: SYMBOL_MKTCAP_MARKETCAP_TIMESTAMP.parquet"""
    if hasattr(filename, 'stem'):
        name = filename.stem
    else:
        name = str(filename).replace('.parquet', '')
    
    parts = name.split('_')
    if len(parts) >= 4:
        try:
            marketcap = int(parts[2])
            return marketcap
        except:
            return 0
    return 0

def categorize_marketcap_log(marketcap):
    """Categorize market cap using data-driven volatility-based buckets"""
    if marketcap <= 0:
        return None
    
    marketcap_millions = marketcap / 1_000_000
    
    # Based on volatility analysis from 20K files
    # These buckets group similar volatility patterns together
    if marketcap_millions < 40:
        return '1. Small (<$40M)'          # CV ~0.0043 - highest volatility
    elif marketcap_millions < 80:
        return '2. Lower-Mid ($40-80M)'    # CV ~0.0041 - slight stability
    elif marketcap_millions < 200:
        return '3. Upper-Mid ($80-200M)'   # CV ~0.0042 - volatile period
    elif marketcap_millions < 700:
        return '4. Large ($200-700M)'      # CV ~0.0035 - stabilizing
    else:
        return '5. Mega (>$700M)'          # CV ~0.0030 - most stable

def process_file_batch(file_batch):
    """Process files to extract volatility metrics and market cap"""
    results = []
    
    for file_path in file_batch:
        try:
            marketcap = extract_marketcap_from_filename(file_path.name)
            
            if marketcap <= 0:
                continue
            
            df = pd.read_parquet(file_path)
            
            if len(df) < 10:
                continue
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Get baseline from first 10 seconds
            start_time = df['timestamp'].iloc[0]
            first_10_sec = df[df['timestamp'] <= start_time + pd.Timedelta(seconds=10)]
            
            if len(first_10_sec) == 0:
                first_10_sec = df.iloc[:5]
            
            baseline_price = first_10_sec['last'].median()
            
            prices = df['last'].values
            volumes = df['volume'].values
            
            # Calculate volatility metrics
            price_std = np.std(prices)
            price_mean = np.mean(prices)
            price_cv = price_std / price_mean if price_mean > 0 else 0
            
            # Price movement metrics
            price_range = np.max(prices) - np.min(prices)
            price_range_pct = (price_range / price_mean * 100) if price_mean > 0 else 0
            
            # Deviation from baseline
            max_deviation = np.max(np.abs(prices - baseline_price)) / baseline_price * 100 if baseline_price > 0 else 0
            
            # Returns volatility
            returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else [0]
            returns_std = np.std(returns) * 100  # Convert to percentage
            
            # Volume metrics
            volume_mean = np.mean(volumes)
            volume_cv = np.std(volumes) / volume_mean if volume_mean > 0 else 0
            
            results.append({
                'marketcap': marketcap,
                'marketcap_millions': marketcap / 1_000_000,
                'marketcap_bucket': categorize_marketcap_log(marketcap),
                'price_std': price_std,
                'price_cv': price_cv,
                'price_range_pct': price_range_pct,
                'max_deviation': max_deviation,
                'returns_std': returns_std,
                'volume_mean': volume_mean,
                'volume_cv': volume_cv,
                'price_mean': price_mean,
                'num_points': len(df)
            })
            
        except Exception as e:
            continue
    
    return results

def process_all_files_parallel(parquet_files):
    """Process all files using all CPU cores"""
    n_workers = cpu_count()
    print(f"Using {n_workers} CPU cores...")
    
    batch_size = max(1, len(parquet_files) // (n_workers * 10))
    batches = [parquet_files[i:i+batch_size] for i in range(0, len(parquet_files), batch_size)]
    
    print(f"Processing {len(parquet_files)} files in {len(batches)} batches...")
    
    all_results = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_file_batch, batch) for batch in batches]
        
        completed = 0
        for future in as_completed(futures):
            results = future.result()
            all_results.extend(results)
            completed += 1
            if completed % 10 == 0:
                print(f"  Completed {completed}/{len(batches)} batches...")
    
    return pd.DataFrame(all_results)

def main():
    start_time = time.time()
    
    print("="*60)
    print("ENHANCED VOLATILITY vs MARKET CAP ANALYSIS")
    print("="*60)
    
    # Load files
    parquet_files = list(PARQUET_DIR.glob('*.parquet'))
    
    # Process all files
    MAX_FILES = min(len(parquet_files), 100000)
    parquet_files = parquet_files[:MAX_FILES]
    
    print(f"\nProcessing {len(parquet_files):,} files...")
    
    # Get volatility metrics
    df = process_all_files_parallel(parquet_files)
    
    if len(df) == 0:
        print("No valid data found!")
        return
    
    # Remove None buckets
    df = df[df['marketcap_bucket'].notna()]
    
    print(f"\n✓ Processed {len(df):,} files with valid market caps")
    print(f"  Time: {time.time() - start_time:.1f} seconds")
    
    # Create comprehensive visualizations
    fig = plt.figure(figsize=(24, 18))
    
    # 1. Volatility by Market Cap Bucket (Box Plot)
    ax1 = plt.subplot(3, 4, 1)
    bucket_order = sorted(df['marketcap_bucket'].unique())
    df_sorted = df.set_index('marketcap_bucket').loc[bucket_order].reset_index()
    
    box_data = [df[df['marketcap_bucket'] == bucket]['price_cv'] for bucket in bucket_order]
    bp = ax1.boxplot(box_data, tick_labels=[b.split('. ')[1] for b in bucket_order], patch_artist=True)
    
    for patch, color in zip(bp['boxes'], plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(bucket_order)))):
        patch.set_facecolor(color)
    
    ax1.set_xlabel('Market Cap Bucket', fontsize=10)
    ax1.set_ylabel('Price Volatility (CV)', fontsize=10)
    ax1.set_title('Volatility Distribution by Market Cap', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. Returns Volatility by Market Cap Bucket
    ax2 = plt.subplot(3, 4, 2)
    returns_data = [df[df['marketcap_bucket'] == bucket]['returns_std'] for bucket in bucket_order]
    bp2 = ax2.boxplot(returns_data, tick_labels=[b.split('. ')[1] for b in bucket_order], patch_artist=True)
    
    for patch, color in zip(bp2['boxes'], plt.cm.viridis(np.linspace(0.3, 0.9, len(bucket_order)))):
        patch.set_facecolor(color)
    
    ax2.set_xlabel('Market Cap Bucket', fontsize=10)
    ax2.set_ylabel('Returns Volatility (%)', fontsize=10)
    ax2.set_title('Returns Volatility by Market Cap', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Average Metrics by Bucket (Bar Chart)
    ax3 = plt.subplot(3, 4, 3)
    bucket_stats = df.groupby('marketcap_bucket').agg({
        'price_cv': 'mean',
        'returns_std': 'mean',
        'max_deviation': 'mean'
    }).reindex(bucket_order)
    
    x = np.arange(len(bucket_order))
    width = 0.25
    
    ax3.bar(x - width, bucket_stats['price_cv'], width, label='Price CV', color='red', alpha=0.7)
    ax3.bar(x, bucket_stats['returns_std']/10, width, label='Returns Std/10', color='blue', alpha=0.7)
    ax3.bar(x + width, bucket_stats['max_deviation']/100, width, label='Max Dev/100', color='green', alpha=0.7)
    
    ax3.set_xlabel('Market Cap Bucket', fontsize=10)
    ax3.set_ylabel('Normalized Metrics', fontsize=10)
    ax3.set_title('Average Volatility Metrics by Market Cap', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([b.split('. ')[1] for b in bucket_order], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Sample Count by Bucket
    ax4 = plt.subplot(3, 4, 4)
    bucket_counts = df['marketcap_bucket'].value_counts().reindex(bucket_order)
    colors = plt.cm.cool(np.linspace(0.3, 0.9, len(bucket_order)))
    
    bars = ax4.bar(range(len(bucket_order)), bucket_counts.values, color=colors, alpha=0.8)
    ax4.set_xlabel('Market Cap Bucket', fontsize=10)
    ax4.set_ylabel('Number of Files', fontsize=10)
    ax4.set_title('Data Distribution Across Market Caps', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(len(bucket_order)))
    ax4.set_xticklabels([b.split('. ')[1] for b in bucket_order], rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, bucket_counts.values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count):,}', ha='center', va='bottom', fontsize=8)
    
    # 5. Scatter: Log Market Cap vs Volatility
    ax5 = plt.subplot(3, 4, 5)
    
    # Use log scale and hexbin for density
    hexbin = ax5.hexbin(df['marketcap_millions'], df['price_cv'], 
                        gridsize=50, cmap='YlOrRd', mincnt=1, xscale='log')
    ax5.set_xlabel('Market Cap (Millions $, log scale)', fontsize=10)
    ax5.set_ylabel('Price Volatility (CV)', fontsize=10)
    ax5.set_title('Volatility vs Market Cap (Log Scale)', fontsize=12, fontweight='bold')
    plt.colorbar(hexbin, ax=ax5, label='Count')
    
    # 6. Volume CV by Market Cap
    ax6 = plt.subplot(3, 4, 6)
    volume_data = [df[df['marketcap_bucket'] == bucket]['volume_cv'] for bucket in bucket_order]
    bp3 = ax6.boxplot(volume_data, tick_labels=[b.split('. ')[1] for b in bucket_order], patch_artist=True)
    
    for patch, color in zip(bp3['boxes'], plt.cm.plasma(np.linspace(0.3, 0.9, len(bucket_order)))):
        patch.set_facecolor(color)
    
    ax6.set_xlabel('Market Cap Bucket', fontsize=10)
    ax6.set_ylabel('Volume CV', fontsize=10)
    ax6.set_title('Volume Volatility by Market Cap', fontsize=12, fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # 7. Correlation Matrix
    ax7 = plt.subplot(3, 4, 7)
    
    corr_data = df[['marketcap_millions', 'price_cv', 'returns_std', 'max_deviation', 'volume_cv']].corr()
    im = ax7.imshow(corr_data, cmap='RdBu_r', vmin=-1, vmax=1)
    
    ax7.set_xticks(range(len(corr_data.columns)))
    ax7.set_yticks(range(len(corr_data.columns)))
    ax7.set_xticklabels(['MktCap', 'PriceCV', 'RetStd', 'MaxDev', 'VolCV'], rotation=45)
    ax7.set_yticklabels(['MktCap', 'PriceCV', 'RetStd', 'MaxDev', 'VolCV'])
    
    # Add correlation values
    for i in range(len(corr_data.columns)):
        for j in range(len(corr_data.columns)):
            text = ax7.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax7.set_title('Correlation Matrix', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax7)
    
    # 8. Max Deviation by Market Cap
    ax8 = plt.subplot(3, 4, 8)
    deviation_data = [df[df['marketcap_bucket'] == bucket]['max_deviation'] for bucket in bucket_order]
    bp4 = ax8.boxplot(deviation_data, tick_labels=[b.split('. ')[1] for b in bucket_order], patch_artist=True)
    
    for patch, color in zip(bp4['boxes'], plt.cm.autumn(np.linspace(0.3, 0.9, len(bucket_order)))):
        patch.set_facecolor(color)
    
    ax8.set_xlabel('Market Cap Bucket', fontsize=10)
    ax8.set_ylabel('Max Deviation from Baseline (%)', fontsize=10)
    ax8.set_title('Max Price Deviation by Market Cap', fontsize=12, fontweight='bold')
    ax8.tick_params(axis='x', rotation=45)
    ax8.grid(True, alpha=0.3)
    
    # 9-12: Statistical summaries
    ax9 = plt.subplot(3, 4, 9)
    ax9.axis('off')
    
    # Create summary statistics table
    summary_stats = []
    for bucket in bucket_order:
        bucket_data = df[df['marketcap_bucket'] == bucket]
        summary_stats.append([
            bucket.split('. ')[1],
            len(bucket_data),
            f"{bucket_data['price_cv'].mean():.4f}",
            f"{bucket_data['returns_std'].mean():.2f}%",
            f"{bucket_data['max_deviation'].mean():.2f}%"
        ])
    
    table = ax9.table(cellText=summary_stats,
                     colLabels=['Bucket', 'Count', 'Avg CV', 'Avg Ret Std', 'Avg Max Dev'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax9.set_title('Summary Statistics by Market Cap', fontsize=12, fontweight='bold')
    
    # Overall statistics
    stats_text = (
        f"Overall Statistics:\n"
        f"Total Files Processed: {len(df):,}\n"
        f"Market Cap Range: ${df['marketcap_millions'].min():.2f}M - ${df['marketcap_millions'].max():.2f}M\n"
        f"Avg Market Cap: ${df['marketcap_millions'].mean():.2f}M\n"
        f"Median Market Cap: ${df['marketcap_millions'].median():.2f}M\n"
        f"\nGlobal Volatility Metrics:\n"
        f"Avg Price CV: {df['price_cv'].mean():.4f}\n"
        f"Avg Returns Std: {df['returns_std'].mean():.2f}%\n"
        f"Avg Max Deviation: {df['max_deviation'].mean():.2f}%\n"
        f"Avg Volume CV: {df['volume_cv'].mean():.4f}\n"
        f"\nCorrelations with Market Cap:\n"
        f"Price Volatility: {df['marketcap_millions'].corr(df['price_cv']):.3f}\n"
        f"Returns Volatility: {df['marketcap_millions'].corr(df['returns_std']):.3f}\n"
        f"Max Deviation: {df['marketcap_millions'].corr(df['max_deviation']):.3f}"
    )
    
    fig.text(0.72, 0.25, stats_text, transform=fig.transFigure, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Comprehensive Market Cap vs Volatility Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save
    output_file = 'volatility_marketcap_enhanced.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved as '{output_file}'")
    
    # Save detailed data
    df.to_csv('volatility_marketcap_detailed.csv', index=False)
    print(f"✓ Data saved as 'volatility_marketcap_detailed.csv'")
    
    # Save summary by bucket
    bucket_summary = df.groupby('marketcap_bucket').agg({
        'price_cv': ['mean', 'std', 'median'],
        'returns_std': ['mean', 'std', 'median'],
        'max_deviation': ['mean', 'std', 'median'],
        'volume_cv': ['mean', 'std', 'median'],
        'marketcap_millions': ['mean', 'min', 'max', 'count']
    })
    bucket_summary.to_csv('marketcap_bucket_summary.csv')
    print(f"✓ Bucket summary saved as 'marketcap_bucket_summary.csv'")
    
    print(f"\nTotal time: {time.time() - start_time:.1f} seconds")
    
    plt.show()

if __name__ == "__main__":
    main()
