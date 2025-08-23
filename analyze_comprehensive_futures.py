#!/usr/bin/env python3
"""
Comprehensive Financial Data Analysis with Market Cap Bucketing
Combines: Deviations, Trajectories, and Volatility Analysis
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
    """Extract market cap from filename
    Futures format: SYMBOL_MKTCAP_12345678_timestamp.parquet
    Market cap is at position 2 for futures files (no SPOT in name)
    """
    if hasattr(filename, 'stem'):
        name = filename.stem
    else:
        name = str(filename).replace('.parquet', '')
    
    parts = name.split('_')
    # For futures files (no SPOT), market cap is at position 2
    if len(parts) >= 4 and 'SPOT' not in parts:
        try:
            return int(parts[2])  # Position 2 for futures files
        except:
            return 0
    return 0

def get_marketcap_bucket(marketcap):
    """Categorize market cap using data-driven volatility-based buckets"""
    if marketcap <= 0:
        return None
    
    m = marketcap / 1_000_000
    
    # Based on volatility analysis from 20K files
    if m < 40:
        return '1. Small (<$40M)'          # CV ~0.0043 - highest volatility
    elif m < 80:
        return '2. Lower-Mid ($40-80M)'    # CV ~0.0041 - slight stability
    elif m < 200:
        return '3. Upper-Mid ($80-200M)'   # CV ~0.0042 - volatile period
    elif m < 700:
        return '4. Large ($200-700M)'      # CV ~0.0035 - stabilizing
    else:
        return '5. Mega (>$700M)'          # CV ~0.0030 - most stable

def process_file_comprehensive(file_path):
    """Process single file for all metrics"""
    try:
        marketcap = extract_marketcap_from_filename(file_path.name)
        if marketcap <= 0:
            return None
        
        df = pd.read_parquet(file_path)
        if len(df) < 10:
            return None
        
        df = df.sort_values('timestamp')
        
        # Baseline from first 10 seconds
        start_time = df['timestamp'].iloc[0]
        first_10_sec = df[df['timestamp'] <= start_time + pd.Timedelta(seconds=10)]
        if len(first_10_sec) == 0:
            first_10_sec = df.iloc[:5]
        
        baseline_price = first_10_sec['last'].median()
        if baseline_price <= 0:
            return None
        
        prices = df['last'].values
        volumes = df['volume'].values
        
        # Deviations from baseline
        deviations = ((prices - baseline_price) / baseline_price) * 100
        
        # Trajectory (normalize to 100 points)
        n_points = 100
        indices = np.linspace(0, len(deviations)-1, n_points, dtype=int)
        trajectory = deviations[indices]
        
        # Volatility metrics
        price_std = np.std(prices)
        price_mean = np.mean(prices)
        price_cv = price_std / price_mean if price_mean > 0 else 0
        
        returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else [0]
        returns_std = np.std(returns) * 100
        
        volume_cv = np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 0
        
        # Additional insightful metrics
        price_range = (max(prices) - min(prices)) / baseline_price * 100
        mean_reversion_score = -np.corrcoef(np.arange(len(deviations)), deviations)[0, 1] if len(deviations) > 1 else 0
        momentum_score = np.mean(np.diff(prices)) / baseline_price * 100 if len(prices) > 1 else 0
        
        # Spike detection with multiple thresholds and cooldown
        thresholds = [1.0, 2.0, 4.0]  # 1%, 2%, 4% thresholds
        # Track up and down separately
        spike_up_counts = {t: 0 for t in thresholds}
        spike_down_counts = {t: 0 for t in thresholds}
        movement_up_counts = {t: 0 for t in thresholds}
        movement_down_counts = {t: 0 for t in thresholds}
        
        # Timing configuration (in milliseconds)
        peak_window_ms = 20  # Find peak/trough within 50ms after trigger
        classification_wait_ms = 100  # Wait 500ms from trigger to classify
        cooldown_ms = 500  # No new detection for 1 second after an event
        
        # Classification thresholds
        movement_recovery_threshold = 0.4  # Below 40% = movement
        spike_recovery_threshold = 0.6     # Above 60% = spike
        # Between 40-60% = neutral/ignored
        
        # Get timestamps if available
        if 'timestamp_ms' in df.columns:
            timestamps = df['timestamp_ms'].values
        else:
            # Fallback: create fake timestamps if not available
            timestamps = np.arange(len(prices)) * 500  # Assume 500ms intervals
        
        for threshold in thresholds:
            last_event_time = -float('inf')  # Initialize to allow first detection
            
            if len(prices) > 2:
                # Pre-calculate price changes for all points
                price_changes = np.zeros(len(prices))
                price_changes[1:] = ((prices[1:] - prices[:-1]) / prices[:-1]) * 100
                
                for i in range(1, len(prices) - 1):
                    # Skip if within cooldown period from last event
                    current_time = timestamps[i]
                    if current_time - last_event_time < cooldown_ms:
                        continue
                    
                    # Get pre-calculated price change
                    price_change_pct = price_changes[i]
                    
                    # Check if move exceeds threshold
                    if abs(price_change_pct) > threshold:
                        # Step 1: Find peak/trough within 50ms window using numpy
                        peak_end_time = current_time + peak_window_ms
                        
                        # Vectorized search for indices in peak window
                        peak_mask = (timestamps[i:] <= peak_end_time)
                        peak_count = np.sum(peak_mask)
                        if peak_count == 0:
                            peak_indices = [i]
                        else:
                            peak_indices = list(range(i, i + peak_count))
                        
                        # Step 2: Find the classification point at exactly 500ms later
                        classification_time = current_time + classification_wait_ms
                        
                        # Vectorized search for classification index
                        future_times = timestamps[i:]
                        classification_mask = future_times >= classification_time
                        if not np.any(classification_mask):
                            continue  # Not enough data
                        
                        first_after_idx = np.argmax(classification_mask)
                        abs_idx = i + first_after_idx
                        
                        # Choose closest point to 500ms mark
                        if abs_idx > i and first_after_idx > 0:
                            time_diff_curr = abs(timestamps[abs_idx] - classification_time)
                            time_diff_prev = abs(timestamps[abs_idx-1] - classification_time)
                            classification_idx = abs_idx - 1 if time_diff_prev < time_diff_curr else abs_idx
                        else:
                            classification_idx = abs_idx
                        
                        if classification_idx == i:
                            continue  # Not enough data to classify
                        
                        # Determine the peak/trough within the 50ms window
                        if price_change_pct > 0:  # Price went up from i-1 to i
                            initial_price = prices[i-1]
                            
                            # Find the maximum price within the 50ms peak window
                            peak_window_prices = [prices[idx] for idx in peak_indices]
                            peak_price = max(peak_window_prices)
                            
                            # Check where price is at 500ms mark
                            recovery_price = prices[classification_idx]
                            
                            # Calculate how much it recovered from the actual peak
                            # If price went from 100 to 102 to 103 (peak) then to 101
                            # Initial move was +2%, peak was +3%, recovery is (103-101)/103 = 1.94%
                            # Recovery ratio is 1.94/3 = 0.65 (65% recovery from peak)
                            actual_peak_change = (peak_price - initial_price) / initial_price * 100
                            recovery_from_peak = (peak_price - recovery_price) / peak_price * 100
                            recovery_ratio = recovery_from_peak / actual_peak_change if actual_peak_change > 0 else 0
                            
                            # Classify UPWARD moves based on recovery ratio
                            if recovery_ratio >= spike_recovery_threshold:  # >= 60% recovery
                                spike_up_counts[threshold] += 1  # Upward spike (went up, came back down)
                                last_event_time = current_time
                            elif recovery_ratio < movement_recovery_threshold:  # < 40% recovery
                                movement_up_counts[threshold] += 1  # Upward movement (went up, stayed up)
                                last_event_time = current_time
                            # else: 40-60% recovery - neutral, don't count
                                
                        else:  # Price went down from i-1 to i
                            initial_price = prices[i-1]
                            
                            # Find the minimum price within the 50ms peak window
                            peak_window_prices = [prices[idx] for idx in peak_indices]
                            trough_price = min(peak_window_prices)
                            
                            # Check where price is at 500ms mark
                            recovery_price = prices[classification_idx]
                            
                            # Calculate how much it recovered from the actual trough
                            # If price went from 100 to 98 to 97 (trough) then to 99
                            # Initial move was -2%, trough was -3%, recovery is (99-97)/97 = 2.06%
                            # Recovery ratio is 2.06/3 = 0.69 (69% recovery from trough)
                            actual_trough_change = abs((trough_price - initial_price) / initial_price * 100)
                            recovery_from_trough = (recovery_price - trough_price) / trough_price * 100
                            recovery_ratio = recovery_from_trough / actual_trough_change if actual_trough_change > 0 else 0
                            
                            # Classify DOWNWARD moves based on recovery ratio
                            if recovery_ratio >= spike_recovery_threshold:  # >= 60% recovery
                                spike_down_counts[threshold] += 1  # Downward spike (went down, came back up)
                                last_event_time = current_time
                            elif recovery_ratio < movement_recovery_threshold:  # < 40% recovery
                                movement_down_counts[threshold] += 1  # Downward movement (went down, stayed down)
                                last_event_time = current_time
                            # else: 40-60% recovery - neutral, don't count
        
        return {
            'marketcap': marketcap,
            'marketcap_millions': marketcap / 1_000_000,
            'bucket': get_marketcap_bucket(marketcap),
            'deviations': deviations,
            'trajectory': trajectory,
            'price_cv': price_cv,
            'returns_std': returns_std,
            'volume_cv': volume_cv,
            'max_deviation': np.max(np.abs(deviations)),
            'end_deviation': deviations[-1],
            'start_deviation': deviations[0],
            'price_range': price_range,
            'mean_reversion': mean_reversion_score,
            'momentum': momentum_score,
            'spike_up_counts': spike_up_counts,
            'spike_down_counts': spike_down_counts,
            'movement_up_counts': movement_up_counts,
            'movement_down_counts': movement_down_counts
        }
    except:
        return None

def process_batch(file_batch):
    """Process batch of files"""
    results = []
    for file_path in file_batch:
        result = process_file_comprehensive(file_path)
        if result:
            results.append(result)
    return results

def process_all_parallel(parquet_files):
    """Process all files in parallel"""
    n_workers = cpu_count()
    print(f"Using {n_workers} CPU cores...")
    
    batch_size = max(1, len(parquet_files) // (n_workers * 10))
    batches = [parquet_files[i:i+batch_size] for i in range(0, len(parquet_files), batch_size)]
    
    all_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]
        
        completed = 0
        for future in as_completed(futures):
            results = future.result()
            all_results.extend(results)
            completed += 1
            if completed % 10 == 0:
                print(f"  Completed {completed}/{len(batches)} batches...")
    
    return all_results

def main():
    start_time = time.time()
    
    print("="*60)
    print("FUTURES MARKET ANALYSIS WITH MARKET CAP BUCKETING")
    print("="*60)
    
    # Load and process only NON-SPOT (futures) files
    all_files = list(PARQUET_DIR.glob('*.parquet'))
    parquet_files = [f for f in all_files if '_SPOT_' not in f.name]
    
    print(f"\nProcessing all {len(parquet_files):,} files...")
    results = process_all_parallel(parquet_files)
    
    if not results:
        print("No valid data found!")
        return
    
    print(f"\n✓ Processed {len(results):,} files with valid data")
    print(f"  Time: {time.time() - start_time:.1f} seconds")
    
    # Organize by bucket
    buckets = {}
    for r in results:
        bucket = r['bucket']
        if bucket not in buckets:
            buckets[bucket] = []
        buckets[bucket].append(r)
    
    bucket_order = sorted(buckets.keys())
    
    # Create mega visualization - 5 rows for 5 heatmaps
    fig = plt.figure(figsize=(24, 24))
    
    # 1. Deviation Heatmap by Market Cap
    ax1 = plt.subplot(5, 3, 1)
    for i, bucket in enumerate(bucket_order):
        bucket_data = buckets[bucket]
        all_devs = []
        for item in bucket_data[:100]:  # Sample for visualization
            all_devs.extend(item['deviations'])
        
        if all_devs:
            hist, bins = np.histogram(all_devs, bins=50, range=(-10, 10))
            hist_norm = hist / hist.max() if hist.max() > 0 else hist
            
            # Plot as horizontal bar
            ax1.barh(bins[:-1] + i*0.1, hist_norm, height=0.15, 
                    alpha=0.7, label=bucket.split('. ')[1])
    
    ax1.set_xlabel('Frequency (normalized)', fontsize=10)
    ax1.set_ylabel('Price Deviation from Baseline (%)', fontsize=10)
    ax1.set_title('Deviation Distribution by Market Cap', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Trajectory Heatmap by Market Cap
    ax2 = plt.subplot(5, 3, 2)
    
    for i, bucket in enumerate(bucket_order):
        bucket_data = buckets[bucket]
        trajectories = np.array([item['trajectory'] for item in bucket_data[:500]])
        
        if len(trajectories) > 0:
            mean_traj = np.mean(trajectories, axis=0)
            ax2.plot(mean_traj, label=bucket.split('. ')[1], linewidth=2)
    
    ax2.set_xlabel('Time Progress (0-100)', fontsize=10)
    ax2.set_ylabel('Avg Deviation from Baseline (%)', fontsize=10)
    ax2.set_title('Average Price Trajectories by Market Cap', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 3. Spike Analysis - 1% Threshold (Up/Down separated)
    ax3 = plt.subplot(5, 3, 3)
    
    threshold_1 = 1.0
    spike_data_1 = []
    for bucket in bucket_order:
        bucket_data = buckets[bucket]
        spike_up = sum(item['spike_up_counts'][threshold_1] for item in bucket_data)
        spike_down = sum(item['spike_down_counts'][threshold_1] for item in bucket_data)
        move_up = sum(item['movement_up_counts'][threshold_1] for item in bucket_data)
        move_down = sum(item['movement_down_counts'][threshold_1] for item in bucket_data)
        
        spike_data_1.append({
            'bucket': bucket.split('. ')[1],
            'spike_up': spike_up,
            'spike_down': spike_down,
            'move_up': move_up,
            'move_down': move_down,
            'total': spike_up + spike_down + move_up + move_down
        })
    
    x = np.arange(len(spike_data_1))
    width = 0.7
    
    # Stack bars for spikes and movements
    spike_up_vals = [d['spike_up'] for d in spike_data_1]
    spike_down_vals = [d['spike_down'] for d in spike_data_1]
    move_up_vals = [d['move_up'] for d in spike_data_1]
    move_down_vals = [d['move_down'] for d in spike_data_1]
    
    # Create stacked bars
    bars1 = ax3.bar(x, spike_up_vals, width, label='Spike Up', color='lightgreen', alpha=0.8)
    bars2 = ax3.bar(x, spike_down_vals, width, bottom=spike_up_vals, label='Spike Down', color='lightcoral', alpha=0.8)
    bars3 = ax3.bar(x, move_up_vals, width, bottom=np.array(spike_up_vals) + np.array(spike_down_vals), 
                    label='Move Up', color='darkgreen', alpha=0.8)
    bars4 = ax3.bar(x, move_down_vals, width, 
                    bottom=np.array(spike_up_vals) + np.array(spike_down_vals) + np.array(move_up_vals),
                    label='Move Down', color='darkred', alpha=0.8)
    
    ax3.set_xlabel('Market Cap Bucket', fontsize=10)
    ax3.set_ylabel('Count', fontsize=10)
    ax3.set_title(f'Up/Down Spikes vs Movements (1% Threshold)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([d['bucket'] for d in spike_data_1], rotation=45, ha='right')
    ax3.legend(fontsize=7, loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Price Movement Direction Analysis
    ax4 = plt.subplot(5, 3, 4)
    
    movement_data = []
    for bucket in bucket_order:
        bucket_data = buckets[bucket]
        up_count = sum(1 for item in bucket_data if item['end_deviation'] > item['start_deviation'] + 0.1)
        down_count = sum(1 for item in bucket_data if item['end_deviation'] < item['start_deviation'] - 0.1)
        flat_count = len(bucket_data) - up_count - down_count
        
        movement_data.append({
            'bucket': bucket.split('. ')[1],
            'up': up_count / len(bucket_data) * 100 if bucket_data else 0,
            'down': down_count / len(bucket_data) * 100 if bucket_data else 0,
            'flat': flat_count / len(bucket_data) * 100 if bucket_data else 0,
            'avg_move': np.mean([item['end_deviation'] - item['start_deviation'] for item in bucket_data]) if bucket_data else 0
        })
    
    x = np.arange(len(movement_data))
    width = 0.25
    
    bars1 = ax4.bar(x - width, [d['up'] for d in movement_data], width, label='Up', color='green', alpha=0.7)
    bars2 = ax4.bar(x, [d['down'] for d in movement_data], width, label='Down', color='red', alpha=0.7)
    bars3 = ax4.bar(x + width, [d['flat'] for d in movement_data], width, label='Flat', color='gray', alpha=0.7)
    
    # Add text on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 5:  # Only show if > 5%
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=7)
    
    ax4.set_xlabel('Market Cap Bucket', fontsize=10)
    ax4.set_ylabel('Percentage (%)', fontsize=10)
    ax4.set_title('Price Movement Direction Distribution', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([d['bucket'] for d in movement_data], rotation=45, ha='right')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Spike Analysis - 2% Threshold
    ax5 = plt.subplot(5, 3, 5)
    
    threshold_2 = 2.0
    spike_data_2 = []
    for bucket in bucket_order:
        bucket_data = buckets[bucket]
        spike_up = sum(item['spike_up_counts'][threshold_2] for item in bucket_data)
        spike_down = sum(item['spike_down_counts'][threshold_2] for item in bucket_data)
        move_up = sum(item['movement_up_counts'][threshold_2] for item in bucket_data)
        move_down = sum(item['movement_down_counts'][threshold_2] for item in bucket_data)
        
        spike_data_2.append({
            'bucket': bucket.split('. ')[1],
            'spike_up': spike_up,
            'spike_down': spike_down,
            'move_up': move_up,
            'move_down': move_down,
            'total': spike_up + spike_down + move_up + move_down
        })
    
    x = np.arange(len(spike_data_2))
    width = 0.7
    
    # Stack bars for spikes and movements
    spike_up_vals = [d['spike_up'] for d in spike_data_2]
    spike_down_vals = [d['spike_down'] for d in spike_data_2]
    move_up_vals = [d['move_up'] for d in spike_data_2]
    move_down_vals = [d['move_down'] for d in spike_data_2]
    
    # Create stacked bars
    bars1 = ax5.bar(x, spike_up_vals, width, label='Spike Up', color='lightgreen', alpha=0.8)
    bars2 = ax5.bar(x, spike_down_vals, width, bottom=spike_up_vals, label='Spike Down', color='lightcoral', alpha=0.8)
    bars3 = ax5.bar(x, move_up_vals, width, bottom=np.array(spike_up_vals) + np.array(spike_down_vals), 
                    label='Move Up', color='darkgreen', alpha=0.8)
    bars4 = ax5.bar(x, move_down_vals, width, 
                    bottom=np.array(spike_up_vals) + np.array(spike_down_vals) + np.array(move_up_vals),
                    label='Move Down', color='darkred', alpha=0.8)
    
    ax5.set_xlabel('Market Cap Bucket', fontsize=10)
    ax5.set_ylabel('Count', fontsize=10)
    ax5.set_title(f'Up/Down Spikes vs Movements (2% Threshold)', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([d['bucket'] for d in spike_data_2], rotation=45, ha='right')
    ax5.legend(fontsize=7, loc='upper right')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Spike Analysis - 4% Threshold  
    ax6 = plt.subplot(5, 3, 6)
    
    threshold_4 = 4.0
    spike_data_4 = []
    for bucket in bucket_order:
        bucket_data = buckets[bucket]
        total_spikes = sum(item['spike_counts'][threshold_4] for item in bucket_data)
        total_movements = sum(item['movement_counts'][threshold_4] for item in bucket_data)
        total = total_spikes + total_movements
        
        spike_data_4.append({
            'bucket': bucket.split('. ')[1],
            'spike_ratio': total_spikes / total * 100 if total > 0 else 0,
            'total_spikes': total_spikes,
            'total_movements': total_movements
        })
    
    x = np.arange(len(spike_data_4))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, [d['spike_ratio'] for d in spike_data_4], width, 
                    label='Spikes', color='purple', alpha=0.7)
    bars2 = ax6.bar(x + width/2, [100 - d['spike_ratio'] for d in spike_data_4], width,
                    label='Movements', color='cyan', alpha=0.7)
    
    for bar, data in zip(bars1, spike_data_4):
        height = bar.get_height()
        if height > 0:
            ax6.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{data["total_spikes"]:,}\n({height:.1f}%)', 
                    ha='center', va='center', fontsize=7, color='white', weight='bold')
    
    for bar, data in zip(bars2, spike_data_4):
        height = bar.get_height()
        if height > 0:
            ax6.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{data["total_movements"]:,}\n({height:.1f}%)', 
                    ha='center', va='center', fontsize=7, color='white', weight='bold')
    
    ax6.set_xlabel('Market Cap Bucket', fontsize=10)
    ax6.set_ylabel('Percentage (%)', fontsize=10)
    ax6.set_title(f'Spikes vs Movements (4% Threshold)', fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels([d['bucket'] for d in spike_data_4], rotation=45, ha='right')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Intraday Volatility Pattern
    ax7 = plt.subplot(5, 3, 7)
    
    # Calculate volatility at different time segments
    time_segments = 10  # Split into 10 segments
    segment_volatilities = {bucket: [] for bucket in bucket_order}
    
    for bucket in bucket_order:
        bucket_data = buckets[bucket]
        segment_vols = [[] for _ in range(time_segments)]
        
        for item in bucket_data[:500]:  # Sample for speed
            dev_array = np.array(item['deviations'])
            seg_size = len(dev_array) // time_segments
            if seg_size > 0:
                for seg in range(time_segments):
                    seg_start = seg * seg_size
                    seg_end = min((seg + 1) * seg_size, len(dev_array))
                    seg_std = np.std(dev_array[seg_start:seg_end])
                    segment_vols[seg].append(seg_std)
        
        # Average across all files for each segment
        for seg_vol_list in segment_vols:
            if seg_vol_list:
                segment_volatilities[bucket].append(np.mean(seg_vol_list))
    
    x_seg = np.arange(time_segments)
    for i, bucket in enumerate(bucket_order):
        if segment_volatilities[bucket]:
            ax7.plot(x_seg[:len(segment_volatilities[bucket])], segment_volatilities[bucket], 
                    marker='o', label=bucket.split('. ')[1], linewidth=2)
    
    ax7.set_xlabel('Time Segment (0=start, 9=end)', fontsize=10)
    ax7.set_ylabel('Avg Volatility (std dev %)', fontsize=10)
    ax7.set_title('Intraday Volatility Pattern', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=7, loc='best')
    ax7.grid(True, alpha=0.3)
    
    # 8. Average Metrics Table
    ax8 = plt.subplot(5, 3, 8)
    ax8.axis('off')
    
    table_data = []
    for bucket in bucket_order:
        bucket_data = buckets[bucket]
        table_data.append([
            bucket.split('. ')[1],
            len(bucket_data),
            f"{np.mean([d['price_cv'] for d in bucket_data]):.4f}",
            f"{np.mean([d['returns_std'] for d in bucket_data]):.2f}",
            f"{np.mean([d['max_deviation'] for d in bucket_data]):.2f}"
        ])
    
    table = ax8.table(cellText=table_data,
                     colLabels=['Bucket', 'Count', 'Avg CV', 'Ret Std%', 'Max Dev%'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax8.set_title('Summary Statistics', fontsize=12, fontweight='bold')
    
    # 9. Mean Reversion vs Momentum Analysis
    ax9 = plt.subplot(5, 3, 9)
    
    for i, bucket in enumerate(bucket_order):
        bucket_data = buckets[bucket]
        if bucket_data:
            mean_revs = [item['mean_reversion'] for item in bucket_data[:1000]]
            momentums = [item['momentum'] for item in bucket_data[:1000]]
            
            # Filter outliers for better visualization
            mean_revs = [m for m in mean_revs if -1 <= m <= 1]
            momentums = [m for m in momentums if -5 <= m <= 5]
            
            if mean_revs and momentums:
                ax9.scatter(mean_revs, momentums, alpha=0.3, s=5, 
                          label=bucket.split('. ')[1])
    
    ax9.set_xlabel('Mean Reversion Score', fontsize=10)
    ax9.set_ylabel('Momentum Score (%)', fontsize=10)
    ax9.set_title('Mean Reversion vs Momentum by Market Cap', fontsize=12, fontweight='bold')
    ax9.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax9.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax9.legend(fontsize=7, loc='best')
    ax9.grid(True, alpha=0.3)
    
    # 10-14. Individual Deviation Heatmaps for each Market Cap Bucket
    for idx, bucket in enumerate(bucket_order):
        ax = plt.subplot(5, 3, 10 + idx)
        
        bucket_data = buckets[bucket]
        # Collect deviations for this bucket
        bucket_devs = []
        bucket_indices = []
        
        for j, item in enumerate(bucket_data[:300]):  # Sample more files for full spectrum
            devs = item['deviations']
            bucket_devs.extend(devs)
            bucket_indices.extend([j] * len(devs))
        
        if bucket_devs:
            # Create 2D histogram with full spectrum
            h = ax.hist2d(bucket_indices, bucket_devs, 
                         bins=[60, 80], cmap='YlOrRd', cmin=1, 
                         range=[[0, 300], [-15, 15]])
            
            bucket_name = bucket.split('. ')[1]
            ax.set_xlabel('File Index', fontsize=9)
            ax.set_ylabel('Deviation (%)', fontsize=9)
            ax.set_title(f'Heatmap: {bucket_name}', fontsize=10, fontweight='bold')
            ax.axhline(y=0, color='cyan', linestyle='--', alpha=0.7)
            plt.colorbar(h[3], ax=ax)
            
            # Add stats text
            mean_dev = np.mean(bucket_devs)
            std_dev = np.std(bucket_devs)
            ax.text(0.02, 0.98, f'μ={mean_dev:.2f}%\nσ={std_dev:.2f}%\nn={len(bucket_data)}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Overall stats
    all_data = []
    for bucket_data in buckets.values():
        all_data.extend(bucket_data)
    
    # Aggregate spike data for all thresholds
    spike_up_1 = sum(d['spike_up_counts'][1.0] for d in all_data)
    spike_down_1 = sum(d['spike_down_counts'][1.0] for d in all_data)
    move_up_1 = sum(d['movement_up_counts'][1.0] for d in all_data)
    move_down_1 = sum(d['movement_down_counts'][1.0] for d in all_data)
    
    spike_up_2 = sum(d['spike_up_counts'][2.0] for d in all_data)
    spike_down_2 = sum(d['spike_down_counts'][2.0] for d in all_data)
    move_up_2 = sum(d['movement_up_counts'][2.0] for d in all_data)
    move_down_2 = sum(d['movement_down_counts'][2.0] for d in all_data)
    
    spike_up_4 = sum(d['spike_up_counts'][4.0] for d in all_data)
    spike_down_4 = sum(d['spike_down_counts'][4.0] for d in all_data)
    move_up_4 = sum(d['movement_up_counts'][4.0] for d in all_data)
    move_down_4 = sum(d['movement_down_counts'][4.0] for d in all_data)
    
    stats_text = (
        f"Overall Statistics:\n"
        f"Total Files: {len(all_data):,}\n"
        f"Market Cap Buckets: {len(bucket_order)}\n"
        f"\nGlobal Metrics:\n"
        f"Avg Price CV: {np.mean([d['price_cv'] for d in all_data]):.4f}\n"
        f"Avg Returns Std: {np.mean([d['returns_std'] for d in all_data]):.2f}%\n"
        f"Avg Price Range: {np.mean([d['price_range'] for d in all_data]):.2f}%\n"
        f"\nSpike Analysis:\n"
        f"1% Threshold: ↑{spike_up_1+move_up_1:,} ↓{spike_down_1+move_down_1:,}\n"
        f"2% Threshold: ↑{spike_up_2+move_up_2:,} ↓{spike_down_2+move_down_2:,}\n"
        f"4% Threshold: ↑{spike_up_4+move_up_4:,} ↓{spike_down_4+move_down_4:,}\n"
        f"\nMarket Cap Range:\n"
        f"Min: ${min(d['marketcap_millions'] for d in all_data):.2f}M\n"
        f"Max: ${max(d['marketcap_millions'] for d in all_data):.2f}M"
    )
    
    # Add overall stats as text on the figure
    fig.text(0.85, 0.25, stats_text, transform=fig.transFigure, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('FUTURES Market Analysis by Market Cap', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save
    output_file = 'comprehensive_analysis_futures.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved as '{output_file}'")
    print(f"Total time: {time.time() - start_time:.1f} seconds")
    
    plt.show()

if __name__ == "__main__":
    main()
