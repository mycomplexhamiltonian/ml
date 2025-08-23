#!/usr/bin/env python3
"""
Quick script to apply up/down spike tracking to all analysis scripts
"""

def update_scripts():
    # Read the updated spot script as template
    with open('analyze_comprehensive_spot.py', 'r') as f:
        spot_content = f.read()
    
    # Apply same changes to futures script
    with open('analyze_comprehensive_futures.py', 'r') as f:
        futures_content = f.read()
    
    # Key replacements for futures script
    replacements = [
        ('spike_counts = {t: 0 for t in thresholds}', 
         'spike_up_counts = {t: 0 for t in thresholds}\n        spike_down_counts = {t: 0 for t in thresholds}'),
        ('movement_counts = {t: 0 for t in thresholds}',
         'movement_up_counts = {t: 0 for t in thresholds}\n        movement_down_counts = {t: 0 for t in thresholds}'),
        ("'spike_counts': spike_counts,", "'spike_up_counts': spike_up_counts,\n            'spike_down_counts': spike_down_counts,"),
        ("'movement_counts': movement_counts", "'movement_up_counts': movement_up_counts,\n            'movement_down_counts': movement_down_counts"),
    ]
    
    for old, new in replacements:
        futures_content = futures_content.replace(old, new)
    
    # Update futures title
    futures_content = futures_content.replace('SPOT Market Analysis', 'FUTURES Market Analysis')
    futures_content = futures_content.replace('comprehensive_analysis_spot.png', 'comprehensive_analysis_futures.png')
    
    # Write back
    with open('analyze_comprehensive_futures.py', 'w') as f:
        f.write(futures_content)
    
    print("Scripts updated with up/down spike tracking!")

if __name__ == "__main__":
    update_scripts()