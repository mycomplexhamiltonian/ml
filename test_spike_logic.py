#!/usr/bin/env python3
"""Test spike detection logic with sample data"""

import numpy as np

def test_spike_detection():
    # Test scenarios
    test_cases = [
        {
            "name": "2% spike (full recovery)",
            "prices": [100, 102, 100],  # +2% then back to 100
            "expected": "spike"
        },
        {
            "name": "2% movement (no recovery)",
            "prices": [100, 102, 102],  # +2% and stays there
            "expected": "movement"
        },
        {
            "name": "2% partial recovery (50%)",
            "prices": [100, 102, 101],  # +2% then recovers 50%
            "expected": "neutral"
        },
        {
            "name": "-2% spike (full recovery)",
            "prices": [100, 98, 100],  # -2% then back to 100
            "expected": "spike"
        },
        {
            "name": "-2% movement (no recovery)",
            "prices": [100, 98, 98],  # -2% and stays there
            "expected": "movement"
        }
    ]
    
    threshold = 1.0  # 1% threshold
    movement_recovery_threshold = 0.4
    spike_recovery_threshold = 0.6
    
    for test in test_cases:
        prices = test["prices"]
        
        # Check price change from tick 0 to tick 1
        price_change_pct = (prices[1] - prices[0]) / prices[0] * 100
        
        if abs(price_change_pct) > threshold:
            print(f"\n{test['name']}")
            print(f"  Price sequence: {prices}")
            print(f"  Initial change: {price_change_pct:.2f}%")
            
            # Classification after 1 tick (simulating 1 second)
            if price_change_pct > 0:  # Price went up
                peak_price = prices[1]
                recovery_price = prices[2]
                recovery_from_peak = (peak_price - recovery_price) / peak_price * 100
                recovery_ratio = recovery_from_peak / price_change_pct
            else:  # Price went down
                trough_price = prices[1]
                recovery_price = prices[2]
                recovery_from_trough = (recovery_price - trough_price) / trough_price * 100
                recovery_ratio = recovery_from_trough / abs(price_change_pct)
            
            print(f"  Recovery ratio: {recovery_ratio:.2f}")
            
            if recovery_ratio >= spike_recovery_threshold:
                result = "SPIKE"
            elif recovery_ratio < movement_recovery_threshold:
                result = "MOVEMENT"
            else:
                result = "NEUTRAL"
            
            print(f"  Classification: {result}")
            print(f"  Expected: {test['expected'].upper()}")
            print(f"  ✓ PASS" if result == test['expected'].upper() else f"  ✗ FAIL")

if __name__ == "__main__":
    test_spike_detection()