#!/usr/bin/env python3
"""Test spike detection logic with max/min within window"""

def test_spike_detection():
    # Test scenarios showing the importance of finding peak/trough
    test_cases = [
        {
            "name": "Spike with continued rise then reversal",
            "prices": [100, 102, 103, 101],  # +2% trigger, continues to 103, then drops
            "description": "Price rises 2%, continues to 3%, then drops back"
        },
        {
            "name": "Movement that keeps going up",
            "prices": [100, 102, 103, 103.5],  # +2% trigger, continues rising
            "description": "Price rises 2% and keeps rising"
        },
        {
            "name": "False positive without max check",
            "prices": [100, 102, 102.5, 102],  # Would look like no recovery without max
            "description": "Price rises 2%, peaks at 2.5%, slight drop"
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
            print(f"  {test['description']}")
            print(f"  Price sequence: {prices}")
            print(f"  Initial change: {price_change_pct:.2f}%")
            
            # OLD METHOD (just looking at single point)
            old_recovery_ratio = 0
            if price_change_pct > 0:
                recovery_from_single = (prices[1] - prices[3]) / prices[1] * 100
                old_recovery_ratio = recovery_from_single / price_change_pct
            
            # NEW METHOD (finding max/min in window)
            window_start = 1
            window_end = 3  # Simulating 1 second window
            
            if price_change_pct > 0:  # Price went up
                peak_price = max(prices[window_start:window_end+1])
                actual_peak_change = (peak_price - prices[0]) / prices[0] * 100
                recovery_from_peak = (peak_price - prices[window_end]) / peak_price * 100
                new_recovery_ratio = recovery_from_peak / actual_peak_change if actual_peak_change > 0 else 0
                
                print(f"  Peak within window: {peak_price} (+{actual_peak_change:.2f}%)")
            else:
                trough_price = min(prices[window_start:window_end+1])
                actual_trough_change = abs((trough_price - prices[0]) / prices[0] * 100)
                recovery_from_trough = (prices[window_end] - trough_price) / trough_price * 100
                new_recovery_ratio = recovery_from_trough / actual_trough_change if actual_trough_change > 0 else 0
                
                print(f"  Trough within window: {trough_price} (-{actual_trough_change:.2f}%)")
            
            print(f"  OLD recovery ratio (single point): {old_recovery_ratio:.2f}")
            print(f"  NEW recovery ratio (with max/min): {new_recovery_ratio:.2f}")
            
            # Classification with new method
            if new_recovery_ratio >= spike_recovery_threshold:
                result = "SPIKE"
            elif new_recovery_ratio < movement_recovery_threshold:
                result = "MOVEMENT"
            else:
                result = "NEUTRAL"
            
            print(f"  Classification: {result}")

if __name__ == "__main__":
    test_spike_detection()