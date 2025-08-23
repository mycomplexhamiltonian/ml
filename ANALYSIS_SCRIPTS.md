# Financial Data Analysis Scripts

## Main Analysis Scripts (3 Total)

### 1. `analyze_volatility_marketcap.py`
**Purpose**: Deep dive into volatility vs market cap relationship
- Uses 5 data-driven market cap buckets based on volatility patterns
- Multiple volatility metrics (CV, returns std, max deviation)
- Box plots, correlation matrices, statistical summaries
- Outputs: `volatility_marketcap_enhanced.png`, detailed CSV files

### 2. `analyze_comprehensive.py`
**Purpose**: All-in-one analysis combining deviations, trajectories, and volatility
- Analyzes price deviations from baseline (first 10 sec median)
- Price trajectories within 10-minute windows
- All metrics split by market cap buckets
- 5 separate deviation heatmaps (one per bucket)
- Output: `comprehensive_analysis.png`

### 3. `analyze_volatility_plot.py`
**Purpose**: Generate detailed volatility vs market cap plot
- Creates binned analysis with fine granularity
- Shows volatility patterns across market cap ranges
- Identifies natural breakpoints
- Output: `volatility_vs_marketcap_complete.png`

## Market Cap Buckets (Data-Driven from Volatility Analysis)

Based on analysis of 20,000+ files showing natural volatility groupings:

1. **Small (<$40M)** - CV ~0.0043 - Highest volatility
2. **Lower-Mid ($40-80M)** - CV ~0.0041 - Stability sweet spot
3. **Upper-Mid ($80-200M)** - CV ~0.0042 - Volatile period (includes $80-100M spike)
4. **Large ($200-700M)** - CV ~0.0035 - Stabilizing
5. **Mega (>$700M)** - CV ~0.0030 - Most stable

## Key Findings

- **$60-80M range**: Unexpected stability pocket
- **$80-100M range**: Highest volatility spike
- **$400-700M transition**: Major drop in volatility (institutional involvement)
- **>$700M**: Stable, mature assets

## Usage

```bash
# For detailed volatility analysis by market cap
python3 analyze_volatility_marketcap.py

# For comprehensive all-in-one analysis
python3 analyze_comprehensive.py

# To regenerate the volatility vs market cap plot
python3 analyze_volatility_plot.py
```

All scripts:
- Use all CPU cores for parallel processing
- Skip files with 0 market cap (data errors)
- Process up to 100K files efficiently
- Export data to CSV for further analysis
