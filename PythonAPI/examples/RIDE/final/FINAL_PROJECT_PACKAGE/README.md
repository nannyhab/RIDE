# RIDE - Autonomous Vehicle Parameter Optimization Dataset

## Dataset Overview
- **Total Points:** 256
- **Scenarios:** 16 complete scenarios
- **Parameters Tested:** 4 gear ratios × 4 tire frictions = 16 configurations per scenario
- **Key Finding:** Gear ratio optimization improves performance by 31.7%

## Files in This Package

### 1. `complete_dataset.json` (RECOMMENDED FOR VIEWING)
**Best for:** Browsing data in text editor, understanding structure
- Human-readable JSON format
- Each data point contains:
  - `track_id`: Scenario number
  - `route_type`: Type of scenario (straight_speed, tight_curves, elevation, mixed)
  - `gear_ratio`: 1.5, 2.5, 4.0, or 6.0
  - `tire_friction`: 0.5, 1.0, 2.0, or 3.0
  - `travel_time`: Performance metric (seconds)
  - Track features: `total_length`, `avg_curvature`, `elevation_range`, etc.

### 2. `complete_dataset.pkl` (FOR PYTHON ANALYSIS)
**Best for:** Loading directly into Python for analysis
```python
import pickle
with open('complete_dataset.pkl', 'rb') as f:
    data = pickle.load(f)
# data is a list of 256 dictionaries
```

### 3. `data_summary.json`
**Best for:** Quick overview of dataset statistics
- Total points per scenario
- Breakdown by scenario type
- List of all scenarios collected

### 4. `parameter_analysis.json`
**Best for:** Pre-computed parameter optimization results
- Gear ratio effects
- Best/worst parameters
- Performance improvements

## Quick Start for Analysis

### Option 1: View in Text Editor
```bash
# Any text editor works
cat complete_dataset.json | less
# Or open in VS Code, Sublime, etc.
```

### Option 2: Python Analysis
```python
import json
import pandas as pd

# Load data
with open('complete_dataset.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Analyze gear ratio effects
print(df.groupby('gear_ratio')['travel_time'].mean())

# Analyze by scenario type
print(df.groupby('route_type')['travel_time'].describe())
```

### Option 3: Excel/Sheets (if needed)
```python
import pandas as pd
df = pd.DataFrame(data)
df.to_csv('dataset.csv', index=False)
# Now open dataset.csv in Excel/Google Sheets
```

## Scenario Details

### Straight Speed (0-4): 80 points
- Tests gear ratio impact on acceleration and top speed
- Minimal curves, flat terrain

### Tight Curves (5-9): 80 points
- Tests handling through curved roads
- Town02 winding routes

### Elevation (10-14): 80 points
- Tests gear ratio on 11-meter climbs
- Town04 highway routes

### Mixed (20): 16 points
- Combined straight, curves, and some elevation

## Key Findings

**Gear Ratio Optimization:**
- Best: 6.0 (56.3s average)
- Worst: 1.5 (82.4s average)
- Improvement: 26.1 seconds (31.7%)

**Tire Friction:**
- Minimal effect (1-2 seconds variation)
- Curves not sharp enough to stress tire limits
- CARLA autopilot drives conservatively

## Contact
Dataset collected by: [Your name]
Northwestern University - [Your course]
Date: December 2025
