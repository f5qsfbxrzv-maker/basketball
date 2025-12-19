"""
Add injury_matchup_advantage to training_data_with_injury_shock.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Input/output paths
INPUT_FILE = Path('data/training_data_with_injury_shock.csv')
OUTPUT_FILE = Path('data/training_data_with_injury_shock_plus_matchup.csv')

print("=" * 80)
print("ADDING injury_matchup_advantage TO TRAINING DATA")
print("=" * 80)
print()

# Load data
print(f"📂 Loading: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"✅ Loaded {len(df)} games with {len(df.columns)} columns")
print()

# Check required columns
required = ['injury_impact_diff', 'injury_shock_diff', 'star_mismatch']
missing = [c for c in required if c not in df.columns]

if missing:
    print(f"❌ Missing required columns: {missing}")
    exit(1)

print("✅ All required injury components present")
print()

# Calculate injury_matchup_advantage using optimized formula
print("📊 Calculating injury_matchup_advantage...")
print()
print("Formula:")
print("  injury_matchup_advantage = (")
print("      0.008127 * injury_impact_diff")
print("    - 0.023904 * injury_shock_diff")
print("    + 0.031316 * star_mismatch")
print("  )")
print()

df['injury_matchup_advantage'] = (
    0.008127 * df['injury_impact_diff']
  - 0.023904 * df['injury_shock_diff']
  + 0.031316 * df['star_mismatch']
)

print("✅ Feature calculated")
print()

# Show statistics
print("Feature Statistics:")
print(df['injury_matchup_advantage'].describe())
print()

# Show distribution
print("Distribution:")
print(f"  Games with positive (home advantage): {(df['injury_matchup_advantage'] > 0).sum()}")
print(f"  Games with negative (away advantage): {(df['injury_matchup_advantage'] < 0).sum()}")
print(f"  Games with neutral (0):               {(df['injury_matchup_advantage'] == 0).sum()}")
print()

# Check correlation with target
target_col = 'target_moneyline_win'
if target_col in df.columns:
    corr = df['injury_matchup_advantage'].corr(df[target_col])
    print(f"Correlation with {target_col}: {corr:+.4f}")
    print()

# Save
print(f"💾 Saving to: {OUTPUT_FILE}")
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved {len(df)} games with {len(df.columns)} columns")
print()

print("=" * 80)
print("SUCCESS")
print("=" * 80)
print()
print(f"New dataset ready: {OUTPUT_FILE}")
print(f"Total features: {len(df.columns)}")
print()
print("Next step: Update tuning script to use this dataset")
