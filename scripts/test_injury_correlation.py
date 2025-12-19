"""
TEST 2: CORRELATION UPGRADE CHECK
==================================

Verify that injury_matchup_advantage is stronger than the 8 individual features.

Goal: Correlation with home_win should be stronger (more negative) than -0.10
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("TEST 2: CORRELATION UPGRADE CHECK")
print("=" * 80)
print()

# Load data with injury features
DATA_PATH = Path('data/training_data_with_injury_shock.csv')

if not DATA_PATH.exists():
    print(f"❌ Data file not found: {DATA_PATH}")
    exit(1)

print(f"📂 Loading: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {len(df)} games")
print()

# Calculate injury_matchup_advantage from raw components
print("📊 Calculating injury_matchup_advantage...")
df['injury_matchup_advantage'] = (
    0.008127 * df['injury_impact_diff']
  - 0.023904 * df['injury_shock_diff']
  + 0.031316 * df['star_mismatch']
)
print("✅ Feature calculated")
print()

# Get target variable
target = 'target_moneyline_win'
if target not in df.columns:
    print(f"❌ Target column '{target}' not found")
    exit(1)

# Calculate correlations
print("=" * 80)
print("CORRELATION WITH HOME WIN")
print("=" * 80)
print()

# Old individual features
old_features = [
    'injury_impact_diff',
    'injury_impact_abs',
    'injury_shock_home',
    'injury_shock_away',
    'injury_shock_diff',
    'home_star_missing',
    'away_star_missing',
    'star_mismatch'
]

print("OLD INDIVIDUAL FEATURES:")
print("-" * 40)
correlations = {}
for feat in old_features:
    if feat in df.columns:
        corr = df[feat].corr(df[target])
        correlations[feat] = corr
        print(f"  {feat:30s}: {corr:+.4f}")
    else:
        print(f"  {feat:30s}: MISSING")

print()
print("NEW COMBINED FEATURE:")
print("-" * 40)
new_corr = df['injury_matchup_advantage'].corr(df[target])
print(f"  injury_matchup_advantage      : {new_corr:+.4f}")
print()

# Compare
print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print()

# Get absolute values for comparison
old_corrs_abs = [abs(c) for c in correlations.values() if not np.isnan(c)]
max_old_corr = max(old_corrs_abs) if old_corrs_abs else 0
new_corr_abs = abs(new_corr)

print(f"Strongest old feature:  {max_old_corr:.4f} (absolute)")
print(f"New combined feature:   {new_corr_abs:.4f} (absolute)")
print()

# Goal check
GOAL_THRESHOLD = 0.10
print(f"Goal: Correlation > {GOAL_THRESHOLD}")
print()

if new_corr_abs > GOAL_THRESHOLD:
    print(f"✅ PASS: {new_corr_abs:.4f} > {GOAL_THRESHOLD}")
else:
    print(f"⚠️  WEAK: {new_corr_abs:.4f} ≤ {GOAL_THRESHOLD}")
    print("   (Still useful but weaker than expected)")

print()

# Is it better than old features?
if new_corr_abs > max_old_corr:
    print(f"✅ UPGRADE: New feature is {new_corr_abs/max_old_corr:.2f}x stronger")
elif new_corr_abs >= max_old_corr * 0.9:
    print(f"✅ COMPARABLE: New feature is {new_corr_abs/max_old_corr:.2%} of best old feature")
else:
    print(f"❌ DOWNGRADE: New feature is weaker ({new_corr_abs/max_old_corr:.2%})")

print()
print("=" * 80)
print("VERDICT")
print("=" * 80)
print()

if new_corr_abs > GOAL_THRESHOLD:
    print("✅ READY FOR PRODUCTION")
    print()
    print("Next steps:")
    print("  1. Regenerate training data with injury_matchup_advantage")
    print("  2. Train 20-feature model (19 + injury feature)")
    print("  3. Verify feature appears in Top 10 importance")
else:
    print("⚠️  PROCEED WITH CAUTION")
    print()
    print("The feature has weak correlation but may still be useful")
    print("in combination with other features (ensemble effect)")

# Show distribution
print()
print("=" * 80)
print("FEATURE DISTRIBUTION")
print("=" * 80)
print()
print(df['injury_matchup_advantage'].describe())
print()

# Show extreme cases
print("Most favorable to home (away injured):")
top_home = df.nlargest(5, 'injury_matchup_advantage')[
    ['injury_matchup_advantage', 'injury_impact_diff', 'injury_shock_diff', 'star_mismatch']
]
print(top_home.to_string())
print()

print("Most favorable to away (home injured):")
top_away = df.nsmallest(5, 'injury_matchup_advantage')[
    ['injury_matchup_advantage', 'injury_impact_diff', 'injury_shock_diff', 'star_mismatch']
]
print(top_away.to_string())
