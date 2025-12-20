"""
🏗️ BUILD MDP TRAINING DATA WITH REAL SCORES
============================================
Merges master_training_data_v6.csv (with actual scores) with 
training_data_GOLD_ELO_22_features.csv (with engineered features)
to create the ultimate MDP training dataset.

Output: training_data_MDP_with_margins.csv
- All 19 Variant D features
- REAL point margins (home_score - away_score)
- Ready for regression training
"""

import pandas as pd
import numpy as np

print("="*80)
print("🏗️  BUILDING MDP TRAINING DATA WITH REAL SCORES")
print("="*80)

# 1. Load master data with scores
print("\n📂 Loading master data with actual scores...")
master_df = pd.read_csv('data/master_training_data_v6.csv')
master_df['date'] = pd.to_datetime(master_df['date'])

print(f"   ✓ Loaded {len(master_df):,} games")
print(f"   ✓ Date range: {master_df['date'].min().date()} to {master_df['date'].max().date()}")
print(f"   ✓ All games have scores: {master_df['home_score'].notna().all()}")

# 2. Load feature data
print("\n📂 Loading engineered features...")
features_df = pd.read_csv('data/training_data_GOLD_ELO_22_features.csv')
features_df['date'] = pd.to_datetime(features_df['date'])

print(f"   ✓ Loaded {len(features_df):,} games")
print(f"   ✓ Features: {len([c for c in features_df.columns if c not in ['date', 'game_date', 'home_team', 'away_team', 'target_moneyline_win', 'season', 'game_id']])}")

# 3. Merge on date + teams
print("\n🔗 Merging datasets...")
merged_df = features_df.merge(
    master_df[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'actual_spread', 'total_points']],
    on=['date', 'home_team', 'away_team'],
    how='left'
)

print(f"   ✓ Merged dataset: {len(merged_df):,} games")

# 4. Calculate margin target (the key!)
merged_df['margin_target'] = merged_df['home_score'] - merged_df['away_score']

# Verify it matches actual_spread
if 'actual_spread' in merged_df.columns:
    matches = (merged_df['margin_target'] == merged_df['actual_spread']).sum()
    print(f"   ✓ Margin verification: {matches}/{len(merged_df)} matches with actual_spread")

# 5. Statistics
print("\n📊 MARGIN STATISTICS:")
print(f"   Games with margins: {merged_df['margin_target'].notna().sum():,}/{len(merged_df):,}")
print(f"   Margin range: {merged_df['margin_target'].min():.0f} to {merged_df['margin_target'].max():.0f}")
print(f"   Margin mean: {merged_df['margin_target'].mean():.2f}")
print(f"   Margin std dev: {merged_df['margin_target'].std():.2f} ⭐ (NBA typical: ~13.5)")
print(f"   Median margin: {merged_df['margin_target'].median():.1f}")

# Distribution
print(f"\n📈 MARGIN DISTRIBUTION:")
bins = [-60, -20, -10, -5, 0, 5, 10, 20, 60]
labels = ['Blowout Loss (<-20)', 'Big Loss (-20 to -10)', 'Close Loss (-10 to -5)', 
          'Tight Loss (-5 to 0)', 'Tight Win (0 to 5)', 'Close Win (5 to 10)', 
          'Big Win (10 to 20)', 'Blowout Win (>20)']

merged_df['margin_bin'] = pd.cut(merged_df['margin_target'], bins=bins, labels=labels)
for label in labels:
    count = (merged_df['margin_bin'] == label).sum()
    pct = count / len(merged_df) * 100
    print(f"   {label:<25}: {count:>5} games ({pct:>5.1f}%)")

# 6. Save
output_path = 'data/training_data_MDP_with_margins.csv'
merged_df.to_csv(output_path, index=False)

print(f"\n✅ SAVED: {output_path}")
print(f"   Size: {len(merged_df):,} games")
print(f"   Columns: {len(merged_df.columns)}")

# Print key columns for verification
key_cols = ['date', 'home_team', 'away_team', 'margin_target', 'home_score', 'away_score', 
            'off_elo_diff', 'def_elo_diff', 'target_moneyline_win']
key_cols = [c for c in key_cols if c in merged_df.columns]

print(f"\n🔍 SAMPLE DATA:")
print(merged_df[key_cols].head(10).to_string())

print("\n" + "="*80)
print("🎯 READY FOR MDP TRAINING!")
print("   Next: Update daily_picks_mdp.py to use this file")
print("   Command: python daily_picks_mdp.py")
print("="*80)
