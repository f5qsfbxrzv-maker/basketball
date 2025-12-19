"""
Create 38-feature dataset by removing:
1. Zero importance: fatigue_mismatch, is_season_opener, endgame_phase
2. Perfect duplicates: season_year_normalized, ewma_foul_synergy_away, season_progress
"""

import pandas as pd

print("="*70)
print("CREATING 38-FEATURE DATASET")
print("="*70)

# Load data
df = pd.read_csv('data/training_data_with_temporal_features.csv')
print(f"\nOriginal: {len(df):,} games, {len(df.columns)} columns")

# Features to remove
features_to_remove = [
    # Zero importance
    'fatigue_mismatch',
    'is_season_opener', 
    'endgame_phase',
    # Perfect duplicates (removing one from each pair)
    'season_year_normalized',  # Keep season_year
    'ewma_foul_synergy_away',  # Keep away_ewma_fta_rate
    'season_progress'          # Keep games_into_season
]

print(f"\nRemoving 6 features:")
for f in features_to_remove:
    print(f"  - {f}")

# Remove features
df_cleaned = df.drop(columns=features_to_remove)

# Verify feature count
exclude_cols = [
    'game_id', 'home_team', 'away_team', 'date', 'season',
    'target_spread', 'target_spread_cover', 'target_moneyline_win', 
    'target_over_under', 'target_game_total'
]
feature_cols = [c for c in df_cleaned.columns if c not in exclude_cols]

print(f"\nNew dataset: {len(df_cleaned):,} games, {len(feature_cols)} features")
print(f"Columns total: {len(df_cleaned.columns)}")

# Save
output_path = 'data/training_data_38features.csv'
df_cleaned.to_csv(output_path, index=False)

print(f"\n✓ Saved: {output_path}")
print(f"\nFeatures retained ({len(feature_cols)}):")
for i, f in enumerate(sorted(feature_cols), 1):
    print(f"  {i:2d}. {f}")

print("\n" + "="*70)
