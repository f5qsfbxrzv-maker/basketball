"""
Refine 38-feature dataset to 36 features:
- KEEP: season_year, season_progress (core time anchors)
- DROP: season_month, games_into_season (redundant with season_progress)
"""

import pandas as pd

print("="*70)
print("REFINING TO 36 FEATURES - CORE TIME ANCHORS")
print("="*70)

# Load 38-feature dataset
df = pd.read_csv('data/training_data_38features.csv')
print(f"\nCurrent: {len(df):,} games")

# Check what we have
print("\nTime anchor features in current dataset:")
time_features = ['season_year', 'season_year_normalized', 'season_progress', 
                 'season_month', 'games_into_season']
for f in time_features:
    status = "✓ Present" if f in df.columns else "✗ Missing"
    print(f"  {f:<30} {status}")

# Apply refinement
features_to_drop = []
features_to_add_back = []

# Check what needs to be dropped
if 'season_month' in df.columns:
    features_to_drop.append('season_month')
if 'games_into_season' in df.columns:
    features_to_drop.append('games_into_season')

print(f"\nChanges needed:")
if features_to_drop:
    print(f"  DROP: {', '.join(features_to_drop)}")
else:
    print("  DROP: None (already removed)")

# Check if season_progress exists
if 'season_progress' not in df.columns:
    print("  ADD BACK: season_progress (was incorrectly removed)")
    # Need to reload from original data
    df_original = pd.read_csv('data/training_data_with_temporal_features.csv')
    if 'season_progress' in df_original.columns:
        df['season_progress'] = df_original['season_progress']
        print("    ✓ Added back season_progress")
else:
    print("  KEEP: season_progress (already present)")

# Drop redundant features
if features_to_drop:
    df = df.drop(columns=features_to_drop)
    print(f"  ✓ Dropped {len(features_to_drop)} features")

# Verify final count
exclude_cols = [
    'game_id', 'home_team', 'away_team', 'date', 'season',
    'target_spread', 'target_spread_cover', 'target_moneyline_win', 
    'target_over_under', 'target_game_total'
]
feature_cols = [c for c in df.columns if c not in exclude_cols]

print(f"\nFinal dataset: {len(df):,} games, {len(feature_cols)} features")

# Save
output_path = 'data/training_data_36features.csv'
df.to_csv(output_path, index=False)

print(f"\n✓ Saved: {output_path}")

print("\nCore Time Anchors (retained):")
print("  ✓ season_year       - Handles 'Era' (pace evolution, rule changes)")
print("  ✓ season_progress   - Handles 'Timing' (early season vs. playoff push)")

print("\n" + "="*70)
