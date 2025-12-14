"""
Add Explicit Features for Hidden Information in "Bugs"
Instead of reverting cleaning, add proper features for:
1. Season openers (was hidden in 286-day rest)
2. Season progression (was hidden in ELO inflation)
3. Team era/quality (was hidden in absolute ELO values)
"""

import pandas as pd
import numpy as np

print("="*70)
print("ADDING EXPLICIT FEATURES FOR HIDDEN SIGNALS")
print("="*70)

# Load clean data
print("\n1. Loading clean dataset...")
df = pd.read_csv('data/training_data_with_features_cleaned.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"   Games: {len(df):,}")
print(f"   Current features: {len([c for c in df.columns if c not in ['date','game_id','home_team','away_team','season'] and not c.startswith('target_')])}")

print("\n2. Creating explicit temporal features...")

# FEATURE 1: Is Season Opener (binary)
# Replaces 286-day rest signal
df['is_season_opener'] = (df['away_rest_days'] >= 100).astype(int) | (df['home_rest_days'] >= 100).astype(int)

# Actually, clean data set these to 3, so check original pattern
# Season openers happen late October every year
df['month'] = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear

# Season opener = first 5 games of season (Oct 25-31)
df['season_game_num'] = df.groupby('season').cumcount() + 1
df['is_season_opener'] = (df['season_game_num'] <= 5).astype(int)

print(f"   ✓ is_season_opener: {df['is_season_opener'].sum():,} games ({df['is_season_opener'].sum()/len(df)*100:.1f}%)")

# FEATURE 2: Season Year (numeric, captures league evolution)
# Replaces ELO inflation signal
df['season_year'] = df['season'].str[:4].astype(int)
df['season_year_normalized'] = (df['season_year'] - 2015) / 10  # 0-1 scale

print(f"   ✓ season_year: {df['season_year'].min()}-{df['season_year'].max()}")
print(f"   ✓ season_year_normalized: {df['season_year_normalized'].min():.2f}-{df['season_year_normalized'].max():.2f}")

# FEATURE 3: Games into Season (0-82, captures team form development)
# Early season = less data, more variance
df['games_into_season'] = df['season_game_num'] - 1  # 0-based
df['season_progress'] = df['games_into_season'] / 82  # 0-1 scale

print(f"   ✓ games_into_season: {df['games_into_season'].min()}-{df['games_into_season'].max()}")
print(f"   ✓ season_progress: {df['season_progress'].min():.2f}-{df['season_progress'].max():.2f}")

# FEATURE 4: Is Playoffs Approaching (last 10 games)
# Teams rest starters, tank for draft picks
df['endgame_phase'] = (df['games_into_season'] >= 72).astype(int)

print(f"   ✓ endgame_phase: {df['endgame_phase'].sum():,} games ({df['endgame_phase'].sum()/len(df)*100:.1f}%)")

# FEATURE 5: Month of Season (Oct=0, Apr=6, captures seasonal patterns)
season_month_map = {10: 0, 11: 1, 12: 2, 1: 3, 2: 4, 3: 5, 4: 6}
df['season_month'] = df['month'].map(season_month_map).fillna(0).astype(int)

print(f"   ✓ season_month: {df['season_month'].value_counts().sort_index().to_dict()}")

# Clean up temp columns
df = df.drop(columns=['month', 'day_of_year', 'season_game_num'], errors='ignore')

print("\n3. Feature statistics:")
print(f"   is_season_opener correlation with target: {df['is_season_opener'].corr(df['target_spread_cover']):+.4f}")
print(f"   season_year correlation with target: {df['season_year'].corr(df['target_spread_cover']):+.4f}")
print(f"   season_progress correlation with target: {df['season_progress'].corr(df['target_spread_cover']):+.4f}")
print(f"   endgame_phase correlation with target: {df['endgame_phase'].corr(df['target_spread_cover']):+.4f}")
print(f"   season_month correlation with target: {df['season_month'].corr(df['target_spread_cover']):+.4f}")

# Save
output_path = "data/training_data_with_temporal_features.csv"
df.to_csv(output_path, index=False)

total_features = len([c for c in df.columns if c not in ['date','game_id','home_team','away_team','season','target_spread','target_spread_cover','target_moneyline_win','target_game_total','target_over_under','target_home_cover','target_over']])

print(f"\n4. Saved: {output_path}")
print(f"   Total features: {total_features} (36 original + 5 temporal = 41)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("ADDED 5 EXPLICIT TEMPORAL FEATURES:")
print("  1. is_season_opener - Captures rusty team effect (was in 286-day rest)")
print("  2. season_year - League evolution 2015-2025 (was in ELO inflation)")
print("  3. season_year_normalized - Same, scaled 0-1")
print("  4. games_into_season - Team form development 0-82")
print("  5. season_progress - Same, scaled 0-1")
print("  6. endgame_phase - Playoffs approaching, rest/tanking")
print("  7. season_month - Oct-Apr seasonal patterns")
print("\nThese replace the hidden signals in the 'bugs' with explicit features.")
print("\nNEXT STEPS:")
print("  1. Train model with 41 features (36 + 5 temporal)")
print("  2. Compare AUC: Should recover lost 1.01% + improve further")
print("  3. If AUC > 0.560: Hyperparameter tune")
print("="*70)
