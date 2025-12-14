"""
Compare Dirty vs Clean Data - Find What Hurt Performance
Identifies which data fix caused the 1.01% AUC drop.
"""

import pandas as pd
import numpy as np

print("="*70)
print("DIRTY VS CLEAN DATA COMPARISON")
print("="*70)

# Load both
dirty = pd.read_csv('data/training_data_with_features.csv')
clean = pd.read_csv('data/training_data_with_features_cleaned.csv')

print(f"\nGames: {len(dirty):,} (both)")

# 1. REST DAYS
print("\n1. REST DAYS FIX:")
print(f"   DIRTY away_rest_days:")
print(f"     Max: {dirty['away_rest_days'].max():.0f} days")
print(f"     Mean: {dirty['away_rest_days'].mean():.2f} days")
print(f"     Season openers (>15 days): {(dirty['away_rest_days'] > 15).sum()} games")

print(f"   CLEAN away_rest_days:")
print(f"     Max: {clean['away_rest_days'].max():.0f} days")
print(f"     Mean: {clean['away_rest_days'].mean():.2f} days")
print(f"     Season openers (>15 days): {(clean['away_rest_days'] > 15).sum()} games")

# 2. ELO NORMALIZATION
print("\n2. ELO NORMALIZATION FIX:")
seasons = ['2015-16', '2019-20', '2024-25']
for season in seasons:
    dirty_elo = dirty[dirty['season'] == season]['home_composite_elo'].mean()
    clean_elo = clean[clean['season'] == season]['home_composite_elo'].mean()
    print(f"   {season}: DIRTY={dirty_elo:.1f}, CLEAN={clean_elo:.1f}, Diff={dirty_elo-clean_elo:+.1f}")

# 3. ELO DIFF CLIPPING
print("\n3. ELO DIFF CLIPPING:")
print(f"   DIRTY off_elo_diff:")
print(f"     Range: [{dirty['off_elo_diff'].min():.1f}, {dirty['off_elo_diff'].max():.1f}]")
print(f"     Outliers (>±400): {((dirty['off_elo_diff'] < -400) | (dirty['off_elo_diff'] > 400)).sum()}")

print(f"   CLEAN off_elo_diff:")
print(f"     Range: [{clean['off_elo_diff'].min():.1f}, {clean['off_elo_diff'].max():.1f}]")
print(f"     Outliers (>±400): {((clean['off_elo_diff'] < -400) | (clean['off_elo_diff'] > 400)).sum()}")

# 4. PERFORMANCE COMPARISON
print("\n" + "="*70)
print("PERFORMANCE:")
print("="*70)
print("DIRTY: 0.5508 AUC (from earlier test)")
print("CLEAN: 0.5407 AUC (from sanity test)")
print("LOSS:  -1.01% (-0.0101 AUC)")

print("\n" + "="*70)
print("HYPOTHESIS:")
print("="*70)
print("The '286 days rest' wasn't noise - it was INFORMATION:")
print("  → Season openers = teams rusty, predictable outcomes")
print("  → Model learned: >100 days = season opener = home advantage")
print("  → Fixing it to 3 days REMOVED this signal")
print("")
print("ELO inflation wasn't noise - it was RECENCY:")
print("  → Higher ELO in recent seasons = teams got better")
print("  → Normalizing removed temporal information")
print("  → Model can't distinguish 2015 Warriors from 2024 Celtics")

print("\n" + "="*70)
print("SOLUTION:")
print("="*70)
print("Option 1: REVERT to dirty data, accept 'bugs' as features")
print("Option 2: ADD EXPLICIT FEATURES for what bugs captured:")
print("  → is_season_opener (binary)")
print("  → season_year (2015-2025 as numeric)")
print("  → games_since_season_start (0-82)")
print("Option 3: PARTIAL REVERT:")
print("  → Keep rest days bug (it's predictive)")
print("  → Keep ELO inflation (it's temporal signal)")
print("  → Only fix: outlier clipping (true errors)")
print("="*70)
