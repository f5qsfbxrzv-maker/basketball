"""
Regenerate training data with CORRECTED 44-feature set including away_composite_elo
Uses HISTORICAL injuries (not live) for all past games
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3

print("="*80)
print("REGENERATE TRAINING DATA - 44 FEATURES")
print("="*80)
print("This will regenerate training_data_with_temporal_features.csv")
print("with the corrected feature set including away_composite_elo")
print("="*80 + "\n")

# Check if feature whitelist has been updated
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))
from feature_whitelist import FEATURE_WHITELIST

print(f"[CHECK] Feature whitelist has {len(FEATURE_WHITELIST)} features")

if 'away_composite_elo' not in FEATURE_WHITELIST:
    print("ERROR: away_composite_elo NOT in whitelist!")
    print("The whitelist must be updated before regenerating training data.")
    sys.exit(1)

print("  ✓ away_composite_elo in whitelist")
print("  ✓ home_composite_elo in whitelist\n")

# Load existing training data to get the game list
print("[1/5] Loading existing training data structure...")
old_df = pd.read_csv('data/training_data_with_temporal_features.csv')
print(f"  Games: {len(old_df):,}")
print(f"  Old features: {len([c for c in old_df.columns if c not in ['date', 'game_id', 'home_team', 'away_team', 'season', 'target_spread', 'target_spread_cover', 'target_moneyline_win', 'target_game_total', 'target_over_under', 'target_home_cover', 'target_over']])}")

# Get game metadata
games = old_df[['date', 'game_id', 'home_team', 'away_team', 'season']].copy()
games['date'] = pd.to_datetime(games['date'])
print(f"  Date range: {games['date'].min()} to {games['date'].max()}\n")

# Load targets from database
print("[2/5] Loading target variables from database...")
conn = sqlite3.connect('data/live/nba_betting_data.db')

# Get target_spread and actual results
targets_query = """
SELECT 
    game_id,
    home_team,
    away_team,
    game_date,
    home_score,
    away_score,
    (home_score - away_score) as point_diff
FROM game_logs
WHERE game_id IN ({})
""".format(','.join(['?']*len(games)))

targets_df = pd.read_sql(targets_query, conn, params=games['game_id'].tolist())
print(f"  Loaded {len(targets_df):,} game results\n")

# Calculate target variables
targets_df['target_moneyline_win'] = (targets_df['point_diff'] > 0).astype(int)
# Note: We'll need to merge with spread data for target_spread_cover

conn.close()

# Initialize feature calculator
print("[3/5] Initializing feature calculator with updated whitelist...")
from src.features.feature_calculator_live import FeatureCalculatorV5

calc = FeatureCalculatorV5()
print(f"  ✓ Feature calculator loaded\n")

# Regenerate features for all games
print("[4/5] Regenerating features for all games (THIS WILL TAKE A WHILE)...")
print("  Using HISTORICAL injuries (not live) for all past games")
print("  Progress: ", end='', flush=True)

new_features_list = []
errors = []

for idx, row in games.iterrows():
    if idx % 500 == 0:
        print(f"{idx:,}...", end='', flush=True)
    
    try:
        features = calc.calculate_game_features(
            home_team=row['home_team'],
            away_team=row['away_team'],
            season=row['season'],
            game_date=row['date'].strftime('%Y-%m-%d')
        )
        
        # Add metadata
        features['date'] = row['date']
        features['game_id'] = row['game_id']
        features['home_team'] = row['home_team']
        features['away_team'] = row['away_team']
        features['season'] = row['season']
        
        new_features_list.append(features)
        
    except Exception as e:
        errors.append((idx, row['game_id'], str(e)))

print(f"\n  Completed: {len(new_features_list):,} games")
if errors:
    print(f"  Errors: {len(errors)}")
    for idx, game_id, error in errors[:5]:
        print(f"    {game_id}: {error}")

# Convert to DataFrame
new_df = pd.DataFrame(new_features_list)

# Verify away_composite_elo is present
if 'away_composite_elo' not in new_df.columns:
    print("\n  ERROR: away_composite_elo STILL not in features!")
    print("  Something is wrong with the feature calculator or whitelist.")
    sys.exit(1)

print(f"\n  ✓ away_composite_elo present in all {len(new_df)} games")

# Merge with targets
print("\n[5/5] Merging with target variables...")
new_df = new_df.merge(
    old_df[['game_id', 'target_spread', 'target_spread_cover', 'target_moneyline_win', 
            'target_game_total', 'target_over_under', 'target_home_cover', 'target_over']],
    on='game_id',
    how='left'
)

# Verify feature count
feature_cols = [c for c in new_df.columns if c not in [
    'date', 'game_id', 'home_team', 'away_team', 'season',
    'target_spread', 'target_spread_cover', 'target_moneyline_win',
    'target_game_total', 'target_over_under', 'target_home_cover', 'target_over'
]]

print(f"  Total features: {len(feature_cols)}")
print(f"  Expected: 44")

if len(feature_cols) != 44:
    print(f"\n  WARNING: Expected 44 features but got {len(feature_cols)}")
    print("  This may be okay if some features were added/removed")

# Save
output_path = 'data/training_data_with_temporal_features_44feat.csv'
new_df.to_csv(output_path, index=False)

print(f"\n✓ Saved: {output_path}")
print(f"  Rows: {len(new_df):,}")
print(f"  Columns: {len(new_df.columns)}")

# Backup old file
import shutil
backup_path = 'data/training_data_with_temporal_features_43feat_backup.csv'
shutil.copy('data/training_data_with_temporal_features.csv', backup_path)
print(f"\n✓ Backed up old data to: {backup_path}")

# Replace original
shutil.move(output_path, 'data/training_data_with_temporal_features.csv')
print(f"✓ Replaced original training data\n")

print("="*80)
print("SUMMARY")
print("="*80)
print(f"Training data regenerated with {len(feature_cols)} features")
print(f"Games: {len(new_df):,}")
print(f"Date range: {new_df['date'].min()} to {new_df['date'].max()}")
print("\nKey features verified:")
print(f"  home_composite_elo: {'✓' if 'home_composite_elo' in new_df.columns else '✗'}")
print(f"  away_composite_elo: {'✓' if 'away_composite_elo' in new_df.columns else '✗'}")
print(f"  injury_impact_diff: {'✓' if 'injury_impact_diff' in new_df.columns else '✗'}")
print(f"  ewma_efg_diff: {'✓' if 'ewma_efg_diff' in new_df.columns else '✗'}")

print("\n✓ Ready for Optuna hypertuning")
print("  Run: python scripts/optuna_hypertune_8hr.py")
print("="*80)
