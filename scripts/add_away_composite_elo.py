"""
Add away_composite_elo to existing training data
This is faster than regenerating all 12,000+ games from scratch
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime

print("="*80)
print("ADD away_composite_elo TO EXISTING TRAINING DATA")
print("="*80)

# Load existing data
print("\n[1/3] Loading existing training data...")
df = pd.read_csv('data/training_data_with_temporal_features.csv')
print(f"  Rows: {len(df):,}")
print(f"  Current features: {len([c for c in df.columns if c not in ['date', 'game_id', 'home_team', 'away_team', 'season', 'target_spread', 'target_spread_cover', 'target_moneyline_win', 'target_game_total', 'target_over_under', 'target_home_cover', 'target_over']])}")

# Check if already has away_composite_elo
if 'away_composite_elo' in df.columns:
    print("  WARNING: away_composite_elo already exists in data")
    print("  Regenerating it anyway...")
    df = df.drop(columns=['away_composite_elo'])

# Initialize ELO system
print("\n[2/3] Loading ELO ratings from database...")
from src.features.off_def_elo_system import OffDefEloSystem

elo_system = OffDefEloSystem('data/live/nba_betting_data.db')
print("  OK - ELO system loaded")

# Add away_composite_elo for each game
print("\n[3/3] Adding away_composite_elo for all games...")
print("  Progress: ", end='', flush=True)

away_elos = []
errors = 0

for idx, row in df.iterrows():
    if idx % 1000 == 0:
        print(f"{idx:,}...", end='', flush=True)
    
    try:
        # Parse game date
        game_date = pd.to_datetime(row['date'])
        
        # Determine season from date
        year = game_date.year
        month = game_date.month
        season = f"{year-1}-{str(year)[-2:]}" if month < 10 else f"{year}-{str(year+1)[-2:]}"
        
        # Get away team ELO as of game date
        away_team_elo = elo_system.get_latest(
            row['away_team'],
            season=season,
            before_date=game_date.strftime('%Y-%m-%d')
        )
        
        if away_team_elo:
            away_elos.append(away_team_elo.composite)
        else:
            # Fallback to 1500 if no ELO found
            away_elos.append(1500.0)
            errors += 1
            
    except Exception as e:
        away_elos.append(1500.0)
        errors += 1

print(f"Done!")
print(f"  OK - Added {len(away_elos):,} away_composite_elo values")
if errors > 0:
    print(f"  WARNING - {errors} games used fallback ELO (1500)")

# Add column
df['away_composite_elo'] = away_elos

# Verify it's added
feature_cols = [c for c in df.columns if c not in [
    'date', 'game_id', 'home_team', 'away_team', 'season',
    'target_spread', 'target_spread_cover', 'target_moneyline_win',
    'target_game_total', 'target_over_under', 'target_home_cover', 'target_over'
]]

print(f"\n{'='*80}")
print("VERIFICATION")
print(f"{'='*80}")
print(f"Total features: {len(feature_cols)} (expected: 44)")
print(f"home_composite_elo: {'OK' if 'home_composite_elo' in df.columns else 'MISSING'}")
print(f"away_composite_elo: {'OK' if 'away_composite_elo' in df.columns else 'MISSING'}")

# Sample check
print(f"\nSample ELO values (first 5 games):")
for i in range(min(5, len(df))):
    print(f"  {df.iloc[i]['home_team']:3s} vs {df.iloc[i]['away_team']:3s}: "
          f"home={df.iloc[i]['home_composite_elo']:.1f}, "
          f"away={df.iloc[i]['away_composite_elo']:.1f}")

# Check for unrealistic ELO values
low_elo = (df['away_composite_elo'] < 1300).sum()
high_elo = (df['away_composite_elo'] > 1700).sum()
default_elo = (df['away_composite_elo'] == 1500).sum()

print(f"\nELO Distribution:")
print(f"  Below 1300: {low_elo} ({low_elo/len(df)*100:.1f}%)")
print(f"  Above 1700: {high_elo} ({high_elo/len(df)*100:.1f}%)")
print(f"  Exactly 1500: {default_elo} ({default_elo/len(df)*100:.1f}%)")

if default_elo / len(df) > 0.1:
    print(f"  WARNING: {default_elo/len(df)*100:.1f}% of games have default ELO")
    print(f"    This may indicate missing ELO data for early season games")

# Backup old file
import shutil
backup_path = 'data/training_data_with_temporal_features_43feat_backup.csv'
if not os.path.exists(backup_path):
    shutil.copy('data/training_data_with_temporal_features.csv', backup_path)
    print(f"\nOK - Backed up original to: {backup_path}")

# Save updated data
df.to_csv('data/training_data_with_temporal_features.csv', index=False)

print(f"\n{'='*80}")
print("SUCCESS")
print(f"{'='*80}")
print(f"OK - Updated training data saved")
print(f"  File: data/training_data_with_temporal_features.csv")
print(f"  Rows: {len(df):,}")
print(f"  Features: {len(feature_cols)}")
print(f"\nOK - Ready for Optuna hypertuning with 44 features")
print(f"  Run: python scripts/optuna_hypertune_8hr.py")
print(f"{'='*80}")
