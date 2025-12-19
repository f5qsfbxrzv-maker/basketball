"""
Generate Training Data for 20-Feature Model
============================================

Generates dataset with:
- 19 v6 features (from feature_calculator_v5)
- 1 injury_matchup_advantage (new optimized formula)

Total: 20 features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_calculator_v5 import FeatureCalculatorV5
import sqlite3
from datetime import datetime

print("=" * 80)
print("GENERATING 20-FEATURE TRAINING DATASET")
print("=" * 80)
print()

# 19 V6 Features (from model_v6_ml.xgb)
V6_FEATURES = [
    'vs_efg_diff',
    'vs_tov',
    'vs_reb_diff',
    'vs_ftr_diff',
    'vs_net_rating',
    'expected_pace',
    'rest_days_diff',
    'is_b2b_diff',
    'h2h_win_rate_l3y',
    'elo_diff',
    'off_elo_diff',
    'def_elo_diff',
    'composite_elo_diff',
    'home_composite_elo',
    'h_off_rating',
    'h_def_rating',
    'a_off_rating',
    'a_def_rating',
    'injury_impact_diff'  # Will be replaced by injury_matchup_advantage
]

# Connect to database
DB_PATH = PROJECT_ROOT / 'data' / 'nba_betting_data.db'
print(f"📂 Connecting to: {DB_PATH}")

if not DB_PATH.exists():
    print(f"❌ Database not found: {DB_PATH}")
    exit(1)

conn = sqlite3.connect(DB_PATH)

# Get historical games
print("📊 Loading historical games...")
games_df = pd.read_sql_query("""
    SELECT game_date, home_team, away_team, home_score, away_score
    FROM game_results
    WHERE game_date >= '2020-01-01'
    ORDER BY game_date
""", conn)

print(f"✅ Loaded {len(games_df)} games")
print()

# Initialize feature calculator
print("🔧 Initializing feature calculator...")
fc = FeatureCalculatorV5(str(DB_PATH))
print("✅ Feature calculator ready")
print()

# Calculate features for each game
print("⚙️  Calculating features (this may take a while)...")
print()

training_data = []
errors = 0

for idx, row in games_df.iterrows():
    if idx % 100 == 0:
        print(f"  Progress: {idx}/{len(games_df)} ({idx/len(games_df)*100:.1f}%)")
    
    try:
        features = fc.calculate_game_features(
            home_team=row['home_team'],
            away_team=row['away_team'],
            game_date=row['game_date']
        )
        
        # Check if we have the required v6 features
        if all(f in features for f in V6_FEATURES):
            # Create row with v6 features
            data_row = {feat: features[feat] for feat in V6_FEATURES}
            
            # Add injury_matchup_advantage (calculated from components)
            injury_impact_diff = features.get('injury_impact_diff', 0)
            injury_shock_diff = features.get('injury_shock_diff', 0)
            star_mismatch = features.get('star_mismatch', 0)
            
            data_row['injury_matchup_advantage'] = (
                0.008127 * injury_impact_diff
              - 0.023904 * injury_shock_diff
              + 0.031316 * star_mismatch
            )
            
            # Add target
            data_row['target_moneyline_win'] = 1 if row['home_score'] > row['away_score'] else 0
            
            # Add metadata
            data_row['game_date'] = row['game_date']
            data_row['home_team'] = row['home_team']
            data_row['away_team'] = row['away_team']
            
            training_data.append(data_row)
        else:
            errors += 1
            
    except Exception as e:
        errors += 1
        if errors <= 5:  # Only print first 5 errors
            print(f"  ⚠️  Error on game {idx}: {e}")

conn.close()

print()
print(f"✅ Feature calculation complete")
print(f"   Success: {len(training_data)} games")
print(f"   Errors: {errors} games")
print()

# Create DataFrame
df = pd.DataFrame(training_data)

# Save
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'training_data_20features.csv'
print(f"💾 Saving to: {OUTPUT_PATH}")
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved {len(df)} rows")
print()

# Display summary
print("=" * 80)
print("DATASET SUMMARY")
print("=" * 80)
print()
print(f"Total samples: {len(df)}")
print(f"Features: {len(V6_FEATURES) + 1} (19 v6 + 1 injury)")
print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
print(f"Home win rate: {df['target_moneyline_win'].mean():.3f}")
print()

print("Feature list:")
feature_cols = [c for c in df.columns if c not in ['target_moneyline_win', 'game_date', 'home_team', 'away_team']]
for i, feat in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {feat}")

print()
print("=" * 80)
print("READY FOR TRAINING")
print("=" * 80)
print()
print("Next step: Run Optuna tuning with this 20-feature dataset")
print(f"  python scripts/tune_20feature_model.py")
