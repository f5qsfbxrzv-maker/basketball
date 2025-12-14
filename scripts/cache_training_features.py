"""
Quick script to cache training features for faster hyperparameter tuning
"""
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import sqlite3
import pandas as pd
import numpy as np
from src.features.feature_calculator_v5 import FeatureCalculatorV5
from config.feature_whitelist import FEATURE_WHITELIST

print("Caching training features for hyperparameter optimization...")

fc = FeatureCalculatorV5()

# Load games
conn = sqlite3.connect('data/live/nba_betting_data.db')
games_df = pd.read_sql("""
    SELECT game_id, game_date, home_team, away_team, home_score, away_score
    FROM game_results
    WHERE date(game_date) >= '2021-01-01'
    AND date(game_date) <= '2024-12-31'
    ORDER BY game_date
""", conn)
conn.close()

print(f"Loaded {len(games_df)} games")

# Generate features
features_list = []
for idx, row in games_df.iterrows():
    if idx % 500 == 0:
        print(f"Progress: {idx}/{len(games_df)} ({100*idx/len(games_df):.1f}%)")
    
    try:
        features = fc.calculate_game_features(
            game_date=row['game_date'],
            home_team=row['home_team'],
            away_team=row['away_team']
        )
        
        if features and isinstance(features, dict):
            feature_values = [features.get(feat, 0.0) for feat in FEATURE_WHITELIST]
            if len(feature_values) == len(FEATURE_WHITELIST):
                home_won = 1 if row['home_score'] > row['away_score'] else 0
                features_list.append(feature_values + [home_won, row['game_date'], 
                                                       row['home_team'], row['away_team']])
    except:
        continue

# Create DataFrame
columns = FEATURE_WHITELIST + ['home_won', 'game_date', 'home_team', 'away_team']
df = pd.DataFrame(features_list, columns=columns)

# Save
Path('data/processed').mkdir(exist_ok=True)
df.to_csv('data/processed/training_features_30.csv', index=False)
print(f"\n✓ Cached {len(df)} games to data/processed/training_features_30.csv")
