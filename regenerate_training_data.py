"""
Regenerate training data with 28 syndicate-level features.

This script:
1. Loads all historical games from game_results
2. Calculates new features (matchup friction, volume efficiency, ELO matchup advantages)
3. Outputs training_data_syndicate_28_features.csv
"""

import sys
import pandas as pd
from datetime import datetime
from tqdm import tqdm

sys.path.append('src')
from features.feature_calculator_v5 import FeatureCalculatorV5

def regenerate_training_data():
    """Generate training data with all 28 syndicate features"""
    
    print("=" * 70)
    print("SYNDICATE TRAINING DATA GENERATOR (28 Features)")
    print("=" * 70)
    
    # Initialize feature calculator
    print("\n[1/4] Initializing feature calculator...")
    calc = FeatureCalculatorV5(db_path="nba_betting_data.db")
    
    # Load historical games
    print("[2/4] Loading historical games...")
    import sqlite3
    conn = sqlite3.connect("nba_betting_data.db")
    
    games_df = pd.read_sql_query("""
        SELECT 
            game_id,
            game_date,
            home_team,
            away_team,
            home_score,
            away_score,
            season,
            point_diff
        FROM game_results
        WHERE game_date IS NOT NULL
        ORDER BY game_date
    """, conn)
    conn.close()
    
    print(f"Loaded {len(games_df)} games")
    
    # Calculate features for each game
    print("[3/4] Calculating syndicate features...")
    
    features_list = []
    errors = 0
    
    for idx, game in tqdm(games_df.iterrows(), total=len(games_df), desc="Processing"):
        try:
            # Calculate features as of game date
            features = calc.calculate_game_features(
                home_team=game['home_team'],
                away_team=game['away_team'],
                season=game['season'],
                game_date=game['game_date'],
                use_recency=True,
                games_back=10,
                decay_rate=0.15
            )
            
            # Add game metadata
            features['game_id'] = game['game_id']
            features['game_date'] = game['game_date']
            features['home_team'] = game['home_team']
            features['away_team'] = game['away_team']
            features['home_score'] = game['home_score']
            features['away_score'] = game['away_score']
            features['point_diff'] = game['point_diff']
            features['home_won'] = 1 if game['point_diff'] > 0 else 0
            
            features_list.append(features)
            
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n  ⚠ Error on game {game['game_id']}: {e}")
    
    print(f"\n✓ Processed {len(features_list)} games ({errors} errors)")
    
    # Convert to DataFrame
    print("[4/4] Saving training data...")
    training_df = pd.DataFrame(features_list)
    
    # Save to CSV
    output_file = "data/training_data_syndicate_28_features.csv"
    training_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Training data saved to: {output_file}")
    print(f"  Rows: {len(training_df)}")
    print(f"  Columns: {len(training_df.columns)}")
    
    # Show feature summary
    print("\n" + "=" * 70)
    print("SYNDICATE FEATURE SUMMARY")
    print("=" * 70)
    
    # Identify syndicate features
    syndicate_features = [
        'off_matchup_advantage', 'def_matchup_advantage', 'net_composite_advantage',
        'effective_shooting_gap', 'turnover_pressure', 'rebound_friction',
        'total_rebound_control', 'whistle_leverage', 'volume_efficiency_diff',
        'injury_leverage'
    ]
    
    existing_syndicate = [f for f in syndicate_features if f in training_df.columns]
    
    print(f"\nSyndicate Features ({len(existing_syndicate)}/10):")
    for feat in existing_syndicate:
        if feat in training_df.columns:
            mean_val = training_df[feat].mean()
            std_val = training_df[feat].std()
            print(f"  ✓ {feat:30s} (mean: {mean_val:6.2f}, std: {std_val:6.2f})")
    
    missing_syndicate = [f for f in syndicate_features if f not in training_df.columns]
    if missing_syndicate:
        print(f"\n⚠ Missing Syndicate Features ({len(missing_syndicate)}):")
        for feat in missing_syndicate:
            print(f"  ✗ {feat}")
    
    print("\n" + "=" * 70)
    print("✓ TRAINING DATA GENERATION COMPLETE")
    print("=" * 70)
    
    return output_file

if __name__ == "__main__":
    regenerate_training_data()
