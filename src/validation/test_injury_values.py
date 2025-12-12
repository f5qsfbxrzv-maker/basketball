"""
Test if injury features are actually generating non-zero values
"""

import pandas as pd
import sqlite3
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.features.feature_calculator_v5 import FeatureCalculatorV5

DB_PATH = "data/live/nba_betting_data.db"

def test_injury_feature_values():
    conn = sqlite3.connect(DB_PATH)
    
    # Get 10 recent games
    games_df = pd.read_sql("""
        SELECT game_date, home_team, away_team
        FROM game_results
        WHERE game_date >= '2024-10-01'
        ORDER BY game_date DESC
        LIMIT 10
    """, conn)
    
    conn.close()
    
    print("="*70)
    print("TESTING INJURY FEATURE VALUES")
    print("="*70)
    
    calc = FeatureCalculatorV5()
    
    non_zero_count = 0
    total_games = 0
    
    for idx, row in games_df.iterrows():
        features = calc.calculate_game_features(
            row['home_team'], 
            row['away_team'], 
            row['game_date']
        )
        
        injury_diff = features.get('injury_impact_diff', 0)
        injury_abs = features.get('injury_impact_abs', 0)
        injury_elo = features.get('injury_elo_interaction', 0)
        
        total_games += 1
        
        if injury_diff != 0 or injury_abs != 0 or injury_elo != 0:
            non_zero_count += 1
            print(f"\n{row['game_date']}: {row['away_team']} @ {row['home_team']}")
            print(f"   injury_impact_diff: {injury_diff:.4f}")
            print(f"   injury_impact_abs: {injury_abs:.4f}")
            print(f"   injury_elo_interaction: {injury_elo:.4f}")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"   Games tested: {total_games}")
    print(f"   Games with non-zero injury features: {non_zero_count}")
    print(f"   Percentage: {(non_zero_count/total_games)*100:.1f}%")
    
    if non_zero_count == 0:
        print(f"\n   ❌ PROBLEM: All injury features are zero!")
        print(f"   This explains why XGBoost assigned 0.0000 importance.")
        print(f"   Need to debug injury impact calculation.")
    else:
        print(f"\n   ✅ Injury features ARE generating values")
        print(f"   XGBoost found them uninformative for prediction.")

if __name__ == "__main__":
    test_injury_feature_values()
