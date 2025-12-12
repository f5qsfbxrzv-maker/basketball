"""
Debug why injury calculations are returning 0
"""

import pandas as pd
import sqlite3
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.features.feature_calculator_v5 import FeatureCalculatorV5

DB_PATH = "data/live/nba_betting_data.db"

def debug_injury_calc():
    print("="*70)
    print("🐛 DEBUGGING INJURY CALCULATION")
    print("="*70)
    
    calc = FeatureCalculatorV5()
    
    # Check if player_stats_df is loaded
    print(f"\n1. player_stats_df loaded: {not calc.player_stats_df.empty}")
    if not calc.player_stats_df.empty:
        print(f"   Rows: {len(calc.player_stats_df)}")
        print(f"   Columns: {calc.player_stats_df.columns.tolist()}")
        print(f"\n   Sample data:")
        print(calc.player_stats_df.head())
    else:
        print("   ❌ player_stats_df is EMPTY!")
    
    # Check historical_inactives for a recent game
    conn = sqlite3.connect(DB_PATH)
    
    injuries_df = pd.read_sql("""
        SELECT game_date, player_name, team_abbreviation, season
        FROM historical_inactives
        WHERE game_date = '2024-10-22'
        LIMIT 10
    """, conn)
    
    print(f"\n2. Injuries on 2024-10-22:")
    print(injuries_df)
    
    # Test the calculation for a specific game
    test_date = '2024-10-22'
    test_home = 'BOS'
    test_away = 'NYK'
    
    print(f"\n3. Testing calculation for {test_away} @ {test_home} on {test_date}:")
    
    # Manually call the historical injury method
    result = calc._calculate_historical_injury_impact(test_home, test_away, test_date)
    
    print(f"   Result: {result}")
    
    # Check what injuries exist for those teams on that date
    team_injuries = pd.read_sql(f"""
        SELECT player_name, team_abbreviation, season
        FROM historical_inactives
        WHERE game_date = '{test_date}'
        AND (team_abbreviation = '{test_home}' OR team_abbreviation = '{test_away}')
    """, conn)
    
    print(f"\n4. Injuries for {test_home}/{test_away} on {test_date}:")
    print(team_injuries)
    
    # Check if player names match
    if not calc.player_stats_df.empty and len(team_injuries) > 0:
        print(f"\n5. Player name matching:")
        for _, inj in team_injuries.iterrows():
            inj_name = inj['player_name']
            # Try exact match
            exact_match = calc.player_stats_df[calc.player_stats_df['player_name'] == inj_name]
            
            if len(exact_match) > 0:
                print(f"   ✅ {inj_name}: EXACT MATCH FOUND")
                print(f"      PIE: {exact_match.iloc[0]['pie']:.4f}")
            else:
                # Try contains
                contains_match = calc.player_stats_df[
                    calc.player_stats_df['player_name'].str.contains(inj_name, case=False, na=False)
                ]
                if len(contains_match) > 0:
                    print(f"   ⚠️ {inj_name}: FUZZY MATCH")
                    print(f"      Matched: {contains_match.iloc[0]['player_name']}")
                    print(f"      PIE: {contains_match.iloc[0]['pie']:.4f}")
                else:
                    print(f"   ❌ {inj_name}: NO MATCH in player_stats")
    
    conn.close()

if __name__ == "__main__":
    debug_injury_calc()
