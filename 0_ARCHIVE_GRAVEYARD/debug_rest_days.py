"""Debug rest days calculation"""
from src.features.feature_calculator_live import FeatureCalculatorV5
import pandas as pd

calc = FeatureCalculatorV5()

print("="*80)
print("DEBUG: Rest Days Calculation")
print("="*80)

# Check if game_logs_df is loaded
print(f"\ngame_logs_df loaded: {not calc.game_logs_df.empty}")
print(f"game_logs_df rows: {len(calc.game_logs_df)}")

if not calc.game_logs_df.empty:
    print(f"\nDate range in game_logs_df:")
    print(f"  Min: {calc.game_logs_df['GAME_DATE'].min()}")
    print(f"  Max: {calc.game_logs_df['GAME_DATE'].max()}")
    
    # Check ORL games
    print(f"\nORL games in game_logs_df:")
    orl_games = calc.game_logs_df[calc.game_logs_df['TEAM_ABBREVIATION'] == 'ORL'].sort_values('GAME_DATE', ascending=False).head(5)
    print(orl_games[['GAME_DATE', 'MATCHUP', 'WL']].to_string(index=False))

# Test the _get_rest_days method directly
print("\n" + "="*80)
print("Testing _get_rest_days method:")
print("="*80)

game_date = '2025-12-13'
for team in ['ORL', 'NYK', 'OKC', 'SAS']:
    rest_days = calc._get_rest_days(team, game_date)
    print(f"{team}: {rest_days} days")

# Test full feature calculation
print("\n" + "="*80)
print("Testing full feature calculation:")
print("="*80)

feats = calc.calculate_game_features('ORL', 'NYK', game_date='2025-12-13')
print(f"home_rest_days: {feats.get('home_rest_days')}")
print(f"away_rest_days: {feats.get('away_rest_days')}")

feats2 = calc.calculate_game_features('OKC', 'SAS', game_date='2025-12-13')
print(f"\nOKC home_rest_days: {feats2.get('home_rest_days')}")
print(f"SAS away_rest_days: {feats2.get('away_rest_days')}")

print("\n" + "="*80)
print("Expected:")
print("  ORL/NYK: 3 days (last played Dec 9)")
print("  OKC/SAS: 2 days (last played Dec 10)")
print("="*80)
