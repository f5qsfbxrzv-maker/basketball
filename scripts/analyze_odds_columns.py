"""
Analyze master_features odds columns to understand the data format
"""
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/live/nba_betting_data.db')

# Get sample from master_features to see the actual column meanings
sample = pd.read_sql("""
SELECT 
    game_date, home_team, away_team,
    opening_spread_home, closing_spread_home,
    opening_spread_visitor, closing_spread_visitor,
    "opening_total.1" as opening_total, 
    "closing_total.1" as closing_total,
    home_score, away_score, point_differential
FROM master_features
WHERE game_date >= '2023-01-01'
  AND opening_spread_home != 0
ORDER BY game_date
LIMIT 20
""", conn)

print('='*100)
print('MASTER_FEATURES - RAW ODDS DATA')
print('='*100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(sample)
print()

# Show specific games to understand the format
print('='*100)
print('EXAMPLE GAMES BREAKDOWN')
print('='*100)

for i in range(min(5, len(sample))):
    example = sample.iloc[i]
    print(f'\nGame {i+1}: {example["away_team"]} @ {example["home_team"]} on {example["game_date"]}')
    print(f'Final Score: {example["away_team"]} {example["away_score"]} - {example["home_team"]} {example["home_score"]}')
    print(f'Point Differential (Home - Away): {example["point_differential"]}')
    print()
    print(f'Opening Spread (Home): {example["opening_spread_home"]}')
    print(f'Closing Spread (Home): {example["closing_spread_home"]}')
    print(f'Opening Spread (Visitor): {example["opening_spread_visitor"]}')
    print(f'Closing Spread (Visitor): {example["closing_spread_visitor"]}')
    print(f'Opening Total: {example["opening_total"]}')
    print(f'Closing Total: {example["closing_total"]}')
    print('-'*100)

# Check what we actually inserted into historical_odds
print('\n' + '='*100)
print('WHAT WE INSERTED INTO HISTORICAL_ODDS')
print('='*100)

inserted = pd.read_sql("""
SELECT game_date, home_team, away_team, spread_line, total_line
FROM historical_odds
WHERE game_date >= '2023-01-01'
ORDER BY game_date
LIMIT 20
""", conn)

print(inserted)

conn.close()
