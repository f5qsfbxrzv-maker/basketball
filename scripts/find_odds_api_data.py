"""
Find real odds from The Odds API in backups
"""

import pandas as pd
import sqlite3

conn = sqlite3.connect('data/backups/odds_history.db')

# Check for real odds from The Odds API
df = pd.read_sql("""
    SELECT * 
    FROM odds_history 
    WHERE game_date < '2025-01-01' 
    AND source = 'The Odds API'
    ORDER BY game_date DESC 
    LIMIT 50
""", conn)

print(f"Found {len(df)} games with real Odds API data in 2024")
print("\nSample:")
print(df[['game_date', 'home_team', 'away_team', 'spread_home_odds', 'spread_away_odds', 'source']].to_string(index=False))

print("\n\nSpread odds distribution:")
print(f"  Home spread odds unique values: {df['spread_home_odds'].nunique()}")
print(f"  Away spread odds unique values: {df['spread_away_odds'].nunique()}")
print(f"\n  Sample home odds: {df['spread_home_odds'].value_counts().head(10).to_dict()}")

# Check date range
print(f"\n\nDate range: {df['game_date'].min()} to {df['game_date'].max()}")

conn.close()

# Also check the other backup
print("\n\n" + "="*80)
print("Checking nba_ODDS_history.db")
print("="*80)

conn2 = sqlite3.connect('data/backups/nba_ODDS_history.db')

df2 = pd.read_sql("""
    SELECT * 
    FROM odds_history 
    WHERE source = 'The Odds API'
    ORDER BY game_date DESC 
    LIMIT 50
""", conn2)

print(f"\nFound {len(df2)} games with real Odds API data")
print("\nSample:")
print(df2[['game_date', 'home_team', 'away_team', 'spread_home_odds', 'spread_away_odds', 'source']].head(20).to_string(index=False))

conn2.close()
