"""
Copy historical odds from live database to main database for backtesting
"""
import sqlite3
import pandas as pd

print("="*70)
print("COPYING ODDS TO MAIN DATABASE")
print("="*70)

# Connect to both databases
live_conn = sqlite3.connect('data/live/nba_betting_data.db')
main_conn = sqlite3.connect('data/nba_betting_data.db')

# Read odds from live database
print("\nReading odds from data/live/nba_betting_data.db...")
odds_df = pd.read_sql("SELECT * FROM historical_odds", live_conn)
print(f"  Found {len(odds_df):,} rows")
print(f"  Date range: {odds_df['game_date'].min()} to {odds_df['game_date'].max()}")

# Create table in main database
print("\nCreating historical_odds table in data/nba_betting_data.db...")
odds_df.to_sql('historical_odds', main_conn, if_exists='replace', index=False)

# Verify
count = pd.read_sql("SELECT COUNT(*) as count FROM historical_odds", main_conn).iloc[0]['count']
print(f"✅ Copied {count:,} rows to main database")

# Show 2024-25 sample
print("\n2024-25 Season Sample:")
sample = pd.read_sql("""
    SELECT game_date, home_team, away_team, home_ml_odds, away_ml_odds, spread_line
    FROM historical_odds
    WHERE game_date >= '2024-10-01'
    ORDER BY game_date
    LIMIT 10
""", main_conn)
print(sample.to_string())

live_conn.close()
main_conn.close()

print("\n" + "="*70)
print("✅ COMPLETE - Odds available for backtesting")
print("="*70)
