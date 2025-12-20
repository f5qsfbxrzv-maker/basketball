import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = r"data\live\nba_betting_data.db"
conn = sqlite3.connect(DB_PATH)

# Check game_logs for recent games
print("=== RECENT GAMES FROM game_logs ===")
df = pd.read_sql("""
    SELECT TEAM_ABBREVIATION, game_date, GAME_ID
    FROM game_logs 
    WHERE TEAM_ABBREVIATION IN ('LAL', 'LAC')
    ORDER BY game_date DESC 
    LIMIT 10
""", conn)
print(df)
print(f"\nMost recent game_logs date: {df['GAME_DATE'].max() if not df.empty else 'EMPTY'}")

# Check game_results for recent games
print("\n=== RECENT GAMES FROM game_results ===")
df2 = pd.read_sql("""
    SELECT TEAM_ABBREVIATION, game_date
    FROM game_results
    WHERE TEAM_ABBREVIATION IN ('LAL', 'LAC')
    ORDER BY game_date DESC
    LIMIT 10
""", conn)
print(df2)
print(f"\nMost recent game_results date: {df2['game_date'].max() if not df2.empty else 'EMPTY'}")

# Check what columns game_results has
print("\n=== game_results SCHEMA ===")
schema = pd.read_sql("PRAGMA table_info(game_results)", conn)
print(schema[['name', 'type']])

conn.close()

print(f"\n=== TODAY'S DATE ===")
print(f"Today: {datetime.now().strftime('%Y-%m-%d')}")
