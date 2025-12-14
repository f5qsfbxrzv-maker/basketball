import sqlite3
import pandas as pd

conn = sqlite3.connect('data/nba_betting_data.db')
cursor = conn.cursor()

# List tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print(f"Available tables: {tables}\n")

# Check for player data with usage
if 'player_season_stats' in tables:
    df = pd.read_sql("SELECT * FROM player_season_stats LIMIT 1", conn)
    print(f"player_season_stats columns: {df.columns.tolist()}")
    
conn.close()
