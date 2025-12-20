import sqlite3
import pandas as pd

conn = sqlite3.connect('data/nba_betting_data.db')

# Check tables
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
print("Available tables:")
print(tables['name'].tolist())

# Check games table structure
print("\nGames table columns:")
cols = pd.read_sql_query("PRAGMA table_info(games)", conn)
print(cols[['name', 'type']].to_string())

# Sample data
print("\nSample games data:")
sample = pd.read_sql_query("SELECT * FROM games LIMIT 3", conn)
print(sample.columns.tolist())

conn.close()
