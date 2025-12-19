import sqlite3
import pandas as pd

conn = sqlite3.connect('data/live/nba_betting_data.db')

# Check tables
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cursor.fetchall()]
print("Tables:", tables)

# Check game_logs schema
print("\ngame_logs columns:")
df = pd.read_sql("SELECT * FROM game_logs LIMIT 1", conn)
print(df.columns.tolist())

# Check if we have STL, BLK, DRB
print("\nSample game_logs data:")
df_sample = pd.read_sql("SELECT * FROM game_logs LIMIT 3", conn)
print(df_sample.columns.tolist())

conn.close()
