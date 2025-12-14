"""Check which database files have current data"""
import sqlite3
import pandas as pd

dbs = [
    'nba_betting_data.db',
    'data/nba_betting_data.db',
    'data/live/nba_betting_data.db'
]

print("="*80)
print("CHECKING ALL DATABASE FILES")
print("="*80)

for db_path in dbs:
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql('SELECT MAX(GAME_DATE) as max_date, COUNT(*) as cnt FROM game_logs', conn)
        print(f"\n{db_path}:")
        print(f"  Latest: {df['max_date'][0]}")
        print(f"  Count: {df['cnt'][0]} rows")
        conn.close()
    except Exception as e:
        print(f"\n{db_path}: ERROR - {e}")
