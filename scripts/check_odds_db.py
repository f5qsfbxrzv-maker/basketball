import sqlite3
import pandas as pd

# Check the odds history database
conn = sqlite3.connect('data/backups/nba_ODDS_history.db')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [t[0] for t in cursor.fetchall()]
print("Tables in nba_ODDS_history.db:", tables)

# Check each table
for table in tables:
    print(f"\n=== {table} ===")
    df = pd.read_sql(f"SELECT * FROM {table} LIMIT 3", conn)
    print(df)
    print(f"Columns: {df.columns.tolist()}")
    print(f"Total rows: {pd.read_sql(f'SELECT COUNT(*) as cnt FROM {table}', conn)['cnt'].iloc[0]}")

conn.close()
