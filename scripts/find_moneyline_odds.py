import sqlite3
import pandas as pd

# Check live database
print("=== LIVE DATABASE ===")
conn = sqlite3.connect('data/live/nba_betting_data.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [t[0] for t in cursor.fetchall()]
print("Tables:", tables)

for table in tables:
    print(f"\n{table}:")
    df = pd.read_sql(f"SELECT * FROM {table} LIMIT 3", conn)
    print(df)
    print(f"Total rows: {len(pd.read_sql(f'SELECT * FROM {table}', conn))}")
    
    # Check 2024-25 season specifically
    if 'date' in df.columns or 'game_date' in df.columns:
        date_col = 'game_date' if 'game_date' in df.columns else 'date'
        df_2024 = pd.read_sql(f"SELECT * FROM {table} WHERE {date_col} >= '2024-10-01' LIMIT 5", conn)
        print(f"2024-25 season sample:\n{df_2024}")

conn.close()

# Check main database
print("\n\n=== MAIN DATABASE ===")
conn2 = sqlite3.connect('data/nba_betting_data.db')
cursor2 = conn2.cursor()
cursor2.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables2 = [t[0] for t in cursor2.fetchall()]
print("Tables:", tables2)
conn2.close()
