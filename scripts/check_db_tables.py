import sqlite3
conn = sqlite3.connect('data/nba_betting_data.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print("Tables in nba_betting_data.db:")
for table in tables:
    print(f"  - {table}")
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"    Rows: {count:,}")
conn.close()
