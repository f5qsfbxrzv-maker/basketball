"""Check database tables in nba_betting_data.db"""
import sqlite3

conn = sqlite3.connect('data/live/nba_betting_data.db')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print("=" * 60)
print("TABLES IN nba_betting_data.db")
print("=" * 60)
for table in tables:
    table_name = table[0]
    print(f"\n{table_name}:")
    
    # Get schema
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"  Rows: {count}")

conn.close()
