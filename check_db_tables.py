import sqlite3
conn = sqlite3.connect('data/live/nba_betting_data.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [row[0] for row in cursor.fetchall()]
print("📊 DATABASE TABLES (data/live/nba_betting_data.db):")
print("=" * 60)
for table in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"  - {table:<30} ({count:,} rows)")
print("=" * 60)
print(f"Total: {len(tables)} tables")
conn.close()
