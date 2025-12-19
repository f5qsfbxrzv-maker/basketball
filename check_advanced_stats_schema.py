import sqlite3

conn = sqlite3.connect('nba_betting_data.db')
cursor = conn.cursor()

# Check schema
cursor.execute("PRAGMA table_info(game_advanced_stats)")
columns = cursor.fetchall()

print("game_advanced_stats columns:")
for col in columns:
    print(f"  {col[1]} ({col[2]})")

# Check sample data
cursor.execute("SELECT COUNT(*) FROM game_advanced_stats")
count = cursor.fetchone()[0]
print(f"\nTotal records: {count}")

if count > 0:
    cursor.execute("SELECT * FROM game_advanced_stats LIMIT 1")
    sample = cursor.fetchone()
    print(f"\nSample record columns: {len(sample)}")

conn.close()
