import sqlite3

DB_PATH = r"data\live\nba_betting_data.db"
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Get all tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cur.fetchall()

print("=== ALL TABLES IN DATABASE ===")
for table in tables:
    table_name = table[0]
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cur.fetchone()[0]
    print(f"{table_name}: {count} rows")

print("\n=== CHECKING ACTIVE_INJURIES ===")
cur.execute("SELECT COUNT(*) FROM active_injuries")
print(f"Total injuries: {cur.fetchone()[0]}")

cur.execute("SELECT team, COUNT(*) FROM active_injuries GROUP BY team ORDER BY COUNT(*) DESC LIMIT 10")
print("\nTeams with most injuries:")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]} players")

print("\n=== SAMPLE INJURY RECORDS ===")
cur.execute("SELECT team, player_name, status, injury_type FROM active_injuries LIMIT 5")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]} - {row[2]} ({row[3]})")

conn.close()
