import sqlite3
conn = sqlite3.connect('nba_betting_data.db')
cur = conn.cursor()

print("Tables in database:")
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
for row in cur.fetchall():
    print(f"  {row[0]}")

print("\nSeasons in game_results:")
cur.execute("SELECT season, COUNT(*) FROM game_results GROUP BY season ORDER BY season")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]} games")

print("\nTotal games:")
cur.execute("SELECT COUNT(*) FROM game_results")
print(f"  {cur.fetchone()[0]} games")

# Check if there are other tables with historical game data
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%game%'")
game_tables = cur.fetchall()
print(f"\nGame-related tables: {[t[0] for t in game_tables]}")

conn.close()
