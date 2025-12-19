import sqlite3

db_path = 'data/live/nba_betting_data.db'
conn = sqlite3.connect(db_path)
cur = conn.cursor()

print(f"Tables in {db_path}:")
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
for row in cur.fetchall():
    print(f"  {row[0]}")

# Check team_stats table for seasons
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%team%'")
team_tables = cur.fetchall()

if team_tables:
    print(f"\nTeam-related tables:")
    for table in team_tables:
        print(f"  {table[0]}")
        try:
            cur.execute(f"SELECT season, COUNT(*) FROM {table[0]} GROUP BY season ORDER BY season")
            seasons = cur.fetchall()
            if seasons:
                print(f"    Seasons: {len(seasons)}")
                for season, count in seasons:
                    print(f"      {season}: {count} records")
        except:
            pass

conn.close()
