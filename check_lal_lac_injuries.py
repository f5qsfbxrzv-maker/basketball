import sqlite3

DB_PATH = r"data\live\nba_betting_data.db"
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

print("=== CURRENT INJURIES FOR LAL AND LAC ===")
cur.execute("""
    SELECT player_name, team_name, status, injury_desc
    FROM active_injuries
    WHERE team_name IN ('Los Angeles Lakers', 'LA Clippers')
    ORDER BY team_name
""")

results = cur.fetchall()
if results:
    for row in results:
        print(f"{row[1]}: {row[0]} - {row[2]} ({row[3]})")
else:
    print("No injuries found for these teams")

print(f"\nTotal: {len(results)} players")

conn.close()
