import sqlite3

conn = sqlite3.connect('data/live/nba_betting_data.db')
cur = conn.cursor()

# Get sample game logs
cur.execute("SELECT GAME_ID, GAME_DATE, TEAM_ABBREVIATION, MATCHUP, PTS, WL FROM game_logs WHERE season='2025-26' ORDER BY GAME_DATE LIMIT 10")
print("Sample game_logs (2025-26):")
print("GAME_ID | DATE | TEAM | MATCHUP | PTS | WL")
for row in cur.fetchall():
    print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]}")

# Check if each GAME_ID has 2 rows (home and away)
cur.execute("""
    SELECT GAME_ID, COUNT(*) as team_count
    FROM game_logs
    WHERE season='2025-26'
    GROUP BY GAME_ID
    LIMIT 5
""")
print("\nGame_ID team counts:")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]} teams")

conn.close()
