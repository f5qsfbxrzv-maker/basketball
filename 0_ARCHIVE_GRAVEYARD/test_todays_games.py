import sqlite3
from datetime import datetime

conn = sqlite3.connect('data/live/nba_betting_data.db')
cursor = conn.cursor()

# Get today's games
today = datetime.now().strftime('%Y-%m-%d')
cursor.execute("""
    SELECT home_team, away_team, game_date 
    FROM games 
    WHERE date(game_date) = date(?)
    LIMIT 10
""", (today,))

games = cursor.fetchall()
print(f"\n=== NBA GAMES FOR {today} ===")
if games:
    for home, away, date in games:
        print(f"{away} @ {home} ({date})")
else:
    print("No games scheduled for today")

conn.close()
