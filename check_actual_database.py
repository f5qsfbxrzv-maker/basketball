import sqlite3
import sys
from pathlib import Path

DB_PATH = Path('data/live/nba_betting_data.db')

conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()

print("="*80)
print("CHECKING ACTUAL DATABASE CONTENTS")
print("="*80)

# Check latest game date
cursor.execute("SELECT MAX(game_date) FROM game_logs")
latest_game = cursor.fetchone()[0]
print(f"\nLatest game in database: {latest_game}")

# Check latest ELO date
cursor.execute("SELECT MAX(game_date) FROM elo_ratings")
latest_elo = cursor.fetchone()[0]
print(f"Latest ELO update: {latest_elo}")

# Get OKC and SAS ELO ratings
print("\n" + "="*80)
print("OKC vs SAS ELO DATA (Latest)")
print("="*80)

for team in ['OKC', 'SAS']:
    cursor.execute("""
        SELECT game_date, off_elo, def_elo, composite_elo
        FROM elo_ratings
        WHERE team = ?
        ORDER BY game_date DESC
        LIMIT 1
    """, (team,))
    
    row = cursor.fetchone()
    if row:
        print(f"\n{team}:")
        print(f"  Date: {row[0]}")
        print(f"  Offensive ELO: {row[1]:.1f}")
        print(f"  Defensive ELO: {row[2]:.1f}")
        print(f"  Composite ELO: {row[3]:.1f}")
    else:
        print(f"\n{team}: NO DATA FOUND")

# Check if there's recent data (last 7 days)
from datetime import datetime, timedelta
recent_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

cursor.execute("""
    SELECT COUNT(*) FROM game_logs
    WHERE game_date >= ?
""", (recent_date,))

recent_games = cursor.fetchone()[0]
print(f"\n" + "="*80)
print(f"Games in last 7 days: {recent_games}")
print("="*80)

conn.close()
