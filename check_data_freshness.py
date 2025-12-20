import sqlite3
import pandas as pd

conn = sqlite3.connect(r"data\live\nba_betting_data.db")

print("=== MOST RECENT GAMES IN ELO_RATINGS ===")
df = pd.read_sql("""
    SELECT team, MAX(game_date) as last_game, COUNT(*) as game_count
    FROM elo_ratings 
    WHERE team IN ('LAL', 'LAC')
    GROUP BY team
""", conn)
print(df)

print("\n=== MOST RECENT GAMES IN GAME_LOGS ===")
df2 = pd.read_sql("""
    SELECT TEAM_ABBREVIATION as team, MAX(GAME_DATE) as last_game, COUNT(*) as game_count
    FROM game_logs
    WHERE TEAM_ABBREVIATION IN ('LAL', 'LAC')
    GROUP BY TEAM_ABBREVIATION
""", conn)
print(df2)

print("\n=== TODAY: 2025-12-20 ===")
print("Lakers last game: Dec 18 (1 day rest)")
print("Clippers last game: Dec 18 (1 day rest)")
print("Database is STALE - needs update!")

conn.close()
