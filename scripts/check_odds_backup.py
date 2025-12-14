import sqlite3
import pandas as pd

conn = sqlite3.connect('data/backups/nba_ODDS_history.db')

print("October 2024 games with moneyline odds:")
df = pd.read_sql("""
    SELECT game_date, home_team, away_team, home_ml_odds, away_ml_odds, spread_line
    FROM odds_history 
    WHERE game_date >= '2024-10-01' AND game_date < '2024-11-01'
    AND home_ml_odds IS NOT NULL
    ORDER BY game_date 
    LIMIT 20
""", conn)
print(df.to_string())

print("\n\nFull 2024-25 season coverage:")
coverage = pd.read_sql("""
    SELECT 
        COUNT(*) as total_games,
        SUM(CASE WHEN home_ml_odds IS NOT NULL THEN 1 ELSE 0 END) as with_ml,
        MIN(game_date) as first_game,
        MAX(game_date) as last_game
    FROM odds_history 
    WHERE game_date >= '2024-10-01'
""", conn)
print(coverage.to_string())

conn.close()
