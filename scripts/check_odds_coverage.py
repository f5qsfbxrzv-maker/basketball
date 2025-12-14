import sqlite3
import pandas as pd

conn = sqlite3.connect('data/nba_betting_data.db')

# Check coverage by season
df = pd.read_sql("""
    SELECT 
        substr(game_date,1,7) as month,
        COUNT(*) as total_games,
        SUM(CASE WHEN home_ml_odds IS NOT NULL THEN 1 ELSE 0 END) as with_ml_odds,
        SUM(CASE WHEN spread_line IS NOT NULL THEN 1 ELSE 0 END) as with_spread
    FROM historical_odds
    WHERE game_date >= '2024-01-01'
    GROUP BY substr(game_date,1,7)
    ORDER BY month DESC
""", conn)

print("Odds Coverage by Month (2024+):")
print(df.to_string(index=False))

conn.close()
