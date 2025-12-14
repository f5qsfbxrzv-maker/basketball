import sqlite3
import pandas as pd

conn = sqlite3.connect('data/live/nba_betting_data.db')

# First check what columns exist
print("Columns in historical_odds:")
df_schema = pd.read_sql("SELECT * FROM historical_odds LIMIT 1", conn)
print(df_schema.columns.tolist())

print("\n2024-25 Historical Odds Sample:")
df = pd.read_sql("""
    SELECT * 
    FROM historical_odds 
    WHERE game_date >= '2024-10-01' 
    ORDER BY game_date 
    LIMIT 10
""", conn)
print(df)

print("\n\nMoneyline coverage check:")
coverage = pd.read_sql("""
    SELECT 
        COUNT(*) as total_games,
        SUM(CASE WHEN home_ml_odds IS NOT NULL THEN 1 ELSE 0 END) as with_ml
    FROM historical_odds 
    WHERE game_date >= '2024-10-01'
""", conn)
print(coverage)

conn.close()
