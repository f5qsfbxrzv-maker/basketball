import sqlite3
import pandas as pd

conn = sqlite3.connect('data/live/nba_betting_data.db')

# Check game_logs for any inactive/injury columns
print("game_logs schema:")
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(game_logs)")
cols = cursor.fetchall()
for col in cols:
    print(f"  {col[1]} ({col[2]})")

# Check if there's any injury-related data in recent games
print("\nSample game_logs (last 5):")
df = pd.read_sql("SELECT * FROM game_logs ORDER BY GAME_DATE DESC LIMIT 5", conn)
print(df.columns.tolist())

# Check historical_inactives more carefully
print("\nhistorical_inactives - recent data:")
inactive_df = pd.read_sql("""
    SELECT game_date, COUNT(*) as count 
    FROM historical_inactives 
    WHERE game_date >= '2024-01-01'
    GROUP BY game_date 
    ORDER BY game_date DESC 
    LIMIT 10
""", conn)
print(inactive_df)

print("\nhistorical_inactives - 2023 data:")
inactive_2023 = pd.read_sql("""
    SELECT game_date, COUNT(*) as count 
    FROM historical_inactives 
    WHERE game_date >= '2023-01-01' AND game_date < '2024-01-01'
    GROUP BY game_date 
    ORDER BY game_date DESC 
    LIMIT 10
""", conn)
print(inactive_2023)

conn.close()
