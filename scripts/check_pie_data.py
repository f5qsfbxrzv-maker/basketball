"""Quick check of PIE data availability"""
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/live/nba_betting_data.db')

# Check player_stats table structure
print("Player Stats Columns:")
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(player_stats)")
print([row[1] for row in cursor.fetchall()])

# Check for superstar PIE values
query = """
SELECT DISTINCT player_name, season, pie 
FROM player_stats 
WHERE player_name IN (
    'Antetokounmpo, Giannis', 
    'Jokic, Nikola', 
    'Embiid, Joel',
    'Doncic, Luka',
    'Curry, Stephen'
) 
ORDER BY player_name, season
"""

df = pd.read_sql(query, conn)
print("\nSuperstar PIE Data:")
print(df)
print(f"\nTotal rows: {len(df)}")
print(f"Non-null PIE: {df['pie'].notna().sum()}")

conn.close()
