import sqlite3
import pandas as pd

conn = sqlite3.connect('data/live/nba_betting_data.db')

# Check sample data
print("Sample game_advanced_stats for LAL:")
df = pd.read_sql_query("""
    SELECT team_abb, game_date, pace, off_rating, def_rating, 
           efg_pct, tov_pct, orb_pct, fta_rate
    FROM game_advanced_stats 
    WHERE team_abb='LAL' 
    ORDER BY game_date DESC 
    LIMIT 3
""", conn)
print(df)

print("\n\nChecking if we have enough data for features:")
print(f"Total LAL games: {pd.read_sql_query('SELECT COUNT(*) FROM game_advanced_stats WHERE team_abb=\"LAL\"', conn).iloc[0,0]}")
print(f"Total GSW games: {pd.read_sql_query('SELECT COUNT(*) FROM game_advanced_stats WHERE team_abb=\"GSW\"', conn).iloc[0,0]}")

conn.close()
