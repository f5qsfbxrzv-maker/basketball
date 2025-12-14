import sqlite3
import pandas as pd

conn = sqlite3.connect('data/live/nba_betting_data.db')

# Check injuries by team
df = pd.read_sql("""
    SELECT team_name, COUNT(*) as injured_count, 
           GROUP_CONCAT(player_name || ' (' || status || ')', ', ') as players
    FROM active_injuries 
    GROUP BY team_name 
    ORDER BY injured_count DESC
""", conn)

print("=" * 80)
print("INJURIES BY TEAM")
print("=" * 80)
print(df.to_string(index=False))
print("\n" + "=" * 80)
print(f"Total teams with injuries: {len(df)}/30")
print(f"Total injured players: {df['injured_count'].sum()}")
print("=" * 80)

# Check if any teams have 0 injuries
all_teams = pd.read_sql("SELECT DISTINCT team_abbreviation FROM game_results LIMIT 30", conn)
print(f"\nTotal NBA teams in database: {len(all_teams)}")

conn.close()
