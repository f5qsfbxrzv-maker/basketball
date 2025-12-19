from nba_api.stats.endpoints import teamgamelogs
import time
import pandas as pd

print("Testing TeamGameLogs endpoint...")

logs = teamgamelogs.TeamGameLogs(season_nullable='2024-25')
time.sleep(0.6)

df = logs.get_data_frames()[0]
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

print(f"Total game records: {len(df)}")
print(f"Latest date: {df['GAME_DATE'].max()}")
print(f"Earliest date: {df['GAME_DATE'].min()}")

recent = df[df['GAME_DATE'] >= '2025-11-21']
print(f"\nGames since Nov 21: {len(recent)}")

if len(recent) > 0:
    print("\nSample recent games:")
    print(recent[['TEAM_ABBREVIATION', 'GAME_DATE', 'MATCHUP', 'WL', 'PTS']].head(10))
else:
    print("\nNo games found after Nov 21")
