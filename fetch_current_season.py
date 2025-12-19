from nba_api.stats.endpoints import leaguegamelog
import time
import pandas as pd

print("Fetching CURRENT season: 2025-26...")

log = leaguegamelog.LeagueGameLog(
    season='2025-26',
    season_type_all_star='Regular Season'
)
time.sleep(0.6)

df = log.get_data_frames()[0]

print(f"\nTotal records: {len(df)}")
print(f"Latest game: {df['GAME_DATE'].max()}")
print(f"Earliest game: {df['GAME_DATE'].min()}")
print(f"\nTeams: {sorted(df['TEAM_ABBREVIATION'].unique())}")
print(f"\nSample recent games:")
print(df[['GAME_DATE', 'MATCHUP', 'TEAM_ABBREVIATION', 'PTS', 'WL']].head(20))
