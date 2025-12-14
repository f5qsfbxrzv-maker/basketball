"""Download current 2025-26 season game logs to fix rest days calculation"""
from src.services._OLD_nba_stats_collector_v2 import NBAStatsCollectorV2
import pandas as pd

print("="*80)
print("DOWNLOADING 2025-26 SEASON GAME LOGS")
print("="*80)

collector = NBAStatsCollectorV2(db_path='nba_betting_data.db')

print('\n[1/1] Downloading game logs for 2025-26 season...')
games = collector.get_game_logs(season='2025-26')
print(f'[OK] Downloaded {len(games)} games')

# Show date range
df = pd.DataFrame(games)
print(f'\nDate range: {df["GAME_DATE"].min()} to {df["GAME_DATE"].max()}')
print(f'Total teams: {df["TEAM_ABBREVIATION"].nunique()}')

# Show recent games for ORL, NYK, OKC, SAS
print("\n" + "="*80)
print("RECENT GAMES FOR TODAY'S TEAMS")
print("="*80)

for team in ['ORL', 'NYK', 'OKC', 'SAS']:
    team_games = df[df['TEAM_ABBREVIATION'] == team].sort_values('GAME_DATE', ascending=False).head(5)
    print(f'\n{team} - Last 5 games:')
    for _, game in team_games.iterrows():
        print(f"  {game['GAME_DATE']}: {game['MATCHUP']} - {game['WL']}")

print("\n" + "="*80)
print("SUCCESS - Game logs updated!")
print("="*80)
print("Rest days calculation should now show 1-2 days instead of 5-7 days")
