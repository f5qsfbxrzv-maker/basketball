"""
Test different NBA API endpoints to find recent game data
"""
from nba_api.stats.endpoints import leaguegamefinder, teamgamelogs
from datetime import datetime
import time

print("="*80)
print("TESTING NBA API ENDPOINTS FOR RECENT GAMES")
print("="*80)

# Test 1: LeagueGameFinder with date range
print("\n[1] LeagueGameFinder with date range...")
try:
    games = leaguegamefinder.LeagueGameFinder(
        season_nullable='2024-25',
        date_from_nullable='11/21/2025',
        date_to_nullable='12/14/2025',
        season_type_nullable='Regular Season'
    )
    time.sleep(0.6)
    df = games.get_data_frames()[0]
    print(f"  Found {len(df)} game records")
    if len(df) > 0:
        print(f"  Latest: {df['GAME_DATE'].max()}")
        print(f"  Earliest: {df['GAME_DATE'].min()}")
except Exception as e:
    print(f"  Error: {e}")

# Test 2: TeamGameLogs for a specific team
print("\n[2] TeamGameLogs for LAL...")
try:
    team_logs = teamgamelogs.TeamGameLogs(
        season_nullable='2024-25',
        season_type_nullable='Regular Season',
        date_from_nullable='11/21/2025',
        date_to_nullable='12/14/2025'
    )
    time.sleep(0.6)
    df = team_logs.get_data_frames()[0]
    print(f"  Found {len(df)} game records")
    if len(df) > 0:
        print(f"  Latest: {df['GAME_DATE'].max()}")
        print(f"  Sample: {df[['TEAM_ABBREVIATION', 'GAME_DATE', 'PTS']].head(3).to_string()}")
except Exception as e:
    print(f"  Error: {e}")

# Test 3: LeagueGameLog without date filter (get all 2024-25)
print("\n[3] LeagueGameLog for entire 2024-25 season...")
try:
    from nba_api.stats.endpoints import leaguegamelog
    game_log = leaguegamelog.LeagueGameLog(
        season='2024-25',
        season_type_all_star='Regular Season'
    )
    time.sleep(0.6)
    df = game_log.get_data_frames()[0]
    print(f"  Total records: {len(df)}")
    print(f"  Latest game date: {df['GAME_DATE'].max()}")
    
    # Filter to recent games
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    recent = df[df['GAME_DATE'] >= '2025-11-21']
    print(f"  Games since Nov 21: {len(recent)}")
    
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
