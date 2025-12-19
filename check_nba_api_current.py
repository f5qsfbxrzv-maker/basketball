import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime
import time

print("Checking NBA API for today's games...")
print("="*80)

try:
    today = datetime.now().strftime('%m/%d/%Y')
    print(f"Fetching scoreboard for: {today}")
    
    sb = scoreboardv2.ScoreboardV2(game_date=today)
    time.sleep(0.6)
    
    games_df = sb.get_data_frames()[0]
    
    if games_df.empty:
        print("No games found for today")
    else:
        print(f"Found {len(games_df)} games today:")
        for idx, row in games_df.iterrows():
            print(f"  {row['VISITOR_TEAM_CITY']} @ {row['HOME_TEAM_CITY']}")
    
    # Try yesterday
    print(f"\nTrying other recent dates...")
    for days_back in [1, 2, 3, 7]:
        from datetime import timedelta
        check_date = (datetime.now() - timedelta(days=days_back)).strftime('%m/%d/%Y')
        try:
            sb = scoreboardv2.ScoreboardV2(game_date=check_date)
            time.sleep(0.6)
            games_df = sb.get_data_frames()[0]
            print(f"  {check_date}: {len(games_df)} games")
        except:
            print(f"  {check_date}: Error fetching")
            
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
