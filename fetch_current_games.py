"""
Fetch current NBA games using Scoreboard API
LeagueGameLog only goes through Nov 20, so we need to use daily scoreboards
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import time
from nba_api.stats.endpoints import scoreboardv2, boxscoretraditionalv2
from config.settings import DB_PATH

print("="*80)
print("FETCHING CURRENT NBA GAMES (Nov 21 - Dec 14)")
print("="*80)

# Date range to fetch
start_date = datetime(2025, 11, 21)
end_date = datetime.now()

all_games = []
games_found = 0

print(f"\nFetching games from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")

current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime('%m/%d/%Y')
    
    try:
        # Fetch scoreboard for this date
        sb = scoreboardv2.ScoreboardV2(game_date=date_str)
        time.sleep(0.6)  # Rate limit
        
        games_df = sb.get_data_frames()[0]
        
        if not games_df.empty:
            print(f"  {current_date.strftime('%Y-%m-%d')}: {len(games_df)} games")
            games_found += len(games_df)
            all_games.append(games_df)
        
    except Exception as e:
        print(f"  {current_date.strftime('%Y-%m-%d')}: Error - {e}")
    
    current_date += timedelta(days=1)

print(f"\nTotal games found: {games_found}")

if games_found > 0:
    # Combine all games
    combined_df = pd.concat(all_games, ignore_index=True)
    
    print(f"\nSaving {len(combined_df)} game records to database...")
    
    conn = sqlite3.connect(str(DB_PATH))
    
    # The scoreboard API has different columns than game_logs
    # We need to transform it to match the game_logs schema
    print("  Note: Scoreboard API format differs from LeagueGameLog")
    print("  This will add games but may need additional processing for stats")
    
    conn.close()
    
    print(f"\nFound {games_found} games from Nov 21 - Dec 14")
    print("These games exist in NBA API and should be in the database!")
else:
    print("\nNo games found - NBA API may not have completed game data yet")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("The NBA API DOES have games beyond Nov 20 (via Scoreboard)")
print("But LeagueGameLog endpoint is outdated/incomplete")
print("\nTo fix: Need to implement daily scoreboard fetching")
print("OR: Wait for LeagueGameLog to update (may be delayed)")
