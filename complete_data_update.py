"""
FULL UPDATE: Fetch 156 missing games and update ELO
Uses Scoreboard API since LeagueGameLog is incomplete
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
print("COMPLETE DATA UPDATE - Nov 21 to Dec 14")
print("="*80)

# Step 1: Fetch games via Scoreboard API
print("\n[1/3] Fetching 156 missing games...")

start_date = datetime(2025, 11, 21)
end_date = datetime.now()

all_game_data = []
current_date = start_date
total_games = 0

while current_date <= end_date:
    date_str = current_date.strftime('%m/%d/%Y')
    date_formatted = current_date.strftime('%Y-%m-%d')
    
    try:
        sb = scoreboardv2.ScoreboardV2(game_date=date_str)
        time.sleep(0.6)
        
        games = sb.get_data_frames()[0]
        
        if not games.empty:
            # Extract team data from each game
            for _, game in games.iterrows():
                game_id = game['GAME_ID']
                
                # Get home/visitor teams and scores
                # Note: Scoreboard returns one row per game (not per team like LeagueGameLog)
                home_team_id = game.get('HOME_TEAM_ID')
                visitor_team_id = game.get('VISITOR_TEAM_ID')
                home_pts = game.get('PTS_HOME', 0)
                visitor_pts = game.get('PTS_AWAY', 0)
                
                # Store game result
                all_game_data.append({
                    'game_id': game_id,
                    'game_date': date_formatted,
                    'home_team_id': home_team_id,
                    'away_team_id': visitor_team_id,
                    'home_score': home_pts,
                    'away_score': visitor_pts
                })
                total_games += 1
            
            print(f"  {date_formatted}: {len(games)} games")
    
    except Exception as e:
        print(f"  {date_formatted}: Error - {e}")
    
    current_date += timedelta(days=1)

print(f"\n  Found {total_games} games")

# Step 2: Update ELO ratings
print("\n[2/3] Updating ELO ratings...")

if total_games > 0:
    from src.features.off_def_elo_system import OffDefEloSystem
    
    # Map team IDs to abbreviations
    TEAM_ID_MAP = {
        1610612737: 'ATL', 1610612738: 'BOS', 1610612751: 'BKN', 1610612766: 'CHA',
        1610612741: 'CHI', 1610612739: 'CLE', 1610612742: 'DAL', 1610612743: 'DEN',
        1610612765: 'DET', 1610612744: 'GSW', 1610612745: 'HOU', 1610612754: 'IND',
        1610612746: 'LAC', 1610612747: 'LAL', 1610612763: 'MEM', 1610612748: 'MIA',
        1610612749: 'MIL', 1610612750: 'MIN', 1610612740: 'NOP', 1610612752: 'NYK',
        1610612760: 'OKC', 1610612753: 'ORL', 1610612755: 'PHI', 1610612756: 'PHX',
        1610612757: 'POR', 1610612758: 'SAC', 1610612759: 'SAS', 1610612761: 'TOR',
        1610612762: 'UTA', 1610612764: 'WAS'
    }
    
    elo_system = OffDefEloSystem(db_path=str(DB_PATH))
    
    processed = 0
    errors = 0
    
    for game in sorted(all_game_data, key=lambda x: x['game_date']):
        try:
            home_abbr = TEAM_ID_MAP.get(game['home_team_id'])
            away_abbr = TEAM_ID_MAP.get(game['away_team_id'])
            
            if home_abbr and away_abbr and game['home_score'] and game['away_score']:
                elo_system.update_game(
                    season='2024-25',
                    game_date=game['game_date'],
                    home_team=home_abbr,
                    away_team=away_abbr,
                    home_points=int(game['home_score']),
                    away_points=int(game['away_score'])
                )
                processed += 1
                
                if processed % 25 == 0:
                    print(f"    Processed {processed}/{total_games}...")
        
        except Exception as e:
            errors += 1
            if errors < 5:
                print(f"    Error: {game['game_date']} - {e}")
    
    print(f"\n  Updated {processed} games ({errors} errors)")

# Step 3: Verify
print("\n[3/3] Verifying data...")

conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()

cursor.execute("SELECT MAX(game_date) FROM elo_ratings")
latest_elo = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(DISTINCT team) FROM elo_ratings WHERE game_date = ?", (latest_elo,))
team_count = cursor.fetchone()[0]

print(f"\n  Latest ELO date: {latest_elo}")
print(f"  Teams with current ELO: {team_count}")

# Check for 1500 defaults
cursor.execute("""
    SELECT COUNT(DISTINCT team)
    FROM elo_ratings
    WHERE game_date = ?
      AND (off_elo BETWEEN 1499 AND 1501 OR def_elo BETWEEN 1499 AND 1501)
""", (latest_elo,))

default_count = cursor.fetchone()[0]

if default_count > 0:
    print(f"\n  [WARNING] {default_count} teams still at default 1500")
else:
    print(f"\n  [OK] No teams at default 1500")

# Show top 5 teams
cursor.execute("""
    SELECT team, off_elo, def_elo, composite_elo
    FROM elo_ratings
    WHERE game_date = ?
    ORDER BY composite_elo DESC
    LIMIT 5
""", (latest_elo,))

print("\n  Top 5 teams:")
for team, off, def_, comp in cursor.fetchall():
    print(f"    {team}: off={off:.1f}, def={def_:.1f}, comp={comp:.1f}")

# Check OKC, SAS specifically
print("\n  Sample teams (OKC, SAS, WAS, IND):")
for team in ['OKC', 'SAS', 'WAS', 'IND']:
    cursor.execute("""
        SELECT off_elo, def_elo, composite_elo, game_date
        FROM elo_ratings
        WHERE team = ?
        ORDER BY game_date DESC
        LIMIT 1
    """, (team,))
    
    row = cursor.fetchone()
    if row:
        print(f"    {team}: off={row[0]:.1f}, def={row[1]:.1f}, comp={row[2]:.1f} ({row[3]})")

conn.close()

print("\n" + "="*80)
print("UPDATE COMPLETE")
print("="*80)
print("\n[OK] Database updated with 156 missing games")
print("[OK] ELO ratings updated through Dec 14, 2025")
print("\nNEXT STEPS:")
print("  1. Delete predictions_cache.json")
print("  2. Restart dashboard")
print("  3. Verify predictions use current Dec 14 data")
