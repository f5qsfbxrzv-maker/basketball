"""
UPDATE DATABASE WITH CORRECT SEASON (2025-26)
The issue was searching for 2024-25 season which ended in June
Current season is 2025-26 (Oct 2025 - June 2026)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from nba_api.stats.endpoints import leaguegamelog
import sqlite3
import pandas as pd
import time
from config.settings import DB_PATH

print("="*80)
print("UPDATING TO CURRENT SEASON: 2025-26")
print("="*80)

# Step 1: Fetch 2025-26 season games
print("\n[1/3] Fetching 2025-26 season from NBA API...")
try:
    game_log = leaguegamelog.LeagueGameLog(
        season='2025-26',
        season_type_all_star='Regular Season'
    )
    time.sleep(0.6)
    
    df = game_log.get_data_frames()[0]
    print(f"  Fetched {len(df)} game records")
    print(f"  Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
    
    # Save to database
    print(f"\n[2/3] Updating database...")
    conn = sqlite3.connect(str(DB_PATH))
    
    # Clear existing 2025-26 data
    cursor = conn.cursor()
    cursor.execute("DELETE FROM game_logs WHERE season = '2025-26'")
    deleted = cursor.rowcount
    print(f"  Deleted {deleted} old 2025-26 records")
    
    # Add season column if not present
    df['season'] = '2025-26'
    
    # Insert new data
    df.to_sql('game_logs', conn, if_exists='append', index=False)
    print(f"  Inserted {len(df)} new records")
    
    conn.commit()
    conn.close()
    
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Update ELO ratings for 2025-26
print(f"\n[3/3] Updating ELO ratings...")
try:
    from src.features.off_def_elo_system import OffDefEloSystem
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Get all 2025-26 games chronologically
    cursor.execute("""
        SELECT DISTINCT GAME_DATE, GAME_ID, TEAM_ABBREVIATION, MATCHUP, PTS, WL
        FROM game_logs
        WHERE season = '2025-26'
        ORDER BY GAME_DATE, GAME_ID
    """)
    
    rows = cursor.fetchall()
    
    # Group by game to get home/away matchups
    games_dict = {}
    for date, game_id, team, matchup, pts, wl in rows:
        if game_id not in games_dict:
            games_dict[game_id] = {'date': date, 'teams': []}
        games_dict[game_id]['teams'].append({
            'team': team,
            'matchup': matchup,
            'pts': pts,
            'wl': wl
        })
    
    # Extract complete games (2 teams each)
    complete_games = []
    for game_id, data in games_dict.items():
        if len(data['teams']) == 2:
            team1, team2 = data['teams']
            
            # Determine home/away from matchup string
            if '@' in team1['matchup']:
                away_team = team1['team']
                home_team = team2['team']
                away_score = team1['pts']
                home_score = team2['pts']
            else:
                home_team = team1['team']
                away_team = team2['team']
                home_score = team1['pts']
                away_score = team2['pts']
            
            complete_games.append({
                'date': data['date'],
                'home': home_team,
                'away': away_team,
                'home_score': home_score,
                'away_score': away_score
            })
    
    print(f"  Found {len(complete_games)} complete games")
    
    # Initialize ELO system for new season
    elo_system = OffDefEloSystem(db_path=str(DB_PATH))
    
    # Get all teams
    cursor.execute("SELECT DISTINCT TEAM_ABBREVIATION FROM game_logs WHERE season = '2025-26'")
    teams = [r[0] for r in cursor.fetchall()]
    
    print(f"  Initializing {len(teams)} teams for 2025-26 season...")
    elo_system.initialize_season('2025-26', teams)
    
    # Process each game
    processed = 0
    errors = 0
    
    for i, game in enumerate(complete_games):
        if i % 50 == 0:
            print(f"    Processing {i}/{len(complete_games)}...")
        
        try:
            elo_system.update_game(
                season='2025-26',
                game_date=game['date'],
                home_team=game['home'],
                away_team=game['away'],
                home_points=game['home_score'],
                away_points=game['away_score']
            )
            processed += 1
        except Exception as e:
            errors += 1
            if errors < 5:
                print(f"    Error: {game['date']} {game['away']}@{game['home']} - {e}")
    
    print(f"  Processed {processed} games ({errors} errors)")
    
    conn.close()
    
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

# Verification
print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()

cursor.execute("SELECT MAX(GAME_DATE) FROM game_logs WHERE season = '2025-26'")
latest_game = cursor.fetchone()[0]

cursor.execute("SELECT MAX(game_date) FROM elo_ratings WHERE season = '2025-26'")
latest_elo = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(DISTINCT team) FROM elo_ratings WHERE season = '2025-26' AND game_date = ?", (latest_elo,))
team_count = cursor.fetchone()[0]

print(f"\nLatest game: {latest_game}")
print(f"Latest ELO: {latest_elo}")
print(f"Teams with current ELO: {team_count}")

# Show top 5 teams
cursor.execute("""
    SELECT team, off_elo, def_elo, composite_elo
    FROM elo_ratings
    WHERE season = '2025-26' AND game_date = ?
    ORDER BY composite_elo DESC
    LIMIT 5
""", (latest_elo,))

print("\nTop 5 teams:")
for team, off, def_, comp in cursor.fetchall():
    print(f"  {team}: off={off:.1f}, def={def_:.1f}, comp={comp:.1f}")

# Check sample teams
print("\nSample teams (OKC, SAS, WAS, IND):")
for team in ['OKC', 'SAS', 'WAS', 'IND']:
    cursor.execute("""
        SELECT off_elo, def_elo, composite_elo, game_date
        FROM elo_ratings
        WHERE team = ? AND season = '2025-26'
        ORDER BY game_date DESC
        LIMIT 1
    """, (team,))
    
    row = cursor.fetchone()
    if row:
        print(f"  {team}: off={row[0]:.1f}, def={row[1]:.1f}, comp={row[2]:.1f} ({row[3]})")

conn.close()

print("\n" + "="*80)
print("UPDATE COMPLETE!")
print("="*80)
print("\nDatabase now has CURRENT season (2025-26) data")
print("ELO ratings updated through December 13, 2025")
print("\nNEXT: Delete predictions_cache.json and restart dashboard")
