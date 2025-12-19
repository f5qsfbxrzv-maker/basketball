"""
Update all data to current date
- Fetch missing games
- Update ELO ratings
- Update injuries
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import sqlite3
from datetime import datetime, timedelta
from config.settings import DB_PATH

print("="*80)
print("FULL DATA UPDATE")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Step 1: Fetch game logs
print("\n[1/4] Updating game logs...")
try:
    from src.collectors.update_game_logs import update_game_logs
    update_game_logs(season='2024-25', last_n_days=None)
    print("[OK] Game logs updated")
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

# Step 2: Parse games from NBA API format and update ELO
print("\n[2/4] Updating ELO ratings...")
try:
    from src.features.off_def_elo_system import OffDefEloSystem
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Get last ELO update
    cursor.execute("SELECT MAX(game_date) FROM elo_ratings")
    last_elo = cursor.fetchone()[0]
    print(f"  Last ELO update: {last_elo}")
    
    if last_elo:
        start_date = datetime.strptime(last_elo, '%Y-%m-%d') + timedelta(days=1)
    else:
        start_date = datetime(2024, 10, 1)
    
    print(f"  Updating from: {start_date.strftime('%Y-%m-%d')}")
    
    # Get all teams
    cursor.execute("SELECT DISTINCT TEAM_ABBREVIATION FROM game_logs")
    teams = [r[0] for r in cursor.fetchall()]
    print(f"  Found {len(teams)} teams")
    
    # Initialize ELO system
    elo_system = OffDefEloSystem(db_path=str(DB_PATH))
    elo_system.initialize_season('2024-25', teams)
    
    # Get games to process (need to reconstruct full games from team rows)
    cursor.execute("""
        SELECT GAME_ID, GAME_DATE, TEAM_ABBREVIATION, MATCHUP, PTS, WL
        FROM game_logs
        WHERE GAME_DATE >= ?
        ORDER BY GAME_DATE, GAME_ID
    """, (start_date.strftime('%Y-%m-%d'),))
    
    rows = cursor.fetchall()
    
    # Group by game_id to reconstruct full games
    games_by_id = {}
    for game_id, date, team, matchup, pts, wl in rows:
        if game_id not in games_by_id:
            games_by_id[game_id] = {'date': date, 'teams': []}
        games_by_id[game_id]['teams'].append({
            'team': team,
            'matchup': matchup,
            'pts': pts,
            'wl': wl
        })
    
    # Process only completed games (2 teams)
    completed_games = []
    for game_id, data in games_by_id.items():
        if len(data['teams']) == 2:
            team1, team2 = data['teams']
            
            # Determine home/away from matchup (@ means away)
            if '@' in team1['matchup']:
                away_team, home_team = team1['team'], team2['team']
                away_score, home_score = team1['pts'], team2['pts']
            else:
                home_team, away_team = team1['team'], team2['team']
                home_score, away_score = team1['pts'], team2['pts']
            
            completed_games.append({
                'date': data['date'],
                'home': home_team,
                'away': away_team,
                'home_score': home_score,
                'away_score': away_score
            })
    
    print(f"  Found {len(completed_games)} completed games to process")
    
    # Process each game
    processed = 0
    errors = 0
    
    for i, game in enumerate(completed_games):
        if i % 50 == 0 and i > 0:
            print(f"    Progress: {i}/{len(completed_games)}")
        
        try:
            elo_system.update_game(
                season='2024-25',
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
                print(f"    Error: {game['date']} {game['away']}@{game['home']}: {e}")
    
    conn.close()
    
    print(f"[OK] Processed {processed} games ({errors} errors)")
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

# Step 3: Update injuries
print("\n[3/4] Updating injuries...")
try:
    from src.services.live_injury_updater import LiveInjuryUpdater
    
    updater = LiveInjuryUpdater(db_path=str(DB_PATH))
    injury_count = updater.update_active_injuries()
    print(f"[OK] Updated {injury_count} injuries")
except Exception as e:
    print(f"[ERROR] {e}")

# Step 4: Verify
print("\n[4/4] Verifying data freshness...")
try:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute("SELECT MAX(GAME_DATE) FROM game_logs")
    latest_game = cursor.fetchone()[0]
    
    cursor.execute("SELECT MAX(game_date) FROM elo_ratings")
    latest_elo = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM active_injuries")
    injury_count = cursor.fetchone()[0]
    
    print(f"\nLatest game: {latest_game}")
    print(f"Latest ELO:  {latest_elo}")
    print(f"Active injuries: {injury_count}")
    
    # Check for default 1500 values
    cursor.execute("""
        SELECT COUNT(DISTINCT team) 
        FROM elo_ratings 
        WHERE game_date = ? 
          AND (off_elo BETWEEN 1499 AND 1501 OR def_elo BETWEEN 1499 AND 1501)
    """, (latest_elo,))
    
    default_teams = cursor.fetchone()[0]
    
    if default_teams > 0:
        print(f"\n[WARNING] {default_teams} teams at default 1500 ELO")
        
        # Show which teams
        cursor.execute("""
            SELECT team, off_elo, def_elo, composite_elo
            FROM elo_ratings
            WHERE game_date = ?
              AND (off_elo BETWEEN 1499 AND 1501 OR def_elo BETWEEN 1499 AND 1501)
        """, (latest_elo,))
        
        print("  Teams with default ELO:")
        for team, off, def_, comp in cursor.fetchall():
            print(f"    {team}: off={off:.1f}, def={def_:.1f}, comp={comp:.1f}")
    else:
        print("\n[OK] No teams at default 1500 ELO")
    
    # Show sample of current ELO
    cursor.execute("""
        SELECT team, off_elo, def_elo, composite_elo
        FROM elo_ratings
        WHERE game_date = ?
        ORDER BY composite_elo DESC
        LIMIT 5
    """, (latest_elo,))
    
    print("\nTop 5 teams by ELO:")
    for team, off, def_, comp in cursor.fetchall():
        print(f"  {team}: off={off:.1f}, def={def_:.1f}, comp={comp:.1f}")
    
    conn.close()
    
except Exception as e:
    print(f"[ERROR] {e}")

print("\n" + "="*80)
print("UPDATE COMPLETE")
print("="*80)
print("\nNext steps:")
print("  1. Delete predictions_cache.json")
print("  2. Restart dashboard")
